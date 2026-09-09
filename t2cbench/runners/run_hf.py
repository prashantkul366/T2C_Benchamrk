"""Generation runner for every HuggingFace causal LM in the benchmark.

Covers four of the six systems with one code path, which is the point -- a
shared runner means the sampling budget, the batching and the decoding are
provably identical across models rather than identical by inspection:

  * CADmium-7B          LoRA on Qwen2.5-Coder-7B-Instruct   -> minimal JSON
  * CADFusion v1.1      LoRA on Meta-Llama-3-8B             -> SkexGen tokens
  * Text-to-CadQuery    full fine-tunes (ricemonster/*)     -> CadQuery
  * general LLMs        Qwen / Llama / Mistral / DeepSeek   -> CadQuery

    python -m t2cbench.runners.run_hf \
        --model chandar-lab/CADmium-7B --base Qwen/Qwen2.5-Coder-7B-Instruct \
        --template native.cadmium --name cadmium-7b \
        --split data/split_a.jsonl --out results/raw/cadmium-7b_splitA.jsonl

Resumable: an existing output file is read back and finished sample_ids are
skipped, so a Colab session that times out costs only the unfinished tail.
"""

from __future__ import annotations

import argparse
import json
import os

import yaml
from tqdm import tqdm

# torch / transformers are imported lazily inside the functions that need them:
# the evaluation host is CPU-only and should not have to install a GPU stack to
# read a config or print --help.


def load_prompts_cfg(path: str = "configs/prompts.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_template(cfg: dict, key: str) -> dict:
    """Return {"system": str | None, "user": str} for a dotted template key.

    A template is either a bare string (user turn only) or a mapping with
    `system` and `user`. CADmium was fine-tuned with a system role carrying its
    JSON schema; dropping it puts the model off its training distribution, so
    the runner has to be able to send one.
    """
    node = cfg
    for part in key.split("."):
        node = node[part]
    if isinstance(node, str):
        return {"system": None, "user": node}
    if isinstance(node, dict) and "user" in node:
        return {"system": node.get("system"), "user": node["user"]}
    raise KeyError(f"template {key!r} must be a string or a mapping with a 'user' key")


def build_model(model_id: str, base: str | None, dtype: str, load_4bit: bool,
                subfolder: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                   "fp32": torch.float32}[dtype]
    kwargs = dict(torch_dtype=torch_dtype, device_map="auto")
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        kwargs.pop("torch_dtype")

    if base:
        # LoRA adapter on top of a separately-hosted base (CADmium, CADFusion).
        from peft import PeftModel
        print(f"loading base {base}")
        model = AutoModelForCausalLM.from_pretrained(base, **kwargs)

        # CADFusion added a [PAD] token and resized the embeddings before training,
        # and saved the resized embed_tokens/lm_head into the adapter -- so loading
        # it onto an unresized base dies with
        #   size mismatch ... checkpoint 128258 vs current model 128256.
        # Read the width the checkpoint expects and match it, rather than assuming.
        n_vocab = _adapter_vocab_size(model_id, subfolder)
        cur = model.get_input_embeddings().weight.shape[0]
        if n_vocab and n_vocab != cur:
            print(f"resizing embeddings {cur} -> {n_vocab} to match the adapter")
            model.resize_token_embeddings(n_vocab)

        print(f"applying adapter {model_id}")
        # CADFusion publishes its adapters under v1_0/ and v1_1/ rather than at
        # the repo root, so the subfolder is not optional there -- without it
        # peft looks for a nonexistent root adapter_config.json and fails.
        model = PeftModel.from_pretrained(model, model_id, subfolder=subfolder)
        model = model.merge_and_unload()
        tok_src = model_id
    else:
        print(f"loading {model_id}")
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        tok_src = model_id

    try:
        tok = AutoTokenizer.from_pretrained(tok_src, padding_side="left")
    except Exception:
        tok = AutoTokenizer.from_pretrained(base or model_id, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    return model, tok


def _adapter_vocab_size(model_id: str, subfolder: str | None) -> int | None:
    """Vocabulary width an adapter's saved embedding expects, or None.

    Peft adapters that include `embed_tokens`/`lm_head` (because the model was
    resized for new special tokens) can only be loaded onto a base of the same
    width. Reading it from the checkpoint header is cheap and avoids hard-coding
    a number that would silently rot if the adapter were republished.
    """
    try:
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
    except Exception:
        return None
    name = "adapter_model.safetensors"
    path = f"{subfolder}/{name}" if subfolder else name
    try:
        local = hf_hub_download(model_id, path)
    except Exception:
        try:
            local = hf_hub_download(model_id, path.replace(".safetensors", ".bin"))
        except Exception:
            return None
    if not local.endswith(".safetensors"):
        return None
    try:
        with safe_open(local, framework="pt") as f:
            for k in f.keys():
                if k.endswith(("embed_tokens.weight", "lm_head.weight")):
                    return int(f.get_slice(k).get_shape()[0])
    except Exception:
        return None
    return None


def format_prompt(tok, template: dict, text: str, use_chat: bool) -> str:
    body = template["user"].format(prompt=text)
    system = template.get("system")
    if use_chat and getattr(tok, "chat_template", None):
        messages = ([{"role": "system", "content": system}] if system else []) \
            + [{"role": "user", "content": body}]
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    # No chat template: a system turn has nowhere to go, so it is prepended.
    return f"{system}\n\n{body}" if system else body


def generate(model, tok, prompts: list[str], gen_cfg: dict, max_new_tokens: int) -> list[list[str]]:
    """Returns, per input prompt, a list of `num_return_sequences` completions."""
    import torch
    with torch.inference_mode():
        return _generate(model, tok, prompts, gen_cfg, max_new_tokens)


def _generate(model, tok, prompts, gen_cfg, max_new_tokens):
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
              max_length=4096).to(model.device)
    n_ret = gen_cfg.get("num_return_sequences", 1)

    kwargs = dict(max_new_tokens=max_new_tokens,
                  num_return_sequences=n_ret,
                  pad_token_id=tok.pad_token_id,
                  do_sample=gen_cfg.get("do_sample", False))
    if kwargs["do_sample"]:
        kwargs.update(temperature=gen_cfg.get("temperature", 0.7),
                      top_p=gen_cfg.get("top_p", 0.95))

    out = model.generate(**enc, **kwargs)
    # Strip the prompt: everything before input length is the echoed prompt.
    trimmed = out[:, enc["input_ids"].shape[1]:]
    texts = tok.batch_decode(trimmed, skip_special_tokens=True)
    return [texts[i * n_ret:(i + 1) * n_ret] for i in range(len(prompts))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--base", default=None, help="base model when --model is a LoRA adapter")
    ap.add_argument("--subfolder", default=None,
                    help="subfolder holding the adapter inside --model (CADFusion uses v1_1)")
    ap.add_argument("--name", required=True, help="short name used in every results table")
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--template", default="general_one_shot",
                    help="dotted key into configs/prompts.yaml, e.g. native.cadmium")
    ap.add_argument("--prompts-cfg", default="configs/prompts.yaml")
    ap.add_argument("--mode", choices=["pass_at_1", "best_of_k"], default="pass_at_1")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--no-chat-template", action="store_true",
                    help="set for base (non-instruct) models and for the native fine-tunes, "
                         "which were trained on raw text without a chat wrapper")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="smoke-test on the first N prompts")
    ap.add_argument("--save-prompt", action="store_true",
                    help="record the exact formatted prompt in the output, so the template "
                         "that was actually applied can be audited rather than assumed")
    args = ap.parse_args()

    cfg = load_prompts_cfg(args.prompts_cfg)
    template = resolve_template(cfg, args.template)
    gen_cfg = cfg["generation"][args.mode]
    max_new = args.max_new_tokens or cfg["generation"]["max_new_tokens"].get(
        args.name.split("-")[0], cfg["generation"]["max_new_tokens"]["default"])

    rows = [json.loads(l) for l in open(args.split) if l.strip()]
    if args.limit:
        rows = rows[:args.limit]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["sample_id"])
                except Exception:
                    pass
        print(f"resuming: {len(done)} sample_ids already generated")
    todo = [r for r in rows if r["sample_id"] not in done]
    print(f"{len(todo)} prompts to generate ({args.mode}, max_new_tokens={max_new})")
    if not todo:
        return

    import torch
    if torch.cuda.is_available():
        torch.manual_seed(gen_cfg.get("seed", 0))
    model, tok = build_model(args.model, args.base, args.dtype, args.load_4bit,
                             subfolder=args.subfolder)

    with open(args.out, "a") as fout:
        for i in tqdm(range(0, len(todo), args.batch_size), desc=args.name):
            batch = todo[i:i + args.batch_size]
            prompts = [format_prompt(tok, template, r["prompt"], not args.no_chat_template)
                       for r in batch]
            try:
                outs = generate(model, tok, prompts, gen_cfg, max_new)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print("OOM -- falling back to batch size 1 for this batch")
                outs = []
                for p in prompts:
                    outs += generate(model, tok, [p], gen_cfg, max_new)

            for r, prompt_str, completions in zip(batch, prompts, outs):
                for k, text in enumerate(completions):
                    rec = {
                        "sample_id": r["sample_id"],
                        "sample_idx": k,
                        "model": args.name,
                        "output": text,
                    }
                    if args.save_prompt:
                        rec["prompt_sent"] = prompt_str
                    fout.write(json.dumps(rec) + "\n")
            fout.flush()

    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
