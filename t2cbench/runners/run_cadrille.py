"""Generation runner for cadrille (text mode).

cadrille is a Qwen2-VL derivative with a custom `Cadrille` class and a collate
function that builds its own chat format, so it cannot share run_hf.py without
reimplementing its prompt construction -- which would silently change what the
model sees. This wrapper imports cadrille's own code and drives it directly.

    python -m t2cbench.runners.run_cadrille \
        --cadrille-repo /content/cadrille \
        --checkpoint maksimko123/cadrille \
        --split data/split_a.jsonl --out results/raw/cadrille_splitA.jsonl

Output is CadQuery source with the result in variable `r`; score it with
`--adapter cadquery`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from tqdm import tqdm

# torch imported lazily -- see run_hf.py


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cadrille-repo", required=True,
                    help="checkout of https://github.com/col14m/cadrille")
    ap.add_argument("--checkpoint", default="maksimko123/cadrille",
                    help="maksimko123/cadrille (SFT) or maksimko123/cadrille-rl (RL)")
    ap.add_argument("--name", default="cadrille")
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--n-points", type=int, default=256,
                    help="point-cloud width cadrille expects; 256 matches its test.py. "
                         "Unused in text mode beyond shaping the zero tensor.")
    ap.add_argument("--n-samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import torch
    repo = os.path.abspath(args.cadrille_repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from cadrille import Cadrille
    from transformers import AutoProcessor

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
        print(f"resuming: {len(done)} already generated")
    todo = [r for r in rows if r["sample_id"] not in done]
    print(f"{len(todo)} prompts to generate")
    if not todo:
        return

    model = Cadrille.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if _has_flash_attn() else "sdpa",
        device_map="auto")
    model.eval()

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28,
        padding_side="left")

    torch.manual_seed(0)
    with open(args.out, "a") as fout:
        for i in tqdm(range(0, len(todo), args.batch_size), desc=args.name):
            batch = todo[i:i + args.batch_size]
            texts = [_chat(processor, r["prompt"]) for r in batch]
            # Match cadrille's own collate(): no truncation, so a long L3 prompt
            # is never silently clipped.
            enc = processor(text=texts, images=None, videos=None,
                            padding=True, return_tensors="pt").to(model.device)

            # Cadrille.forward calls is_pc.sum() and is_img.sum() unconditionally,
            # so these are required even in pure text mode -- passing None raises
            # AttributeError on the first batch. Zeros mean "no point cloud, no
            # image", which is exactly what text mode is.
            n = len(batch)
            point_clouds = torch.zeros(n, args.n_points, 3, device=model.device)
            is_pc = torch.zeros(n, dtype=torch.bool, device=model.device)
            is_img = torch.zeros(n, dtype=torch.bool, device=model.device)

            kwargs = dict(max_new_tokens=args.max_new_tokens,
                          num_return_sequences=args.n_samples,
                          pad_token_id=processor.tokenizer.pad_token_id)
            if args.temperature > 0:
                kwargs.update(do_sample=True, temperature=args.temperature, top_p=0.95)
            else:
                kwargs.update(do_sample=False)

            with torch.inference_mode():
                out = model.generate(input_ids=enc["input_ids"],
                                     attention_mask=enc["attention_mask"],
                                     point_clouds=point_clouds,
                                     is_pc=is_pc, is_img=is_img, **kwargs)
            trimmed = out[:, enc["input_ids"].shape[1]:]
            decoded = processor.batch_decode(trimmed, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)

            for j, r in enumerate(batch):
                for k in range(args.n_samples):
                    fout.write(json.dumps({
                        "sample_id": r["sample_id"],
                        "sample_idx": k,
                        "model": args.name,
                        "output": decoded[j * args.n_samples + k],
                    }) + "\n")
            fout.flush()

    print(f"wrote -> {args.out}")


def _chat(processor, prompt: str) -> str:
    """cadrille's text-mode chat format, matching its collate() in cadrille.py."""
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _has_flash_attn() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False


if __name__ == "__main__":
    main()
