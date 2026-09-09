"""Generation runner for Text2CAD v1.0.

Text2CAD is not a HuggingFace causal LM -- it is a BERT encoder feeding a custom
CAD-sequence decoder, driven through `model.test_decode(...)`. This wrapper
loads it exactly as `Cad_VLM/test_user_input.py` does and emits the predicted
`cad_vec` as JSON, so the CadVecAdapter can score it alongside everything else.

    python -m t2cbench.runners.run_text2cad \
        --text2cad-repo /content/Text2CAD \
        --checkpoint /content/weights/Text2CAD_1.0.pth \
        --split data/split_a.jsonl --out results/raw/text2cad_splitA.jsonl

Sampling note: `sampling_type: max` gives topk_index=1, which is this model's
greedy decode and therefore its pass@1. `--n-samples 5` reproduces the paper's
best-of-5 by sweeping topk_index 1..5, matching Cad_VLM/test.py.

The checkpoint lives in the gated SadilKhan/Text2CAD dataset repo -- accept the
licence on the HF page and export HF_TOKEN before downloading. It is the only
gated asset the benchmark cannot route around.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from tqdm import tqdm

# torch imported lazily -- see run_hf.py

DEFAULT_CONFIG = {
    "text_encoder": {
        "text_embedder": {"model_name": "bert_large_uncased", "max_seq_len": 512,
                          "cache_dir": None},
        "adaptive_layer": {"in_dim": 1024, "out_dim": 1024, "num_heads": 8, "dropout": 0.1},
    },
    "cad_decoder": {"tdim": 1024, "cdim": 256, "num_layers": 8, "num_heads": 8,
                    "dropout": 0.1, "ca_level_start": 2},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--text2cad-repo", required=True,
                    help="checkout of https://github.com/SadilKhan/Text2CAD")
    ap.add_argument("--checkpoint", required=True, help="Text2CAD_1.0.pth")
    ap.add_argument("--name", default="text2cad")
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir", default=None, help="HF cache for bert-large-uncased")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-samples", type=int, default=1,
                    help="1 = greedy pass@1; 5 = the paper's best-of-5 topk sweep")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import torch
    repo = os.path.abspath(args.text2cad_repo)
    for p in (repo, os.path.join(repo, "Cad_VLM")):
        if p not in sys.path:
            sys.path.insert(0, p)

    from CadSeqProc.utility.macro import MAX_CAD_SEQUENCE_LENGTH
    from Cad_VLM.models.text2cad import Text2CAD

    cfg = DEFAULT_CONFIG
    if args.cache_dir:
        cfg["text_encoder"]["text_embedder"]["cache_dir"] = args.cache_dir
    cad_cfg = dict(cfg["cad_decoder"])
    cad_cfg["cad_seq_len"] = MAX_CAD_SEQUENCE_LENGTH

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Text2CAD(text_config=cfg["text_encoder"], cad_config=cad_cfg).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = {}
    for k, v in ckpt["model_state_dict"].items():
        state[".".join(k.split(".")[1:]) if k.split(".")[0] == "module" else k] = v
    model.load_state_dict(state, strict=False)
    if "epoch" in ckpt:
        print(f"checkpoint trained for {ckpt['epoch']} epochs")
    model.eval()

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

    with open(args.out, "a") as fout, torch.no_grad():
        for i in tqdm(range(0, len(todo), args.batch_size), desc=args.name):
            batch = todo[i:i + args.batch_size]
            texts = [r["prompt"] for r in batch]
            for k in range(args.n_samples):
                pred = model.test_decode(
                    texts=texts,
                    maxlen=MAX_CAD_SEQUENCE_LENGTH,
                    nucleus_prob=0,
                    topk_index=k + 1,   # topk_index=1 is greedy
                    device=device,
                )
                vecs = pred["cad_vec"].cpu().numpy()
                for j, r in enumerate(batch):
                    fout.write(json.dumps({
                        "sample_id": r["sample_id"],
                        "sample_idx": k,
                        "model": args.name,
                        # serialised so CadVecAdapter has the same string-in
                        # interface as every other adapter
                        "output": json.dumps(vecs[j].astype(int).tolist()),
                    }) + "\n")
            fout.flush()

    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
