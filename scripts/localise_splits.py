"""Rewrite a split file's ground-truth paths for the machine doing the scoring.

    python scripts/localise_splits.py --split data/split_a.jsonl --out data/split_a.local.jsonl
    python scripts/localise_splits.py --split data/split_b.jsonl --out data/split_b.local.jsonl \
        --cadprompt-dir /path/to/CAD_Code_Generation

Generation and scoring happen on different machines -- a GPU box for the models,
a pythonocc box for the geometry -- and `gt_mesh` is an absolute path baked in
when the split was built. It never survives the trip: split B points into
whatever CADPrompt clone the generating machine had, and split A has been seen
carrying a bare `GT_ROOT/...` placeholder from an older builder. Either way
`t2cbench.evaluate` then fails to open a single reference.

The uid is the durable key, so paths are re-derived from it rather than
patched: split A from the DeepCAD test mesh snapshot on HuggingFace, split B
from a local CADPrompt checkout. Nothing else in the row is touched, and the
script refuses to write a file it could not fully resolve -- a split that is
quietly 3% unresolvable would show up as a model's invalidity, not as a missing
file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

DEEPCAD_MESH_REPO = "maksimko123/deepcad_test_mesh"


def find_cadprompt_root(start: str | None) -> str | None:
    """CADPrompt clones nest inconsistently; find the directory of uid dirs."""
    if not start or not os.path.isdir(start):
        return None
    for root, dirs, _ in os.walk(os.path.abspath(start)):
        hits = [d for d in dirs if d.isdigit() and len(d) == 8
                and os.path.exists(os.path.join(root, d, "Ground_Truth.stl"))]
        if len(hits) >= 10:
            return root
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cadprompt-dir", default=None, help="needed for split B")
    ap.add_argument("--cache", default=None, help="HuggingFace cache dir")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.split) if l.strip()]
    if not rows:
        sys.exit(f"{args.split} is empty")
    split = rows[0].get("split", "A")
    uids = sorted({r["uid"] for r in rows})
    print(f"{len(rows)} rows, {len(uids)} uids, split {split}")

    resolved: dict[str, str] = {}
    if split == "A":
        from huggingface_hub import hf_hub_download
        for i, uid in enumerate(uids, 1):
            try:
                resolved[uid] = hf_hub_download(
                    DEEPCAD_MESH_REPO, f"{uid}.stl", repo_type="dataset",
                    cache_dir=args.cache)
            except Exception as e:
                print(f"  !! {uid}: {type(e).__name__}")
            if i % 25 == 0 or i == len(uids):
                print(f"  fetched {i}/{len(uids)}", flush=True)
    else:
        root = find_cadprompt_root(args.cadprompt_dir)
        if not root:
            sys.exit("--cadprompt-dir must point at a CAD_Code_Generation checkout "
                     "(clone https://github.com/Kamel773/CAD_Code_Generation)")
        print(f"  CADPrompt root: {root}")
        for uid in uids:
            p = os.path.join(root, uid, "Ground_Truth.stl")
            if os.path.exists(p):
                resolved[uid] = p

    missing = [u for u in uids if u not in resolved]
    if missing:
        sys.exit(f"could not resolve {len(missing)}/{len(uids)} uids, e.g. "
                 f"{missing[:5]} -- refusing to write a partial split, because the "
                 f"gap would surface later as model invalidity rather than as a "
                 f"missing file")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(dict(r, gt_mesh=resolved[r["uid"]])) + "\n")
    print(f"all {len(uids)} uids resolved -> {args.out}")


if __name__ == "__main__":
    main()
