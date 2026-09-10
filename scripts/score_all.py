"""Score every raw prediction file, picking each model's adapter from the registry.

    python scripts/score_all.py --work results --data data
"""
import argparse, glob, os, re, subprocess, sys
import yaml


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", default="results")
    ap.add_argument("--data", default="data")
    ap.add_argument("--models", default="configs/models.yaml")
    ap.add_argument("--workers", type=int, default=8)
    # 60 s, matching t2cbench.evaluate's own default. This used to hard-code 20,
    # which silently overrode it: on a box running 8 scoring workers, 12 of 18
    # known-good samples "failed" purely on contention, and a TIMEOUT is counted
    # against the model.
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--force", action="store_true", help="rescore files already scored")
    args = ap.parse_args()

    registry = yaml.safe_load(open(args.models))
    raws = sorted(glob.glob(os.path.join(args.work, "raw", "*.jsonl")))
    if not raws:
        sys.exit(f"no prediction files in {args.work}/raw")

    for raw in raws:
        stem = os.path.basename(raw)[:-6]
        name = re.split(r"_split[AB]", stem)[0]
        key = name.replace("_0shot", "")
        adapter = registry.get(key, {}).get("adapter", "cadquery")
        split_letter = "a" if "_splitA" in stem else "b"
        scored = os.path.join(args.work, "scored", f"{stem}.jsonl")
        if os.path.exists(scored) and not args.force:
            print(f"skip (already scored): {stem}")
            continue
        print(f"\n>>> {stem}   adapter={adapter}")
        subprocess.run([
            sys.executable, "-m", "t2cbench.evaluate",
            "--predictions", raw,
            "--split", os.path.join(args.data, f"split_{split_letter}.jsonl"),
            "--adapter", adapter, "--model-name", name,
            "--keep-meshes", os.path.join(args.work, "meshes", name),
            "--out", scored, "--workers", str(args.workers),
            "--timeout", str(args.timeout),
        ], check=True)


if __name__ == "__main__":
    main()
