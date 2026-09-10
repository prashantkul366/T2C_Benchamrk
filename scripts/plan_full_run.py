"""Emit the generation commands for the full 900-prompt benchmark.

    python scripts/plan_full_run.py --work results --data data \
        --cadrille-repo /content/cadrille \
        --text2cad-repo /content/Text2CAD --text2cad-ckpt /path/Text2CAD_1.0.pth

Prints one command per (model, split), in the order to run them. Add `--run` to
execute them instead of printing, or pipe the output to a shell.

Why this exists rather than a hand-written list: run_all.sh kept its own roster,
and it drifted. By the time anyone looked it was missing five of the thirteen
models (text2cad, both cadrille checkpoints, t2cq-mistral-7b, qwen25-coder-32b),
had lost CADFusion's `--subfolder v1_1`, and named a Mistral base and template
that the checkpoint's own adapter_config.json contradicts. Every one of those
would have burned A100 hours and produced a row that looked like a model result.

So the roster lives in exactly one place -- configs/models.yaml -- and the
commands are built by the same `build_gen_cmd` the smoke test has driven through
five runs. A model that smoke-tests correctly is launched here identically, only
without `--limit`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from smoke_test import build_gen_cmd, registry, SPLITS   # noqa: E402

# Specialised systems first: they are the benchmark's subject. The general LLMs
# after them, base-models-of-the-fine-tunes first, so that if GPU time runs out
# the three controlled ablations (CADmium/CADFusion/cadrille vs their own bases)
# are already complete.
ORDER = ["text2cad", "cadmium-7b", "cadfusion-v1.1", "cadrille", "cadrille-rl",
         "t2cq-qwen-3b", "t2cq-mistral-7b",
         "qwen25-coder-7b", "llama3-8b-instruct", "qwen2-vl-2b",
         "mistral-7b-instruct", "deepseek-coder-6.7b", "qwen25-coder-32b"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="results")
    ap.add_argument("--data", default="data")
    ap.add_argument("--models", default=None,
                    help="comma-separated subset; default is every model in the registry")
    ap.add_argument("--cadrille-repo", default=None)
    ap.add_argument("--text2cad-repo", default=None)
    ap.add_argument("--text2cad-ckpt", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--run", action="store_true",
                    help="execute the commands instead of printing them")
    args = ap.parse_args()

    reg = registry()
    repos = {"cadrille": args.cadrille_repo,
             "text2cad": args.text2cad_repo,
             "text2cad_ckpt": args.text2cad_ckpt}

    names = ([m.strip() for m in args.models.split(",") if m.strip()] if args.models
             else [m for m in ORDER if m in reg] + [m for m in reg if m not in ORDER])

    raw_dir = os.path.join(args.work, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    planned, skipped = 0, []
    for name in names:
        cfg = reg.get(name)
        if not isinstance(cfg, dict):
            skipped.append((name, "not in configs/models.yaml"))
            continue
        for split in SPLITS:
            split_file = os.path.join(args.data, f"split_{split.lower()}.jsonl")
            if not os.path.exists(split_file):
                skipped.append((f"{name} split{split}", f"missing {split_file}"))
                continue
            out = os.path.join(raw_dir, f"{name}_split{split}_pass_at_1.jsonl")
            cmd = build_gen_cmd(name, cfg, split_file, out, None, repos,
                                batch_size=args.batch_size)
            if cmd is None:
                need = {"cadrille": "--cadrille-repo",
                        "text2cad": "--text2cad-repo and --text2cad-ckpt"}
                skipped.append((f"{name} split{split}",
                                f"runner {cfg['runner']!r} needs {need.get(cfg['runner'], 'a path')}"))
                continue
            planned += 1
            if args.run:
                print(f"\n=== [{planned}] {name} split{split} ===", flush=True)
                subprocess.run(cmd, check=False)
            else:
                print(" ".join(cmd))

    if skipped:
        print("\n# skipped:", file=sys.stderr)
        for what, why in skipped:
            print(f"#   {what}: {why}", file=sys.stderr)
    print(f"\n# {planned} (model, split) jobs planned", file=sys.stderr)


if __name__ == "__main__":
    main()
