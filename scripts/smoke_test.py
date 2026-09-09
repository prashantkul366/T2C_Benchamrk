"""Smoke-test every specialised model on both splits and print one pasteable report.

The point is to answer, cheaply and before committing GPU hours: for each
(model, split), did the model load, did it receive the prompt format it was
trained on, did it emit the representation it is supposed to emit, and does that
representation turn into scoreable geometry?

It drives the *real* runners and the *real* adapters as subprocesses rather than
reimplementing them, so a pass here means the full benchmark path works, not
that a parallel copy of it works.

Three phases, because the two geometry kernels cannot be installed the same way:

    # 1. GPU, plain pip environment
    python scripts/smoke_test.py --phase generate --work $T2C_WORK -n 4

    # 2. scores whatever adapters this environment can actually import.
    #    In the pip env that is the CadQuery models; after `mamba install
    #    pythonocc-core` it is the sequence models. Run it in both.
    python scripts/smoke_test.py --phase score --work $T2C_WORK

    # 3. consolidated verdict over everything scored so far
    python scripts/smoke_test.py --phase report --work $T2C_WORK

Phase 2 is deliberately idempotent and additive: run it again in another
environment and it fills in the models it previously had to skip.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import glob
import subprocess
import sys
import textwrap

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPECIALISED = ["text2cad", "cadmium-7b", "cadfusion-v1.1", "cadrille",
               "cadrille-rl", "t2cq-qwen-3b", "t2cq-mistral-7b"]
SPLITS = ["A", "B"]

# What each representation must look like. A model can load fine, run fine and
# still emit the wrong thing entirely -- usually because it got a prompt format
# it was never trained on -- and that is invisible without checking the shape of
# the output itself.
FORMAT_SIGNATURES = {
    "cadquery":     dict(name="CadQuery Python",
                         must_match=r"(cq\.|cadquery|Workplane)"),
    "minimal_json": dict(name="minimal JSON",
                         must_match=r'"parts"'),
    "skexgen":      dict(name="SkexGen tokens",
                         must_match=r"(<extrude_end>|<curve_end>|<sketch_end>)"),
    "cadvec":       dict(name="(N,2) int cad_vec",
                         must_match=r"^\s*\[\s*\[\s*-?\d+\s*,\s*-?\d+\s*\]"),
}


def registry() -> dict:
    with open(os.path.join(REPO_ROOT, "configs", "models.yaml")) as f:
        return yaml.safe_load(f)


def raw_path(work: str, model: str, split: str) -> str:
    return os.path.join(work, "smoke", "raw", f"{model}_split{split}.jsonl")


def scored_path(work: str, model: str, split: str) -> str:
    return os.path.join(work, "smoke", "scored", f"{model}_split{split}.jsonl")


def split_path(work: str, split: str) -> str:
    return os.path.join(work, "data", f"split_{split.lower()}.jsonl")


def read_jsonl(path: str) -> list:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# --------------------------------------------------------------------------- #
# Phase 1: generate
# --------------------------------------------------------------------------- #

def build_gen_cmd(name: str, cfg: dict, split_file: str, out: str, n: int,
                  repos: dict) -> list[str] | None:
    py = sys.executable
    runner = cfg["runner"]

    if runner == "hf":
        cmd = [py, "-m", "t2cbench.runners.run_hf",
               "--model", cfg["weights"], "--name", name,
               "--template", cfg.get("template", "general_one_shot"),
               "--split", split_file, "--out", out,
               "--mode", "pass_at_1", "--batch-size", "4", "--limit", str(n),
               "--save-prompt"]
        if cfg.get("lora"):
            cmd += ["--base", cfg["base_model"]]
        if cfg.get("subfolder"):
            cmd += ["--subfolder", cfg["subfolder"]]
        if not cfg.get("chat_template", True):
            cmd += ["--no-chat-template"]
        if cfg.get("load_4bit"):
            cmd += ["--load-4bit"]
        return cmd

    if runner == "cadrille":
        if not repos.get("cadrille"):
            return None
        return [py, "-m", "t2cbench.runners.run_cadrille",
                "--cadrille-repo", repos["cadrille"],
                "--checkpoint", cfg["weights"], "--name", name,
                "--split", split_file, "--out", out,
                "--batch-size", "4", "--n-samples", "1",
                "--temperature", "0", "--limit", str(n), "--save-prompt"]

    if runner == "text2cad":
        if not repos.get("text2cad") or not repos.get("text2cad_ckpt"):
            return None
        return [py, "-m", "t2cbench.runners.run_text2cad",
                "--text2cad-repo", repos["text2cad"],
                "--checkpoint", repos["text2cad_ckpt"], "--name", name,
                "--split", split_file, "--out", out,
                "--batch-size", "4", "--n-samples", "1", "--limit", str(n),
                "--save-prompt", "--stub-occ"]
    return None


def make_smoke_split(work: str, split: str, n: int) -> str:
    """Write a sub-split of n rows covering n DISTINCT shapes.

    `--limit n` takes the first n rows, and the split files are ordered
    uid-then-level, so `--limit 3` on split A means one shape at three prompt
    levels. That tests the plumbing but says nothing about behaviour across
    shapes, and it makes every model's score hostage to a single object. Pick
    distinct uids instead, cycling the level so the level axis is still sampled.
    """
    src = split_path(work, split)
    rows = read_jsonl(src)
    levels = ["L0", "L1", "L2", "L3"]
    by_uid: dict[str, list] = {}
    for r in rows:
        by_uid.setdefault(r["uid"], []).append(r)

    picked = []
    for i, uid in enumerate(sorted(by_uid)):
        if len(picked) >= n:
            break
        want = levels[i % len(levels)]
        cand = [r for r in by_uid[uid] if r.get("level") == want] or by_uid[uid]
        picked.append(cand[0])

    dst_dir = os.path.join(work, "smoke", "splits")
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, f"split_{split.lower()}_smoke{n}.jsonl")
    with open(dst, "w") as f:
        for r in picked:
            f.write(json.dumps(r) + "\n")
    print(f"[split {split}] smoke subset: {len(picked)} distinct shapes "
          f"({', '.join(sorted({r['uid'] for r in picked}))})")
    return dst


def phase_generate(work: str, models: list[str], n: int, repos: dict,
                   timeout: int, force: bool, dry_run: bool = False) -> None:
    reg = registry()
    os.makedirs(os.path.join(work, "smoke", "raw"), exist_ok=True)
    subsets = {}
    for split in SPLITS:
        if os.path.exists(split_path(work, split)):
            subsets[split] = make_smoke_split(work, split, n)

    for model in models:
        cfg = reg.get(model)
        if cfg is None:
            print(f"!! unknown model {model!r}, skipping")
            continue
        for split in SPLITS:
            out = raw_path(work, model, split)
            sf = subsets.get(split, split_path(work, split))
            if not os.path.exists(sf):
                _note(work, model, split, "SKIP", f"split file missing: {sf}")
                continue
            if os.path.exists(out) and not force and not dry_run:
                print(f"[gen] {model} split{split}: already generated, skipping")
                continue

            cmd = build_gen_cmd(model, cfg, sf, out, n, repos)
            if cmd is None:
                if not dry_run:
                    _note(work, model, split, "SKIP",
                          f"runner {cfg['runner']!r} needs a repo/checkpoint path that was not provided")
                continue

            print(f"\n{'='*78}\n[gen] {model}  split{split}  (n={n})\n{'='*78}", flush=True)
            # A dry run must not touch disk -- deleting the outputs it claims it
            # is only going to describe is a nasty surprise.
            if dry_run:
                print("  " + " ".join(cmd))
                continue
            if os.path.exists(out):
                os.remove(out)   # runners resume; a forced re-run must start clean
            r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                               timeout=timeout)
            if r.returncode != 0 or not os.path.exists(out):
                tail = (r.stderr or r.stdout or "").strip().splitlines()
                _note(work, model, split, "GEN_FAIL", "\n".join(tail[-12:]))
                print("  FAILED:\n" + "\n".join("    " + l for l in tail[-12:]))
            else:
                got = len(read_jsonl(out))
                print(f"  ok, {got} generations -> {out}")
                _note(work, model, split, "GENERATED", f"{got} rows")
            # free VRAM before the next model
            _free_gpu()


def _free_gpu() -> None:
    try:
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Phase 2: score whatever this environment can
# --------------------------------------------------------------------------- #

def available_adapters() -> dict:
    """Which adapters this interpreter can actually run.

    cadquery ships OCP; the sequence adapters need pythonocc (`OCC`). The two
    are installed by different package managers, so an environment usually has
    one of them, and pretending otherwise produces confusing failures.
    """
    avail = {}
    for mod, adapters in (("cadquery", ["cadquery"]),
                          ("OCC", ["minimal_json", "cadvec", "skexgen"])):
        try:
            __import__(mod)
            ok = True
        except Exception:
            ok = False
        for a in adapters:
            avail[a] = ok
    return avail


def phase_score(work: str, models: list[str], workers: int, timeout: float,
                force: bool) -> None:
    reg = registry()
    avail = available_adapters()
    print("adapters runnable in this environment:",
          {k: ("yes" if v else "no") for k, v in avail.items()}, "\n")

    os.makedirs(os.path.join(work, "smoke", "scored"), exist_ok=True)
    for model in models:
        cfg = reg.get(model)
        if cfg is None:
            continue
        adapter = cfg["adapter"]
        for split in SPLITS:
            raw = raw_path(work, model, split)
            out = scored_path(work, model, split)
            if not os.path.exists(raw):
                continue
            if os.path.exists(out) and not force:
                print(f"[score] {model} split{split}: already scored, skipping")
                continue
            if not avail.get(adapter):
                print(f"[score] {model} split{split}: adapter {adapter!r} not importable here "
                      f"-- re-run this phase in the other environment")
                continue

            print(f"[score] {model} split{split}  adapter={adapter}", flush=True)
            cmd = [sys.executable, "-m", "t2cbench.evaluate",
                   "--predictions", raw, "--split", split_path(work, split),
                   "--adapter", adapter, "--model-name", model,
                   "--out", out, "--workers", str(workers),
                   "--timeout", str(timeout), "--no-boolean-iou", "--no-backfill"]
            env = dict(os.environ)
            r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, env=env)
            if r.returncode != 0:
                tail = (r.stderr or "").strip().splitlines()[-10:]
                print("  SCORING FAILED:\n" + "\n".join("    " + l for l in tail))
                _note(work, model, split, "SCORE_FAIL", "\n".join(tail))


# --------------------------------------------------------------------------- #
# Phase 3: consolidated report
# --------------------------------------------------------------------------- #

def phase_report(work: str, models: list[str], excerpt: int) -> None:
    import numpy as np
    reg = registry()
    notes = _load_notes(work)
    rows, details = [], []

    for model in models:
        cfg = reg.get(model)
        if cfg is None:
            continue
        adapter = cfg["adapter"]
        for split in SPLITS:
            raw_p, sc_p = raw_path(work, model, split), scored_path(work, model, split)
            rec = {"model": model, "split": split, "adapter": adapter,
                   "n_gen": 0, "fmt": "-", "valid": "-", "cd": None,
                   "f1": None, "iou": None, "verdict": "SKIP", "why": ""}

            if not os.path.exists(raw_p):
                note = notes.get((model, split))
                rec["why"] = (note or {}).get("detail", "not generated")[:90]
                rec["verdict"] = "FAIL" if note and note["status"] == "GEN_FAIL" else "SKIP"
                rows.append(rec); continue

            raws = read_jsonl(raw_p)
            rec["n_gen"] = len(raws)
            outs = [r.get("output", "") for r in raws]
            nonempty = [o for o in outs if o and o.strip()]

            sig = FORMAT_SIGNATURES.get(adapter, {})
            hits = sum(1 for o in nonempty if re.search(sig.get("must_match", "."), o,
                                                        re.MULTILINE | re.DOTALL))
            rec["fmt"] = f"{hits}/{len(outs)}"

            if os.path.exists(sc_p):
                sc = read_jsonl(sc_p)
                ok = sum(1 for r in sc if r.get("validity") == "OK")
                usable = sum(1 for r in sc if r.get("validity") in ("OK", "NON_MANIFOLD"))
                rec["valid"] = f"{ok}/{len(sc)}"
                v = [r for r in sc if r.get("scored") and r.get("cd") is not None]
                if v:
                    rec["cd"] = float(np.median([r["cd"] for r in v]))
                    rec["f1"] = float(np.mean([r["f1_002"] for r in v]))
                    rec["iou"] = float(np.mean([r["iou_voxel"] for r in v]))
                rec["verdict"], rec["why"] = _verdict(rec, len(sc), ok, usable, hits, len(outs))
            else:
                rec["verdict"] = "GEN-ONLY"
                rec["why"] = "generated but not scored in any environment yet"
                if hits == 0 and nonempty:
                    rec["verdict"] = "FAIL"
                    rec["why"] = f"output does not look like {sig.get('name', adapter)}"

            rows.append(rec)
            if split == "A":
                details.append((model, adapter, sig.get("name", adapter),
                                raws[0] if raws else None, excerpt))

    _print_report(rows, details, work)


def _verdict(rec, n_scored, ok, usable, fmt_hits, n_out):
    if n_out and fmt_hits == 0:
        return "FAIL", "output is not the expected representation"
    if usable == 0:
        return "FAIL", "no sample produced scoreable geometry"
    if rec["cd"] is None:
        return "FAIL", "geometry built but no metric could be computed"
    if ok / max(n_scored, 1) < 0.5:
        return "WARN", f"only {ok}/{n_scored} fully valid (small n; may be fine)"
    if rec["cd"] > 200:
        return "WARN", f"CD median {rec['cd']:.0f} is very high -- check the prompt format"
    return "PASS", ""


def _print_report(rows, details, work):
    print("\n" + "=" * 100)
    print("T2C-BENCH SMOKE TEST REPORT".center(100))
    print("=" * 100)
    print(f"{'model':17s} {'sp':3s} {'adapter':13s} {'gen':>4s} {'fmt':>6s} "
          f"{'valid':>6s} {'CDmed':>8s} {'F1':>6s} {'IoU':>6s}  verdict")
    print("-" * 100)
    for r in rows:
        cd = f"{r['cd']:8.2f}" if r["cd"] is not None else "       -"
        f1 = f"{r['f1']:6.3f}" if r["f1"] is not None else "     -"
        iou = f"{r['iou']:6.3f}" if r["iou"] is not None else "     -"
        print(f"{r['model']:17s} {r['split']:3s} {r['adapter']:13s} {r['n_gen']:4d} "
              f"{r['fmt']:>6s} {r['valid']:>6s} {cd} {f1} {iou}  {r['verdict']}"
              + (f" ({r['why']})" if r["why"] else ""))
    print("-" * 100)
    counts = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print("summary:", ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    print("""
  fmt   = generations matching the representation the adapter expects
  valid = adapter produced a fully valid solid (OK); NON_MANIFOLD counts as usable, not OK
  CDmed = median Chamfer x1000 after canonicalisation; lower is better
  GEN-ONLY = generated but this environment could not import that adapter""")

    print("\n" + "=" * 100)
    print("RAW OUTPUT SAMPLES  (split A, first prompt of each model)".center(100))
    print("=" * 100)
    for model, adapter, fmt_name, row, excerpt in details:
        print(f"\n--- {model}  [expects: {fmt_name}] " + "-" * max(4, 60 - len(model) - len(fmt_name)))
        if row is None:
            print("    (nothing generated)")
            continue
        sent = row.get("prompt_sent")
        if sent:
            # The prompt is the thing most likely to be wrong and least likely to
            # announce it: a model given a format it was never trained on still
            # produces confident-looking output.
            head, tail = sent[:260], sent[-200:]
            print("  PROMPT SENT:")
            print(textwrap.indent(head, "    "))
            if len(sent) > 460:
                print(f"    ... [{len(sent) - 460} chars elided] ...")
                print(textwrap.indent(tail, "    "))
            print("  OUTPUT:")
        out = (row.get("output") or "").strip()
        if not out:
            print("    (EMPTY OUTPUT)")
            continue
        body = out[:excerpt]
        print(textwrap.indent(body, "    "))
        if len(out) > excerpt:
            print(f"    ... [{len(out) - excerpt} more chars]")
    print("\n" + "=" * 100)
    print(f"artifacts under {os.path.join(work, 'smoke')}")


# --------------------------------------------------------------------------- #

def _note(work, model, split, status, detail) -> None:
    p = os.path.join(work, "smoke", "notes.jsonl")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps({"model": model, "split": split,
                            "status": status, "detail": detail}) + "\n")


def _load_notes(work) -> dict:
    p = os.path.join(work, "smoke", "notes.jsonl")
    notes = {}
    if os.path.exists(p):
        for r in read_jsonl(p):
            notes[(r["model"], r["split"])] = r     # last write wins
    return notes


# --------------------------------------------------------------------------- #
# Phase 4: export a self-contained blob for scoring somewhere else
# --------------------------------------------------------------------------- #

def phase_export(work: str, models: list[str], max_chars: int) -> None:
    """Print every generation as one JSON document.

    Generation needs a GPU; scoring needs pythonocc. Those rarely live in the
    same place, so this makes the handoff explicit: run `generate` where the GPU
    is, paste this blob where the geometry kernel is, and score there. The blob
    carries the sample_id (which contains the uid), so ground-truth meshes can be
    re-fetched on the scoring side without shipping any geometry.
    """
    reg = registry()
    notes = _load_notes(work)
    payload = {"format": "t2c-smoke-export/1", "generations": [], "failures": []}

    # The recorded prompt is the *formatted* one, which for CADmium is mostly a
    # 2.3 kB JSON schema. Carry the split's plain description too, so a reader
    # (or a figure) can see the task without unpicking chat scaffolding.
    task = {}
    for split in SPLITS:
        d = os.path.join(work, "smoke", "splits")
        files = sorted(glob.glob(os.path.join(d, f"split_{split.lower()}_smoke*.jsonl"))) \
            if os.path.isdir(d) else []
        files = files or ([split_path(work, split)] if os.path.exists(split_path(work, split)) else [])
        for fp in files:
            for r in read_jsonl(fp):
                task.setdefault(r["sample_id"], r.get("prompt", ""))

    for model in models:
        cfg = reg.get(model)
        if cfg is None:
            continue
        for split in SPLITS:
            raw = raw_path(work, model, split)
            if not os.path.exists(raw):
                note = notes.get((model, split))
                payload["failures"].append({
                    "model": model, "split": split,
                    "status": (note or {}).get("status", "NOT_GENERATED"),
                    "detail": (note or {}).get("detail", "")[:800],
                })
                continue
            for r in read_jsonl(raw):
                out = r.get("output") or ""
                sent = r.get("prompt_sent") or ""
                payload["generations"].append({
                    "model": model,
                    "split": split,
                    "adapter": cfg["adapter"],
                    "sample_id": r["sample_id"],
                    "sample_idx": r.get("sample_idx", 0),
                    # head+tail is enough to confirm the template applied; the
                    # middle of a 2 kB system message carries no information here
                    "prompt_head": sent[:300],
                    "prompt_tail": sent[-200:] if len(sent) > 500 else "",
                    "prompt_len": len(sent),
                    "prompt_text": task.get(r["sample_id"], ""),
                    "output": out[:max_chars],
                    "output_len": len(out),
                    "truncated": len(out) > max_chars,
                })

    n_models = len({g["model"] for g in payload["generations"]})
    print("\n" + "=" * 78)
    print("COPY EVERYTHING BETWEEN THE MARKERS AND PASTE IT BACK FOR SCORING")
    print("=" * 78)
    print("<<<T2C_SMOKE_EXPORT_BEGIN>>>")
    print(json.dumps(payload, separators=(",", ":")))
    print("<<<T2C_SMOKE_EXPORT_END>>>")
    print("=" * 78)
    size = len(json.dumps(payload, separators=(",", ":")))
    print(f"{len(payload['generations'])} generations from {n_models} models, "
          f"{len(payload['failures'])} model/split failures, {size/1024:.1f} KB")
    out_file = os.path.join(work, "smoke", "export.json")
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"also written to {out_file} (attach the file if the paste is too large)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", required=True,
                    choices=["generate", "score", "report", "export"])
    ap.add_argument("--work", default=os.environ.get("T2C_WORK", "."))
    ap.add_argument("--models", default=",".join(SPECIALISED))
    ap.add_argument("-n", "--n-prompts", type=int, default=4,
                    help="prompts per (model, split). 4 is enough to see the format.")
    ap.add_argument("--cadrille-repo", default=None)
    ap.add_argument("--text2cad-repo", default=None)
    ap.add_argument("--text2cad-ckpt", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--gen-timeout", type=int, default=3600)
    ap.add_argument("--excerpt", type=int, default=600)
    ap.add_argument("--max-output-chars", type=int, default=4000,
                    help="export: cap per-generation output so the blob stays pasteable")
    ap.add_argument("--force", action="store_true", help="redo work already on disk")
    ap.add_argument("--dry-run", action="store_true",
                    help="generate phase: print the commands instead of running them")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    repos = {"cadrille": args.cadrille_repo,
             "text2cad": args.text2cad_repo,
             "text2cad_ckpt": args.text2cad_ckpt}

    if args.phase == "generate":
        phase_generate(args.work, models, args.n_prompts, repos, args.gen_timeout,
                       args.force, args.dry_run)
    elif args.phase == "score":
        phase_score(args.work, models, args.workers, args.timeout, args.force)
    elif args.phase == "report":
        phase_report(args.work, models, args.excerpt)
    else:
        phase_export(args.work, models, args.max_output_chars)


if __name__ == "__main__":
    main()
