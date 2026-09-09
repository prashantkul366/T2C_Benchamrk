"""Score a smoke-test export blob on a machine that has the geometry kernels.

Generation needs a GPU; scoring needs pythonocc (sequence adapters) and cadquery
(CadQuery adapters). This closes the loop: take the blob printed by
`smoke_test.py --phase export`, re-fetch the ground-truth meshes by uid, run the
real adapters, and print the same verdict table the in-place smoke test would.

    python scripts/score_export.py --export export.json --cadprompt-dir /path/to/CADPrompt

Ground truth is re-fetched rather than shipped: the sample_id carries the uid, so
split A meshes come from the DeepCAD test snapshot on HuggingFace and split B
meshes from the local CADPrompt checkout. Nothing geometric has to survive the
copy-paste.

Adapters this interpreter cannot import are reported as GEN-ONLY rather than
failed -- pythonocc and cadquery are installed by different package managers and
an environment usually has only one.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import textwrap
from collections import defaultdict

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from t2cbench.adapters import get_adapter                      # noqa: E402
from t2cbench.metrics.geometry import compare_meshes, load_mesh  # noqa: E402
from t2cbench.metrics.topology import compare_topology          # noqa: E402

DEEPCAD_MESH_REPO = "maksimko123/deepcad_test_mesh"

FORMAT_SIGNATURES = {
    "cadquery":     ("CadQuery Python",   r"(cq\.|cadquery|Workplane)"),
    "minimal_json": ("minimal JSON",      r'"parts"'),
    "skexgen":      ("SkexGen tokens",    r"(<extrude_end>|<curve_end>|<sketch_end>)"),
    "cadvec":       ("(N,2) int cad_vec", r"^\s*\[\s*\[\s*-?\d+\s*,\s*-?\d+\s*\]"),
}


def available_adapters() -> dict:
    avail = {}
    for mod, names in (("cadquery", ["cadquery"]),
                       ("OCC", ["minimal_json", "cadvec", "skexgen"])):
        try:
            __import__(mod)
            ok = True
        except Exception:
            ok = False
        for a in names:
            avail[a] = ok
    return avail


def load_export(path: str | None) -> dict:
    """Accept a JSON file, or raw pasted text with the BEGIN/END markers."""
    text = open(path).read() if path else sys.stdin.read()
    m = re.search(r"<<<T2C_SMOKE_EXPORT_BEGIN>>>\s*(.*?)\s*<<<T2C_SMOKE_EXPORT_END>>>",
                  text, re.DOTALL)
    if m:
        text = m.group(1)
    return json.loads(text)


# --------------------------------------------------------------------------- #
# Ground truth resolution
# --------------------------------------------------------------------------- #

class GroundTruth:
    def __init__(self, cadprompt_dir: str | None, cache: str | None):
        self.cadprompt_dir = cadprompt_dir
        self.cache = cache
        self._memo: dict[str, str | None] = {}

    def path_for(self, sample_id: str, split: str) -> str | None:
        uid = sample_id.split("/")[1]
        key = f"{split}/{uid}"
        if key not in self._memo:
            self._memo[key] = (self._split_a(uid) if split == "A" else self._split_b(uid))
        return self._memo[key]

    def _split_a(self, uid: str) -> str | None:
        try:
            from huggingface_hub import hf_hub_download
            return hf_hub_download(DEEPCAD_MESH_REPO, f"{uid}.stl",
                                   repo_type="dataset", cache_dir=self.cache)
        except Exception as e:
            print(f"  !! could not fetch GT mesh for {uid}: {type(e).__name__}")
            return None

    def _split_b(self, uid: str) -> str | None:
        if not self.cadprompt_dir:
            return None
        p = os.path.join(self.cadprompt_dir, uid, "Ground_Truth.stl")
        return p if os.path.exists(p) else None


def find_cadprompt_root(start: str | None) -> str | None:
    """CADPrompt clones nest inconsistently; find the uid directories."""
    if not start or not os.path.isdir(start):
        return None
    for root, dirs, _ in os.walk(os.path.abspath(start)):
        hits = [d for d in dirs if d.isdigit() and len(d) == 8
                and os.path.exists(os.path.join(root, d, "Ground_Truth.stl"))]
        if len(hits) >= 10:
            return root
    return None


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", default=None, help="export.json; omit to read stdin")
    ap.add_argument("--cadprompt-dir", default=None,
                    help="CAD_Code_Generation checkout, for split B ground truth")
    ap.add_argument("--cache", default=None, help="HuggingFace cache dir")
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--excerpt", type=int, default=500)
    ap.add_argument("--out", default=None, help="write per-sample scores as JSONL")
    args = ap.parse_args()

    payload = load_export(args.export)
    gens = payload.get("generations", [])
    fails = payload.get("failures", [])
    print(f"loaded {len(gens)} generations, {len(fails)} model/split failures\n")

    avail = available_adapters()
    print("adapters importable here:",
          {k: ("yes" if v else "NO") for k, v in avail.items()}, "\n")

    cp = find_cadprompt_root(args.cadprompt_dir)
    if args.cadprompt_dir and not cp:
        print(f"!! no CADPrompt uid directories under {args.cadprompt_dir}; "
              f"split B cannot be scored\n")
    gt = GroundTruth(cp, args.cache)

    tmp = tempfile.mkdtemp(prefix="t2c_score_")
    per_sample, buckets = [], defaultdict(list)

    for i, g in enumerate(gens):
        adapter_name = g["adapter"]
        key = (g["model"], g["split"])
        rec = dict(g)
        rec["validity"] = None

        if not avail.get(adapter_name):
            rec["validity"] = "GEN-ONLY"
            buckets[key].append(rec); per_sample.append(rec); continue

        gt_path = gt.path_for(g["sample_id"], g["split"])
        if not gt_path:
            rec["validity"] = "NO_GT"
            buckets[key].append(rec); per_sample.append(rec); continue

        stem = g["sample_id"].replace("/", "_") + f"_{g.get('sample_idx', 0)}"
        out_stl = os.path.join(tmp, f"{g['model']}__{stem}.stl")
        ad = get_adapter(adapter_name, timeout_s=args.timeout)
        res = ad.run(g["output"], out_stl)
        rec["validity"] = res.validity.value
        rec["diagnostic"] = res.diagnostic[:200]

        if res.mesh_path and os.path.exists(res.mesh_path):
            try:
                m = compare_meshes(load_mesh(res.mesh_path), load_mesh(gt_path),
                                   compute_boolean_iou=False,
                                   absolute_scale=(g["split"] == "B"))
                rec.update(cd=m.cd, f1=m.f1_002, iou=m.iou_voxel, hd95=m.hd95,
                           iou_method=m.iou_voxel_method, bbox_err=m.bbox_rel_err)
                t = compare_topology(load_mesh(res.mesh_path), load_mesh(gt_path))
                rec["euler_match"] = t["euler_match"]
            except Exception as e:
                rec["score_error"] = f"{type(e).__name__}: {e}"
        buckets[key].append(rec)
        per_sample.append(rec)
        print(f"  [{i+1:3d}/{len(gens)}] {g['model']:16s} {g['split']} "
              f"{g['sample_id']:22s} {rec['validity']}", flush=True)

    _report(buckets, fails, gens, args.excerpt)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            for r in per_sample:
                f.write(json.dumps(r, default=str) + "\n")
        print(f"\nper-sample scores -> {args.out}")


def _report(buckets, fails, gens, excerpt):
    print("\n" + "=" * 104)
    print("SMOKE TEST SCORING".center(104))
    print("=" * 104)
    print(f"{'model':17s} {'sp':3s} {'adapter':13s} {'n':>3s} {'fmt':>5s} {'OK':>5s} "
          f"{'CDmed':>9s} {'F1':>6s} {'IoU':>6s}  verdict")
    print("-" * 104)

    for (model, split), rows in sorted(buckets.items()):
        adapter = rows[0]["adapter"]
        name, pat = FORMAT_SIGNATURES.get(adapter, (adapter, "."))
        n = len(rows)
        fmt_hits = sum(1 for r in rows if r.get("output") and
                       re.search(pat, r["output"], re.MULTILINE | re.DOTALL))
        ok = sum(1 for r in rows if r.get("validity") == "OK")
        usable = sum(1 for r in rows if r.get("validity") in ("OK", "NON_MANIFOLD"))
        genonly = sum(1 for r in rows if r.get("validity") == "GEN-ONLY")
        nogt = sum(1 for r in rows if r.get("validity") == "NO_GT")
        scored = [r for r in rows if r.get("cd") is not None]

        cd = f"{np.median([r['cd'] for r in scored]):9.2f}" if scored else "        -"
        f1 = f"{np.mean([r['f1'] for r in scored]):6.3f}" if scored else "     -"
        iou = f"{np.mean([r['iou'] for r in scored]):6.3f}" if scored else "     -"

        if genonly == n:
            verdict = "GEN-ONLY (adapter not importable here)"
        elif nogt == n:
            verdict = "NO GROUND TRUTH (could not resolve meshes)"
        elif fmt_hits == 0:
            verdict = "FAIL - output is not " + name
        elif usable == 0:
            verdict = "FAIL - no scoreable geometry"
        elif not scored:
            verdict = "FAIL - geometry built but unscoreable"
        elif ok < n * 0.5:
            verdict = f"WARN - only {ok}/{n} fully valid"
        else:
            verdict = "PASS"
        print(f"{model:17s} {split:3s} {adapter:13s} {n:3d} {fmt_hits:2d}/{n:<2d} "
              f"{ok:2d}/{n:<2d} {cd} {f1} {iou}  {verdict}")

    for f in fails:
        print(f"{f['model']:17s} {f['split']:3s} {'-':13s} {'-':>3s} {'-':>5s} {'-':>5s} "
              f"{'-':>9s} {'-':>6s} {'-':>6s}  {f['status']}")
    print("-" * 104)
    print("""
  fmt   = generations matching the representation the adapter expects
  OK    = adapter built a fully valid solid (NON_MANIFOLD counts as usable, not OK)
  CDmed = median Chamfer x1000 after canonicalisation to the unit cube; lower is better
  Split A is DeepCAD-normalised (no absolute scale); split B is CADPrompt (real dimensions)

  The verdict judges the PIPELINE, not the model: PASS means this system loaded, got a
  prompt it recognises, emitted its own representation and produced scoreable geometry.
  Whether CD/F1/IoU are *good* is what the full benchmark is for -- at n=3 they are far
  too noisy to rank anything.""")

    print("\n" + "=" * 104)
    print("PROMPT + OUTPUT SAMPLE PER MODEL (split A)".center(104))
    print("=" * 104)
    seen = set()
    for g in gens:
        if g["split"] != "A" or g["model"] in seen:
            continue
        seen.add(g["model"])
        name, _ = FORMAT_SIGNATURES.get(g["adapter"], (g["adapter"], "."))
        print(f"\n--- {g['model']}  [expects: {name}]  prompt_len={g.get('prompt_len', '?')}")
        if g.get("prompt_head"):
            print("  PROMPT (head):")
            print(textwrap.indent(g["prompt_head"], "    "))
            if g.get("prompt_tail"):
                print("  PROMPT (tail):")
                print(textwrap.indent(g["prompt_tail"], "    "))
        print("  OUTPUT:")
        out = (g.get("output") or "").strip()
        print(textwrap.indent(out[:excerpt] if out else "(EMPTY)", "    "))
        if len(out) > excerpt:
            print(f"    ... [{len(out) - excerpt} more chars]")


if __name__ == "__main__":
    main()
