"""Evaluate one system's predictions against the split ground truth.

    python -m t2cbench.evaluate \
        --predictions results/raw/cadmium-7b_splitA.jsonl \
        --split data/split_a.jsonl \
        --adapter minimal_json \
        --out results/scored/cadmium-7b_splitA.jsonl

Predictions file: JSONL, one object per generation, with
    {"sample_id": ..., "output": "<raw model output>", "sample_idx": 0}
`sample_idx` distinguishes the k generations of a best-of-k run. Everything else
in the row is carried through to the scored output.

Runs entirely on CPU. The geometry kernel, not the GPU, is the bottleneck.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from t2cbench.adapters import get_adapter
from t2cbench.metrics.geometry import compare_meshes, load_mesh
from t2cbench.metrics.topology import compare_topology
from t2cbench.metrics.validity import Validity


def score_one(task: dict) -> dict:
    """Adapt one raw output to a mesh and score it. Runs in a worker process."""
    adapter = get_adapter(task["adapter"], timeout_s=task["timeout_s"])
    row = {
        "sample_id": task["sample_id"],
        "sample_idx": task.get("sample_idx", 0),
        "uid": task.get("uid"),
        "level": task.get("level"),
        "split": task.get("split"),
        "complexity_bin": task.get("complexity_bin"),
        "contamination": task.get("contamination"),
    }

    workdir = task.get("mesh_dir")
    tmp = None
    if workdir is None:
        tmp = tempfile.TemporaryDirectory(prefix="t2c_")
        workdir = tmp.name
    stem = task["sample_id"].replace("/", "_") + f"_{task.get('sample_idx', 0)}"
    out_stl = os.path.join(workdir, stem + ".stl")

    try:
        res = adapter.run(task["output"], out_stl, extra=task.get("extra", {}))
        row.update({
            "validity": res.validity.value,
            "diagnostic": res.diagnostic[:500],
            "adapter_seconds": round(res.elapsed_s, 3),
            "pred_mesh": res.mesh_path,
        })

        # Geometry is scored whenever a usable mesh exists. A NON_MANIFOLD mesh
        # yields CD/F1 but its volumetric numbers are flagged, not silently used.
        if res.mesh_path and os.path.exists(res.mesh_path):
            try:
                pred = load_mesh(res.mesh_path)
                gt = load_mesh(task["gt_mesh"])
                gm = compare_meshes(
                    pred, gt,
                    compute_boolean_iou=task.get("boolean_iou", True),
                    absolute_scale=task.get("absolute_scale", False),
                    align_icp=task.get("align_icp", False),
                )
                row.update(gm.to_dict())
                row["topology"] = compare_topology(pred, gt)
                row["scored"] = True
            except Exception as e:
                row["scored"] = False
                row["score_error"] = f"{type(e).__name__}: {e}"
        else:
            row["scored"] = False
    finally:
        if tmp is not None:
            tmp.cleanup()

    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--adapter", required=True,
                    choices=["cadquery", "cadvec", "minimal_json", "skexgen"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--keep-meshes", default=None,
                    help="directory to keep predicted STLs in (needed for the qualitative figure)")
    ap.add_argument("--no-boolean-iou", action="store_true",
                    help="skip mesh-boolean IoU; it is slow and only used to reconcile with cadrille")
    ap.add_argument("--align-icp", action="store_true", help="ablation only, see docs section 4.1")
    ap.add_argument("--no-backfill", action="store_true",
                    help="do not score unseen split rows as NO_OUTPUT. Only for deliberate "
                         "subset runs such as the smoke test -- a full run MUST back-fill, "
                         "or a model that skipped prompts scores as if it had never faced them.")
    args = ap.parse_args()

    split = {r["sample_id"]: r for r in _read_jsonl(args.split)}
    preds = _read_jsonl(args.predictions)
    print(f"{len(preds)} predictions over {len(split)} split rows")

    if args.keep_meshes:
        os.makedirs(args.keep_meshes, exist_ok=True)

    tasks, missing = [], 0
    for p in preds:
        sid = p.get("sample_id")
        meta = split.get(sid)
        if meta is None:
            missing += 1
            continue
        tasks.append({
            "sample_id": sid,
            "sample_idx": p.get("sample_idx", 0),
            "output": p.get("output", ""),
            "adapter": args.adapter,
            "timeout_s": args.timeout,
            "gt_mesh": meta["gt_mesh"],
            "uid": meta.get("uid"),
            "level": meta.get("level"),
            "split": meta.get("split"),
            "complexity_bin": meta.get("complexity_bin"),
            "contamination": meta.get("contamination"),
            "absolute_scale": meta.get("absolute_scale", False),
            "boolean_iou": not args.no_boolean_iou,
            "align_icp": args.align_icp,
            "mesh_dir": args.keep_meshes,
            "extra": p.get("extra", {}),
        })
    if missing:
        print(f"WARNING: {missing} predictions had no matching sample_id in the split")

    # Every prompt in the split must appear. A system that emitted nothing for a
    # prompt scores NO_OUTPUT -- silently having fewer rows would inflate its
    # scores, which is exactly the failure mode this benchmark exists to avoid.
    seen = {(t["sample_id"], t["sample_idx"]) for t in tasks}
    if args.no_backfill:
        print(f"back-fill disabled: scoring only the {len(tasks)} supplied predictions")
        seen = None
    n_expected_samples = max([t["sample_idx"] for t in tasks], default=0) + 1
    for sid, meta in ({}.items() if seen is None else split.items()):
        for k in range(n_expected_samples):
            if (sid, k) not in seen:
                tasks.append({
                    "sample_id": sid, "sample_idx": k, "output": "",
                    "adapter": args.adapter, "timeout_s": args.timeout,
                    "gt_mesh": meta["gt_mesh"], "uid": meta.get("uid"),
                    "level": meta.get("level"), "split": meta.get("split"),
                    "complexity_bin": meta.get("complexity_bin"),
                    "contamination": meta.get("contamination"),
                    "absolute_scale": meta.get("absolute_scale", False),
                    "boolean_iou": False, "align_icp": False,
                    "mesh_dir": args.keep_meshes, "extra": {},
                })

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(score_one, t) for t in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="scoring"):
            try:
                rows.append(fut.result())
            except Exception as e:
                rows.append({"validity": Validity.EXEC_FAIL.value,
                             "scored": False, "score_error": str(e)})

    model = args.model_name or os.path.basename(args.predictions).split("_")[0]
    for r in rows:
        r["model"] = model

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")

    _summary(rows)
    print(f"\nwrote {len(rows)} scored rows -> {args.out}")


def _summary(rows: list) -> None:
    import numpy as np
    n = len(rows)
    counts = defaultdict(int)
    for r in rows:
        counts[r.get("validity", "NO_OUTPUT")] += 1
    ok = counts.get("OK", 0) + counts.get("NON_MANIFOLD", 0)
    print(f"\n  n={n}  P(usable)={ok/n:.3f}  IR={1-counts.get('OK',0)/n:.3f}")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {k:16s} {v:5d}  {v/n*100:5.1f}%")
    # A TIMEOUT is counted as a model failure, so a meaningful timeout rate means
    # the numbers are partly measuring the machine. Say so rather than letting it
    # pass as a model result.
    n_timeout = counts.get("TIMEOUT", 0)
    if n_timeout and n_timeout / n > 0.02:
        print(f"\n  !! {n_timeout} TIMEOUTs ({n_timeout/n*100:.1f}%) -- these are scored as")
        print( "     failures. Re-run with a larger --timeout and/or fewer --workers")
        print( "     before trusting these numbers; a loaded machine looks like a bad model.")

    cds = [r["cd"] for r in rows if r.get("scored") and r.get("cd") is not None]
    if cds:
        print(f"  CD median={np.median(cds):.3f}  mean={np.mean(cds):.3f}  (x1000, n={len(cds)})")
        f1 = [r["f1_002"] for r in rows if r.get("scored")]
        # Same rule as the report tables: IoU only where the occupancy grid can
        # represent the reference (closed, and thicker than a couple of voxels).
        iou = [r["iou_voxel"] for r in rows if r.get("scored")
               and r.get("gt_closed", True) and not r.get("gt_thin", False)]
        iou_txt = (f"{np.mean(iou):.4f} (n={len(iou)})" if iou
                   else "n/a (no reference the voxel grid can represent)")
        print(f"  F1@0.02 mean={np.mean(f1):.4f}   IoU(voxel) mean={iou_txt}")


def _read_jsonl(path: str) -> list:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


if __name__ == "__main__":
    main()
