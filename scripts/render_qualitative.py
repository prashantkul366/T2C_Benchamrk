"""Render qualitative comparison grids straight from a scored benchmark run.

    # every model on six shapes, at one prompt level
    python scripts/render_qualitative.py --mode grid  --level L3 \
        --uids 00178990,00308160,00617877 --work results --data data --out fig.png

    # one shape, every level: the within-model level effect, made visible
    python scripts/render_qualitative.py --mode sweep --uid 00308160 \
        --work results --data data --out fig.png

The tables say cadrille scores 0.0145 at L0 and 0.5963 at L3. They do not show
that the L0 output is a featureless slab and the L3 output is the part. Looking
at the geometry is how a reader tells "the model cannot design" apart from "the
model cannot parse", and it is the figure reviewers in this field look for
first.

Works off `results/raw` + `results/scored` + the split file, rebuilding only the
handful of meshes a figure needs -- so a run scored with `--no-keep-meshes`
(6,300 STLs is 1-2 GB) can still be drawn. CD and IoU annotations are read back
from the scored rows rather than recomputed, so a cell can never disagree with
the table it illustrates.

Needs both geometry kernels -- see scripts/setup_scoring_env.sh.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from t2cbench.adapters import get_adapter                      # noqa: E402
from render_smoke_figure import render, task_prompt, GT_GREY, INK, INK_2, MUTED, SURFACE  # noqa: E402

LEVELS = ["L0", "L1", "L2", "L3"]

# Specialised systems and general LLMs get different hues, so a reader can see
# the two families separate (or fail to) without consulting the legend.
SPEC_COLOUR = "#2a78d6"
GEN_COLOUR = "#eb6834"
DEAD_COLOUR = "#b9b7b1"


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def raw_output(rec):
    o = rec.get("outputs") or [rec.get("output", "")]
    return o[0] if isinstance(o, list) else o


def collect(work, data, models, sample_ids):
    """raw output + recorded validity/CD/IoU for each (model, sample_id)."""
    want = set(sample_ids)
    out = {}
    for m in models:
        rawp = os.path.join(work, "raw", f"{m}_splitA_pass_at_1.jsonl")
        scop = os.path.join(work, "scored", f"{m}_splitA_pass_at_1.jsonl")
        if not os.path.exists(rawp):
            continue
        raws = {r["sample_id"]: raw_output(r) for r in read_jsonl(rawp)
                if r["sample_id"] in want}
        sco = {}
        if os.path.exists(scop):
            sco = {r["sample_id"]: r for r in read_jsonl(scop)
                   if r["sample_id"] in want}
        for sid in sample_ids:
            s = sco.get(sid, {})
            out[(m, sid)] = {"raw": raws.get(sid), "validity": s.get("validity", "?"),
                             "cd": s.get("cd"), "iou": s.get("iou_voxel")}
    return out


def build_meshes(cells, adapters, tmp, timeout):
    for (m, sid), rec in cells.items():
        if not rec["raw"] or rec["validity"] not in ("OK", "NON_MANIFOLD", "INVALID_SOLID"):
            rec["mesh"] = None
            continue
        stem = f"{m}__{sid.replace('/', '_')}.stl"
        dest = os.path.join(tmp, stem)
        try:
            res = get_adapter(adapters[m], timeout_s=timeout).run(rec["raw"], dest)
            rec["mesh"] = res.mesh_path if res.mesh_path and os.path.exists(res.mesh_path) else None
        except Exception:
            rec["mesh"] = None
        print(f"  {stem}: {'ok' if rec['mesh'] else 'no mesh'} ({rec['validity']})", flush=True)


def label_for(rec):
    if rec["mesh"] and rec["cd"] is not None:
        iou = f" · IoU {rec['iou']:.2f}" if rec.get("iou") is not None else ""
        return [f"CD {rec['cd']:.1f}{iou}"]
    return None


def draw(fig_title, subtitle, row_labels, col_labels, col_captions, gts, grid,
         families, out_path, row_notes=None):
    """grid[(row_label, col_label)] -> cell record. gts[col_label] -> GT mesh path."""
    row_notes = row_notes or {}
    nrows, ncols = len(row_labels) + 1, len(col_labels)
    fig = plt.figure(figsize=(2.9 * ncols + 1.4, 2.35 * nrows + 2.2), facecolor=SURFACE)
    gs = fig.add_gridspec(nrows + 1, ncols, height_ratios=[1.6] + [3] * nrows,
                          hspace=0.06, wspace=0.02,
                          left=0.095, right=0.985, top=0.925, bottom=0.035)

    fig.text(0.095, 0.982, fig_title, ha="left", va="top", fontsize=17,
             color=INK, fontweight="600")
    # wrapped to the figure width: an unwrapped subtitle runs off the canvas at
    # the narrow end of the size range these grids span
    wrap_at = max(70, int(11.5 * (2.9 * ncols + 1.4)))
    fig.text(0.095, 0.962, "\n".join(textwrap.wrap(subtitle, wrap_at)[:2]),
             ha="left", va="top", fontsize=10.5, color=INK_2)

    for c, col in enumerate(col_labels):
        ax = fig.add_subplot(gs[0, c]); ax.axis("off")
        ax.text(0.5, 1.0, col, ha="center", va="top", fontsize=11,
                color=INK, fontweight="600", transform=ax.transAxes)
        cap = col_captions.get(col, "")
        if cap:
            ax.text(0.5, 0.80, "\n".join(textwrap.wrap(cap, 44)[:5]),
                    ha="center", va="top", fontsize=7.8, color=MUTED,
                    transform=ax.transAxes)

    for c, col in enumerate(col_labels):
        ax = fig.add_subplot(gs[1, c], projection="3d")
        if gts.get(col):
            render(ax, gts[col], GT_GREY)
        else:
            ax.set_axis_off()
        if c == 0:
            ax.text2D(-0.14, 0.5, "GROUND TRUTH", transform=ax.transAxes, rotation=90,
                      va="center", ha="center", fontsize=9.5, color=INK, fontweight="600")

    row_axes = {}
    for r, row in enumerate(row_labels):
        colour = GEN_COLOUR if families.get(row) == "gen" else SPEC_COLOUR
        for c, col in enumerate(col_labels):
            ax = fig.add_subplot(gs[r + 2, c], projection="3d")
            rec = grid.get((row, col))
            if rec and rec.get("mesh"):
                render(ax, rec["mesh"], colour, label_for(rec))
            else:
                ax.set_axis_off()
                v = (rec or {}).get("validity", "—")
                ax.text2D(0.5, 0.52, v.replace("_", "\n") if v != "?" else "—",
                          transform=ax.transAxes, ha="center", va="center",
                          fontsize=8.5, color=DEAD_COLOUR, fontweight="600")
            if c == 0:
                row_axes[row] = ax

    # Placed after the axes exist so the text sits at each row's true vertical
    # centre. Drawing it inside the 3-D axes put long names far enough outside
    # the frame that neighbouring rows overlapped.
    fig.canvas.draw()
    for row, ax in row_axes.items():
        bb = ax.get_position()
        colour = GEN_COLOUR if families.get(row) == "gen" else SPEC_COLOUR
        note = row_notes.get(row, "")
        fig.text(0.030, (bb.y0 + bb.y1) / 2, row, rotation=90, va="center", ha="center",
                 fontsize=9.5, color=colour, fontweight="600")
        if note:
            fig.text(0.052, (bb.y0 + bb.y1) / 2, note, rotation=90, va="center",
                     ha="center", fontsize=7.5, color=MUTED)

    fig.savefig(out_path, dpi=145, facecolor=SURFACE)
    print(f"\nwrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["grid", "sweep"], default="grid")
    ap.add_argument("--work", default="results")
    ap.add_argument("--data", default="data")
    ap.add_argument("--models", default=None, help="comma-separated; default all in the run")
    ap.add_argument("--uids", default=None, help="grid mode: comma-separated 8-digit uids")
    ap.add_argument("--uid", default=None, help="sweep mode: one uid")
    ap.add_argument("--level", default="L3", help="grid mode: which prompt level")
    ap.add_argument("--registry", default=os.path.join(REPO_ROOT, "configs/models.yaml"))
    ap.add_argument("--families", default=os.path.join(REPO_ROOT, "configs/model_families.json"))
    ap.add_argument("--tables", default=None,
                    help="tables dir; used to annotate each row with that model's "
                         "aggregate Score, so a single good cell is not read as typical")
    ap.add_argument("--title", default=None)
    ap.add_argument("--subtitle", default=None)
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    reg = yaml.safe_load(open(args.registry))
    fam_cfg = json.load(open(args.families))
    gen_set = set(fam_cfg.get("General LLMs (zero-shot)", []))

    present = sorted({os.path.basename(p).split("_splitA")[0]
                      for p in glob.glob(os.path.join(args.work, "raw", "*_splitA_*.jsonl"))})
    models = [m.strip() for m in args.models.split(",")] if args.models else present
    models = [m for m in models if m in present]
    if not models:
        sys.exit("no models found in --work/raw")
    adapters = {m: reg.get(m.replace("_0shot", ""), {}).get("adapter", "cadquery") for m in models}
    families = {m: ("gen" if m in gen_set else "spec") for m in models}

    split = {r["sample_id"]: r for r in read_jsonl(os.path.join(args.data, "split_a.jsonl"))}

    if args.mode == "grid":
        uids = [u.strip() for u in (args.uids or "").split(",") if u.strip()]
        if not uids:
            sys.exit("--uids is required in grid mode")
        sample_ids = [f"A/{u}/{args.level}" for u in uids]
        col_labels = uids
        sid_of = dict(zip(col_labels, sample_ids))
        title = args.title or f"Predicted geometry at prompt level {args.level}"
        subtitle = args.subtitle or (
            f"Every system, same {len(uids)} shapes, same prompt level. Grey = ground truth. "
            f"Empty cell = no solid produced (reason shown). Each row carries that model's "
            f"Score over all 125 {args.level} prompts, so single cells are not read as typical.")
    else:
        if not args.uid:
            sys.exit("--uid is required in sweep mode")
        sample_ids = [f"A/{args.uid}/{lv}" for lv in LEVELS]
        col_labels = LEVELS
        sid_of = dict(zip(col_labels, sample_ids))
        title = args.title or f"The same shape described four ways — uid {args.uid}"
        subtitle = args.subtitle or (
            "L0 is a one-line caption; L3 transcribes every coordinate. The ground-truth row is "
            "identical across all four columns — only the prompt changes. Cells are one shape; "
            "each row carries the model's pooled Score over all 500 prompts for context.")

    col_captions = {}
    for col, sid in sid_of.items():
        row = split.get(sid, {})
        if args.mode == "sweep":
            col_captions[col] = (row.get("prompt", "") or "")[:200]
        else:
            col_captions[col] = (row.get("prompt", "") or "")[:150]

    gts = {}
    for col, sid in sid_of.items():
        gts[col] = (split.get(sid) or {}).get("gt_mesh")

    # A grid shows one shape per cell; a reader will over-generalise from a
    # lucky one unless the model's aggregate is right there next to it. Every
    # row therefore carries the Score for exactly the slice being drawn.
    row_notes = {}
    tdir = args.tables or os.path.join(args.work, "tables")
    tcsv = os.path.join(tdir, "01_main_splitA_pass1.csv")
    if os.path.exists(tcsv):
        import csv as _csv
        want_level = args.level if args.mode == "grid" else "ALL"
        with open(tcsv) as f:
            for row in _csv.DictReader(f):
                if row.get("level") == want_level and row.get("model") in models:
                    try:
                        lbl = "Score" if want_level == "ALL" else f"Score {want_level}"
                        row_notes[row["model"]] = f"{lbl} {float(row['Score']):.3f}"
                    except (TypeError, ValueError):
                        pass
    else:
        print(f"  (no {tcsv}; rows will not carry aggregate Scores)")

    cells = collect(args.work, args.data, models, sample_ids)
    tmp = tempfile.mkdtemp(prefix="t2c_qual_")
    print(f"building meshes in {tmp}")
    build_meshes(cells, adapters, tmp, args.timeout)

    grid = {(m, col): cells.get((m, sid_of[col])) for m in models for col in col_labels}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    draw(title, subtitle, models, col_labels, col_captions, gts, grid, families,
         args.out, row_notes)


if __name__ == "__main__":
    main()
