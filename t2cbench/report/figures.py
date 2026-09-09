"""Benchmark figures.

    python -m t2cbench.report.figures --scored 'results/scored/*.jsonl' --out results/figures

Three figures:
  fig1_failure_modes      stacked bars, the 8-way validity breakdown per model
  fig2_level_sensitivity  Score and CD vs prompt level, faceted by model family
  fig3_qualitative        rendered prediction grid, models x examples, CD/IoU annotated

Colour follows the validated categorical palette in this module's PALETTE dict
(hues assigned in fixed order, never cycled; adjacent-pair CVD and normal-vision
floors verified with the dataviz validator). These are static figures for a
paper, so there is no hover layer and no dark-mode variant -- both are properties
of a chart rendered in a viewer's theme, which a PNG in a PDF is not. The
contrast WARN on three light-surface hues is relieved the required way: every
segment carries a visible label and every figure has a CSV twin in
results/tables/.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Validated categorical slots, fixed order -- never cycled, never re-ordered.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
STATUS_GOOD = "#0ca30c"

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

LEVEL_ORDER = ["L0", "L1", "L2", "L3"]
VALIDITY_ORDER = ["OK", "NON_MANIFOLD", "INVALID_SOLID", "EMPTY_SOLID",
                  "EXEC_FAIL", "TIMEOUT", "PARSE_FAIL", "NO_OUTPUT"]

# The 8 validity codes collapse to 4 groups for the figure. Two reasons, and
# the first is the important one:
#
#  1. Eight stacked segments cannot be told apart by colour. In a stack any two
#     segments can end up adjacent (zero-width segments collapse), so the
#     all-pairs separation gate applies, and no 8-hue set clears it. Four does.
#  2. The four groups are the actual story: did the model emit something
#     parseable, did it run, did it build a solid. CadQuery emitters fail at
#     execution and sequence emitters fail at geometry -- that contrast is what
#     the figure is for, and it survives the collapse.
#
# The full 8-way breakdown stays in results/tables/06_failure_modes.csv, which
# is also the table-view relief for the sub-3:1 aqua.
VALIDITY_GROUPS = {
    "Valid geometry": ["OK"],
    "Built, bad geometry": ["NON_MANIFOLD", "INVALID_SOLID", "EMPTY_SOLID"],
    "Failed to execute": ["EXEC_FAIL", "TIMEOUT"],
    "Unusable output": ["PARSE_FAIL", "NO_OUTPUT"],
}
# Slots 1, 2, 3, 7 -- validated all-pairs in light mode (worst CVD dE 9.2,
# worst normal-vision dE 16.3). Slot 6 green is deliberately unused: it sits
# only dE 9.7 from the status green and the two read as the same colour.
GROUP_COLORS = dict(zip(VALIDITY_GROUPS, [PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[6]]))

LEVEL_LABEL = {
    "L0": "L0\nabstract", "L1": "L1\nbeginner",
    "L2": "L2\nintermediate", "L3": "L3\nexpert",
}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=1.0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lbl.set_color(INK_2)


# --------------------------------------------------------------------------- #
# Fig 1 -- failure modes
# --------------------------------------------------------------------------- #

def fig_failure_modes(df: pd.DataFrame, out_path: str) -> None:
    """Which systems fail, and how.

    The reason this figure earns its place: a single invalidity ratio makes a
    model that emits unparseable prose look identical to one that emits a clean
    sequence which builds an empty solid. Those are different problems.
    """
    d = df[(df["split"] == "A") & (df["sample_idx"] == 0)]
    n_prompts = d.groupby("model").size().max()
    ct = pd.crosstab(d["model"], d["validity"], normalize="index") * 100
    for v in VALIDITY_ORDER:
        if v not in ct.columns:
            ct[v] = 0.0
    grouped = pd.DataFrame(
        {g: ct[[c for c in codes if c in ct.columns]].sum(axis=1)
         for g, codes in VALIDITY_GROUPS.items()}
    ).sort_values("Valid geometry")

    fig, ax = plt.subplots(figsize=(11, 0.52 * len(grouped) + 2.1), facecolor=SURFACE)
    left = np.zeros(len(grouped))
    for g in VALIDITY_GROUPS:
        w = grouped[g].values
        ax.barh(grouped.index, w, left=left, height=0.62,
                color=GROUP_COLORS[g], label=g,
                edgecolor=SURFACE, linewidth=2)  # 2px surface gap between fills
        # Visible label on every segment wide enough -- the relief the sub-3:1
        # aqua requires, and a secondary encoding besides hue.
        for i, (l, ww) in enumerate(zip(left, w)):
            if ww >= 7.0:
                ax.text(l + ww / 2, i, f"{ww:.0f}", ha="center", va="center",
                        fontsize=8.5, color="white", fontweight="600")
        left += w

    ax.set_xlim(0, 100)
    ax.set_xlabel(f"share of the {n_prompts} Split-A prompts (%)", fontsize=10, color=INK_2)
    ax.set_title("Where each system fails", fontsize=13, color=INK,
                 fontweight="600", loc="left", pad=14)
    ax.text(0, 1.015,
            "pass@1, Split A. The 8-way validity taxonomy grouped into 4; "
            "full breakdown in 06_failure_modes.csv.",
            transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4,
              frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------- #
# Fig 2 -- level sensitivity
# --------------------------------------------------------------------------- #

def fig_level_sensitivity(df: pd.DataFrame, out_path: str,
                          families: dict | None = None) -> None:
    """Score and CD against prompt level, faceted by model family.

    This is the figure that separates genuine text understanding from grammar
    transcription: a system trained on expert-level prompts can post an
    excellent L3 number and collapse at L0, and that collapse is invisible in
    any pooled table.

    Faceting is not cosmetic -- it keeps each panel within the 8-slot
    categorical cap instead of cycling hues.
    """
    d = df[(df["split"] == "A") & (df["sample_idx"] == 0)]
    families = families or _default_families(sorted(d["model"].unique()))

    rows = []
    for (model, level), g in d.groupby(["model", "level"]):
        v = g[g["scored"].fillna(False) & g["cd"].notna()]
        rows.append({
            "model": model, "level": level,
            "Score": float(v["f1_002"].sum() / len(g)) if len(g) else 0.0,
            "CD_median": float(np.median(v["cd"])) if len(v) else np.nan,
        })
    agg = pd.DataFrame(rows)

    fams = [f for f in families if any(m in set(d["model"]) for m in families[f])]
    fig, axes = plt.subplots(2, len(fams), figsize=(6.0 * len(fams), 8.2),
                             facecolor=SURFACE, squeeze=False)

    for col, fam in enumerate(fams):
        models = [m for m in families[fam] if m in set(agg["model"])]
        if len(models) > len(PALETTE):
            print(f"  NOTE: family {fam!r} has {len(models)} models; "
                  f"showing the first {len(PALETTE)} and folding the rest into 'Other'")
        for metric, row, ylabel, better in (
            ("Score", 0, "Score  =  P(valid) x F1@0.02", "higher is better"),
            ("CD_median", 1, "median Chamfer distance (x1000)", "lower is better"),
        ):
            ax = axes[row][col]
            # Scale must be set before any label placement: _decollide reads
            # ax.get_yscale(), and nudging log-spaced labels with linear offsets
            # throws them off the line entirely.
            if metric == "CD_median":
                vals = agg[agg["model"].isin(models)]["CD_median"].values
                if len(vals[np.isfinite(vals) & (vals > 0)]):
                    ax.set_yscale("log")
            ends = []
            for i, m in enumerate(models[:len(PALETTE)]):
                s = agg[agg["model"] == m].set_index("level").reindex(LEVEL_ORDER)
                ax.plot(range(4), s[metric].values, marker="o", markersize=8,
                        linewidth=2, color=PALETTE[i], label=m,
                        markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
                y = s[metric].values[-1]
                if np.isfinite(y):
                    ends.append((y, m, PALETTE[i]))

            # Direct labels only where they stay legible: at 5+ series the end
            # points collide and the legend carries identity instead.
            if len(models) <= 4 and ends:
                for y, m, colour in _decollide(ends, ax):
                    ax.annotate(m, (3, y), xytext=(8, 0), textcoords="offset points",
                                fontsize=8, color=colour, va="center", fontweight="600")
            ax.set_xticks(range(4))
            ax.set_xticklabels([LEVEL_LABEL[l] for l in LEVEL_ORDER], fontsize=9)
            ax.set_ylabel(ylabel, fontsize=10, color=INK_2)
            if metric == "CD_median" and ax.get_yscale() != "log":
                ax.text(0.5, 0.5, "no valid geometry produced",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=10, color=MUTED)
            if row == 0:
                ax.set_title(fam, fontsize=12, color=INK, fontweight="600", loc="left", pad=10)
            ax.text(0.0, -0.34, better, transform=ax.transAxes,
                    fontsize=8, color=MUTED, va="top")
            ax.set_xlim(-0.25, 3.55 if len(models) <= 4 else 3.15)
            _style(ax)
            # A legend is always present for 2+ series: identity must never rest
            # on colour alone, and at 5+ series it is the only carrier.
            if len(models) >= 2 and row == 1:
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.42),
                          ncol=min(3, len(models)), frameon=False,
                          fontsize=8, labelcolor=INK_2)

    fig.suptitle("Prompt level is the axis that separates these systems",
                 fontsize=14, color=INK, fontweight="600", x=0.01, ha="left", y=0.995)
    fig.text(0.01, 0.962,
             "L3 prompts are near-transcriptions of the CAD program; L0 is a one-line caption. "
             "A flat line is a system that actually reads the text.",
             fontsize=9, color=MUTED, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _decollide(ends, ax, min_gap_frac: float = 0.045):
    """Nudge end-of-line labels apart so converged series stay readable.

    Without this, two models that finish at the same score print on top of each
    other and both become unreadable -- which is worse than no label at all.
    """
    lo, hi = ax.get_ylim()
    span = abs(hi - lo) or 1.0
    if ax.get_yscale() == "log":
        return sorted(ends, key=lambda e: e[0])
    gap = span * min_gap_frac
    out, prev = [], None
    for y, m, c in sorted(ends, key=lambda e: e[0]):
        if prev is not None and y - prev < gap:
            y = prev + gap
        out.append((y, m, c))
        prev = y
    return out


def _default_families(models: list[str]) -> dict:
    specialised_keys = ("text2cad", "cadmium", "cadfusion", "cadrille", "t2cq", "cadquery-sft")
    spec = [m for m in models if any(k in m.lower() for k in specialised_keys)]
    general = [m for m in models if m not in spec]
    out = {}
    if spec:
        out["Specialised text-to-CAD"] = spec
    if general:
        out["General LLMs (zero-shot)"] = general
    return out


# --------------------------------------------------------------------------- #
# Fig 3 -- qualitative grid
# --------------------------------------------------------------------------- #

def fig_qualitative(df: pd.DataFrame, out_path: str, split_path: str,
                    mesh_root: str, n_examples: int = 6,
                    models: list[str] | None = None, seed: int = 0) -> None:
    """Rendered predictions, models x examples, with GT in the top row.

    Examples are chosen to *span* the difficulty range rather than to flatter
    anyone: one per complexity bin plus the two prompts with the widest
    between-model spread in Score, so the grid shows both agreement and
    disagreement.
    """
    import trimesh
    from t2cbench.metrics.geometry import canonicalize

    split = {r["sample_id"]: r for r in _read_jsonl(split_path)}
    d = df[(df["split"] == "A") & (df["sample_idx"] == 0)].copy()
    models = models or sorted(d["model"].unique())

    chosen = _pick_examples(d, split, n_examples, seed)
    nrows, ncols = len(models) + 1, len(chosen)
    fig = plt.figure(figsize=(2.5 * ncols, 2.5 * nrows), facecolor=SURFACE)

    for c, sid in enumerate(chosen):
        meta = split[sid]
        ax = fig.add_subplot(nrows, ncols, c + 1, projection="3d")
        _render(ax, meta["gt_mesh"], color="#c3c2b7")
        ax.set_title(f"{meta['level']} · {meta.get('complexity_bin','')}\n{meta['uid']}",
                     fontsize=8, color=INK_2, pad=2)
        if c == 0:
            ax.text2D(-0.18, 0.5, "Ground truth", transform=ax.transAxes, rotation=90,
                      va="center", ha="center", fontsize=10, color=INK, fontweight="600")

    for r, model in enumerate(models):
        for c, sid in enumerate(chosen):
            ax = fig.add_subplot(nrows, ncols, (r + 1) * ncols + c + 1, projection="3d")
            row = d[(d["model"] == model) & (d["sample_id"] == sid)]
            usable = len(row) and str(row.iloc[0].get("validity")) in ("OK", "NON_MANIFOLD")
            mesh_path = _find_pred_mesh(mesh_root, model, sid) if usable else None
            if len(row) and mesh_path:
                rr = row.iloc[0]
                _render(ax, mesh_path, color=PALETTE[r % len(PALETTE)])
                cd = rr.get("cd"); iou = rr.get("iou_voxel")
                if pd.notna(cd):
                    ax.text2D(0.5, -0.06, f"CD {cd:.2f} · IoU {iou:.2f}",
                              transform=ax.transAxes, ha="center", fontsize=7.5, color=INK_2)
            else:
                v = str(row.iloc[0].get("validity", "NO_OUTPUT")) if len(row) else "NO_OUTPUT"
                ax.text2D(0.5, 0.5, v.replace("_", "\n"), transform=ax.transAxes,
                          ha="center", va="center", fontsize=8, color=MUTED)
                ax.set_axis_off()
                ax.set_facecolor(SURFACE)
                ax.patch.set_alpha(0.0)
            if c == 0:
                ax.text2D(-0.18, 0.5, model, transform=ax.transAxes, rotation=90,
                          va="center", ha="center", fontsize=9, color=INK, fontweight="600")

    fig.suptitle(f"Same prompt, {len(models)} systems", fontsize=14, color=INK,
                 fontweight="600", x=0.01, ha="left")
    fig.tight_layout(rect=[0.02, 0, 1, 0.98])
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _render(ax, mesh_path: str, color: str) -> None:
    import trimesh
    from t2cbench.metrics.geometry import canonicalize
    try:
        m = canonicalize(trimesh.load(mesh_path, force="mesh"))
        ax.plot_trisurf(m.vertices[:, 0], m.vertices[:, 1], m.vertices[:, 2],
                        triangles=m.faces, color=color, edgecolor="none",
                        shade=True, alpha=1.0, linewidth=0)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
        ax.view_init(elev=26, azim=-52)
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        ax.text2D(0.5, 0.5, "render\nfailed", transform=ax.transAxes,
                  ha="center", va="center", fontsize=8, color=MUTED)
    ax.set_axis_off()
    ax.set_facecolor(SURFACE)


def _pick_examples(d: pd.DataFrame, split: dict, n: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    # A model that failed a prompt has no F1, but "everyone failed this one" is
    # itself informative, so failures count as 0 rather than being dropped --
    # filtering on non-null F1 would discard every prompt some model missed,
    # which on a hard benchmark is nearly all of them.
    dd = d.copy()
    dd["f1_filled"] = dd["f1_002"].fillna(0.0)
    spread = dd.groupby("sample_id")["f1_filled"].agg(["std", "mean"])
    spread["n_models"] = dd.groupby("sample_id")["model"].nunique()
    spread = spread[spread["n_models"] >= max(1, dd["model"].nunique() - 1)]

    chosen: list[str] = []
    for b in ["simple", "moderate", "complex", "very_complex"]:
        pool = [s for s in spread.index
                if split.get(s, {}).get("complexity_bin") == b and s not in chosen]
        if pool:
            chosen.append(str(rng.choice(pool)))
    # Fill the remaining slots with the widest between-model disagreement,
    # preferring shapes not already shown so the grid spans distinct parts
    # rather than the same uid at four prompt levels.
    seen_uids = {split.get(s, {}).get("uid") for s in chosen}
    ranked = list(spread.sort_values("std", ascending=False, na_position="last").index)
    for pass_ in (0, 1):
        for s in ranked:
            if len(chosen) >= n:
                break
            if s in chosen:
                continue
            uid = split.get(s, {}).get("uid")
            if pass_ == 0 and uid in seen_uids:
                continue
            chosen.append(str(s))
            seen_uids.add(uid)
    return chosen[:n]


def _find_pred_mesh(root: str, model: str, sample_id: str) -> str | None:
    """Locate THIS model's mesh for this sample, or nothing.

    Every candidate path is namespaced by model. There is deliberately no
    recursive-glob fallback: a bare `**/<stem>_0.stl` search matches whichever
    model happens to be first on disk, so a system that produced no geometry at
    all would be shown rendering another system's part. A figure that invents
    output for a model that failed is worse than a blank cell.
    """
    if not root:
        return None
    stem = sample_id.replace("/", "_")
    for cand in (os.path.join(root, model, f"{stem}_0.stl"),
                 os.path.join(root, f"{model}_{stem}_0.stl")):
        if os.path.exists(cand):
            return cand
    return None


# --------------------------------------------------------------------------- #

def _read_jsonl(path: str) -> list:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scored", default="results/scored/*.jsonl")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--split", default="data/split_a.jsonl")
    ap.add_argument("--mesh-root", default="results/meshes")
    ap.add_argument("--n-examples", type=int, default=6)
    ap.add_argument("--families", default="configs/model_families.json")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows = []
    for fp in sorted(glob.glob(args.scored)):
        rows += _read_jsonl(fp)
    if not rows:
        raise SystemExit(f"no scored rows matched {args.scored!r}")
    df = pd.DataFrame(rows)
    print(f"loaded {len(df)} rows, {df['model'].nunique()} models")

    families = None
    if os.path.exists(args.families):
        with open(args.families) as f:
            families = json.load(f)

    fig_failure_modes(df, os.path.join(args.out, "fig1_failure_modes.png"))
    fig_level_sensitivity(df, os.path.join(args.out, "fig2_level_sensitivity.png"), families)
    if os.path.isdir(args.mesh_root):
        fig_qualitative(df, os.path.join(args.out, "fig3_qualitative.png"),
                        args.split, args.mesh_root, args.n_examples)
    else:
        print(f"  skipping fig3: no mesh dir at {args.mesh_root} "
              f"(re-run t2cbench.evaluate with --keep-meshes)")


if __name__ == "__main__":
    main()
