"""Aggregate scored rows into the benchmark tables.

    python -m t2cbench.report.tables --scored 'results/scored/*.jsonl' --out results/tables

Aggregation rules that keep the numbers honest:

  * Invalid samples are counted in the denominator of every rate, never dropped.
  * CD has no defined value for an invalid sample, so it is reported over valid
    samples only -- always printed next to P(OK) so a low CD bought by answering
    few prompts is visible.
  * `Score = P(OK) x F1@0.02` is the single ranking number, because it cannot be
    improved by refusing to answer.
  * best-of-k selects by minimum CD per (sample_id), i.e. an oracle, matching
    what Text2CAD / cadrille / CADFusion do. It is reported separately from
    pass@1 and never mixed into it.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

LEVEL_ORDER = ["L0", "L1", "L2", "L3"]
BIN_ORDER = ["simple", "moderate", "complex", "very_complex"]


def load_scored(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no files matched {pattern!r}")
    rows = []
    for fp in files:
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    print(f"loaded {len(df)} scored rows from {len(files)} files, "
          f"{df['model'].nunique()} models")
    return df


def select_pass_at_1(df: pd.DataFrame) -> pd.DataFrame:
    """First generation only."""
    return df[df["sample_idx"] == 0].copy()


def select_best_of_k(df: pd.DataFrame) -> pd.DataFrame:
    """Oracle selection by minimum CD within each (model, sample_id).

    A sample_id with no valid generation at all keeps one representative row so
    that it still counts as a failure in the denominator.
    """
    out = []
    for (_, _), grp in df.groupby(["model", "sample_id"], sort=False):
        valid = grp[grp["scored"].fillna(False) & grp["cd"].notna()]
        out.append(valid.loc[valid["cd"].idxmin()] if len(valid) else grp.iloc[0])
    return pd.DataFrame(out).reset_index(drop=True)


# Every column `aggregate` emits, so an empty slice still has the right shape.
AGG_COLUMNS = ["n", "IR_%", "P_ok", "P_usable", "CD_median", "CD_mean",
               "CD_mean_trim5", "F1_002", "F1_005", "IoU_voxel", "IoU_n",
               "HD95_median", "n_scored", "IoU_bool", "IoU_bool_coverage",
               "Score"]


def aggregate(df: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Core aggregation. `by` defaults to model only."""
    by = by or ["model"]
    # An empty slice is normal, not an error: best-of-5 tables are empty in a
    # pass@1-only run, and a contamination or complexity slice can be empty for a
    # model that failed everything in it. An empty frame also loses its columns,
    # so groupby(by) raises KeyError before any of this runs. Both used to take
    # the whole report down -- after every GPU hour had already been spent.
    if df.empty or any(c not in df.columns for c in by):
        return pd.DataFrame(columns=by + AGG_COLUMNS)

    rows = []
    for keys, g in df.groupby(by, dropna=False, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        n = len(g)
        ok = (g["validity"] == "OK").sum()
        usable = g["validity"].isin(["OK", "NON_MANIFOLD"]).sum()
        v = g[g["scored"].fillna(False) & g["cd"].notna()]

        # IoU only over ground truth an occupancy grid can represent: it must
        # bound a volume, and be thicker than a couple of voxels. Averaging over
        # the rest drags every model towards zero on the ~3% of open-shell
        # references and on the thin plates that score 0 against any prediction,
        # then ranks models partly on that artefact. Both flags come from
        # compare_meshes; scored files predating them fall back to "usable".
        iou_ok = pd.Series(True, index=v.index)
        if "gt_closed" in v.columns:
            iou_ok &= v["gt_closed"].fillna(True).astype(bool)
        if "gt_thin" in v.columns:
            iou_ok &= ~v["gt_thin"].fillna(False).astype(bool)
        vi = v[iou_ok]

        rec = dict(zip(by, keys))
        rec.update({
            "n": n,
            "IR_%": 100.0 * (1 - ok / n) if n else 100.0,
            "P_ok": ok / n if n else 0.0,
            "P_usable": usable / n if n else 0.0,
            "CD_median": float(np.median(v["cd"])) if len(v) else np.nan,
            "CD_mean": float(np.mean(v["cd"])) if len(v) else np.nan,
            "CD_mean_trim5": _trimmed_mean(v["cd"].values, 0.05) if len(v) else np.nan,
            "F1_002": float(np.mean(v["f1_002"])) if len(v) else 0.0,
            "F1_005": float(np.mean(v["f1_005"])) if len(v) else 0.0,
            "IoU_voxel": float(np.mean(vi["iou_voxel"])) if len(vi) else np.nan,
            "IoU_n": len(vi),
            "HD95_median": float(np.median(v["hd95"])) if len(v) else np.nan,
            "n_scored": len(v),
        })
        if "iou_boolean" in g.columns:
            b = v["iou_boolean"].dropna()
            rec["IoU_bool"] = float(np.mean(b)) if len(b) else np.nan
            rec["IoU_bool_coverage"] = len(b) / n if n else 0.0

        # The ranking number. F1 over ALL prompts (failures contribute 0), which
        # is identical to P(ok-and-scored) x mean F1 over scored.
        rec["Score"] = float(v["f1_002"].sum() / n) if n else 0.0
        rows.append(rec)

    if not rows:
        return pd.DataFrame(columns=by + AGG_COLUMNS)

    out = pd.DataFrame(rows)
    return out.sort_values("Score", ascending=False).reset_index(drop=True)


def _trimmed_mean(x: np.ndarray, frac: float) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    k = int(len(x) * frac)
    return float(np.mean(x[:len(x) - k])) if len(x) - k > 0 else float("nan")


# --------------------------------------------------------------------------- #
# The five tables
# --------------------------------------------------------------------------- #

def table_main(df: pd.DataFrame) -> pd.DataFrame:
    """Split A, pass@1, per level plus pooled."""
    a = select_pass_at_1(df[df["split"] == "A"])
    per_level = aggregate(a, ["model", "level"])
    pooled = aggregate(a, ["model"]); pooled["level"] = "ALL"
    out = pd.concat([per_level, pooled], ignore_index=True)
    out["level"] = pd.Categorical(out["level"], LEVEL_ORDER + ["ALL"], ordered=True)
    return out.sort_values(["model", "level"]).reset_index(drop=True)


def table_best_of_k(df: pd.DataFrame) -> pd.DataFrame:
    """Split A, oracle best-of-k -- the like-for-like comparison with published numbers."""
    a = df[df["split"] == "A"]
    if a["sample_idx"].max() == 0:
        return pd.DataFrame()
    best = select_best_of_k(a)
    per_level = aggregate(best, ["model", "level"])
    pooled = aggregate(best, ["model"]); pooled["level"] = "ALL"
    out = pd.concat([per_level, pooled], ignore_index=True)
    out["level"] = pd.Categorical(out["level"], LEVEL_ORDER + ["ALL"], ordered=True)
    return out.sort_values(["model", "level"]).reset_index(drop=True)


def table_contamination(df: pd.DataFrame) -> pd.DataFrame:
    """Split B clean vs contaminated, and the per-model memorisation gap.

    The gap is the point of this table: a system that only looks good on shapes
    it was trained on shows a large positive delta here.
    """
    b = select_pass_at_1(df[df["split"] == "B"])
    if b.empty:
        return pd.DataFrame()
    agg = aggregate(b, ["model", "contamination"])
    piv = agg.pivot(index="model", columns="contamination",
                    values=["Score", "CD_median", "F1_002", "IR_%", "n"])
    piv.columns = [f"{a}_{c}" for a, c in piv.columns]
    piv = piv.reset_index()
    if "Score_contaminated" in piv and "Score_clean" in piv:
        piv["memorisation_gap"] = piv["Score_contaminated"] - piv["Score_clean"]
        piv = piv.sort_values("memorisation_gap", ascending=False)
    return piv.reset_index(drop=True)


def table_ablation(df: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Base LLM vs its fine-tuned descendant, same weights, same prompts."""
    a = aggregate(select_pass_at_1(df[df["split"] == "A"]), ["model"]).set_index("model")
    rows = []
    for base, tuned in pairs:
        if base not in a.index or tuned not in a.index:
            continue
        rows.append({
            "base": base, "finetuned": tuned,
            "Score_base": a.at[base, "Score"], "Score_ft": a.at[tuned, "Score"],
            "Score_delta": a.at[tuned, "Score"] - a.at[base, "Score"],
            "CD_median_base": a.at[base, "CD_median"], "CD_median_ft": a.at[tuned, "CD_median"],
            "IR_base_%": a.at[base, "IR_%"], "IR_ft_%": a.at[tuned, "IR_%"],
        })
    return pd.DataFrame(rows)


def table_complexity(df: pd.DataFrame) -> pd.DataFrame:
    a = select_pass_at_1(df[df["split"] == "A"])
    out = aggregate(a, ["model", "complexity_bin"])
    out["complexity_bin"] = pd.Categorical(out["complexity_bin"], BIN_ORDER, ordered=True)
    return out.sort_values(["model", "complexity_bin"]).reset_index(drop=True)


def table_validity(df: pd.DataFrame) -> pd.DataFrame:
    """Failure-mode breakdown -- which systems fail how."""
    a = select_pass_at_1(df[df["split"] == "A"])
    ct = pd.crosstab(a["model"], a["validity"], normalize="index") * 100
    return ct.round(2).reset_index()


# --------------------------------------------------------------------------- #

# IoU_n rides alongside IoU_voxel because that mean is taken over a subset --
# references the voxel grid can actually represent -- and a subset mean without
# its n invites the reader to compare two numbers computed over different rows.
MAIN_COLS = ["model", "level", "n", "IR_%", "CD_median", "CD_mean", "F1_002",
             "IoU_voxel", "IoU_n", "HD95_median", "Score"]


def to_markdown(df: pd.DataFrame, cols: list[str] | None = None, floatfmt: int = 4) -> str:
    if df.empty:
        return "_(no data)_\n"
    d = df[[c for c in (cols or df.columns) if c in df.columns]].copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].round(floatfmt)
    return d.to_markdown(index=False) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scored", default="results/scored/*.jsonl")
    ap.add_argument("--out", default="results/tables")
    ap.add_argument("--ablation-pairs", default="configs/ablation_pairs.json")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = load_scored(args.scored)
    for col in ("sample_idx", "scored", "cd"):
        if col not in df.columns:
            df[col] = 0 if col == "sample_idx" else (False if col == "scored" else np.nan)

    pairs = []
    if os.path.exists(args.ablation_pairs):
        with open(args.ablation_pairs) as f:
            pairs = [tuple(p) for p in json.load(f)]

    tables = {
        "01_main_splitA_pass1": (table_main(df), MAIN_COLS),
        "02_best_of_k_splitA": (table_best_of_k(df), MAIN_COLS),
        "03_contamination_splitB": (table_contamination(df), None),
        "04_ablation_base_vs_finetuned": (table_ablation(df, pairs), None),
        "05_by_complexity": (table_complexity(df),
                             ["model", "complexity_bin", "n", "IR_%", "CD_median", "F1_002", "Score"]),
        "06_failure_modes": (table_validity(df), None),
    }

    md = ["# T2C-Bench results\n",
          "Generated by `t2cbench.report.tables`. "
          "CD is x1000, symmetric squared Chamfer after canonicalisation to the unit cube. "
          "`Score = P(valid) x mean F1@0.02` over **all** prompts, so failures count as zero.\n"]
    for name, (t, cols) in tables.items():
        t.to_csv(os.path.join(args.out, f"{name}.csv"), index=False)
        md.append(f"\n## {name.replace('_', ' ')}\n")
        md.append(to_markdown(t, cols))
        print(f"  wrote {name}.csv  ({len(t)} rows)")

    md_path = os.path.join(args.out, "RESULTS.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"\nwrote {md_path}")


if __name__ == "__main__":
    main()
