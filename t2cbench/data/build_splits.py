"""Build the two evaluation splits.

Split A -- Text2CAD test, 125 deduplicated uids x 4 prompt levels = 500 prompts
Split B -- CADPrompt, 200 objects x 2 prompt variants = 400 prompts

Run once on CPU. Mesh signatures dominate the runtime (~30 min for 8k meshes on
16 processes); everything is cached so re-running is cheap.

    python -m t2cbench.data.build_splits --stage all --out data/

The Text2CAD L0-L3 prompts come from `text2cad_v1.1.csv`. The canonical copy
lives in the gated `SadilKhan/Text2CAD` dataset repo; `ricemonster/NeurIPS11092`
carries an ungated mirror of the same file, which is the default here so the
split can be built without clearing a licence gate. Point --t2c-csv at the
gated copy instead if you prefer the primary source.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from t2cbench.data.dedup import (
    shape_signature, greedy_dedup, complexity_bin, Signature, DEFAULT_EPS,
)

LEVELS = {"L0": "abstract", "L1": "beginner", "L2": "intermediate", "L3": "expert"}

T2C_CSV_REPO = "ricemonster/NeurIPS11092"
T2C_CSV_FILE = "text2cad_v1.1.csv"
DEEPCAD_MESH_REPO = "maksimko123/deepcad_test_mesh"
CADQUERY_GT_REPO = "maksimko123/text2cad"

N_UIDS = 125
SEED = 0


# --------------------------------------------------------------------------- #
# Downloads
# --------------------------------------------------------------------------- #

def fetch_assets(cache: str) -> dict:
    """Pull the ungated assets. Returns local paths."""
    from huggingface_hub import hf_hub_download, snapshot_download

    os.makedirs(cache, exist_ok=True)
    paths = {}

    paths["t2c_csv"] = hf_hub_download(
        T2C_CSV_REPO, T2C_CSV_FILE, repo_type="model", cache_dir=cache)

    paths["deepcad_meshes"] = snapshot_download(
        DEEPCAD_MESH_REPO, repo_type="dataset", cache_dir=cache)

    zip_path = hf_hub_download(
        CADQUERY_GT_REPO, "text2cad.zip", repo_type="dataset", cache_dir=cache)
    cq_dir = os.path.join(cache, "text2cad_cadquery")
    if not os.path.isdir(os.path.join(cq_dir, "cadquery")):
        import zipfile
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(cq_dir)
    paths["cadquery_gt"] = cq_dir
    return paths


# --------------------------------------------------------------------------- #
# Split A: Text2CAD test, 125 x L0-L3
# --------------------------------------------------------------------------- #

def _signature_job(args):
    uid, path = args
    try:
        import trimesh
        m = trimesh.load(path, force="mesh")
        if m.faces is None or len(m.faces) < 4:
            return uid, None
        sig = shape_signature(m, uid=uid, seed=SEED)
        return uid, sig.vector
    except Exception:
        return uid, None


def build_split_a(paths: dict, out_dir: str, n_uids: int = N_UIDS,
                  eps: float = DEFAULT_EPS, workers: int = 16) -> str:
    mesh_dir = _find_mesh_dir(paths["deepcad_meshes"])
    uids = sorted(f[:-4] for f in os.listdir(mesh_dir) if f.endswith(".stl"))
    print(f"[split A] {len(uids)} DeepCAD test meshes")

    # --- prompts ---------------------------------------------------------- #
    print(f"[split A] reading {T2C_CSV_FILE} (1.3 GB, streamed)")
    wanted = set(uids)
    rows = []
    usecols = ["uid"] + list(LEVELS.values())
    for chunk in pd.read_csv(paths["t2c_csv"], usecols=usecols, chunksize=20000):
        # csv uid is "0035/00359148"; mesh uid is "00359148"
        chunk["short_uid"] = chunk["uid"].astype(str).str.split("/").str[-1]
        rows.append(chunk[chunk["short_uid"].isin(wanted)])
    prompts = pd.concat(rows, ignore_index=True).drop_duplicates("short_uid")
    prompts = prompts.set_index("short_uid")
    print(f"[split A] prompts found for {len(prompts)} / {len(uids)} uids")

    # keep only uids with all four levels non-empty
    def _complete(u):
        if u not in prompts.index:
            return False
        return all(isinstance(prompts.at[u, c], str) and prompts.at[u, c].strip()
                   for c in LEVELS.values())

    uids = [u for u in uids if _complete(u)]
    print(f"[split A] {len(uids)} uids have all four levels")

    # --- signatures ------------------------------------------------------- #
    sig_cache = os.path.join(out_dir, "cache_signatures.pkl")
    if os.path.exists(sig_cache):
        with open(sig_cache, "rb") as f:
            sig_map = pickle.load(f)
        print(f"[split A] loaded {len(sig_map)} cached signatures")
    else:
        sig_map = {}
        jobs = [(u, os.path.join(mesh_dir, f"{u}.stl")) for u in uids]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_signature_job, j) for j in jobs]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="signatures"):
                uid, vec = fut.result()
                if vec is not None:
                    sig_map[uid] = vec
        os.makedirs(out_dir, exist_ok=True)
        with open(sig_cache, "wb") as f:
            pickle.dump(sig_map, f)
    print(f"[split A] {len(sig_map)} usable meshes")

    # --- dedup ------------------------------------------------------------ #
    ordered = [u for u in uids if u in sig_map]
    sigs = [Signature(u, sig_map[u]) for u in ordered]
    kept, cluster = greedy_dedup(sigs, eps=eps)
    print(f"[split A] dedup eps={eps}: {len(ordered)} -> {len(kept)} unique shapes")

    # --- complexity + stratified sample ----------------------------------- #
    cq_dir = os.path.join(paths["cadquery_gt"], "cadquery")
    meta = {u: _complexity_from_cadquery(os.path.join(cq_dir, f"{u}.py")) for u in kept}
    bins = {}
    for u in kept:
        bins.setdefault(complexity_bin(*meta[u]), []).append(u)
    print("[split A] bin sizes:", {k: len(v) for k, v in sorted(bins.items())})

    rng = np.random.default_rng(SEED)
    chosen = _stratified_sample(bins, n_uids, rng)
    print(f"[split A] sampled {len(chosen)} uids")

    # --- emit -------------------------------------------------------------- #
    records = []
    for uid in chosen:
        n_ext, n_curves = meta[uid]
        cq_path = os.path.join(cq_dir, f"{uid}.py")
        for level, col in LEVELS.items():
            records.append({
                "sample_id": f"A/{uid}/{level}",
                "split": "A",
                "uid": uid,
                "level": level,
                "prompt": str(prompts.at[uid, col]).strip(),
                "gt_mesh": os.path.join(mesh_dir, f"{uid}.stl"),
                "gt_cadquery": cq_path if os.path.exists(cq_path) else None,
                "complexity_bin": complexity_bin(n_ext, n_curves),
                "n_extrusions": n_ext,
                "n_curves": n_curves,
                "dedup_cluster": cluster.get(uid),
                "contamination": "clean",   # DeepCAD test split
                "absolute_scale": False,    # normalised DeepCAD units
            })

    path = os.path.join(out_dir, "split_a.jsonl")
    _write_jsonl(path, records)
    print(f"[split A] wrote {len(records)} prompts -> {path}")
    return path


def _stratified_sample(bins: dict, n: int, rng) -> list[str]:
    """Proportional allocation with largest-remainder rounding."""
    total = sum(len(v) for v in bins.values())
    names = sorted(bins)
    exact = {b: len(bins[b]) / total * n for b in names}
    alloc = {b: int(np.floor(exact[b])) for b in names}
    rem = n - sum(alloc.values())
    for b in sorted(names, key=lambda b: exact[b] - alloc[b], reverse=True)[:rem]:
        alloc[b] += 1

    chosen = []
    for b in names:
        pool = sorted(bins[b])
        k = min(alloc[b], len(pool))
        chosen += list(rng.choice(pool, size=k, replace=False)) if k else []
    # top up if a bin was too small to meet its allocation
    if len(chosen) < n:
        rest = sorted(set(sum(bins.values(), [])) - set(chosen))
        chosen += list(rng.choice(rest, size=min(n - len(chosen), len(rest)), replace=False))
    return sorted(chosen)


def _complexity_from_cadquery(py_path: str) -> tuple[int, int]:
    """(n_extrusions, n_curves) read off the ground-truth CadQuery program.

    Uses cadrille's CadQuery ground truth, which is available for 171k DeepCAD
    uids, so no extra download is needed. Falls back to (1, 4) -- a simple
    single-extrusion box -- when the file is missing, which biases such uids
    toward the 'simple' bin rather than dropping them.
    """
    try:
        with open(py_path) as f:
            src = f.read()
    except Exception:
        return 1, 4
    n_ext = src.count(".extrude(")
    n_curves = src.count(".circle(") + src.count(".segment(") + src.count(".arc(") \
        + src.count(".polygon(") + src.count(".rect(")
    return max(n_ext, 1), max(n_curves, 1)


# --------------------------------------------------------------------------- #
# Split B: CADPrompt
# --------------------------------------------------------------------------- #

def build_split_b(cadprompt_dir: str, deepcad_mesh_dir: str, out_dir: str) -> str:
    """CADPrompt, with the contamination flag that makes it usable.

    187 of the 200 uids are in DeepCAD's train/val split, so they were seen by
    every fine-tuned system here and by none of the general LLMs. Flagging each
    row turns a misleading aggregate into a memorisation probe.
    """
    test_uids = {f[:-4] for f in os.listdir(deepcad_mesh_dir) if f.endswith(".stl")}
    uids = sorted(d for d in os.listdir(cadprompt_dir)
                  if d.isdigit() and os.path.isdir(os.path.join(cadprompt_dir, d)))
    print(f"[split B] {len(uids)} CADPrompt objects")

    variants = {
        "plain": "Natural_Language_Descriptions_Prompt.txt",
        "measured": "Natural_Language_Descriptions_Prompt_with_specific_measurements.txt",
    }

    records, n_clean = [], 0
    for uid in uids:
        d = os.path.join(cadprompt_dir, uid)
        gt_mesh = os.path.join(d, "Ground_Truth.stl")
        if not os.path.exists(gt_mesh):
            continue
        clean = uid in test_uids
        n_clean += clean
        for vname, fname in variants.items():
            p = os.path.join(d, fname)
            if not os.path.exists(p):
                continue
            with open(p, encoding="utf-8", errors="replace") as f:
                prompt = f.read().strip()
            records.append({
                "sample_id": f"B/{uid}/{vname}",
                "split": "B",
                "uid": uid,
                "level": {"plain": "L1", "measured": "L3"}[vname],  # approximate mapping
                "variant": vname,
                "prompt": prompt,
                "gt_mesh": gt_mesh,
                "gt_cadquery": os.path.join(d, "Python_Code.py"),
                "gt_properties": os.path.join(d, "Ground_Truth.json"),
                "contamination": "clean" if clean else "contaminated",
                "absolute_scale": True,   # CADPrompt carries real dimensions
            })

    print(f"[split B] {n_clean} clean / {len(uids) - n_clean} contaminated uids")
    path = os.path.join(out_dir, "split_b.jsonl")
    _write_jsonl(path, records)
    print(f"[split B] wrote {len(records)} prompts -> {path}")
    return path


# --------------------------------------------------------------------------- #

def _find_mesh_dir(root: str) -> str:
    for dirpath, _, files in os.walk(root):
        if any(f.endswith(".stl") for f in files):
            return dirpath
    raise FileNotFoundError(f"no .stl files under {root}")


def _write_jsonl(path: str, records: list) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["all", "a", "b", "fetch"], default="all")
    ap.add_argument("--out", default="data")
    ap.add_argument("--cache", default="data/hf_cache")
    ap.add_argument("--cadprompt-dir", default=None,
                    help="path to CAD_Code_Generation/CADPrompt (git clone Kamel773/CAD_Code_Generation)")
    ap.add_argument("--n-uids", type=int, default=N_UIDS)
    ap.add_argument("--eps", type=float, default=DEFAULT_EPS)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    paths = fetch_assets(args.cache)
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
    if args.stage == "fetch":
        return

    if args.stage in ("all", "a"):
        build_split_a(paths, args.out, n_uids=args.n_uids, eps=args.eps, workers=args.workers)

    if args.stage in ("all", "b"):
        if not args.cadprompt_dir:
            raise SystemExit("--cadprompt-dir is required for split B")
        build_split_b(args.cadprompt_dir, _find_mesh_dir(paths["deepcad_meshes"]), args.out)


if __name__ == "__main__":
    main()
