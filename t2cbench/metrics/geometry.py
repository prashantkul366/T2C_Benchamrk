"""Geometric metrics for T2C-Bench.

One canonicalisation, one Chamfer distance, one IoU -- applied identically to
every system's output. See docs/02_benchmark_design.md section 4.

Canonicalisation (cadrille's convention, the only translation-invariant one of
the three used in the literature):

    centre bbox at origin -> scale so max extent == 1 -> translate to [0,1]^3

Deliberately NO ICP by default. DeepCAD ground truth is in a canonical frame and
the prompts describe coordinate systems explicitly, so orientation is part of
what is being tested. Text2CAD, cadrille and CADFusion all evaluate without ICP;
turning it on would both mask real errors and break comparability with published
numbers. `align_icp=True` is available as an ablation.
"""

from __future__ import annotations

import numpy as np
import trimesh
from dataclasses import dataclass, asdict, field
from scipy.spatial import cKDTree

# Protocol constants. Changing any of these invalidates cross-run comparison.
N_POINTS = 8192
VOXEL_RES = 64
F1_TAUS = (0.02, 0.05)
CD_SCALE = 1000.0
RNG_SEED = 0


def _warn_if_slow_backend() -> None:
    """`contains` is ~100x slower without embree; say so once, loudly."""
    try:
        import embreex  # noqa: F401
    except Exception:
        import warnings
        warnings.warn(
            "embreex is not installed: trimesh will use a pure-Python ray engine and "
            "voxel IoU will take ~90s per sample instead of ~0.5s. `pip install embreex`.",
            RuntimeWarning, stacklevel=2)


_warn_if_slow_backend()


# --------------------------------------------------------------------------- #
# Canonicalisation
# --------------------------------------------------------------------------- #

def canonicalize(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Centre, unit-scale and place a mesh in [0,1]^3.

    Applied identically to ground truth and prediction. Returns a copy.
    """
    m = mesh.copy()
    lo, hi = m.bounds
    m.apply_translation(-(lo + hi) / 2.0)
    extent = float(np.max(m.extents))
    if extent > 1e-9:
        m.apply_scale(1.0 / extent)
    m.apply_translation([0.5, 0.5, 0.5])
    return m


def _concat(mesh) -> trimesh.Trimesh:
    """Collapse a Scene or list of meshes into a single Trimesh."""
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError("scene contains no triangle geometry")
        return trimesh.util.concatenate(geoms)
    if isinstance(mesh, (list, tuple)):
        return trimesh.util.concatenate(list(mesh))
    return mesh


def load_mesh(path: str) -> trimesh.Trimesh:
    m = _concat(trimesh.load(path, force="mesh"))
    if m.faces is None or len(m.faces) == 0:
        raise ValueError(f"no faces in {path}")
    return m


def sample_surface(mesh: trimesh.Trimesh, n: int = N_POINTS, seed: int = RNG_SEED) -> np.ndarray:
    """Deterministic surface sampling. Seeded so runs are reproducible."""
    rng = np.random.default_rng(seed)
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=rng.integers(2**31))
    return np.asarray(pts, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Point-set metrics
# --------------------------------------------------------------------------- #

def chamfer_distance(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """Symmetric squared-distance Chamfer, the convention used by Text2CAD,
    cadrille and CADFusion alike:

        CD = mean_i min_j ||a_i - b_j||^2  +  mean_j min_i ||b_j - a_i||^2

    Caller multiplies by CD_SCALE (1000) for reporting.
    """
    d_ab, _ = cKDTree(pts_b).query(pts_a, k=1)
    d_ba, _ = cKDTree(pts_a).query(pts_b, k=1)
    return float(np.mean(d_ab**2) + np.mean(d_ba**2))


def f_score(pts_pred: np.ndarray, pts_gt: np.ndarray, tau: float) -> tuple[float, float, float]:
    """Point-cloud F-score at distance threshold tau.

    precision = fraction of predicted points within tau of the GT surface
    recall    = fraction of GT points within tau of the predicted surface
    """
    d_pred, _ = cKDTree(pts_gt).query(pts_pred, k=1)
    d_gt, _ = cKDTree(pts_pred).query(pts_gt, k=1)
    precision = float(np.mean(d_pred < tau))
    recall = float(np.mean(d_gt < tau))
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def hausdorff(pts_a: np.ndarray, pts_b: np.ndarray, pct: float = 95.0) -> float:
    """Symmetric percentile Hausdorff. pct=100 gives the true Hausdorff distance.

    HD95 is preferred over HD100 because a single stray triangle from a
    tessellation artefact should not dominate the number.
    """
    d_ab, _ = cKDTree(pts_b).query(pts_a, k=1)
    d_ba, _ = cKDTree(pts_a).query(pts_b, k=1)
    return float(max(np.percentile(d_ab, pct), np.percentile(d_ba, pct)))


# --------------------------------------------------------------------------- #
# Volumetric metrics
# --------------------------------------------------------------------------- #

def _grid_centres(res: int) -> np.ndarray:
    step = 1.0 / res
    return np.stack(np.meshgrid(
        *[np.arange(res) * step + step / 2.0] * 3, indexing="ij"
    ), axis=-1).reshape(-1, 3)


def is_closed(mesh: trimesh.Trimesh) -> bool:
    """Does this mesh bound a volume -- i.e. is "inside" well defined?

    `is_watertight` is stricter than that: it also rejects a *closed multi-body
    assembly*, because merging coincident STL vertices where two bodies touch
    leaves a few edges shared by four faces. Those meshes have no holes and a
    well-defined interior, and every system in this benchmark can emit them, so
    treating them as open would push a whole class of correct output onto the
    dilating occupancy back-end for no reason. Overlapping bodies, whose shared
    face is duplicated, still fail here.
    """
    if mesh.is_watertight:
        return True
    try:
        if len(trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)):
            return False
        parts = mesh.split(only_watertight=False)
        return bool(parts) and all(p.is_watertight for p in parts)
    except Exception:
        return False


def voxel_iou(mesh_pred: trimesh.Trimesh, mesh_gt: trimesh.Trimesh,
              res: int = VOXEL_RES, return_method: bool = False):
    """Voxel IoU on a shared res^3 grid over [0,1]^3.

    Primary IoU for the benchmark. Unlike mesh-boolean IoU this always returns
    a number, so a model cannot improve its mean IoU by producing geometry that
    breaks the boolean kernel.

    Two occupancy back-ends, and BOTH meshes always use the same one:

      "contains"  exact point-in-solid test. Requires closed geometry (see
                  `is_closed`: watertight, or a closed multi-body assembly).
      "voxelize"  surface voxelisation + flood fill. Works on open meshes, but
                  dilates the solid by roughly half a voxel.

    Mixing them would compare a dilated occupancy against an exact one and
    systematically flatter whichever side got dilated, so the choice is made
    once for the pair: exact only when *both* meshes are closed.

    Install `embreex`. Without it trimesh falls back to a pure-Python ray engine
    and `contains` goes from ~0.5 s to ~90 s per sample, which makes a full
    benchmark run take weeks instead of hours.
    """
    centres = _grid_centres(res)
    method = "contains" if (is_closed(mesh_pred) and is_closed(mesh_gt)) else "voxelize"

    occ_pred = _occupancy(mesh_pred, centres, res, method)
    occ_gt = _occupancy(mesh_gt, centres, res, method)
    if occ_pred is None or occ_gt is None:  # back-end failed on one side
        occ_pred = _occupancy(mesh_pred, centres, res, "voxelize")
        occ_gt = _occupancy(mesh_gt, centres, res, "voxelize")
        method = "voxelize"
    if occ_pred is None or occ_gt is None:
        return (0.0, "failed") if return_method else 0.0

    inter = np.count_nonzero(occ_pred & occ_gt)
    union = np.count_nonzero(occ_pred | occ_gt)
    iou = float(inter / union) if union > 0 else 0.0
    return (iou, method) if return_method else iou


def _occupancy(mesh: trimesh.Trimesh, centres: np.ndarray, res: int, method: str):
    """Boolean occupancy over the shared grid, or None if the back-end fails."""
    try:
        if method == "contains":
            return np.asarray(mesh.contains(centres), dtype=bool)
        vg = mesh.voxelized(pitch=1.0 / res)
        try:
            vg = vg.fill()
        except Exception:
            pass  # open shell: keep the surface voxels rather than failing
        return np.asarray(vg.is_filled(centres), dtype=bool)
    except Exception:
        return None


def boolean_iou(mesh_pred: trimesh.Trimesh, mesh_gt: trimesh.Trimesh) -> float | None:
    """Mesh-boolean IoU -- cadrille's definition, kept only to reconcile with
    their published numbers.

    Returns None when the boolean kernel fails. Aggregators MUST treat None as
    a failure (score 0 or excluded-and-reported), not silently skip it; cadrille's
    bare `except: pass` turns these into missing values, which inflates their
    reported mean IoU.
    """
    try:
        inter_vol = 0.0
        for a in mesh_gt.split(only_watertight=False):
            for b in mesh_pred.split(only_watertight=False):
                inter = a.intersection(b)
                if inter is not None and hasattr(inter, "volume"):
                    inter_vol += float(inter.volume)
        v_gt = sum(float(m.volume) for m in mesh_gt.split(only_watertight=False))
        v_pred = sum(float(m.volume) for m in mesh_pred.split(only_watertight=False))
        union = v_gt + v_pred - inter_vol
        if union <= 0:
            return None
        return float(inter_vol / union)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #

@dataclass
class GeomMetrics:
    cd: float                       # x1000, symmetric squared chamfer
    f1_002: float
    f1_005: float
    precision_002: float
    recall_002: float
    iou_voxel: float
    iou_voxel_method: str          # "contains" (exact) or "voxelize" (dilated)
    iou_boolean: float | None
    hd95: float
    hd100: float
    # False when the *ground truth* is an open shell rather than a closed solid.
    # 3.2% of the DeepCAD test meshes are (measured over 220 of them). "Inside"
    # is then undefined for the reference itself, so IoU on those samples is not
    # comparable with the rest and aggregation reports it apart. CD, F1 and
    # Hausdorff are surface metrics and stay valid either way.
    gt_closed: bool = True
    # absolute-scale fidelity, None when either mesh has no meaningful units
    bbox_rel_err: float | None = None
    volume_rel_err: float | None = None
    area_rel_err: float | None = None
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def compare_meshes(mesh_pred: trimesh.Trimesh,
                   mesh_gt: trimesh.Trimesh,
                   n_points: int = N_POINTS,
                   seed: int = RNG_SEED,
                   compute_boolean_iou: bool = True,
                   absolute_scale: bool = False,
                   align_icp: bool = False) -> GeomMetrics:
    """Full metric suite for one (prediction, ground truth) mesh pair.

    Args:
        absolute_scale: additionally compute bbox/volume/area relative error in
            the meshes' own units, BEFORE canonicalisation. Only meaningful when
            both sides carry real dimensions (CADPrompt, CadQuery emitters).
        align_icp: ablation only -- see module docstring.
    """
    raw_pred, raw_gt = mesh_pred, mesh_gt

    p = canonicalize(mesh_pred)
    g = canonicalize(mesh_gt)

    if align_icp:
        from trimesh.registration import icp as _icp
        try:
            matrix, _, _ = _icp(sample_surface(p, 2048, seed), sample_surface(g, 2048, seed),
                                scale=False, max_iterations=50)
            p.apply_transform(matrix)
        except Exception:
            pass

    pts_p = sample_surface(p, n_points, seed)
    pts_g = sample_surface(g, n_points, seed)

    cd = chamfer_distance(pts_p, pts_g) * CD_SCALE
    f1_002, prec, rec = f_score(pts_p, pts_g, F1_TAUS[0])
    f1_005, _, _ = f_score(pts_p, pts_g, F1_TAUS[1])

    iou_v, iou_method = voxel_iou(p, g, return_method=True)
    m = GeomMetrics(
        cd=cd,
        f1_002=f1_002,
        f1_005=f1_005,
        precision_002=prec,
        recall_002=rec,
        iou_voxel=iou_v,
        iou_voxel_method=iou_method,
        iou_boolean=boolean_iou(p, g) if compute_boolean_iou else None,
        hd95=hausdorff(pts_p, pts_g, 95.0),
        hd100=hausdorff(pts_p, pts_g, 100.0),
        gt_closed=is_closed(g),
    )

    if absolute_scale:
        m.bbox_rel_err = _rel_err(raw_pred.extents, raw_gt.extents)
        m.volume_rel_err = _rel_err(abs(raw_pred.volume), abs(raw_gt.volume))
        m.area_rel_err = _rel_err(raw_pred.area, raw_gt.area)

    return m


def _rel_err(pred, gt) -> float | None:
    pred = np.atleast_1d(np.asarray(pred, dtype=float))
    gt = np.atleast_1d(np.asarray(gt, dtype=float))
    denom = np.linalg.norm(gt)
    if denom < 1e-12:
        return None
    return float(np.linalg.norm(pred - gt) / denom)
