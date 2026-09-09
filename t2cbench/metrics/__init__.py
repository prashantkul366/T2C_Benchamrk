"""Metrics: one protocol applied identically to every system."""

from t2cbench.metrics.geometry import (
    GeomMetrics, compare_meshes, canonicalize, load_mesh, sample_surface,
    chamfer_distance, f_score, hausdorff, voxel_iou, boolean_iou,
    N_POINTS, VOXEL_RES, F1_TAUS, CD_SCALE, RNG_SEED,
)
from t2cbench.metrics.topology import TopoMetrics, topology, compare_topology
from t2cbench.metrics.validity import (
    Validity, VALIDITY_ORDER, AdapterResult, invalidity_ratio, validity_breakdown,
)

__all__ = [
    "GeomMetrics", "compare_meshes", "canonicalize", "load_mesh", "sample_surface",
    "chamfer_distance", "f_score", "hausdorff", "voxel_iou", "boolean_iou",
    "N_POINTS", "VOXEL_RES", "F1_TAUS", "CD_SCALE", "RNG_SEED",
    "TopoMetrics", "topology", "compare_topology",
    "Validity", "VALIDITY_ORDER", "AdapterResult", "invalidity_ratio", "validity_breakdown",
]
