"""Topology metrics.

Cheap to compute and they catch what Chamfer distance is bad at. A plate with
three holes and a plate with four holes differ by a tiny fraction of surface
area -- CD barely moves -- but the Euler characteristic changes, and for a
manufactured part that difference is the whole ballgame.
"""

from __future__ import annotations

import numpy as np
import trimesh
from dataclasses import dataclass, asdict


@dataclass
class TopoMetrics:
    n_solids: int
    n_faces: int
    n_vertices: int
    n_edges: int
    euler_characteristic: int
    is_watertight: bool
    is_volume: bool
    volume: float
    area: float

    def to_dict(self) -> dict:
        return asdict(self)


def topology(mesh: trimesh.Trimesh) -> TopoMetrics:
    """Topological summary of a tessellated mesh.

    Note these are counts on the *triangulation*, not on the B-rep. They are
    comparable between prediction and GT because both go through the same
    tessellation settings, but they are not the CAD face/edge counts. B-rep
    counts come from the adapter where the kernel is available.
    """
    try:
        parts = mesh.split(only_watertight=False)
        n_solids = max(1, len(parts))
    except Exception:
        n_solids = 1

    v = int(len(mesh.vertices))
    f = int(len(mesh.faces))
    try:
        e = int(len(mesh.edges_unique))
    except Exception:
        e = 0

    try:
        chi = int(mesh.euler_number)
    except Exception:
        chi = v - e + f

    return TopoMetrics(
        n_solids=n_solids,
        n_faces=f,
        n_vertices=v,
        n_edges=e,
        euler_characteristic=chi,
        is_watertight=bool(mesh.is_watertight),
        is_volume=bool(mesh.is_volume),
        volume=float(abs(mesh.volume)) if mesh.is_volume else 0.0,
        area=float(mesh.area),
    )


def compare_topology(pred: trimesh.Trimesh, gt: trimesh.Trimesh) -> dict:
    """Prediction-vs-GT topological agreement."""
    tp, tg = topology(pred), topology(gt)
    return {
        "pred": tp.to_dict(),
        "gt": tg.to_dict(),
        "euler_match": tp.euler_characteristic == tg.euler_characteristic,
        "n_solids_match": tp.n_solids == tg.n_solids,
        "watertight_match": tp.is_watertight == tg.is_watertight,
        "genus_delta": _genus(tg) - _genus(tp) if (tp.is_watertight and tg.is_watertight) else None,
    }


def _genus(t: TopoMetrics) -> float:
    """Genus for a closed orientable surface: chi = 2 - 2g (per component)."""
    return (2 * t.n_solids - t.euler_characteristic) / 2.0
