"""Adapter contract: raw model output -> mesh on disk + a validity label.

Every adapter runs the actual geometry construction in a *subprocess* with a
hard wall-clock timeout. Both OpenCASCADE and CadQuery leak memory across
repeated calls and both can hang indefinitely on degenerate input, so in-process
execution turns one bad sample into a dead harness. A timeout is a recorded
result, not a crash.

The adapter is considered part of the system under test. If a model emits code
that will not execute, that is the model's failure and it is scored as such.
The only thing the adapter is allowed to do is the format-level plumbing the
system's own published pipeline does -- nothing that repairs the geometry.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
from abc import ABC, abstractmethod

from t2cbench.metrics.validity import AdapterResult, Validity

# 60s, not 20s. Roughly 2-3s of every call is process fork plus the CadQuery/OCP
# import, before any geometry is attempted, and that grows sharply when many
# workers contend. A TIMEOUT is scored as a model failure, so a timeout budget
# that is merely tight quietly turns machine load into a worse benchmark score.
DEFAULT_TIMEOUT_S = 60.0

# "fork" on POSIX: it does not re-import __main__, so adapters work from a
# notebook cell, a heredoc, or `python -c` -- all of which break "spawn" with
# `FileNotFoundError: <stdin>`. The child does geometry and exits immediately,
# so the usual fork-with-threads hazard does not arise. Windows has no fork.
# Override with T2CBENCH_MP_START if a platform needs it.
_START_METHOD = os.environ.get(
    "T2CBENCH_MP_START",
    "fork" if "fork" in mp.get_all_start_methods() else "spawn",
)


class Adapter(ABC):
    """Base adapter. Subclasses implement `_build`, which runs in a subprocess."""

    name: str = "base"

    def __init__(self, timeout_s: float = DEFAULT_TIMEOUT_S):
        self.timeout_s = timeout_s

    # -- subclass API ------------------------------------------------------- #

    @staticmethod
    @abstractmethod
    def _build(raw: str, out_stl: str, extra: dict) -> dict:
        """Turn `raw` into an STL at `out_stl`.

        MUST be a staticmethod with no closure over `self`: it is executed in a
        forked subprocess. Returns a dict with at least {"validity": str}.
        Raising is fine -- the parent maps it to EXEC_FAIL.
        """

    # -- driver ------------------------------------------------------------- #

    def run(self, raw: str, out_stl: str, extra: dict | None = None) -> AdapterResult:
        extra = extra or {}
        if raw is None or not str(raw).strip():
            return AdapterResult(Validity.NO_OUTPUT, diagnostic="empty model output")

        os.makedirs(os.path.dirname(os.path.abspath(out_stl)), exist_ok=True)
        ctx = mp.get_context(_START_METHOD)
        q: mp.Queue = ctx.Queue()
        p = ctx.Process(target=_worker, args=(type(self), raw, out_stl, extra, q))

        t0 = time.time()
        p.start()
        p.join(self.timeout_s)
        elapsed = time.time() - t0

        if p.is_alive():
            p.terminate()
            p.join(5)
            if p.is_alive():
                p.kill()
                p.join()
            return AdapterResult(Validity.TIMEOUT, elapsed_s=elapsed,
                                 diagnostic=f"exceeded {self.timeout_s}s")

        if q.empty():
            # Worker died without reporting: segfault in the kernel, OOM kill.
            return AdapterResult(Validity.EXEC_FAIL, elapsed_s=elapsed,
                                 diagnostic=f"worker died, exitcode={p.exitcode}")

        payload = q.get()
        validity = Validity(payload.get("validity", Validity.EXEC_FAIL.value))
        # A NON_MANIFOLD result still has usable geometry for CD/F1, so the mesh
        # path is kept whenever a file actually exists. The aggregator decides
        # which metrics a non-watertight mesh is allowed to contribute to.
        has_file = os.path.exists(out_stl) and os.path.getsize(out_stl) > 0
        return AdapterResult(
            validity=validity,
            mesh_path=out_stl if has_file else None,
            diagnostic=payload.get("diagnostic", ""),
            n_solids=int(payload.get("n_solids", 0)),
            elapsed_s=elapsed,
        )


def _worker(cls, raw: str, out_stl: str, extra: dict, q: "mp.Queue") -> None:
    """Subprocess entry point."""
    try:
        result = cls._build(raw, out_stl, extra)
        if "validity" not in result:
            result["validity"] = Validity.OK.value
        q.put(result)
    except TimeoutError as e:
        q.put({"validity": Validity.TIMEOUT.value, "diagnostic": str(e)})
    except SyntaxError as e:
        q.put({"validity": Validity.PARSE_FAIL.value, "diagnostic": f"{type(e).__name__}: {e}"})
    except Exception as e:
        q.put({
            "validity": Validity.EXEC_FAIL.value,
            "diagnostic": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc()[-2000:],
        })


# --------------------------------------------------------------------------- #
# Shared post-construction checks, used by every adapter
# --------------------------------------------------------------------------- #

def finalize_solid(shape, out_stl: str, linear_deflection: float = 0.001,
                   angular_deflection: float = 0.5) -> dict:
    """Validate an OCC shape, tessellate it, write the STL.

    Tessellation parameters are fixed across all systems -- they move CD in the
    4th decimal, so they must not vary between models.
    """
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.StlAPI import StlAPI_Writer
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_SOLID

    if shape is None:
        return {"validity": Validity.EMPTY_SOLID.value, "diagnostic": "adapter produced no shape"}

    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    volume = props.Mass()
    if abs(volume) < 1e-12:
        return {"validity": Validity.EMPTY_SOLID.value, "diagnostic": f"volume={volume}"}

    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    n_solids = 0
    while exp.More():
        n_solids += 1
        exp.Next()

    if not BRepCheck_Analyzer(shape).IsValid():
        return {"validity": Validity.INVALID_SOLID.value,
                "diagnostic": "BRepCheck_Analyzer rejected the shape",
                "n_solids": n_solids}

    BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
    writer = StlAPI_Writer()
    writer.SetASCIIMode(True)
    if not writer.Write(shape, out_stl):
        return {"validity": Validity.EXEC_FAIL.value, "diagnostic": "STL write failed",
                "n_solids": n_solids}

    return _check_mesh(out_stl, n_solids)


def finalize_mesh_file(out_stl: str, n_solids: int = 1) -> dict:
    """For adapters that write an STL directly (CadQuery exporters)."""
    return _check_mesh(out_stl, n_solids)


# A watertight solid can still tessellate to an STL with a handful of unmatched
# edges -- the classic case is the poles of a sphere or a revolved surface, which
# leave exactly a couple of pinholes. Counting that as NON_MANIFOLD would charge
# an invalidity to every model that emits curved geometry while box-only models
# go free, which is a bias in the metric, not a property of the models. So a
# *tessellation-scale* number of boundary edges is repaired and the sample counts
# as OK; anything larger is a genuinely open shell and stays NON_MANIFOLD.
#
# Two conditions must both hold, because an edge count alone is not enough: a
# cube missing one face also has only 4 boundary edges, and fill_holes() would
# happily close it and hand back a "watertight" solid that is not the model's.
# So the patched area must also be negligible -- a pole pinhole adds ~0 area, a
# missing face adds a sixth of the surface.
TESSELLATION_PINHOLE_EDGES = 8
TESSELLATION_PINHOLE_FRAC = 0.001
TESSELLATION_PATCH_AREA_FRAC = 0.005


def _check_mesh(out_stl: str, n_solids: int) -> dict:
    import trimesh
    if not os.path.exists(out_stl) or os.path.getsize(out_stl) == 0:
        return {"validity": Validity.EMPTY_SOLID.value, "diagnostic": "no STL written"}
    try:
        m = trimesh.load(out_stl, force="mesh")
    except Exception as e:
        return {"validity": Validity.EXEC_FAIL.value, "diagnostic": f"unreadable STL: {e}"}
    if m.faces is None or len(m.faces) < 4:
        return {"validity": Validity.EMPTY_SOLID.value,
                "diagnostic": f"degenerate mesh, {0 if m.faces is None else len(m.faces)} faces"}

    if m.is_watertight:
        return {"validity": Validity.OK.value, "n_solids": n_solids}

    try:
        n_boundary = len(trimesh.grouping.group_rows(m.edges_sorted, require_count=1))
    except Exception:
        n_boundary = None

    # A closed multi-body assembly is not an open shell. Every one of these
    # systems can emit several bodies, and when two of them touch, merging
    # coincident STL vertices leaves a handful of edges shared by four faces
    # instead of two -- enough for `is_watertight` to say False even though the
    # shape has no holes at all and a well-defined volume. Counting that as
    # invalid would charge an invalidity to every model that emits assemblies
    # while single-body models go free: the same bias the pinhole rule exists to
    # prevent, arriving through a different door. So: no boundary edges and every
    # connected component closed => the solid is fine.
    if n_boundary == 0:
        try:
            parts = m.split(only_watertight=False)
        except Exception:
            parts = []
        if parts and all(p.is_watertight for p in parts):
            n_shared = len(trimesh.grouping.group_rows(m.edges_sorted, require_count=2))
            n_unique = len(trimesh.grouping.unique_rows(m.edges_sorted)[0])
            return {"validity": Validity.OK.value, "n_solids": n_solids,
                    "diagnostic": f"closed {len(parts)}-body assembly; touching bodies leave "
                                  f"{n_unique - n_shared} merged edge(s) shared by more than "
                                  f"two faces"}
        return {"validity": Validity.NON_MANIFOLD.value, "n_solids": n_solids,
                "diagnostic": "no boundary edges, but the mesh is not a union of closed "
                              "bodies -- self-intersecting or duplicated faces"}

    budget = max(TESSELLATION_PINHOLE_EDGES, int(len(m.edges) * TESSELLATION_PINHOLE_FRAC))
    if n_boundary is not None and n_boundary <= budget:
        repaired = m.copy()
        try:
            repaired.process(validate=True)
            repaired.fill_holes()
        except Exception:
            repaired = m
        patch_frac = abs(repaired.area - m.area) / m.area if m.area > 0 else 1.0
        if repaired.is_watertight and patch_frac <= TESSELLATION_PATCH_AREA_FRAC:
            repaired.export(out_stl)
            return {"validity": Validity.OK.value, "n_solids": n_solids,
                    "diagnostic": f"repaired {n_boundary} tessellation pinhole edge(s), "
                                  f"patched area {patch_frac*100:.3f}%"}
        if repaired.is_watertight:
            return {"validity": Validity.NON_MANIFOLD.value, "n_solids": n_solids,
                    "diagnostic": f"open mesh: {n_boundary} boundary edges, closing them would "
                                  f"add {patch_frac*100:.1f}% surface area -- a real hole, not a pinhole"}

    # Still usable for CD/F1, so the mesh is kept -- but flagged, and volumetric
    # metrics fall back to the dilating voxel back-end for both sides.
    where = (f"{n_boundary} boundary edges (budget {budget})"
             if n_boundary is not None else "boundary edges could not be counted")
    return {"validity": Validity.NON_MANIFOLD.value, "n_solids": n_solids,
            "diagnostic": f"open mesh: {where}"}
