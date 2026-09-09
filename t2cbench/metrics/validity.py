"""Validity taxonomy -- the decomposed invalidity ratio.

A single scalar "IR" hides the thing that actually distinguishes these systems.
CadQuery emitters fail by raising during execution; sequence emitters fail by
producing a syntactically fine sequence that builds an empty solid. Both show up
as "invalid" in the published numbers, which makes the two failure regimes
indistinguishable. We keep them apart.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, asdict


class Validity(str, Enum):
    OK = "OK"                        # non-empty, positive-volume solid
    PARSE_FAIL = "PARSE_FAIL"        # could not extract the expected representation
    EXEC_FAIL = "EXEC_FAIL"          # parsed, raised during construction
    TIMEOUT = "TIMEOUT"              # exceeded wall-clock budget
    EMPTY_SOLID = "EMPTY_SOLID"      # built, zero volume / no faces
    INVALID_SOLID = "INVALID_SOLID"  # BRepCheck_Analyzer rejects it
    NON_MANIFOLD = "NON_MANIFOLD"    # tessellated mesh is not watertight
    NO_OUTPUT = "NO_OUTPUT"          # model emitted nothing at all

    @property
    def is_ok(self) -> bool:
        return self is Validity.OK


# Ordered for stacked-bar plots: OK first, then increasingly upstream failures.
VALIDITY_ORDER = [
    Validity.OK,
    Validity.NON_MANIFOLD,
    Validity.INVALID_SOLID,
    Validity.EMPTY_SOLID,
    Validity.EXEC_FAIL,
    Validity.TIMEOUT,
    Validity.PARSE_FAIL,
    Validity.NO_OUTPUT,
]


@dataclass
class AdapterResult:
    """What every adapter returns."""
    validity: Validity
    mesh_path: str | None = None
    diagnostic: str = ""
    n_solids: int = 0
    elapsed_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.validity.is_ok

    def to_dict(self) -> dict:
        d = asdict(self)
        d["validity"] = self.validity.value
        return d


def invalidity_ratio(results) -> float:
    """IR = 1 - P(OK). Invalid samples are counted, never dropped."""
    results = list(results)
    if not results:
        return 1.0
    n_ok = sum(1 for r in results if _validity_of(r).is_ok)
    return 1.0 - n_ok / len(results)


def validity_breakdown(results) -> dict:
    """Counts per validity code, in plot order. Sums to len(results)."""
    results = list(results)
    counts = {v.value: 0 for v in VALIDITY_ORDER}
    for r in results:
        counts[_validity_of(r).value] += 1
    return counts


def _validity_of(r) -> Validity:
    if isinstance(r, Validity):
        return r
    if isinstance(r, AdapterResult):
        return r.validity
    if isinstance(r, dict):
        return Validity(r.get("validity", Validity.NO_OUTPUT.value))
    if isinstance(r, str):
        return Validity(r)
    raise TypeError(f"cannot read validity from {type(r)}")
