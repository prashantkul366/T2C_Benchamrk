"""Output-format adapters: raw model output -> mesh + validity label."""

from t2cbench.adapters.base import Adapter, DEFAULT_TIMEOUT_S
from t2cbench.adapters.cadquery_adapter import CadQueryAdapter, extract_code
from t2cbench.adapters.cadseq_adapter import (
    CadVecAdapter,
    MinimalJsonAdapter,
    SkexGenAdapter,
    extract_json,
    clean_skexgen,
)

ADAPTERS = {
    "cadquery": CadQueryAdapter,
    "cadvec": CadVecAdapter,
    "minimal_json": MinimalJsonAdapter,
    "skexgen": SkexGenAdapter,
}


def get_adapter(name: str, **kwargs) -> Adapter:
    if name not in ADAPTERS:
        raise KeyError(f"unknown adapter {name!r}; choose from {sorted(ADAPTERS)}")
    return ADAPTERS[name](**kwargs)


__all__ = [
    "Adapter", "DEFAULT_TIMEOUT_S", "ADAPTERS", "get_adapter",
    "CadQueryAdapter", "CadVecAdapter", "MinimalJsonAdapter", "SkexGenAdapter",
    "extract_code", "extract_json", "clean_skexgen",
]
