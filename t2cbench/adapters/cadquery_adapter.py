"""CadQuery adapter -- cadrille, Text-to-CadQuery, CADSmith and all general LLMs.

Three dialects have to be handled by one adapter, and the differences are real:

  cadrille          : bare script, result always in `r`, integer coords ~100 scale,
                      no export call.  `r=w0.sketch()...finalize().extrude(28)`
  Text-to-CadQuery  : full script with `import cadquery as cq`, named part
                      variables, and its own `exporters.export(part, 'x.stl')`.
  general LLMs      : anything. Prose, markdown fences, `show_object(...)`,
                      `result = ...`, or a bare expression.

Result discovery order is fixed and applied identically to every system, so no
dialect gets a bespoke rescue path:
    1. an STL the script exported itself
    2. `r`  (cadrille's contract)
    3. `result`, `part`, `solid`, `shape`, `obj`, `assembly`  -- common LLM names
    4. the last bound Workplane / Shape / Assembly in the namespace

Extraction is deliberately generous about *formatting* (fences, prose) and
deliberately strict about *geometry*: nothing here repairs a broken model.
"""

from __future__ import annotations

import os
import re

from t2cbench.adapters.base import Adapter, finalize_mesh_file
from t2cbench.metrics.validity import Validity

# Names an LLM plausibly binds its result to, in priority order.
RESULT_NAMES = ("r", "result", "part", "solid", "shape", "obj",
                "model", "assembly", "res", "final", "workpiece")

_FENCE_RE = re.compile(r"```(?:python|py|cadquery)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_UNCLOSED_FENCE_RE = re.compile(r"```(?:python|py|cadquery)?\s*\n(.*)", re.DOTALL | re.IGNORECASE)


def extract_code(raw: str) -> str:
    """Pull Python out of a model response.

    Handles fenced blocks (including an unterminated final fence, which is what
    a truncated generation looks like), and falls back to the raw text when the
    model emitted bare code.
    """
    if raw is None:
        return ""
    text = str(raw).strip()

    blocks = _FENCE_RE.findall(text)
    if blocks:
        # Concatenate all fenced blocks: models often split imports from body.
        return "\n\n".join(b.strip() for b in blocks)

    m = _UNCLOSED_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()

    # No fences. If it looks like prose followed by code, start at the first
    # import/assignment that mentions cadquery.
    lines = text.splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith(("import ", "from ")) or "cq.Workplane" in s or "cadquery" in s:
            return "\n".join(lines[i:]).strip()
    return text


class CadQueryAdapter(Adapter):
    name = "cadquery"

    @staticmethod
    def _build(raw: str, out_stl: str, extra: dict) -> dict:
        import cadquery as cq

        import shutil
        import tempfile

        code = extract_code(raw)
        if not code.strip():
            return {"validity": Validity.PARSE_FAIL.value,
                    "diagnostic": "no python code found in output"}

        # compile separately so a syntax error is PARSE_FAIL, not EXEC_FAIL
        try:
            compiled = compile(code, "<model_output>", "exec")
        except SyntaxError as e:
            return {"validity": Validity.PARSE_FAIL.value,
                    "diagnostic": f"SyntaxError line {e.lineno}: {e.msg}"}

        # A PRIVATE scratch directory per call. Generated scripts export to
        # relative paths ("Ground_Truth.stl"), so the process must chdir
        # somewhere -- and it must not be the shared output directory: with N
        # workers running concurrently, "files that appeared while I ran" would
        # pick up another worker's STL and move it to this sample's output.
        with tempfile.TemporaryDirectory(prefix="t2c_cq_") as sandbox:
            os.chdir(sandbox)

            ns: dict = {
                "__name__": "__main__",
                "cq": cq,
                "cadquery": cq,
                "show_object": lambda *a, **k: None,   # CQ-editor no-op
                "log": lambda *a, **k: None,
                "debug": lambda *a, **k: None,
            }
            exec(compiled, ns)

            # 1. the script exported a mesh itself (Text-to-CadQuery style)
            produced = sorted(f for f in os.listdir(sandbox) if f.lower().endswith(".stl"))
            if produced:
                shutil.move(os.path.join(sandbox, produced[0]), out_stl)
                return finalize_mesh_file(out_stl)

            # 2-4. find the result object in the namespace
            obj = _find_result(ns, cq)
            if obj is None:
                return {"validity": Validity.EMPTY_SOLID.value,
                        "diagnostic": "script ran but produced no CadQuery result object"}

            return _export(obj, out_stl, cq)


def _find_result(ns: dict, cq):
    """Locate the result object, preferring the conventional names."""
    candidates = (cq.Workplane, cq.Assembly, cq.Shape, cq.Sketch)

    for name in RESULT_NAMES:
        v = ns.get(name)
        if isinstance(v, candidates):
            return v

    # Last bound candidate wins -- dicts preserve insertion order, so this is
    # the final assignment in the script.
    last = None
    for k, v in ns.items():
        if k.startswith("__"):
            continue
        if isinstance(v, candidates):
            last = v
    return last


def _export(obj, out_stl: str, cq) -> dict:
    """Export whatever we found, then run the shared mesh checks.

    Note CadQuery is built on OCP, not pythonocc's `OCC` bindings -- only the
    sequence adapters need pythonocc-core, and this path must not import it.
    """
    shape = obj
    if isinstance(shape, cq.Assembly):
        shape = shape.toCompound()
    if isinstance(shape, cq.Workplane):
        vals = [v for v in shape.vals() if isinstance(v, cq.Shape)]
        if not vals:
            return {"validity": Validity.EMPTY_SOLID.value,
                    "diagnostic": "Workplane holds no solids"}
        shape = vals[0] if len(vals) == 1 else cq.Compound.makeCompound(vals)
    if isinstance(shape, cq.Sketch):
        return {"validity": Validity.EMPTY_SOLID.value,
                "diagnostic": "result is a 2D Sketch, never extruded"}

    try:
        cq.exporters.export(shape, out_stl, tolerance=0.001, angularTolerance=0.5)
    except TypeError:
        cq.exporters.export(shape, out_stl)

    n_solids = 1
    try:
        n_solids = max(1, len(shape.Solids()))
    except Exception:
        pass
    return finalize_mesh_file(out_stl, n_solids=n_solids)
