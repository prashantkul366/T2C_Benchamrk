"""Adapters for the sequence-emitting systems: Text2CAD, CADmium, CADFusion.

All three go through Text2CAD's `CadSeqProc` (CADmium vendors a copy of it), so
they share `create_cad_model()` -> OCC solid -> the shared finalisation checks in
base.py. This matters for fairness: the same kernel, the same tessellation
settings and the same validity rules apply to sequence models and CadQuery
models alike.

`T2CBENCH_CADSEQ_PATH` must point at a checkout of SadilKhan/Text2CAD (the repo,
not the gated weights) so `CadSeqProc` is importable.
"""

from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

from t2cbench.adapters.base import Adapter, finalize_solid
from t2cbench.metrics.validity import Validity

N_BIT = 8


def _ensure_cadseq_on_path() -> None:
    root = os.environ.get("T2CBENCH_CADSEQ_PATH")
    if not root:
        raise RuntimeError(
            "set T2CBENCH_CADSEQ_PATH to a checkout of https://github.com/SadilKhan/Text2CAD"
        )
    root = os.path.abspath(root)
    for p in (root, os.path.join(root, "CadSeqProc")):
        if p not in sys.path:
            sys.path.insert(0, p)


# --------------------------------------------------------------------------- #
# Text2CAD: (N, 2) int vector sequence
# --------------------------------------------------------------------------- #

class CadVecAdapter(Adapter):
    """Text2CAD's native output.

    `raw` is a JSON-encoded (N, 2) integer array -- the runner serialises the
    model's `cad_vec` tensor to JSON so that every adapter has the same
    string-in interface.
    """

    name = "cadvec"

    @staticmethod
    def _build(raw: str, out_stl: str, extra: dict) -> dict:
        _ensure_cadseq_on_path()
        from CadSeqProc.cad_sequence import CADSequence

        try:
            vec = np.asarray(json.loads(raw), dtype=np.int64)
        except Exception as e:
            return {"validity": Validity.PARSE_FAIL.value,
                    "diagnostic": f"cad_vec is not valid JSON int array: {e}"}
        if vec.ndim != 2 or vec.shape[1] != 2:
            return {"validity": Validity.PARSE_FAIL.value,
                    "diagnostic": f"expected (N,2) cad_vec, got {vec.shape}"}

        seq = CADSequence.from_vec(vec, bit=N_BIT, post_processing=True)
        seq.create_cad_model()
        return finalize_solid(seq.cad_model, out_stl)


# --------------------------------------------------------------------------- #
# CADmium: minimal JSON
# --------------------------------------------------------------------------- #

class MinimalJsonAdapter(Adapter):
    """CADmium's minimal-JSON output.

    Extraction is brace-balanced rather than regex-based: LLM output routinely
    contains prose before and after the JSON, and nested braces defeat a lazy
    regex. Only formatting is repaired -- a JSON object that parses but
    describes broken geometry is allowed to fail downstream.
    """

    name = "minimal_json"

    @staticmethod
    def _build(raw: str, out_stl: str, extra: dict) -> dict:
        _ensure_cadseq_on_path()
        from CadSeqProc.cad_sequence import CADSequence

        obj = extract_json(raw)
        if obj is None:
            return {"validity": Validity.PARSE_FAIL.value,
                    "diagnostic": "no balanced JSON object in output"}
        if "parts" not in obj:
            # Some checkpoints emit the parts dict without the wrapper.
            if any(k.startswith("part_") for k in obj):
                obj = {"parts": obj}
            else:
                return {"validity": Validity.PARSE_FAIL.value,
                        "diagnostic": f"no 'parts' key; got {list(obj)[:6]}"}

        # CADmium's own system message tells the model to keep the numbering
        # sequential "even if some are null", and the model duly emits
        # "part_2": null -- which CadSeqProc's loader then dereferences and dies
        # on. Dropping null parts is a formatting repair, not a geometric one:
        # a null part carries no geometry, so removing it cannot invent any.
        # Failing the whole sample for it would penalise CADmium for obeying its
        # own prompt, while the CadQuery adapters already ignore the prose that
        # surrounds their code block.
        parts = obj["parts"]
        dropped = 0
        if isinstance(parts, dict):
            kept = {k: v for k, v in parts.items() if isinstance(v, dict) and v}
            dropped = len(parts) - len(kept)
            if not kept:
                return {"validity": Validity.PARSE_FAIL.value,
                        "diagnostic": f"'parts' has no non-empty part ({len(parts)} null)"}
            obj = dict(obj, parts=kept)

        seq = CADSequence.from_minimal_json(obj)
        seq.create_cad_model()
        res = finalize_solid(seq.cad_model, out_stl)
        if dropped:
            note = f"dropped {dropped} null part(s)"
            res["diagnostic"] = f"{note}; {res['diagnostic']}" if res.get("diagnostic") else note
        return res


def extract_json(text: str) -> dict | None:
    """First balanced {...} that parses as a dict containing 'parts'.

    Falls back to the first balanced object that parses at all.
    """
    if not text:
        return None
    text = _strip_fences(str(text))
    first_any = None
    for start in (i for i, c in enumerate(text) if c == "{"):
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                    except Exception:
                        break
                    if isinstance(obj, dict):
                        if "parts" in obj:
                            return obj
                        if first_any is None:
                            first_any = obj
                    break
    return first_any


def _strip_fences(text: str) -> str:
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else text


# --------------------------------------------------------------------------- #
# CADFusion: SkexGen command-sequence tokens
# --------------------------------------------------------------------------- #

class SkexGenAdapter(Adapter):
    """CADFusion's output.

    Format (see CADFusion data/sl_data/convert.py):
        line,x,y <curve_end> arc,x1,y1,x2,y2 <curve_end> circle,... <curve_end>
        <loop_end> <face_end> <sketch_end>
        add,ext_v...,ext_T...,ext_R...,scale,offset <extrude_end>

    Goes through CADFusion's own pipeline verbatim -- `CADparser(bit=6)` ->
    `write_obj_sample` -> `OBJParser` -> `OBJReconverter` -> boolean ops --
    so nothing here is a reimplementation that could drift from their results.
    `T2CBENCH_CADFUSION_PATH` must point at a checkout of microsoft/CADFusion.
    """

    name = "skexgen"
    BIT = 6  # CADFusion calls CADparser(bit=6); see src/rendering_utils/parser.py

    @staticmethod
    def _build(raw: str, out_stl: str, extra: dict) -> dict:
        import tempfile
        from pathlib import Path

        root = os.environ.get("T2CBENCH_CADFUSION_PATH")
        if not root:
            raise RuntimeError(
                "set T2CBENCH_CADFUSION_PATH to a checkout of https://github.com/microsoft/CADFusion"
            )
        root = os.path.abspath(root)
        for p in (os.path.join(root, "src"), root):
            if p not in sys.path:
                sys.path.insert(0, p)

        from rendering_utils.parser import CADparser, write_obj_sample
        from geometry.obj_parser import OBJParser
        from utils.obj_reconverter import OBJReconverter
        from OCC.Core.BRepCheck import BRepCheck_Analyzer

        seq = clean_skexgen(raw)
        if not seq:
            return {"validity": Validity.PARSE_FAIL.value,
                    "diagnostic": "no SkexGen tokens found in output"}

        se_datas = CADparser(SkexGenAdapter.BIT).perform(seq)
        if not se_datas:
            return {"validity": Validity.PARSE_FAIL.value,
                    "diagnostic": "CADparser rejected the token sequence"}

        with tempfile.TemporaryDirectory(prefix="cadfusion_") as tmp:
            write_obj_sample(tmp, se_datas)
            objs = sorted(Path(tmp).glob("*_param.obj"))
            if not objs:
                return {"validity": Validity.PARSE_FAIL.value,
                        "diagnostic": "parser produced no OBJ blocks"}

            cur_solid = None
            for obj in objs:
                _, faces, meta = OBJParser(obj).parse_file(1.0)
                conv = OBJReconverter()
                ext_solid, _, _ = conv.parse_obj(faces, meta)
                op = meta["set_op"]
                if op in ("NewBodyFeatureOperation", "JoinFeatureOperation"):
                    cur_solid = ext_solid if cur_solid is None else conv.my_op(cur_solid, ext_solid, "fuse")
                elif op == "CutFeatureOperation":
                    cur_solid = conv.my_op(cur_solid, ext_solid, "cut")
                elif op == "IntersectFeatureOperation":
                    cur_solid = conv.my_op(cur_solid, ext_solid, "common")
                else:
                    return {"validity": Validity.EXEC_FAIL.value,
                            "diagnostic": f"unknown set operation {op!r}"}
                if cur_solid is None or not BRepCheck_Analyzer(cur_solid).IsValid():
                    return {"validity": Validity.INVALID_SOLID.value,
                            "diagnostic": f"invalid solid after {obj.name}"}

        return finalize_solid(cur_solid, out_stl)


_SKEX_TOKENS = ("<curve_end>", "<loop_end>", "<face_end>", "<sketch_end>", "<extrude_end>")


def clean_skexgen(raw: str) -> str:
    """Trim a raw generation down to the token sequence.

    Keeps everything from the first geometry command to the last
    `<extrude_end>`; an unterminated trailing sketch-extrude block is dropped,
    since a partial block cannot be parsed and counting it as a parse failure
    is the honest outcome.
    """
    if not raw:
        return ""
    text = str(raw).replace("\n", " ").strip()
    starts = [text.find(k) for k in ("line,", "arc,", "circle,") if text.find(k) >= 0]
    if not starts:
        return ""
    start = min(starts)
    end = text.rfind("<extrude_end>")
    if end < 0:
        return ""
    return " ".join(text[start:end + len("<extrude_end>")].split())
