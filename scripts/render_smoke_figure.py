"""Render a smoke-test export as a parts grid: ground truth, every model, the prompt.

    python scripts/render_smoke_figure.py --export export.json \
        --cadprompt-dir /path/to/CAD_Code_Generation --out fig.png --split A

The tables say a model scored CD 240; they do not say it produced a 200:1 sliver
where the answer was a disc. Looking at the parts is how you tell "the pipeline
is broken" apart from "the model is bad", and it takes seconds.

Needs both geometry kernels -- see scripts/setup_scoring_env.sh.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from t2cbench.adapters import get_adapter                        # noqa: E402
from t2cbench.metrics.geometry import compare_meshes, load_mesh, canonicalize  # noqa: E402

# Validated categorical slots, fixed order -- one hue per model, never cycled.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GT_GREY = "#a8a69e"
INK, INK_2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
DEEPCAD_MESH_REPO = "maksimko123/deepcad_test_mesh"


def task_prompt(raw: str) -> str:
    """Pull the task description out of a formatted prompt.

    The recorded prompt is what the model actually received, which for CADmium
    is 2.3 kB of JSON schema in a system turn. Showing that next to a picture of
    a part is useless -- the reader wants the sentence describing the shape, so
    the chat scaffolding and any system turn are stripped.
    """
    if not raw:
        return ""
    t = raw
    # last user turn wins for chat-formatted prompts
    for marker in ("<|im_start|>user", "### Instruction:", "[INST]"):
        if marker in t:
            t = t.rsplit(marker, 1)[1]
            break
    for end in ("<|im_end|>", "### Response:", "[/INST]", "<|im_start|>assistant"):
        t = t.split(end)[0]
    t = re.sub(r"<\|[^>]*\|>", " ", t)
    t = re.sub(r"^\s*(system|user|assistant)\b", " ", t.strip(), flags=re.I)
    return re.sub(r"\s+", " ", t).strip()


def load_export(path):
    text = open(path).read()
    m = re.search(r"<<<T2C_SMOKE_EXPORT_BEGIN>>>\s*(.*?)\s*<<<T2C_SMOKE_EXPORT_END>>>",
                  text, re.DOTALL)
    return json.loads(m.group(1) if m else text)


def find_cadprompt_root(start):
    if not start or not os.path.isdir(start):
        return None
    for root, dirs, _ in os.walk(os.path.abspath(start)):
        if len([d for d in dirs if d.isdigit() and len(d) == 8
                and os.path.exists(os.path.join(root, d, "Ground_Truth.stl"))]) >= 10:
            return root
    return None


def gt_path(sample_id, split, cadprompt_root):
    uid = sample_id.split("/")[1]
    if split == "A":
        from huggingface_hub import hf_hub_download
        try:
            return hf_hub_download(DEEPCAD_MESH_REPO, f"{uid}.stl", repo_type="dataset")
        except Exception:
            return None
    if not cadprompt_root:
        return None
    p = os.path.join(cadprompt_root, uid, "Ground_Truth.stl")
    return p if os.path.exists(p) else None


def render(ax, mesh_path, colour, label_lines=None):
    ax.set_facecolor(SURFACE)
    ax.patch.set_alpha(0.0)
    try:
        m = canonicalize(load_mesh(mesh_path))
        ax.plot_trisurf(m.vertices[:, 0], m.vertices[:, 1], m.vertices[:, 2],
                        triangles=m.faces, color=colour, edgecolor="none",
                        shade=True, linewidth=0)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
        ax.view_init(elev=26, azim=-52)
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        ax.text2D(0.5, 0.5, "render\nfailed", transform=ax.transAxes,
                  ha="center", va="center", fontsize=8, color=MUTED)
    ax.set_axis_off()
    if label_lines:
        ax.text2D(0.5, -0.02, "\n".join(label_lines), transform=ax.transAxes,
                  ha="center", va="top", fontsize=7.5, color=INK_2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", required=True)
    ap.add_argument("--cadprompt-dir", default=None)
    ap.add_argument("--split", default="A", choices=["A", "B"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--prompt-chars", type=int, default=230)
    args = ap.parse_args()

    payload = load_export(args.export)
    gens = [g for g in payload["generations"] if g["split"] == args.split]
    fails = {f["model"] for f in payload.get("failures", []) if f["split"] == args.split}
    if not gens:
        raise SystemExit(f"no generations for split {args.split}")

    cp_root = find_cadprompt_root(args.cadprompt_dir)
    models = sorted({g["model"] for g in gens})
    samples = sorted({g["sample_id"] for g in gens})
    tmp = tempfile.mkdtemp(prefix="t2c_fig_")

    # Build every mesh once, scoring as we go so the cells can be annotated.
    cells, gts, prompts = {}, {}, {}
    for sid in samples:
        gts[sid] = gt_path(sid, args.split, cp_root)
    for g in gens:
        sid = sid_ = g["sample_id"]
        # prompt_text is the split's plain description when the export carries
        # it; older exports only have the formatted prompt, so fall back.
        prompts.setdefault(sid, g.get("prompt_text") or g.get("prompt_tail")
                           or g.get("prompt_head", ""))
        stem = f"{g['model']}__{sid.replace('/', '_')}"
        out = os.path.join(tmp, stem + ".stl")
        res = get_adapter(g["adapter"], timeout_s=args.timeout).run(g["output"], out)
        rec = {"validity": res.validity.value, "mesh": None, "cd": None, "iou": None}
        if res.mesh_path and os.path.exists(res.mesh_path):
            rec["mesh"] = res.mesh_path
            if gts[sid]:
                try:
                    m = compare_meshes(load_mesh(res.mesh_path), load_mesh(gts[sid]),
                                       compute_boolean_iou=False)
                    rec["cd"], rec["iou"] = m.cd, m.iou_voxel
                except Exception:
                    pass
        cells[(g["model"], sid)] = rec
        print(f"  built {stem}: {rec['validity']}", flush=True)

    nrows, ncols = len(models) + 1, len(samples)
    fig = plt.figure(figsize=(3.4 * ncols, 2.5 * nrows + 1.8), facecolor=SURFACE)
    gs = fig.add_gridspec(nrows + 1, ncols, height_ratios=[1.9] + [3] * nrows,
                          hspace=0.05, wspace=0.02)

    # Prompt strip along the top -- the whole point is seeing part and prompt together.
    for c, sid in enumerate(samples):
        ax = fig.add_subplot(gs[0, c]); ax.axis("off")
        txt = task_prompt(prompts.get(sid, ""))[:args.prompt_chars]
        ax.text(0.5, 0.98, sid, ha="center", va="top", fontsize=10,
                color=INK, fontweight="600", transform=ax.transAxes)
        ax.text(0.5, 0.80, "\n".join(textwrap.wrap(txt, 46)[:6]) + "…",
                ha="center", va="top", fontsize=8, color=INK_2, transform=ax.transAxes)

    for c, sid in enumerate(samples):
        ax = fig.add_subplot(gs[1, c], projection="3d")
        render(ax, gts[sid], GT_GREY) if gts[sid] else ax.set_axis_off()
        if c == 0:
            ax.text2D(-0.12, 0.5, "GROUND\nTRUTH", transform=ax.transAxes, rotation=90,
                      va="center", ha="center", fontsize=10, color=INK, fontweight="600")

    for r, model in enumerate(models):
        colour = PALETTE[r % len(PALETTE)]
        for c, sid in enumerate(samples):
            ax = fig.add_subplot(gs[r + 2, c], projection="3d")
            rec = cells.get((model, sid))
            if rec and rec["mesh"]:
                lab = []
                if rec["cd"] is not None:
                    lab.append(f"CD {rec['cd']:.1f} · IoU {rec['iou']:.3f}")
                if rec["validity"] != "OK":
                    lab.append(rec["validity"])
                render(ax, rec["mesh"], colour, lab)
            else:
                ax.set_axis_off(); ax.patch.set_alpha(0.0)
                why = rec["validity"] if rec else ("GEN_FAIL" if model in fails else "—")
                ax.text2D(0.5, 0.5, why.replace("_", "\n"), transform=ax.transAxes,
                          ha="center", va="center", fontsize=9, color=MUTED)
            if c == 0:
                ax.text2D(-0.12, 0.5, model, transform=ax.transAxes, rotation=90,
                          va="center", ha="center", fontsize=9.5,
                          color=colour, fontweight="600")

    title = ("Split A — DeepCAD test, normalised units"
             if args.split == "A" else "Split B — CADPrompt, real dimensions")
    fig.suptitle(f"Smoke test parts · {title}", fontsize=14, color=INK,
                 fontweight="600", x=0.01, ha="left", y=0.995)
    fig.text(0.01, 0.978,
             "All parts canonicalised to the unit cube, so only shape is compared. "
             "CD is x1000, lower is better; IoU higher is better.",
             fontsize=9, color=MUTED, ha="left")
    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
