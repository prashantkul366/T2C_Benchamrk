"""Generate the Colab notebooks from the cell definitions below.

Keeping the notebooks generated rather than hand-edited means the setup steps
stay identical across them -- a drifting install cell between two model
notebooks is exactly how "the same protocol" quietly stops being the same.

    python scripts/make_notebooks.py
"""

from __future__ import annotations

import json
import os

REPO = "https://github.com/prashantkul366/T2C_Benchamrk"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip().splitlines(True)}


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "A100"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 0,
    }


MOUNT = f"""
# Mount Drive so predictions survive a Colab timeout, then get the harness.
from google.colab import drive
drive.mount('/content/drive')

import os
WORK = '/content/drive/MyDrive/t2c_bench'
os.makedirs(WORK, exist_ok=True)
os.environ['T2C_WORK'] = WORK

!git clone -q {REPO} /content/t2cbench_repo || (cd /content/t2cbench_repo && git pull -q)
%cd /content/t2cbench_repo
!pip install -q -e . 2>/dev/null || pip install -q -r requirements.txt
print('work dir:', WORK)
"""

EVAL_DEPS = """
# CPU-side evaluation dependencies. embreex matters: without it the exact
# point-in-solid test falls back to a pure-Python ray engine and voxel IoU goes
# from ~0.5s to ~90s per sample.
!pip install -q trimesh rtree embreex manifold3d scipy pandas matplotlib tabulate pyyaml tqdm
!pip install -q cadquery
import trimesh, embreex
print('trimesh', trimesh.__version__, '| embreex present')
"""


# --------------------------------------------------------------------------- #

NB_00 = notebook([
    md("""
# 00 · Build the evaluation splits

Runs on **CPU** — no GPU needed, so use a CPU runtime and save your GPU quota.

Produces:
* `split_a.jsonl` — 125 geometrically-deduplicated DeepCAD test shapes × L0–L3 = **500 prompts**
* `split_b.jsonl` — 200 CADPrompt objects × 2 prompt variants = **400 prompts**, each flagged
  `clean` or `contaminated`

The dedup pass is the slow part (~30 min for 8k meshes on 16 processes) and is cached,
so re-running is cheap.
"""),
    code(MOUNT),
    code(EVAL_DEPS),
    md("""
### CADPrompt

Cloned from the paper's repo. Its 200 directories are named with DeepCAD uids, which is
what lets us check contamination at all.
"""),
    code("""
!git clone -q --depth 1 https://github.com/Kamel773/CAD_Code_Generation /content/CADPrompt || true
CADPROMPT = '/content/CAD_Code_Generation/CADPrompt'
print(len(os.listdir(CADPROMPT)), 'CADPrompt objects')
"""),
    md("""
### Build both splits

The Text2CAD L0–L3 prompts come from `text2cad_v1.1.csv`. The canonical copy is in the
**gated** `SadilKhan/Text2CAD` repo; `ricemonster/NeurIPS11092` mirrors the same file
ungated, which is what the builder uses by default, so this step needs no token.
"""),
    code("""
!python -m t2cbench.data.build_splits \\
    --stage all \\
    --out $T2C_WORK/data \\
    --cache $T2C_WORK/hf_cache \\
    --cadprompt-dir /content/CAD_Code_Generation/CADPrompt \\
    --n-uids 125 --workers 16
"""),
    md("### Sanity-check what was built"),
    code("""
import json, collections, pandas as pd
a = [json.loads(l) for l in open(f'{WORK}/data/split_a.jsonl')]
b = [json.loads(l) for l in open(f'{WORK}/data/split_b.jsonl')]
print(f'Split A: {len(a)} prompts over {len({r["uid"] for r in a})} uids')
print('  by level     :', dict(collections.Counter(r['level'] for r in a)))
print('  by complexity:', dict(collections.Counter(r['complexity_bin'] for r in a)))
print(f'Split B: {len(b)} prompts over {len({r["uid"] for r in b})} uids')
print('  contamination:', dict(collections.Counter(r['contamination'] for r in b)))

ex = a[0]
print(f"\\nExample -- {ex['sample_id']} ({ex['complexity_bin']})")
for lv in ['L0','L1','L2','L3']:
    p = next(r['prompt'] for r in a if r['uid']==ex['uid'] and r['level']==lv)
    print(f"  {lv}: {p[:150]}{'...' if len(p)>150 else ''}")
"""),
    md("""
Both split files now live in Drive and every later notebook reads them from there.
**Do not rebuild them between model runs** — a different split means the models are no
longer being compared on the same thing.
"""),
])


NB_01 = notebook([
    md("""
# 01 · Run the specialised text-to-CAD systems

One model per session on an **A100**. Predictions stream to Drive and every runner is
resumable, so a Colab timeout costs only the unfinished tail.

Set `MODEL` in the config cell and run top to bottom. Order below is cheapest first.

| MODEL | VRAM | ~time for 900 prompts |
|---|---|---|
| `text2cad` | <2 GB | ~25 min |
| `cadrille` / `cadrille-rl` | ~5 GB | ~40 min |
| `t2cq-qwen-3b` | ~8 GB | ~25 min |
| `cadmium-7b` | ~17 GB | ~50 min |
| `t2cq-mistral-7b` | ~16 GB | ~40 min |
| `cadfusion-v1.1` | ~18 GB | ~60 min |
"""),
    code(MOUNT),
    code("""
#@title Configuration
MODEL = 'cadmium-7b'  #@param ['text2cad','cadmium-7b','cadfusion-v1.1','cadrille','cadrille-rl','t2cq-qwen-3b','t2cq-mistral-7b']
SPLIT = 'A'           #@param ['A','B']
MODE  = 'pass_at_1'   #@param ['pass_at_1','best_of_k']

import yaml, os
cfg = yaml.safe_load(open('configs/models.yaml'))[MODEL]
SPLIT_FILE = f"{os.environ['T2C_WORK']}/data/split_{SPLIT.lower()}.jsonl"
OUT = f"{os.environ['T2C_WORK']}/results/raw/{MODEL}_split{SPLIT}_{MODE}.jsonl"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
print(yaml.dump(cfg, sort_keys=False))
print('split :', SPLIT_FILE)
print('output:', OUT)
"""),
    code("""
!pip install -q transformers accelerate peft bitsandbytes sentencepiece
import torch; print(torch.cuda.get_device_name(0), f'{torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB')
"""),
    md("""
### Gated weights

Two assets need a licence accepted on their HuggingFace page:

* **Text2CAD checkpoint** — `SadilKhan/Text2CAD` (needed only for `MODEL='text2cad'`)
* **Meta-Llama-3-8B** — the base CADFusion adapts (needed only for `cadfusion-v1.1`)

Everything else is open. Skip this cell for the other models.
"""),
    code("""
from huggingface_hub import login
login()   # paste a token with read access to the repos you accepted
"""),
    md("### Run"),
    code("""
if cfg['runner'] == 'hf':
    base = f"--base {cfg['base_model']}" if cfg.get('lora') else ''
    sub  = f"--subfolder {cfg['subfolder']}" if cfg.get('subfolder') else ''
    chat = '' if cfg.get('chat_template', True) else '--no-chat-template'
    q4   = '--load-4bit' if cfg.get('load_4bit') else ''
    tmpl = cfg.get('template', 'general_one_shot')
    !python -m t2cbench.runners.run_hf \\
        --model {cfg['weights']} {base} {sub} \\
        --name {MODEL} --template {tmpl} {chat} {q4} \\
        --split {SPLIT_FILE} --out {OUT} --mode {MODE} --batch-size 8

elif cfg['runner'] == 'cadrille':
    !git clone -q --depth 1 https://github.com/col14m/cadrille /content/cadrille || true
    !pip install -q qwen-vl-utils
    n = 5 if MODE == 'best_of_k' else 1
    t = 0.7 if MODE == 'best_of_k' else 0.0
    !python -m t2cbench.runners.run_cadrille \\
        --cadrille-repo /content/cadrille --checkpoint {cfg['weights']} \\
        --name {MODEL} --split {SPLIT_FILE} --out {OUT} \\
        --n-samples {n} --temperature {t} --batch-size 16

elif cfg['runner'] == 'text2cad':
    !git clone -q --depth 1 https://github.com/SadilKhan/Text2CAD /content/Text2CAD || true
    from huggingface_hub import hf_hub_download
    ckpt = hf_hub_download(cfg['weights'], cfg['checkpoint_file'], repo_type='dataset')
    n = 5 if MODE == 'best_of_k' else 1
    !python -m t2cbench.runners.run_text2cad \\
        --text2cad-repo /content/Text2CAD --checkpoint {ckpt} \\
        --name {MODEL} --split {SPLIT_FILE} --out {OUT} --n-samples {n}
"""),
    md("### Check the output before you close the session"),
    code("""
import json
rows = [json.loads(l) for l in open(OUT)]
print(f'{len(rows)} generations, {len({r["sample_id"] for r in rows})} unique prompts')
print('\\n--- first output ---')
print(rows[0]['output'][:900])
"""),
    md("""
Now either switch `MODEL` and re-run, or move to notebook **03** to score everything.
Scoring is CPU-only, so do it in a separate CPU runtime.
"""),
])


NB_02 = notebook([
    md("""
# 02 · Run the general-LLM baselines

Same prompts, same sampling budget, same adapter, same metrics as the specialised systems.

The first three models are the **base models** of CADmium, CADFusion and cadrille. Running
them turns the benchmark into three controlled ablations — same weights, same prompts, the
only difference being the CAD fine-tune. That comparison is the strongest claim this
benchmark can make and it costs nothing extra.
"""),
    code(MOUNT),
    code("""
#@title Configuration
MODEL = 'qwen25-coder-7b'  #@param ['qwen25-coder-7b','llama3-8b-instruct','qwen2-vl-2b','mistral-7b-instruct','deepseek-coder-6.7b','qwen25-coder-32b']
SPLIT = 'A'                #@param ['A','B']
SHOT  = 'general_one_shot' #@param ['general_one_shot','general_zero_shot']
MODE  = 'pass_at_1'        #@param ['pass_at_1','best_of_k']

import yaml, os
cfg = yaml.safe_load(open('configs/models.yaml'))[MODEL]
tag = MODEL + ('_0shot' if SHOT.endswith('zero_shot') else '')
SPLIT_FILE = f"{os.environ['T2C_WORK']}/data/split_{SPLIT.lower()}.jsonl"
OUT = f"{os.environ['T2C_WORK']}/results/raw/{tag}_split{SPLIT}_{MODE}.jsonl"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
print(yaml.dump(cfg, sort_keys=False)); print('output:', OUT)
"""),
    code("""
!pip install -q transformers accelerate bitsandbytes sentencepiece
import torch; print(torch.cuda.get_device_name(0))
# llama3-8b-instruct is gated -- accept Meta's licence, then log in.
# from huggingface_hub import login; login()
"""),
    code("""
q4 = '--load-4bit' if cfg.get('load_4bit') else ''
!python -m t2cbench.runners.run_hf \\
    --model {cfg['weights']} --name {tag} --template {SHOT} {q4} \\
    --split {SPLIT_FILE} --out {OUT} --mode {MODE} --batch-size 8
"""),
    md("""
### Look at what it actually produced

Worth eyeballing: general LLMs fail in ways the fine-tuned systems do not — prose outside the
code fence, a `Sketch` that is never extruded, imports of libraries that are not installed.
The 8-way validity taxonomy in notebook 03 separates those from real geometric errors.
"""),
    code("""
import json
rows = [json.loads(l) for l in open(OUT)]
print(f'{len(rows)} generations')
for r in rows[:2]:
    print('='*70); print(r['output'][:700])
"""),
])


NB_03 = notebook([
    md("""
# 03 · Score everything, build the tables and figures

**CPU runtime** — no GPU. The geometry kernel is the bottleneck, not the GPU.

This notebook turns raw model outputs into meshes, scores every mesh under the single
shared protocol, and emits the six tables and three figures.
"""),
    code(MOUNT),
    code(EVAL_DEPS),
    md("""
### pythonocc-core

Needed only by the sequence adapters (Text2CAD, CADmium, CADFusion) — CadQuery uses OCP
instead. On Colab, conda is the reliable route.
"""),
    code("""
!pip install -q condacolab
import condacolab; condacolab.install()   # this restarts the runtime; re-run the cells above after it
"""),
    code("""
!mamba install -q -y -c conda-forge pythonocc-core=7.7.0
!git clone -q --depth 1 https://github.com/SadilKhan/Text2CAD /content/Text2CAD || true
!git clone -q --depth 1 https://github.com/microsoft/CADFusion /content/CADFusion || true
import os
os.environ['T2CBENCH_CADSEQ_PATH'] = '/content/Text2CAD'
os.environ['T2CBENCH_CADFUSION_PATH'] = '/content/CADFusion'
"""),
    md("""
### Score every raw prediction file

The adapter is chosen from `configs/models.yaml`, so each system is scored through its own
output format but against the *same* metrics. Invalid outputs are scored as failures, never
dropped — that is the whole point.
"""),
    code("""
import glob, os, yaml, re
WORK = os.environ['T2C_WORK']
models = yaml.safe_load(open('configs/models.yaml'))

for raw in sorted(glob.glob(f'{WORK}/results/raw/*.jsonl')):
    stem = os.path.basename(raw)[:-6]
    name = re.split(r'_split[AB]', stem)[0]
    key  = name.replace('_0shot', '')
    adapter = models.get(key, {}).get('adapter', 'cadquery')
    split_letter = 'a' if '_splitA' in stem else 'b'
    scored = f'{WORK}/results/scored/{stem}.jsonl'
    if os.path.exists(scored):
        print('skip (done):', stem); continue
    print(f'>>> {stem}  adapter={adapter}')
    !python -m t2cbench.evaluate \\
        --predictions {raw} \\
        --split {WORK}/data/split_{split_letter}.jsonl \\
        --adapter {adapter} --model-name {name} \\
        --keep-meshes {WORK}/results/meshes/{name} \\
        --out {scored} --workers 8 --timeout 20
"""),
    md("### Tables"),
    code("""
!python -m t2cbench.report.tables \\
    --scored '{WORK}/results/scored/*.jsonl' \\
    --out {WORK}/results/tables \\
    --ablation-pairs configs/ablation_pairs.json

from IPython.display import Markdown, display
display(Markdown(open(f'{WORK}/results/tables/RESULTS.md').read()))
"""),
    md("### Figures"),
    code("""
!python -m t2cbench.report.figures \\
    --scored '{WORK}/results/scored/*.jsonl' \\
    --out {WORK}/results/figures \\
    --split {WORK}/data/split_a.jsonl \\
    --mesh-root {WORK}/results/meshes \\
    --families configs/model_families.json

from IPython.display import Image, display
for f in ['fig1_failure_modes','fig2_level_sensitivity','fig3_qualitative']:
    display(Image(f'{WORK}/results/figures/{f}.png'))
"""),
    md("""
### Reading the results

* **Score = P(valid) × F1@0.02** is the ranking column. It cannot be gamed by answering few
  prompts well, which median-CD-over-valid-only can.
* **Compare published numbers against the best-of-5 table, not pass@1.** Text2CAD, CADFusion
  and cadrille all report oracle best-of-N.
* **The contamination table is a memorisation probe**, not a leaderboard — n=13 clean
  CADPrompt objects is far too small to rank on. The clean↔contaminated *gap* per model is
  the number to read.
* **CADFusion carries a "split unverified" caveat**: it trains on SkexGen, whose train/test
  boundary is not DeepCAD's, so some Split A shapes may be in its training data.
"""),
])


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, nb in [("00_setup_data", NB_00),
                     ("01_run_specialised_models", NB_01),
                     ("02_run_general_llms", NB_02),
                     ("03_evaluate_and_report", NB_03)]:
        path = os.path.join(OUT_DIR, f"{name}.ipynb")
        with open(path, "w") as f:
            json.dump(nb, f, indent=1)
        print(f"wrote {path} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
