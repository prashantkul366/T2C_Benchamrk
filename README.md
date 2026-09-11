# T2C-Bench

**A fair benchmark for text-to-CAD generation.**

Six systems — Text2CAD, CADmium-7B, CADFusion, cadrille, Text-to-CadQuery, and general LLMs
(Qwen, Llama, Mistral, DeepSeek) — evaluated on 900 prompts under one protocol, with Chamfer
distance, IoU, invalidity ratio, F-score, Hausdorff and topology metrics.

---

## Why this exists

The five published text-to-CAD systems cannot be compared from their papers. Three concrete
reasons, all verified by reading their code rather than their tables:

**1. They emit five incompatible representations.**

| System | Output |
|---|---|
| Text2CAD | `(N,2)` int vector sequence, 8-bit quantised |
| CADmium | minimal JSON (`parts` → `sketch` → `extrusion`) |
| CADFusion | SkexGen command tokens (`line,x,y <curve_end> …`) |
| cadrille | CadQuery code, result in `r`, integer ~100 scale |
| Text-to-CadQuery | CadQuery script that exports its own STL |
| general LLMs | whatever you ask for, plus prose |

**2. Their four eval scripts compute three different Chamfer distances.**

| Source | Normalisation | Points | Translation-invariant? |
|---|---|---|---|
| Text2CAD | `pts / (max(pts) − min(pts))`, flattened, no centering | 8192 | **no** |
| cadrille | centre bbox, max extent → 1, place at (0.5,0.5,0.5) | 8192 | yes |
| CADFusion | `pts / max(abs(pts))`, no centering | **2000** | **no** |

Two of the three give a *different* CD for a translated copy of the same solid. And all three
headline numbers are **best-of-N oracle** scores (N=5, min CD per sample), which a
single-sample general LLM is not.

**3. 93.5% of CADPrompt is in the training set of the fine-tuned models.**

CADPrompt's 200 objects are named with DeepCAD uids. Checked against the DeepCAD test split:

```
in DeepCAD TEST split :  13 / 200   ( 6.5%)
in DeepCAD train/val  : 187 / 200   (93.5%)   <- seen by Text2CAD, CADmium, cadrille
```

Reporting one averaged CADPrompt number hands the fine-tuned systems a memorisation advantage
over the general LLMs and calls it a result.

T2C-Bench re-runs every system under one protocol and reports all three problems explicitly.

---

## The design in one picture

```
prompt ──► [system] ──► native output ──► [adapter] ──► OCC solid ──► mesh ──► metrics
                                              │
                                              └──► one of 8 validity codes
```

A mesh is the only thing all six representations agree on, so that is where the measurement
plane goes. Key decisions:

- **The adapter is part of the system.** Code that will not execute is the model's failure.
  Invalid outputs are scored, never dropped.
- **Same sampling budget for everyone.** pass@1 is the headline; best-of-5 is a separate table
  for comparing against published numbers.
- **`Score = P(valid) × F1@0.02`** is the ranking column. A model that answers 10% of prompts
  perfectly would top a median-CD-over-valid-only leaderboard; it cannot top this.
- **No ICP.** DeepCAD ground truth is in a canonical frame and the prompts describe coordinate
  systems, so orientation is part of what is tested. Available as `--align-icp` for ablation.
- **IR is decomposed 8 ways.** CadQuery emitters fail at execution; sequence emitters fail at
  geometry. A scalar invalidity ratio hides exactly that difference.

Full protocol: **[docs/02_benchmark_design.md](docs/02_benchmark_design.md)**.
What each system is, where its weights are, what it emits: **[docs/01_model_survey.md](docs/01_model_survey.md)**.
Everything in one place — design, codebase map, all results, findings with their causal reasoning,
the fairness audit, and the threats to validity: **[docs/03_paper_reference.md](docs/03_paper_reference.md)**.

---

## Evaluation set — 900 prompts

**Split A — Text2CAD test, 125 shapes × 4 prompt levels = 500 prompts.**
125 uids drawn from the 8,046-shape DeepCAD test split after geometric deduplication (D2 shape
distribution + covariance eigenvalues + solidity), then stratified across four complexity bins.
Dedup matters: on a 220-mesh sample of the test split, 33% of shapes are flat plates
(thinnest/longest extent < 0.15) and 32% tessellate to <=24 triangles. At the calibrated
eps = 0.08 the pass folds ~11% of shapes as near-identical - enough to thin the duplicate
mass without merging distinct parts (eps is measured against the nearest-neighbour distance
distribution; see `t2cbench/data/dedup.py`).

| Level | Column | What the prompt looks like |
|---|---|---|
| L0 | `abstract` | *"A rectangular block with a flat top and bottom."* |
| L1 | `beginner` | design steps, no measurements |
| L2 | `intermediate` | generalised geometry, some detail abstracted |
| L3 | `expert` | every line's start/end point, sketch scale, extrusion depth |

L3 is nearly a transcription of the CAD program. **A system can win at L3 by learning a
grammar and be useless at L0** — which is why nothing here is reported pooled-only.

**Split B — CADPrompt, 200 objects × 2 prompt variants = 400 prompts**, every row flagged
`clean` (13) or `contaminated` (187). Used as a **memorisation probe**, not a leaderboard:
the per-model clean↔contaminated gap is the number to read.

---

## Quick start

You need a Colab A100 for generation; everything else is CPU.

```bash
git clone https://github.com/prashantkul366/T2C_Benchamrk
cd T2C_Benchamrk
pip install -r requirements.txt
conda install -c conda-forge pythonocc-core=7.7.0   # sequence adapters only
```

Then run the notebooks in order:

| Notebook | Runtime | What it does |
|---|---|---|
| `00_setup_data.ipynb` | CPU | builds both splits (~40 min, cached) |
| `01_run_specialised_models.ipynb` | A100 | one specialised system per session |
| `02_run_general_llms.ipynb` | A100 | one general LLM per session |
| `03_evaluate_and_report.ipynb` | CPU | scores everything, emits tables + figures |

Or from the command line:

```bash
# 1. build the splits
python -m t2cbench.data.build_splits --stage all --out data \
    --cadprompt-dir /path/to/CAD_Code_Generation/CADPrompt

# 2. generate (GPU)
python -m t2cbench.runners.run_hf \
    --model chandar-lab/CADmium-7B --base Qwen/Qwen2.5-Coder-7B-Instruct \
    --name cadmium-7b --template native.cadmium \
    --split data/split_a.jsonl --out results/raw/cadmium-7b_splitA.jsonl

# 3. score (CPU)
python -m t2cbench.evaluate \
    --predictions results/raw/cadmium-7b_splitA.jsonl \
    --split data/split_a.jsonl --adapter minimal_json \
    --model-name cadmium-7b --keep-meshes results/meshes/cadmium-7b \
    --out results/scored/cadmium-7b_splitA.jsonl

# 4. report
python -m t2cbench.report.tables  --scored 'results/scored/*.jsonl' --out results/tables
python -m t2cbench.report.figures --scored 'results/scored/*.jsonl' --out results/figures \
    --mesh-root results/meshes
```

---

## Metrics

| Metric | Definition | Reported |
|---|---|---|
| **CD** | symmetric `mean(d²)` both directions, ×1000, after canonicalisation | median (headline), mean, trimmed mean |
| **F1@τ** | point-cloud F-score, τ = 0.02 and 0.05 of the unit diagonal | mean |
| **IoU** | voxel IoU at 64³ | mean |
| **IoU-bool** | mesh-boolean IoU (cadrille's definition) | cross-check only |
| **HD95** | 95th-percentile symmetric Hausdorff | median |
| **IR** | `1 − P(OK)` plus an 8-way breakdown | % |
| **Topology** | Euler characteristic match, solid/face counts, watertightness | rates |
| **Scale** | bbox / volume / area relative error, pre-canonicalisation | CADPrompt only |
| **Sequence F1** | line / arc / circle / extrusion | separate table — only 3 of 6 systems can score it |

Canonicalisation is `centre bbox → scale max extent to 1 → place in [0,1]³`, applied
identically to prediction and ground truth.

Voxel IoU is primary over boolean IoU because `trimesh`'s boolean intersection fails on a large
fraction of CAD output, and cadrille's implementation swallows those failures in a bare
`except: pass` — converting them to *missing values* rather than zeros, which inflates the
reported mean.

---

## Outputs

1. Main table — Split A, pass@1, per level
2. Best-of-5 table — for comparison with published numbers
3. Contamination table — clean vs contaminated, with the memorisation gap
4. Ablation table — base LLM vs its fine-tuned descendant (Qwen2.5-Coder-7B→CADmium,
   Llama-3-8B→CADFusion, Qwen2-VL-2B→cadrille)
5. Complexity table, failure-mode table
6. `fig1_failure_modes.png` — where each system fails
7. `fig2_level_sensitivity.png` — CD and Score vs L0→L3
8. `fig3_qualitative.png` — rendered predictions, models × prompts, GT on top

---

## Assets

| Asset | Source | Size | Gated |
|---|---|---|---|
| Text2CAD checkpoint | `SadilKhan/Text2CAD` | 92 MB | **yes** |
| Text2CAD L0–L3 prompts | `ricemonster/NeurIPS11092` (ungated mirror) | 1.3 GB | no |
| CADmium-7B adapter | `chandar-lab/CADmium-7B` | 646 MB | no |
| CADFusion v1.1 adapter | `microsoft/CADFusion` | 4.3 GB | no |
| Llama-3-8B (CADFusion base) | `meta-llama/Meta-Llama-3-8B` | 16 GB | **yes** |
| cadrille SFT / RL | `maksimko123/cadrille`, `-rl` | 4.4 GB | no |
| DeepCAD test meshes | `maksimko123/deepcad_test_mesh` | 260 MB | no |
| CadQuery GT, 171k uids | `maksimko123/text2cad` | 83 MB | no |
| CADPrompt | `Kamel773/CAD_Code_Generation` | ~1 GB | no |

Only two gates to clear by hand. The Text2CAD *prompts* route around theirs via the mirror;
only the *checkpoint* still needs the licence accepted.

---

## Known limitations

Stated up front because a reviewer will find them anyway:

1. **CADFusion's split is unverified.** It trains on SkexGen, whose train/test boundary is not
   DeepCAD's. Until the SkexGen split is obtained, some Split A shapes may be in its training
   data. Flagged in the results.
2. **Prompt-distribution mismatch.** CADmium and cadrille were trained on their own paraphrases,
   not Text2CAD L0–L3, so their Split A numbers include a distribution-shift penalty Text2CAD
   does not pay.
3. **Ground truth is DeepCAD** — sketch-extrude only. Nothing here measures fillets, chamfers,
   lofts, revolves or assemblies.
4. **n=13 clean CADPrompt** is too small to rank on; it is a probe, not a leaderboard.
5. **Tessellation tolerance** (`linear_deflection=0.001`) moves CD in the 4th decimal. Fixed
   across all systems, but it is a free parameter.

---

## Repository layout

```
configs/          model registry, prompt templates, ablation pairs
docs/             model survey, benchmark design, complete study reference
notebooks/        four Colab notebooks (generated by scripts/make_notebooks.py)
t2cbench/
  adapters/       native output -> mesh, sandboxed with hard timeouts
  data/           split builders, geometric dedup
  metrics/        geometry, topology, validity taxonomy
  runners/        generation drivers
  report/         tables and figures
  evaluate.py     scoring driver
```

## Citation

Built on: [Text2CAD](https://github.com/SadilKhan/Text2CAD) (NeurIPS'24),
[CADFusion](https://github.com/microsoft/CADFusion) (ICML'25),
[CADmium](https://github.com/chandar-lab/CADmium) (TMLR),
[cadrille](https://github.com/col14m/cadrille),
[Text-to-CadQuery](https://github.com/Text-to-CadQuery/Text-to-CadQuery),
[CADPrompt](https://github.com/Kamel773/CAD_Code_Generation) (ICLR'25),
and [DeepCAD](https://github.com/ChrisWu1997/DeepCAD).
