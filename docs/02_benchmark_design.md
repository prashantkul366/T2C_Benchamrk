# T2C-Bench: a fair text-to-CAD benchmark

## 0. The problem in one paragraph

Five text-to-CAD systems emit five incompatible representations (int vector sequence, minimal JSON,
SkexGen tokens, two dialects of CadQuery). Their four evaluation scripts compute three different
Chamfer distances and two different invalidity ratios, all at different sample counts, most of them
best-of-N. 93.5% of CADPrompt sits in the training set of the fine-tuned models but not of the
general LLMs. Any table that copies published numbers into shared rows is measuring the protocol,
not the models.

T2C-Bench fixes all of that by re-running every system under one protocol.

---

## 1. Design principles

1. **One measurement plane.** Every system's output is reduced to a *mesh* before any metric is
   computed. Meshes are the only thing all six representations agree on.
2. **The adapter is part of the system.** If a model emits code that does not execute, that is the
   model's failure, not an excuse to drop the sample. Invalid outputs are counted, never silently
   skipped.
3. **Identical sampling budget.** Same N, same temperature policy, same max tokens, for everyone.
4. **Contamination is reported, not hidden.** Every row carries a clean/contaminated flag.
5. **Prompt-level fairness.** No system gets its native prompt style as the headline while others
   get an out-of-distribution one. Results are always broken out by level. The native scaffold is
   a property of the **checkpoint**, not the paper: it is transcribed from that checkpoint's own
   inference script and cross-checked against its `adapter_config.json`. Text-to-CadQuery is why
   this is a rule — its Qwen and GPT-2 releases take `### Instruction:`, but its Mistral release is
   a LoRA on Mistral-7B-**Instruct**-v0.3 prompted with `[INST]` markers. Given the wrong one it
   does not degrade gracefully: it echoes the instruction back and emits no code at all, which
   reads in the table as "cannot do CAD".
6. **Nothing is normalised away that matters.** Absolute scale is reported alongside
   scale-invariant metrics, because a part that is right up to a scale factor is still a wrong part.

---

## 2. Evaluation set

### 2.1 Split A — Text2CAD test, 125 uids × 4 levels = 500 prompts

Source: DeepCAD/Text2CAD test split (8,046 uids), L0–L3 prompts from `text2cad_v1.1.csv`.

Selection pipeline (`t2cbench/data/build_splits.py`):

1. Start from the 8,035 uids that have both a test-split prompt row and a buildable GT mesh.
2. **Deduplicate geometrically.** DeepCAD test carries a lot of near-identical plates and blocks --
   measured on a 220-mesh sample, 33% are flat (thinnest/longest extent < 0.15) and 32% tessellate
   to <=24 triangles. For each uid: normalise the GT mesh (centre bbox, scale max extent to 1),
   sample 4,096 surface points, compute a rotation-tolerant signature (sorted eigenvalues of the
   covariance, D2 shape-distribution histogram over 64 bins, volume/convex-hull-volume ratio,
   #solids). Greedily drop any uid whose signature is within ε of one already kept.
   **ε = 0.08** is calibrated against the measured nearest-neighbour distance distribution
   (p5 = 0.055, median = 0.114) and folds ~11% of shapes; the full calibration table is in
   `t2cbench/data/dedup.py`. The default errs toward keeping shapes, because over-merging silently
   shrinks the evaluation set's diversity and is the harder error to notice.
3. **Stratify by complexity** into 4 bins using CAD-sequence complexity (number of extrusions ×
   number of curves, from the GT minimal JSON): `simple / moderate / complex / very_complex`.
4. Sample **125** uids proportionally to the deduplicated bin populations, seeded (`seed=0`), so the
   set is reproducible and spans the difficulty range.
5. Emit all four level prompts per uid → **500 prompts**.

Recorded per uid: `uid, level, prompt, gt_mesh_path, gt_cadquery_path, gt_minimal_json,
complexity_bin, n_extrusions, n_curves, dedup_cluster_id`.

### 2.2 Split B — CADPrompt, 200 objects × 2 prompt variants = 400 prompts

All 200, both the plain and the with-measurements prompt. Each row carries:

- `contamination: clean` (13 uids, in DeepCAD test) or `contaminated` (187 uids, in DeepCAD train/val)
- `gt_mesh`: the provided `Ground_Truth.stl`
- `gt_code`: the expert-written `Python_Code.py`

**Reporting rule:** the CADPrompt headline number is computed on the *clean* 13 only, with the
contaminated 187 reported beside it as a separate column. n=13 is too small to rank models on, so
Split A is the primary benchmark and CADPrompt Split B is a **contamination probe**: the
(contaminated − clean) gap per model is the memorisation estimate. That is a more honest and more
interesting use of CADPrompt than a single averaged 200-row score.

Total per system: **900 prompts** (500 + 400).

---

## 3. The universal interface

```
prompt ──► [system] ──► native output ──► [adapter] ──► OCC solid ──► mesh (STL) ──► metrics
                                              │
                                              └──► validity label
```

`t2cbench/adapters/` has one adapter per representation, each implementing:

```python
class Adapter:
    def run(self, raw: str, out_stl: str, extra: dict = None) -> AdapterResult:
        """raw model output -> (mesh on disk | None, Validity, diagnostic, timing)"""
```

| Adapter | Handles | Mechanism |
|---|---|---|
| `CadVecAdapter` | Text2CAD | `CADSequence.from_vec(vec, bit=8, post_processing=True).create_cad_model()` |
| `MinimalJsonAdapter` | CADmium | brace-balanced JSON extraction → `CADSequence.from_minimal_json` |
| `SkexGenAdapter` | CADFusion | `CADparser(bit=6)` → `write_obj_sample` → `OBJParser` → `OBJReconverter` → boolean ops (their pipeline verbatim) |
| `CadQueryAdapter` | cadrille, Text-to-CadQuery, general LLMs | fenced-code extraction → sandboxed `exec` → `r.val()` or last `Workplane`/`Assembly` |

All adapters run in a **subprocess with a hard timeout** (default 60 s), because OCC and CadQuery
both leak and both can hang on degenerate input. A timeout is a recorded failure, not a crash of the
harness -- but it is scored as a *model* failure, so the budget is deliberately generous: roughly
2-3 s of every call is process fork plus the kernel import before any geometry runs, and that grows
under worker contention. At 20 s, 12 of 18 known-good samples "failed" on a loaded machine.
`evaluate` warns when more than 2% of samples time out, because at that point the numbers are partly
measuring the machine.

### 3.1 Validity codes (the IR decomposition)

A single "invalidity ratio" hides which system is failing at what. We record:

| Code | Meaning |
|---|---|
| `OK` | a non-empty, positive-volume solid was produced |
| `PARSE_FAIL` | output could not be parsed into the expected representation |
| `EXEC_FAIL` | parsed, but raised during construction |
| `TIMEOUT` | exceeded the wall-clock budget |
| `EMPTY_SOLID` | built, but zero volume / no faces |
| `INVALID_SOLID` | `BRepCheck_Analyzer` says not valid |
| `NON_MANIFOLD` | mesh is genuinely open after tessellation (see below) |
| `NO_OUTPUT` | the model emitted nothing at all |

**Tessellation pinholes are not invalidity.** A watertight solid can still tessellate to an STL
with a couple of unmatched edges — the poles of a sphere or a revolved surface are the usual
culprits. Counting that as `NON_MANIFOLD` would charge an invalidity to every system that emits
curved geometry while box-only systems go free, which is a property of the metric, not of the
models. So a mesh is repaired and scored `OK` only when **both**: (a) it has at most
`max(8, 0.1% of edges)` boundary edges, and (b) closing them adds less than 0.5% surface area.
A cube missing one face passes (a) but fails (b), so it stays `NON_MANIFOLD` — the area test is
what stops the repair from inventing geometry.

**Neither are closed multi-body assemblies.** Every system here can emit several bodies. When two
of them touch, merging coincident STL vertices leaves a handful of edges shared by four faces
instead of two, and `is_watertight` returns False even though the shape has no holes and a
well-defined volume. That is the same bias as the pinhole case arriving through a different door —
it charges an invalidity to every model that emits assemblies while single-body models go free. So
a mesh with **zero boundary edges whose every connected component is watertight** is `OK`. Two
solids sharing a whole coincident face do *not* pass (the shared face is duplicated, so the merged
component is not closed) and stay `NON_MANIFOLD`: the rule waves through bodies that touch, not
bodies that overlap. On the third smoke run this alone moved CADmium from 1/3 to 3/3 valid on
split A.

**Null parts are a formatting slip, not a geometry error.** CADmium's own system message instructs
the model to keep part numbering sequential *"even if some are null"*, and `CadSeqProc`'s loader
then dereferences the null and raises. The adapter drops null and empty parts before building and
records that it did; if nothing survives, the sample is `PARSE_FAIL`. Dropping a part that carries
no geometry cannot invent geometry, and failing the sample would penalise CADmium for obeying its
own prompt while the CadQuery adapters already ignore the prose around their code block.

**IR = 1 − P(OK)**, and the breakdown is reported as a stacked bar (grouped into 4 for legibility; the full 8 stay in the CSV). This is one of the more
informative figures in the whole benchmark — CadQuery-emitting models fail at `EXEC_FAIL`,
sequence-emitting models fail at `EMPTY_SOLID`, and that difference is invisible in a single number.

---

## 4. Metric protocol

### 4.1 Canonicalisation (applied identically to GT and prediction)

```
mesh → centre at bbox centre → scale so max extent = 1 → translate to [0,1]³
```

This is **cadrille's** convention (the only translation-invariant one of the three), extended to GT
as well. Both meshes go through the identical function; no model's native convention is privileged.

Rationale for choosing scale-invariant canonicalisation as the primary: Text2CAD, CADmium and
cadrille all emit shapes in DeepCAD's normalised unit space and have no notion of millimetres, so a
millimetre-space metric would score them on information they were never given. Absolute scale is
still measured — see §4.5.

### 4.2 Primary geometric metrics

Sample **8,192** points per surface (`trimesh.sample.sample_surface`, seeded).

| Metric | Definition | Report |
|---|---|---|
| **CD** | `mean(d²)_gt→pred + mean(d²)_pred→gt`, ×1000 | **median** (headline), mean, and mean-trimmed-5% |
| **F1@τ** | point-cloud F-score, τ = 0.02 and τ = 0.05 of the unit diagonal | mean |
| **IoU** | voxel IoU at **64³** after canonicalisation | mean |
| **IoU-bool** | mesh-boolean IoU (cadrille's definition) | mean, cross-check only |
| **HD95** | 95th-percentile symmetric Hausdorff | median |
| **IR** | `1 − P(OK)`, plus the 8-way breakdown | % |

Why median CD is the headline: CD has an unbounded right tail, so one catastrophic sample moves the
mean by more than fifty good ones. Both are reported, and the mean is what reveals whether a model
fails *gracefully* or *catastrophically*.

Both meshes always use the **same** voxel occupancy back-end: exact point-in-solid (`contains`)
when both are closed, surface-voxelisation-plus-fill otherwise. Mixing them would compare a
dilated occupancy against an exact one and flatter whichever side got dilated. Which back-end was
used is recorded per row as `iou_voxel_method`. "Closed" here means `is_closed` — watertight, or a
closed multi-body assembly (§3.1) — so an assembly is not pushed onto the dilating path for a
merged edge.

**Two kinds of ground truth cannot carry an IoU, and are excluded from that column only.** Both are
flagged per row, and CD, F1 and Hausdorff — surface metrics — stay on the full set:

- **Open-shell references.** 3.2% of the DeepCAD test meshes (7 of 220 measured) have boundary
  edges, so "inside" is undefined for the reference itself.
- **Sub-voxel plates.** A canonicalised mesh spans 1.0 in its longest dimension, so at 64³ a part
  thinner than ~1/32 of its length occupies less than two voxels and scores IoU ≈ 0 against *every*
  prediction. CADPrompt `00000633` is 192:1: IoU is 0.000 at both 64³ and 128³ and 0.006 at 256³,
  while F1@0.02 still separates a good fit (0.66) from a bad one (0.03). Raising the resolution
  costs 64× and does not fix it. About a third of this corpus is flat or slab-like, so this is not
  a corner case — it is the main reason the headline ranking is `P(valid) × mean F1@0.02` and not
  anything built on IoU.

Why voxel IoU is primary over boolean IoU: `trimesh`'s boolean intersection fails on a large
fraction of non-watertight CAD output, and cadrille's implementation swallows that in a bare
`except: pass`, silently converting failures into *missing values* rather than zeros — which
inflates the reported mean IoU. Voxel IoU always returns a number. Boolean IoU is kept only to
reconcile with published numbers.

### 4.3 Topology metrics

Cheap, and they separate "geometrically close" from "actually the same part":

- `n_solids`, `n_faces`, `n_edges`, `n_vertices` — and exact-match rate against GT
- **Euler characteristic match** χ = V − E + F — catches wrong hole counts that CD barely registers
- `is_watertight`, `is_volume`
- **Volume / surface-area relative error**

### 4.4 Sequence-level metrics (secondary, partial coverage)

Text2CAD's line/arc/circle/extrusion F1 is only computable for systems that emit a CAD sequence.

| System | Sequence F1 available |
|---|---|
| Text2CAD | yes (native) |
| CADmium | yes (minimal JSON → CADSequence) |
| CADFusion | yes (SkexGen → CADSequence) |
| cadrille | **no** — CadQuery is not losslessly invertible to a sketch-extrude sequence |
| Text-to-CadQuery, general LLMs | no |

Because coverage is partial, sequence F1 goes in a **separate table**, clearly marked, and never in
the headline ranking. Putting an N/A-riddled column in the main table would let a reader rank
systems on a metric half of them are structurally unable to score on.

### 4.5 Scale fidelity (reported, not ranked)

For the CadQuery-emitting systems and CADPrompt (which has real dimensions), also report
`bbox_relative_error` **before** canonicalisation. This is the only metric that distinguishes a part
that is manufacturable from one that is merely the right shape. It is marked N/A for the
normalised-space models, and it is not part of the primary ranking, because they were never trained
to produce absolute dimensions.

### 4.6 Aggregation

- Per level (L0/L1/L2/L3) and pooled, for Split A.
- Per complexity bin.
- Per contamination class, for Split B.
- **Invalid samples are included as failures**, never dropped. For CD, where an invalid sample has
  no defined value, report (a) median over valid only *and* (b) a **coverage-adjusted score**
  `CD@P = median CD over valid, reported alongside P(OK)`, plus a single scalar
  **`Score = P(OK) × F1@0.02`** for a ranking that cannot be gamed by emitting nothing.

That last point matters: a model that answers 10% of prompts perfectly and fails the rest would top
a median-CD-over-valid-only leaderboard. `P(OK) × F1` cannot be gamed that way.

---

## 5. Sampling protocol

| Setting | Value |
|---|---|
| Primary | **pass@1**, temperature 0, greedy (or the model's deterministic default) |
| Secondary | **best-of-5**, temperature 0.7, top-p 0.95, seeded — oracle-selected by CD, to compare with published best-of-N numbers |
| Max new tokens | 1024 for CadQuery emitters, 768 for cadrille (its trained limit), 272 tokens for Text2CAD (architectural limit), 512 for CADFusion (`MAX_LENGTH`) |
| Retries | none — first output counts |

Both tables are produced. Published numbers should be compared against the **best-of-5** table, and
model quality judged on **pass@1**.

---

## 6. General-LLM baselines

Prompted zero-shot to emit CadQuery, with an identical prompt template for every model
(`configs/prompt_templates.yaml`). The template states the task, requires a single fenced Python
block, requires the result in variable `r`, and gives one worked example (1-shot) so that failures
measure CAD ability rather than format compliance. A **0-shot** variant is also run to quantify how
much of the gap is format compliance vs. geometry.

Default roster, chosen to be **size-matched to the fine-tuned systems** so the comparison isolates
fine-tuning rather than parameter count:

| Model | Why |
|---|---|
| `Qwen/Qwen2.5-Coder-7B-Instruct` | **the exact base model of CADmium-7B** — the single most informative baseline in the benchmark |
| `meta-llama/Meta-Llama-3-8B-Instruct` | **the base model of CADFusion** |
| `Qwen/Qwen2-VL-2B-Instruct` | **the base model of cadrille** |
| `mistralai/Mistral-7B-Instruct-v0.3` | requested; also the Text-to-CadQuery Mistral base |
| `deepseek-ai/deepseek-coder-6.7b-instruct` | requested, strong code model |
| `Qwen/Qwen2.5-Coder-32B-Instruct` (4-bit) | scale reference |

The first three turn the benchmark into three **controlled ablations**: base vs. fine-tuned, same
weights, same prompts, same metrics. That is the strongest claim this benchmark can make, and it
costs nothing extra to run.

Optional closed-model row (`gpt-4o`, `claude-*`, `gemini-*`) is supported by the runner but off by
default — it costs money and needs keys, so it is your call.

---

## 7. Deliverables

1. **Main table** — Split A, pass@1, per level: `IR%, CD median, CD mean, F1@0.02, IoU, HD95, Score`
2. **Best-of-5 table** — same, for comparison with published numbers
3. **Contamination table** — Split B clean vs contaminated, with the per-model memorisation gap
4. **Ablation table** — base LLM vs its fine-tuned descendant (3 pairs)
5. **Sequence-F1 table** — the three sequence-emitting systems only
6. **Failure-mode figure** — stacked bars of the 7 validity codes per model
7. **Qualitative figure** — a grid: rows = models, columns = hand-picked prompts spanning L0–L3 and
   easy→hard, cells = rendered prediction, with GT in the top row and CD/IoU annotated per cell
8. **Level-sensitivity figure** — CD and F1 vs prompt level, one line per model. This is the plot
   that shows which systems only work when the prompt is a transcription of the answer.

---

## 8. Threats to validity (state these in the paper)

1. **CADFusion split unverified.** SkexGen's train/test boundary ≠ DeepCAD's. Until the SkexGen
   split is obtained, some Split A uids may be in CADFusion's training data.
2. **Prompt-distribution mismatch.** CADmium and cadrille were trained on their own paraphrases of
   DeepCAD sequences, not on Text2CAD L0–L3. Their Split A numbers include a distribution-shift
   penalty that Text2CAD does not pay. Mitigated by also reporting each system on its *native*
   prompt style for the same uids (an extra column, `native_prompt`).
3. **GT is DeepCAD.** DeepCAD is sketch-extrude only. Nothing here measures fillets, chamfers,
   revolves, lofts, or assemblies. Do not claim general CAD ability.
4. **Tessellation tolerance** (`linear_deflection=0.001`) affects CD at the 4th decimal. Fixed
   across all systems, but it is a free parameter.
5. **n=13 clean CADPrompt** is too small for ranking; it is used only as a contamination probe.
6. **The 1-shot example** in the general-LLM prompt is itself a design choice that advantages
   general LLMs relative to a 0-shot setting; both are reported.
7. **The token budget is spent in a representation-dependent currency.** "1024 tokens for
   everyone" sounds equal and is not: CADmium's pretty-printed JSON costs ~3.1 chars/token against
   ~2.6 for CadQuery code, so the same budget buys it far less geometry. Measured on the first full
   run, 109 of its 500 Split A outputs were truncated mid-number and scored `PARSE_FAIL` — 22% of
   its samples failing on our budget rather than its ability — and its longest *complete* output
   was 1,019 tokens, i.e. the cap bound exactly at the boundary. Its budget is now 2048. **Any
   re-parameterisation of the output format requires re-checking this**, and the check is cheap:
   count outputs that do not end in the format's own terminator.
8. **Text2CAD cannot read long prompts.** Its BERT encoder is capped at 512 tokens
   (`max_seq_len: 512`), which **10% of Split A prompts — all L3 — exceed**. This is architectural
   to the published model, not a choice of ours, so it is reported rather than fixed; but its L3
   column must be read as "the part of the prompt that fitted".
9. **CADFusion truncates at its own limit.** 4.6% of its Split A outputs end without
   `<extrude_end>` at the 512-token `MAX_LENGTH` taken from its own inference code. Kept as native,
   but it means a few points of its invalidity are budget, not ability.
10. **Batch-size determinism is assumed, not proven.** All models are generated at batch size 8
    with left padding, so any batching effect applies equally and is a reproducibility caveat
    rather than a fairness one — but greedy decoding under left padding is not guaranteed
    bit-identical to batch size 1, and that has not been verified on a GPU.

### 8.1 Fairness properties that were verified, not assumed

Checked against the recorded artefacts of the first full 6,300-generation run:

| Property | Evidence |
|---|---|
| Every model saw the same prompts | identical 500 Split A `sample_id`s across all 7; the split's task text appears verbatim inside every recorded `prompt_sent` |
| No input truncation | longest prompt ~2,900 tokens against the runner's 4,096 cap |
| Decoding identical | `do_sample=False` for all; each model's own sampling config is explicitly overridden |
| Adapter leniency not tilted | extraction repair changed **0/500** outputs for cadrille and both Text-to-CadQuery models; it *rescued* 46 CADmium samples. The generous path exists for every representation and is used only where needed |
| Machine load absent from results | 1 `TIMEOUT` in 6,300 samples (0.02%) |
| Scoring path agrees with itself | `evaluate.py` and `score_export.py` match to printed precision on identical inputs |

---

## 9. Execution plan on one A100 40 GB

| Stage | Where | Time (est., 900 prompts) |
|---|---|---|
| 0. Build splits | CPU, once | ~40 min (mesh dedup dominates) |
| 1. Text2CAD | A100 | ~25 min |
| 2. cadrille SFT + RL | A100 | ~40 min |
| 3. CADmium-7B | A100 | ~50 min |
| 4. CADFusion v1.1 | A100 | ~60 min |
| 5. Text-to-CadQuery (2 models) | A100 | ~50 min |
| 6. General LLMs (6 models) | A100 | ~4 h |
| 7. Metrics (all systems) | CPU, 16 proc | ~2 h |
| 8. Figures + tables | CPU | ~10 min |

Roughly **9–10 GPU-hours** for pass@1, ~5× that for best-of-5 generation (metrics scale similarly).
One model per Colab session, predictions written to Drive as JSONL, evaluation entirely on CPU.
