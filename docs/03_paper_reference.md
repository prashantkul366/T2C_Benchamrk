# T2C-Bench — complete study reference

**Purpose of this file.** Everything needed to write the paper, in one place: what was built, what
was run, what was measured, what was found, why each design decision was made, and what is still
open. Numbers here were re-derived from the recorded artefacts of the run, not from memory. Where a
figure in an earlier doc disagrees with this one, **this one is correct** (see §14.1 for the
corrections).

Status: **complete**. All 13 systems have a full 900-prompt run generated and scored — 11,700
samples, 0 timeouts, 0 harness errors. CADmium has been regenerated against the corrected token
budget and Llama-3-8B-Instruct's licence gate is cleared, so every row and every ablation pair is
populated. Outstanding: the best-of-5 table and the sequence-F1 table (§16.3).

---

## 1. What the study is

**One sentence.** Seven published text-to-CAD systems and six general LLMs, re-run from scratch
under a single protocol on 900 prompts, scored on one measurement plane, with every incomparability
in the prior literature made explicit rather than averaged away.

**The claim the paper can make.** Not "model X is best" — the top of the leaderboard is a
statistical tie. The claims that survive scrutiny are:

1. The published numbers in this field are not comparable, and we show quantitatively by how much.
2. Prompt specificity, not model identity, is the dominant axis of variation. The spread across
   L0→L3 within a single model is up to **41×**; the spread across models at a fixed level is under 2×.
3. Failure is representation-structured: which *kind* of failure a system has is predicted by what
   it emits, not by how good it is. A scalar invalidity ratio destroys this signal.
4. Four controlled base-vs-fine-tuned ablations (same weights, same prompts, same metrics) show what
   CAD fine-tuning actually buys: reliability, not always geometry — and one RL variant scores below
   its own un-fine-tuned base.
5. Benchmark protocol choices — a token budget, a metric definition — move the ranking by more than
   the difference between the top two models. We demonstrate this on our own benchmark.

Point 5 is unusual and is the strongest methodological contribution: we audited our own harness and
found two protocol defects that each changed the answer, and we report both.

---

## 2. Research questions

| RQ | Question | Where answered |
|----|----------|----------------|
| RQ1 | Can the five published systems be compared at all from their papers? | §4 — no, and we quantify why |
| RQ2 | Under one protocol, how do they actually rank? | §12.1 — a three-way tie at the top |
| RQ3 | How much of a system's score is prompt specificity rather than CAD ability? | §12.2, §13.1, §13.3 |
| RQ4 | Does CAD fine-tuning beat a general code LLM of the same size? | §12.3, §12.4, §13.3, §13.4 |
| RQ5 | How much of the reported performance on CADPrompt is memorisation? | §12.5, §13.6 — the control refutes the probe |
| RQ6 | Where does each representation fail, and is that a property of the model or the format? | §12.7, §13.5 |
| RQ7 | How sensitive is the ranking to benchmark design choices? | §13.8 — more sensitive than to the models |

---

## 3. Systems under test

### 3.1 The seven specialised systems

| Key | Display | Venue | Architecture | Params | Weights | Emits |
|---|---|---|---|---|---|---|
| `text2cad` | Text2CAD v1.0 | NeurIPS'24 Spotlight | BERT-large encoder → custom CAD-sequence decoder (not an LLM) | ~90M | `SadilKhan/Text2CAD`, `text2cad_v1.0/Text2CAD_1.0.pth` (92 MB, **gated**) | `(N,2)` int vector sequence, 8-bit quantised, max len 272 |
| `cadmium-7b` | CADmium-7B | TMLR 2026 | LoRA r=64 α=16, all 7 proj modules, on Qwen2.5-Coder-7B-Instruct | 7B | `chandar-lab/CADmium-7B` (646 MB adapter) | minimal JSON (`parts`→`sketch`→`extrusion`) |
| `cadfusion-v1.1` | CADFusion v1.1 | ICML 2025 | LoRA r=32 α=32, q/v only, on Meta-Llama-3-8B | 8B | `microsoft/CADFusion`, subfolder `v1_1` (4.3 GB) | SkexGen command tokens |
| `cadrille` | cadrille (SFT) | 2025 | Qwen2-VL-2B-Instruct + point-cloud encoder, full fine-tune | 2B | `maksimko123/cadrille` (4.4 GB) | CadQuery code, result in `r`, integer ~100 scale |
| `cadrille-rl` | cadrille (RL) | 2025 | same, RL-tuned | 2B | `maksimko123/cadrille-rl` (4.4 GB) | same |
| `t2cq-qwen-3b` | Text-to-CadQuery (Qwen2.5-3B) | IEEE T-ASE sub. | full fine-tune of Qwen2.5-3B | 3B | `ricemonster/qwen2.5-3B-SFT` | CadQuery script that exports its own STL |
| `t2cq-mistral-7b` | Text-to-CadQuery (Mistral-7B) | IEEE T-ASE sub. | LoRA on **Mistral-7B-Instruct-v0.3** | 7B | `ricemonster/Mistral-7B-lora` | same |

### 3.2 The six general LLMs

Chosen to be **size-matched** to the fine-tuned systems, and the first three are the *exact base
models* of three of them — which turns the comparison into controlled ablations.

| Key | Weights | Role |
|---|---|---|
| `qwen25-coder-7b` | `Qwen/Qwen2.5-Coder-7B-Instruct` | **base of CADmium-7B** — most informative single baseline |
| `llama3-8b-instruct` | `meta-llama/Meta-Llama-3-8B-Instruct` | **base family of CADFusion** (gated; outstanding) |
| `qwen2-vl-2b` | `Qwen/Qwen2-VL-2B-Instruct` | **base of cadrille** |
| `mistral-7b-instruct` | `mistralai/Mistral-7B-Instruct-v0.3` | also the Text-to-CadQuery Mistral base |
| `deepseek-coder-6.7b` | `deepseek-ai/deepseek-coder-6.7b-instruct` | strong code model, unrelated lineage |
| `qwen25-coder-32b` | `Qwen/Qwen2.5-Coder-32B-Instruct` (4-bit NF4) | scale reference |

Ablation pairs are declared in `configs/ablation_pairs.json`:
`qwen25-coder-7b → cadmium-7b`, `llama3-8b-instruct → cadfusion-v1.1`,
`qwen2-vl-2b → cadrille`, `qwen2-vl-2b → cadrille-rl`.

### 3.3 Two checkpoint-level facts that are easy to get wrong

Both were found by reading `adapter_config.json` and the repos' own inference scripts, not the
papers, and both silently produce a wrong network or a wrong score if missed:

- **Text-to-CadQuery Mistral is a LoRA on Mistral-7B-*Instruct*-v0.3, not the base.** Merging onto
  `Mistral-7B-v0.3` yields a different network. It is also the one release in that family prompted
  with `[INST]` markers rather than the `### Instruction:` scaffold used by their Qwen and GPT-2
  checkpoints. Given the wrong scaffold it does not degrade gracefully — it echoes the instruction
  back and emits no code at all, which in a results table reads as *"this model cannot do CAD."*
- **CADFusion's adapters live under `v1_0/` and `v1_1/`, not at the repo root.** Loading the root
  silently gets you nothing or the wrong round count. `v1_1` is 9 rounds and is the one to report.

---

## 4. Why the published numbers cannot be tabulated together (RQ1)

This section is a paper contribution in its own right. All of it was verified by reading the four
evaluation scripts.

### 4.1 Five incompatible output representations

| System | Native output | Path to geometry |
|---|---|---|
| Text2CAD | `(N,2)` int vector, 8-bit quantised | `CADSequence.from_vec(vec, bit=8, post_processing=True).create_cad_model()` |
| CADmium | minimal JSON | `CADSequence.from_minimal_json` (vendored Text2CAD `CadSeqProc`) |
| CADFusion | SkexGen token string | `CADparser(bit=6)` → OBJ → `OBJReconverter` → boolean ops |
| cadrille | CadQuery code, result in `r` | `exec` → `r.val()` → `.tessellate(0.001, 0.1)` |
| Text-to-CadQuery | CadQuery script exporting its own STL | `exec` |
| general LLMs | CadQuery + prose | fenced-block extraction → `exec` |

They converge on exactly one thing: **a boundary-representation solid, hence a mesh.** That is the
only fair place to put the measurement plane.

### 4.2 Three different Chamfer distances, called "CD" by all four papers

| Source | Normalisation | N points | Translation-invariant? |
|---|---|---|---|
| Text2CAD `utils.py:1291` | `pts / (max(pts) − min(pts))` over the **flattened** array, **no centring** | 8192 | **no** |
| cadrille `evaluate.py:31` | centre bbox, max extent → 1, translate to (0.5,0.5,0.5) | 8192 | **yes** |
| CADFusion `chamfer_dist.py` | `pts / max(abs(pts))`, **no centring** | **2000** | **no** |
| CADmium | delegates to Text2CAD `eval_seq.py` + CAD-MLLM metrics | 8192 | no |

Two of the three assign a *different* CD to a translated copy of the same solid. One uses a quarter
of the sample points. These are not one metric with three implementations; they are three metrics
sharing a name.

### 4.3 Two different invalidity ratios

- Text2CAD IR = fraction of samples where `cd < 0`, i.e. the OCC solid failed to build.
- cadrille IR = fraction where no valid mesh file appeared — which *includes 3-second exec timeouts*,
  so part of their IR is a property of their machine.
- cadrille additionally reports IR at "skip 0..4", trimming the worst *k* outliers from mean CD.

### 4.4 All three headline numbers are best-of-N oracle scores

- Text2CAD: 5 samples, `choose_best_index` over CD.
- CADFusion: 5 samples, temperature 0.3.
- cadrille: N samples, then `argmin(cd)` / `max(iou)` per uid.

A single-sample general LLM compared against any of these is being cheated. We fix N across every
system and report pass@1 and best-of-5 as **separate tables**.

### 4.5 A contamination problem that decides the field's credibility

CADPrompt's 200 objects are named with DeepCAD uids. Checked against the DeepCAD test split
(`maksimko123/deepcad_test_mesh`, 8,046 STL files):

```
CADPrompt uids in DeepCAD TEST  :  13 / 200   ( 6.5%)
CADPrompt uids in train/val     : 187 / 200   (93.5%)
  ├─ confirmed in Text2CAD TRAIN : 132
  ├─ confirmed in Text2CAD VAL   :   6
  └─ DeepCAD train, outside cadrille's 90,737-uid subset : 49
```

**187 of 200 CADPrompt objects were in the training set of Text2CAD, CADmium and cadrille** — and
were not in the training set of Qwen/Llama/Mistral/DeepSeek in any comparable way. Reporting one
averaged CADPrompt number hands the fine-tuned systems a memorisation advantage and calls it a
result.

---

## 5. Evaluation set

Total per system: **900 prompts** = 500 (Split A) + 400 (Split B).

### 5.1 Split A — primary benchmark, 125 shapes × 4 prompt levels = 500 prompts

Source: DeepCAD/Text2CAD test split (8,046 uids), L0–L3 prompts from `text2cad_v1.1.csv`.
Built by `t2cbench/data/build_splits.py --stage a`.

Pipeline:

1. Start from the 8,035 uids that have both a test-split prompt row and a buildable GT mesh.
2. **Geometric deduplication.** DeepCAD test carries a lot of near-identical plates and blocks —
   measured on a 220-mesh sample, **33% are flat** (thinnest/longest extent < 0.15) and **32%
   tessellate to ≤24 triangles**. For each uid: canonicalise the GT mesh, sample 4,096 surface
   points, compute a rotation-tolerant signature (sorted covariance eigenvalues + D2 shape-
   distribution histogram over 64 bins + volume/convex-hull-volume ratio + number of solids).
   Greedily drop any uid within ε of one already kept.
   **ε = 0.08**, calibrated against the measured nearest-neighbour distance distribution
   (p5 = 0.055, median = 0.114); it folds **~11%** of shapes. Calibration table in
   `t2cbench/data/dedup.py`. The default deliberately errs toward *keeping* shapes: over-merging
   silently shrinks diversity and is the harder error to notice.
3. **Stratify by complexity** into 4 bins from CAD-sequence complexity (n_extrusions × n_curves,
   read from the GT minimal JSON): `simple / moderate / complex / very_complex`.
4. Sample **125** uids proportionally to deduplicated bin populations, `seed=0`.
5. Emit all four level prompts per uid → **500 prompts**.

Realised bin sizes (×4 levels): simple 128, moderate 176, complex 100, very_complex 96.

Per-row fields: `uid, level, prompt, gt_mesh, gt_cadquery_path, gt_minimal_json, complexity_bin,
n_extrusions, n_curves, dedup_cluster_id, sample_id`.

**The four prompt levels** (verified against the annotation CSV columns):

| Level | CSV column | Content | Median BERT tokens |
|---|---|---|---|
| L0 | `abstract` | one-line VLM caption — *"A rectangular block with a flat top and bottom."* | 31 |
| L1 | `beginner` | layperson design steps, no measurements or jargon | 52 |
| L2 | `intermediate` | generalised geometric description, some detail abstracted | 135 |
| L3 | `expert` | full geometry with relative values — coordinate system, every line's start/end point, sketch scale, extrusion depth | **539** |

L3 is very nearly a literal transcription of the CAD program. This spread is the entire point of the
benchmark: **a model can win at L3 by learning a transcription grammar and be useless at L0.**

### 5.2 Split B — contamination probe, 200 objects × 2 prompt variants = 400 prompts

Source: CADPrompt (`Kamel773/CAD_Code_Generation`, ICLR 2025). Both prompt variants per object —
plain, and with specific measurements (roughly an L1/L3 axis). Built by `--stage b`.

Each row carries `contamination: clean` (13 uids → **26 rows**) or `contaminated`
(187 uids → **374 rows**), plus `gt_mesh` (`Ground_Truth.stl`) and `gt_code` (`Python_Code.py`).

**Reporting rule: Split B is not a leaderboard.** n=13 clean uids is far too small to rank on. The
number to read is the per-model **(contaminated − clean) gap**, which is a memorisation estimate.
This is a more honest and more interesting use of CADPrompt than a single averaged 200-row score.

---

## 6. The universal interface

```
prompt ──► [system] ──► native output ──► [adapter] ──► OCC solid ──► mesh (STL) ──► metrics
                                              │
                                              └──► one of 8 validity codes
```

`t2cbench/adapters/`, one adapter per representation:

| Adapter | Handles | Mechanism |
|---|---|---|
| `CadVecAdapter` | Text2CAD | `CADSequence.from_vec(vec, bit=8, post_processing=True).create_cad_model()` |
| `MinimalJsonAdapter` | CADmium | brace-balanced JSON extraction → `CADSequence.from_minimal_json` |
| `SkexGenAdapter` | CADFusion | `CADparser(bit=6)` → `write_obj_sample` → `OBJParser` → `OBJReconverter` → booleans (their pipeline verbatim) |
| `CadQueryAdapter` | cadrille, Text-to-CadQuery, general LLMs | fenced-code extraction → sandboxed `exec` → `r.val()` or last `Workplane`/`Assembly` |

**Design rule: the adapter is part of the system.** If a model emits code that will not execute,
that is the model's failure, not grounds for dropping the sample. Invalid outputs are counted as
failures, never silently skipped.

All adapters run in a **subprocess with a hard timeout** (`DEFAULT_TIMEOUT_S = 60`), because OCC and
CadQuery both leak and both can hang on degenerate input. The budget is deliberately generous:
roughly 2–3 s of every call is process fork plus kernel import before any geometry runs, and that
grows under worker contention — at 20 s, 12 of 18 known-good samples "failed" on a loaded machine.
`evaluate` warns when more than 2% of samples time out, because at that point the numbers are partly
measuring the machine. **Observed: 1 timeout in 6,300 samples (0.02%).**

### 6.1 The eight validity codes

A single scalar "invalidity ratio" hides which system fails at what.

| Code | Meaning |
|---|---|
| `OK` | non-empty, positive-volume solid produced |
| `PARSE_FAIL` | output could not be parsed into the expected representation |
| `EXEC_FAIL` | parsed, but raised during construction |
| `TIMEOUT` | exceeded the wall-clock budget |
| `EMPTY_SOLID` | built, but zero volume / no faces |
| `INVALID_SOLID` | `BRepCheck_Analyzer` says not valid |
| `NON_MANIFOLD` | mesh genuinely open after tessellation |
| `NO_OUTPUT` | model emitted nothing at all |

**IR = 1 − P(OK).**

### 6.2 Three leniency rules, and why each is not a thumb on the scale

Each of these exists because *not* having it would charge an invalidity to one representation for a
reason that has nothing to do with CAD ability.

**Tessellation pinholes are not invalidity.** A watertight solid can still tessellate to an STL with
a few unmatched edges — sphere poles and revolved surfaces are the usual culprits. Counting that as
`NON_MANIFOLD` charges an invalidity to every system emitting curved geometry while box-only systems
go free. A mesh is repaired and scored `OK` only when **both**: (a) it has at most
`max(8, 0.1% of edges)` boundary edges (`TESSELLATION_PINHOLE_EDGES = 8`,
`TESSELLATION_PINHOLE_FRAC = 0.001`), and (b) closing them adds less than **0.5%** surface area
(`TESSELLATION_PATCH_AREA_FRAC = 0.005`). A cube missing one face passes (a) but fails (b) and stays
`NON_MANIFOLD` — **the area test is what stops the repair from inventing geometry.**

**Closed multi-body assemblies are not invalidity.** When two bodies touch, merging coincident STL
vertices leaves edges shared by four faces instead of two, and `is_watertight` returns False even
though the shape has no holes and a well-defined volume. A mesh with **zero boundary edges whose
every connected component is watertight** is `OK`. Two solids sharing a whole coincident face do
*not* pass (the shared face is duplicated, so the merged component is not closed) — the rule waves
through bodies that *touch*, not bodies that *overlap*. On the third smoke run this alone moved
CADmium from 1/3 to 3/3 valid on Split A.

**Null parts are a formatting slip, not a geometry error.** CADmium's own system message instructs
the model to keep part numbering sequential *"even if some are null"*, and `CadSeqProc`'s loader then
dereferences the null and raises. The adapter drops null and empty parts before building and records
that it did; if nothing survives, the sample is `PARSE_FAIL`. Dropping a part carrying no geometry
cannot invent geometry, and failing the sample would penalise CADmium for obeying its own prompt
while the CadQuery adapters already ignore the prose around their code block.

**Symmetry was measured, not assumed** — see §14.3.

---

## 7. Metric protocol

### 7.1 Canonicalisation, applied identically to GT and prediction

```
mesh → centre at bbox centre → scale so max extent = 1 → translate to [0,1]³
```

This is **cadrille's** convention — the only translation-invariant one of the three in the
literature — extended to the ground truth as well. Both meshes go through the identical function; no
model's native convention is privileged.

Why scale-invariant is primary: Text2CAD, CADmium and cadrille all emit shapes in DeepCAD's
normalised unit space and have no notion of millimetres, so a millimetre-space metric would score
them on information they were never given. Absolute scale is still measured separately (§7.5).

**No ICP.** DeepCAD ground truth is in a canonical frame and the prompts describe coordinate
systems, so orientation is part of what is being tested. Available as `--align-icp` for ablation.

### 7.2 Primary geometric metrics

Constants (`t2cbench/metrics/geometry.py`): `N_POINTS = 8192`, `VOXEL_RES = 64`, `CD_SCALE = 1000.0`,
`RNG_SEED = 0`, `IOU_MIN_VOXELS = 2.0`; tessellation `linear_deflection = 0.001`.

| Metric | Definition | Reported |
|---|---|---|
| **CD** | `mean(d²)_gt→pred + mean(d²)_pred→gt`, ×1000, post-canonicalisation | **median** (headline), mean, mean-trimmed-5% |
| **F1@τ** | point-cloud F-score, τ = 0.02 and 0.05 of the unit diagonal | mean |
| **IoU** | voxel IoU at **64³** | mean, with coverage `IoU_n` |
| **IoU-bool** | mesh-boolean IoU (cadrille's definition) | cross-check only |
| **HD95** | 95th-percentile symmetric Hausdorff | median |
| **IR** | `1 − P(OK)` + the 8-way breakdown | % |

**Why median CD is the headline.** CD has an unbounded right tail, so one catastrophic sample moves
the mean by more than fifty good ones. Both are reported, and the *mean* is what reveals whether a
model fails gracefully or catastrophically — e.g. CADmium's L3 median CD is 0.116 but its mean is
1.787, a 15× gap that says the failures it does have are severe.

**The voxel back-end is forced to match.** Both meshes always use the same occupancy back-end: exact
point-in-solid (`contains`) when both are closed, surface-voxelisation-plus-fill otherwise. Mixing
them would compare a dilated occupancy against an exact one and flatter whichever side got dilated.
Recorded per row as `iou_voxel_method`. "Closed" means `is_closed` — watertight *or* a closed
multi-body assembly — so an assembly is not pushed onto the dilating path for a merged edge.

### 7.3 Two kinds of ground truth cannot carry an IoU

Both are flagged per row and **excluded from the IoU column only**; CD, F1 and Hausdorff — surface
metrics — stay on the full set.

- **Open-shell references.** 3.2% of DeepCAD test meshes (7 of 220 measured) have boundary edges, so
  "inside" is undefined for the reference itself.
- **Sub-voxel plates.** A canonicalised mesh spans 1.0 in its longest dimension, so at 64³ a part
  thinner than ~1/32 of its length occupies under two voxels and scores IoU ≈ 0 against *every*
  prediction. CADPrompt `00000633` is 192:1: IoU is 0.000 at both 64³ and 128³ and 0.006 at 256³,
  while F1@0.02 still cleanly separates a good fit (0.66) from a bad one (0.03). Raising resolution
  costs 64× and does not fix it.

About a third of this corpus is flat or slab-like, so this is not a corner case. **It is the main
reason the headline ranking is `P(valid) × mean F1@0.02` and not anything built on IoU.**

The coverage figures make the point concretely — `IoU_n` on Split A ranges from **203** (CADmium) to
**423** (t2cq-qwen-3b) out of 500. A mean IoU computed over 203 samples and one computed over 423 are
not the same statistic, and ranking on them would be ranking partly on which subset survived.

**Why voxel IoU over boolean IoU:** `trimesh`'s boolean intersection fails on a large fraction of
non-watertight CAD output, and cadrille's implementation swallows that in a bare `except: pass`,
silently converting failures into *missing values* rather than zeros — which inflates their reported
mean IoU. Voxel IoU always returns a number. Boolean IoU is kept only to reconcile with published
numbers.

### 7.4 Topology metrics

`n_solids`, `n_faces`, `n_edges`, `n_vertices` with exact-match rate; **Euler characteristic match**
χ = V − E + F (catches wrong hole counts that CD barely registers); `is_watertight`, `is_volume`;
volume and surface-area relative error.

### 7.5 Scale fidelity — reported, not ranked

`bbox_relative_error` computed **before** canonicalisation, for the CadQuery-emitting systems and
CADPrompt (which has real dimensions). This is the only metric distinguishing a part that is
manufacturable from one that is merely the right shape. Marked N/A for the normalised-space models —
they were never trained to produce absolute dimensions, so ranking on it would be a category error.

### 7.6 Sequence-level metrics — separate table, partial coverage

Text2CAD's line/arc/circle/extrusion F1 is only computable for systems that emit a CAD sequence:
available for Text2CAD, CADmium and CADFusion; **structurally impossible** for cadrille,
Text-to-CadQuery and the general LLMs, because CadQuery is not losslessly invertible to a
sketch-extrude sequence. It therefore goes in a clearly-marked separate table and **never** in the
headline ranking — an N/A-riddled column in the main table would let a reader rank systems on a
metric half of them cannot score on.

### 7.7 The ranking metric

```
Score = P(validity == OK) × mean F1@0.02,  summed over ALL prompts (failures contribute 0)
```

Two properties make this the right choice:

- **It cannot be gamed by abstention.** A model answering 10% of prompts perfectly and failing the
  rest would top a median-CD-over-valid-only leaderboard. It cannot top this.
- **Credit requires a closed solid.** A `NON_MANIFOLD` mesh is *measurable* but is not a closed
  solid and so is not manufacturable; it earns no ranking credit. This is not cosmetic — on the
  first full run, including `NON_MANIFOLD` **inverted the top two** (see §13.8).

**Every Score carries a 95% percentile-bootstrap CI over prompts** (`_bootstrap_ci`, seeded), because
the top of this leaderboard is a tie and a bare ranking hides that.

### 7.8 Aggregation

Per level (L0–L3) and pooled for Split A; per complexity bin; per contamination class for Split B.
**Invalid samples are included as failures, never dropped.** For CD, where an invalid sample has no
defined value, report (a) median over valid only *and* (b) always alongside P(OK).

---

## 8. Sampling protocol

| Setting | Value |
|---|---|
| Primary | **pass@1**, `do_sample=False`, temperature 0, greedy |
| Secondary | **best-of-5**, temp 0.7, top-p 0.95, seed 0 — oracle-selected by CD, for comparison with published best-of-N |
| Batch size | 8, left padding, all models |
| Retries | none — first output counts |

### 8.1 Token budgets — the one setting that legitimately differs

```yaml
max_new_tokens:
  default:   1024
  cadmium:   2048   # 2x its longest complete output; 1024 was binding
  cadrille:   768   # its trained generation length
  cadfusion:  512   # src/test/utils.py MAX_LENGTH -- theirs, kept as native
  text2cad:   272   # MAX_CAD_SEQUENCE_LENGTH, architectural
```

These are architectural or trained limits, not tuning knobs: raising them past the trained limit
produces garbage, lowering them truncates. The rule is *a model's own native limit where it has one,
otherwise a budget large enough that it never binds.*

**"1024 for everyone" sounds equal and is not.** The budget is spent in a representation-dependent
currency: CADmium's pretty-printed JSON costs **~3.1 chars/token** against **~2.6 for CadQuery code**,
so the same nominal budget buys it far less geometry. See §13.8 for what this cost us.

---

## 9. General-LLM prompting

Identical template for every general model — `general_one_shot` in `configs/prompts.yaml`. It states
the task, requires a single fenced Python block, requires the result in variable `r`, and gives one
worked example, so that failures measure CAD ability rather than format compliance. A
`general_zero_shot` variant is also supported, to quantify how much of the gap was only ever
formatting. Fine-tuned systems are excluded from the 0-shot variant (gated on `runner != "hf"` *and*
a `native.` template — gating on the template alone would sweep cadrille in, since it declares none).

Each fine-tuned system gets its **own native scaffold**, transcribed from that checkpoint's inference
script and cross-checked against its `adapter_config.json` — not written fresh. `native.cadmium`'s
system message is byte-for-byte their `cadmium.src.utils.prompts.SYSTEM_MESSAGE`; `native.cadfusion`
is verbatim from `src/test/inference.py`; `native.text2cadquery` is their training scaffold;
`native.text2cadquery_mistral` is the `[INST]` form including their literal leading `<s>`.

---

## 10. Execution — hardware, environments, paths

### 10.1 Division of labour

Generation needs a GPU; scoring needs a geometry kernel. They cannot share a process, because the
two kernels are installed by different package managers and conflict:

- **`pythonocc-core` 7.7.0 (conda, `OCC.*`)** — required by the sequence adapters (`cadvec`,
  `minimal_json`, `skexgen`).
- **`cadquery` / OCP (pip)** — required by the `cadquery` adapter.

Never in the same process. The scoring env is built by `scripts/setup_scoring_env.sh`.

### 10.2 Where things live

| What | Path |
|---|---|
| Colab working root | `/content/drive/MyDrive/t2c_bench` (referred to as `WORK`) |
| Splits | `WORK/data/split_a.jsonl`, `WORK/data/split_b.jsonl` |
| Raw generations | `WORK/results/raw/<model>_split{A,B}_pass_at_1.jsonl` |
| Scored rows | `<work>/results/scored/<model>_split{A,B}_pass_at_1.jsonl` |
| Tables | `<work>/results/tables/*.csv` + `RESULTS.md` |
| Handoff tarball (run 1) | `raw_predictions.tar.gz` → Drive file id `1gxIEm1NLx5CfVOAjeFnf67Bpb_5WTAc9` |
| Code | `https://github.com/prashantkul366/T2C_Benchamrk` |

Ground-truth mesh paths are **absolute** and baked in when a split is built, so they never survive
the trip between machines. `scripts/localise_splits.py` re-derives them from the durable key (the
uid) — Split A from the HuggingFace `maksimko123/deepcad_test_mesh` snapshot, Split B from a local
CADPrompt checkout — and **refuses to write a partially-resolved split**, because a split that is
quietly 3% unresolvable surfaces later as *model invalidity*, not as a missing file.

### 10.3 Required environment variables for scoring

```
T2CBENCH_CADSEQ_PATH=<path>/Text2CAD      # vendored CadSeqProc, for cadvec + minimal_json
T2CBENCH_CADFUSION_PATH=<path>/CADFusion  # for skexgen
```

**The CADFusion import root is `src/rendering_utils`, not `src`** — their modules do
`from geometry.arc import Arc`. Pointing at `src` yields 100% `EXEC_FAIL` with no obvious cause.

### 10.4 Measured wall-clock (A100 40 GB, pass@1, 900 prompts/model)

| Stage | Time |
|---|---|
| Build splits (CPU, once) | ~40 min (mesh dedup dominates) |
| Text2CAD | ~25 min |
| cadrille SFT + RL | ~40 min |
| CADmium-7B | ~50 min |
| CADFusion v1.1 | ~60 min |
| Text-to-CadQuery ×2 | ~50 min |
| General LLMs ×5 @ 7B | ~4 h |
| `qwen25-coder-32b` (4-bit) | **1 h 34 m (Split A) + 33 m (Split B)** |
| Scoring, 14 files (CPU, 4 workers) | ~2.5 h |
| Tables + figures | ~10 min |

Roughly **9–10 GPU-hours** for pass@1 across the full roster.

---

## 11. Codebase map — what each file does

```
configs/
  models.yaml            13-model registry: weights, base_model, adapter, runner, template,
                         params, LoRA flag, gating, max_new_tokens. Adding a model is one
                         block here -- nothing else in the harness knows model names.
  prompts.yaml           every prompt template + the generation settings + the token budgets.
                         Native templates are transcribed from each repo, not written fresh.
  ablation_pairs.json    the 4 base->fine-tuned pairs for the ablation table.
  model_families.json    grouping for figures: specialised vs general.

docs/
  01_model_survey.md     per-system survey: architecture, weights, sizes, gating, output format,
                         training data, test split, native inference command. Plus the asset
                         checklist and the contamination analysis.
  02_benchmark_design.md the protocol: splits, adapters, validity codes, metrics, aggregation,
                         sampling, threats to validity.
  03_paper_reference.md  this file.

t2cbench/
  data/
    build_splits.py      --stage a builds Split A (dedup -> stratify -> sample -> 4 levels);
                         --stage b builds Split B from a CADPrompt checkout. Asset fetching is
                         gated so a network blip cannot fail a rebuild of an already-present asset.
    dedup.py             the rotation-tolerant shape signature + the epsilon calibration table.
  adapters/
    base.py              Adapter ABC; subprocess isolation with hard timeout; finalize_solid()
                         (OCC shape -> STL at linear_deflection=0.001); the mesh-closure rules
                         (pinhole repair, multi-body assembly, area guard).
    cadseq_adapter.py    CadVecAdapter (Text2CAD), MinimalJsonAdapter (CADmium),
                         SkexGenAdapter (CADFusion). Null-part dropping; multi-closed-curve
                         loop diagnostic; CADFusion import root handling.
    cadquery_adapter.py  CadQueryAdapter: fenced-block extraction, sandboxed exec, result
                         resolution (r.val(), else last Workplane/Assembly), tessellation.
  metrics/
    geometry.py          canonicalise(); symmetric squared CD; F1@tau; voxel IoU (forced-matching
                         back-end); boolean IoU; symmetric percentile Hausdorff; is_closed();
                         gt_too_thin_for_iou(). All constants live here.
    topology.py          solid/face/edge/vertex counts, Euler characteristic, watertightness,
                         volume/area relative error.
    validity.py          the 8-code Validity enum, AdapterResult, invalidity_ratio(),
                         validity_breakdown().
  runners/
    run_hf.py            the generic HuggingFace driver: LoRA merge onto the declared base,
                         4-bit loading, chat-template on/off, left-padded batching, resumable
                         output, per-model token budget lookup, auto-class fallback for
                         vision-language checkpoints not registered for causal LM.
    run_cadrille.py      drives cadrille's own repo code (it builds its own chat template).
    run_text2cad.py      drives the Text2CAD custom net (BERT encoder + CAD decoder).
  report/
    tables.py            the six tables + RESULTS.md. Score computation, bootstrap CIs,
                         IoU exclusion logic, empty-slice guards.
    figures.py           failure-mode bars, level-sensitivity lines, qualitative render grid.
  evaluate.py            scoring driver: predictions x split -> adapter -> mesh -> metrics ->
                         scored JSONL. Timeout warning at >2%.

scripts/
  setup_scoring_env.sh   builds the pythonocc scoring environment.
  build/smoke:
    smoke_test.py        build_gen_cmd() -- the single place that turns a models.yaml entry into
                         a generation command. Everything else calls it, so smoke and full run
                         cannot drift apart.
    COLAB_SMOKE_CELL.py  the paste-into-Colab smoke cell; exports a JSON report.
  plan_full_run.py       emits (or runs) one command per (model, split) from the registry, in a
                         fixed order, via build_gen_cmd. Exists because run_all.sh drifted --
                         see 13.8.
  localise_splits.py     re-derives gt_mesh paths from uid on the scoring machine.
  score_all.py           scores every raw file with the right adapter in the right env.
  score_export.py        compact JSON export of a scoring run, for pasting between machines.
  render_smoke_figure.py the parts-grid qualitative renderer.
  make_notebooks.py      generates the four Colab notebooks so they cannot drift from the scripts.

notebooks/
  00_setup_data.ipynb              CPU  -- builds both splits
  01_run_specialised_models.ipynb  A100 -- one specialised system per session
  02_run_general_llms.ipynb        A100 -- one general LLM per session
  03_evaluate_and_report.ipynb     CPU  -- scores everything, emits tables + figures
```

**One structural note worth a sentence in the paper.** The roster lives in exactly one place
(`configs/models.yaml`) and every command is built by one function (`build_gen_cmd`). This is not
tidiness for its own sake — the earlier hand-maintained `run_all.sh` had silently drifted to missing
5 of 13 models, dropping CADFusion's `--subfolder v1_1`, and naming a Mistral base that the
checkpoint's own `adapter_config.json` contradicts. Every one of those would have burned A100 hours
and produced a row that *looked like a model result*.

---

## 12. Results

All numbers: Split A, **pass@1**, greedy, n = 500 prompts per model (125 shapes × 4 levels).
CD is ×1000 symmetric squared Chamfer after canonicalisation. `Score = P(OK) × mean F1@0.02` over all
500 prompts, failures counted as zero. CIs are 95% percentile bootstrap over prompts.

**Run of record: 13 models × 900 prompts = 11,700 scored samples, 0 timeouts, 0 harness errors.**

### 12.1 Main table — Split A, pooled (RQ2)

| Model | family | IR % | P(OK) | CD med | F1@0.02 | IoU | IoU n | **Score** | 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| **text2cad** | spec | 10.8 | 0.892 | 51.51 | 0.2235 | 0.2145 | 416 | **0.2018** | [0.1786, 0.2263] |
| **cadmium-7b** | spec | 52.6 | 0.474 | **26.31** | **0.3938** | **0.4125** | 257 | **0.1991** | [0.1687, 0.2302] |
| **cadrille** | spec | 13.4 | 0.866 | 93.25 | 0.2271 | 0.2122 | 412 | **0.1984** | [0.1691, 0.2265] |
| t2cq-mistral-7b | spec | 10.6 | 0.894 | 44.34 | 0.1688 | 0.1905 | 413 | 0.1512 | [0.1332, 0.1696] |
| t2cq-qwen-3b | spec | **8.0** | **0.920** | 44.60 | 0.1621 | 0.1805 | 423 | 0.1488 | [0.1318, 0.1665] |
| cadfusion-v1.1 | spec | 16.2 | 0.838 | 51.24 | 0.1300 | 0.1340 | 379 | 0.1081 | [0.0961, 0.1210] |
| deepseek-coder-6.7b | gen | 55.8 | 0.442 | 37.38 | 0.2037 | 0.1994 | 200 | 0.0901 | [0.0744, 0.1061] |
| qwen25-coder-32b | gen | 59.2 | 0.408 | 36.47 | 0.1968 | 0.2059 | 194 | 0.0813 | [0.0655, 0.0971] |
| qwen25-coder-7b | gen | 67.8 | 0.322 | 40.47 | 0.2169 | 0.2260 | 142 | 0.0714 | [0.0568, 0.0857] |
| llama3-8b-instruct | gen | 82.2 | 0.178 | 35.58 | 0.2201 | 0.2513 | 82 | 0.0392 | [0.0286, 0.0509] |
| qwen2-vl-2b | gen | 92.2 | 0.078 | 30.16 | 0.1967 | 0.1974 | 35 | 0.0153 | [0.0093, 0.0225] |
| mistral-7b-instruct | gen | 93.4 | 0.066 | 42.81 | 0.1415 | 0.1363 | 30 | 0.0093 | [0.0056, 0.0136] |
| cadrille-rl | spec | 52.0 | 0.480 | 145.02 | 0.0113 | 0.0000 | 213 | 0.0054 | [0.0040, 0.0070] |

**The top three are a statistical tie.** text2cad 0.2018, cadmium-7b 0.1991, cadrille 0.1984 — a
spread of 0.0034 against CIs roughly ±0.025 wide, all three overlapping almost completely. The paper
must report a three-way tie, not a winner. (After the token-budget fix, CADmium moved from 0.1633
into that tie; before it, the same model ranked third by a clear margin. See §13.8.)

**Note the IoU coverage column.** `IoU_n` ranges from 30 (mistral-7b-instruct) to 423
(t2cq-qwen-3b) out of 500. A mean IoU over 30 samples and one over 423 are not the same statistic.
This is why IoU cannot be the ranking metric, and why conditional metrics must always be read
beside P(OK).

### 12.2 Per-level table — Split A (the most important table in the study)

Score by prompt level:

| Model | family | L0 | L1 | L2 | L3 | pooled |
|---|---|---|---|---|---|---|
| text2cad | spec | 0.0914 | 0.1300 | 0.1650 | 0.4209 | 0.2018 |
| cadmium-7b | spec | 0.0528 | 0.0583 | 0.1052 | **0.5801** | 0.1991 |
| cadrille | spec | 0.0145 | 0.0246 | 0.1582 | **0.5963** | 0.1984 |
| t2cq-mistral-7b | spec | 0.1039 | 0.1453 | 0.1569 | 0.1986 | 0.1512 |
| t2cq-qwen-3b | spec | 0.1180 | 0.1305 | 0.1538 | 0.1931 | 0.1488 |
| cadfusion-v1.1 | spec | **0.1290** | 0.1261 | 0.1033 | 0.0738 | 0.1081 |
| deepseek-coder-6.7b | gen | 0.1085 | **0.1247** | 0.1114 | 0.0157 | 0.0901 |
| qwen25-coder-32b | gen | 0.0887 | 0.1276 | 0.1071 | 0.0017 | 0.0813 |
| qwen25-coder-7b | gen | 0.0544 | 0.1006 | 0.0882 | 0.0424 | 0.0714 |
| llama3-8b-instruct | gen | 0.0380 | 0.0643 | 0.0544 | **0.0000** | 0.0392 |
| qwen2-vl-2b | gen | 0.0015 | 0.0181 | 0.0160 | 0.0258 | 0.0153 |
| mistral-7b-instruct | gen | 0.0121 | 0.0154 | 0.0098 | **0.0000** | 0.0093 |
| cadrille-rl | spec | 0.0112 | 0.0101 | 0.0003 | 0.0000 | 0.0054 |

**Two crossings, and they carry the paper.**

1. **At L0 and L1 the best general LLM is competitive with, and sometimes beats, the specialised
   systems.** At L1, deepseek-coder-6.7b (0.1247) and qwen25-coder-32b (0.1276) beat cadmium-7b
   (0.0583) and cadrille (0.0246) by 2–5×, and sit within noise of text2cad (0.1300) and
   cadfusion-v1.1 (0.1261). At L0, deepseek (0.1085) beats every specialised system except
   cadfusion and the two Text-to-CadQuery models.
2. **At L3 the general LLMs collapse to zero and the specialised systems peak.** llama3 and mistral
   score exactly **0.0000**; qwen25-coder-32b scores 0.0017. Meanwhile cadrille reaches 0.5963 and
   cadmium 0.5801 — a **350×** gap between the best specialised and the best general model on the
   same 125 prompts.

So the headline is not "fine-tuning helps". It is: **fine-tuning on DeepCAD buys almost nothing when
the prompt is vague, and buys everything when the prompt is a transcription.** A benchmark that
reports only a pooled number over a prompt corpus skewed to either end will reach the opposite
conclusion from one skewed to the other.

### 12.3 Ablation — base LLM vs its own fine-tuned descendant (RQ4)

Same weights, same prompts, same metrics; paired bootstrap over prompts.

| Base | Fine-tuned | Score base | Score FT | **Δ** | 95% CI | sig. | IR base % | IR FT % |
|---|---|---|---|---|---|---|---|---|
| Qwen2-VL-2B-Instruct | **cadrille** | 0.0153 | 0.1984 | **+0.1831** | [+0.1542, +0.2128] | ✔ | 92.2 | 13.4 |
| Qwen2.5-Coder-7B-Instruct | **cadmium-7b** | 0.0714 | 0.1991 | **+0.1277** | [+0.0946, +0.1637] | ✔ | 67.8 | 52.6 |
| Llama-3-8B-Instruct | **cadfusion-v1.1** | 0.0392 | 0.1081 | **+0.0689** | [+0.0526, +0.0865] | ✔ | 82.2 | 16.2 |
| Qwen2-VL-2B-Instruct | **cadrille-rl** | 0.0153 | 0.0054 | **−0.0099** | [−0.0170, −0.0034] | ✔ | 92.2 | 52.0 |

Three fine-tunes beat their own base significantly. **The RL variant is significantly *worse* than
the un-fine-tuned model it descends from** — cadrille-rl loses to plain Qwen2-VL-2B-Instruct.

### 12.4 What the ablation gain is actually made of — matched-subset analysis

The headline Δ conflates two things. Restricting to the prompts **both** models built successfully
separates them:

| Base → fine-tuned | n both OK | CD base | CD FT | F1 base | F1 FT |
|---|---|---|---|---|---|
| qwen25-coder-7b → cadmium-7b | 77 | 41.22 | **39.47** | 0.2080 | **0.2825** |
| llama3-8b-instruct → cadfusion-v1.1 | 80 | **32.47** | 55.19 | **0.2330** | 0.1401 |
| qwen2-vl-2b → cadrille | 36 | **32.36** | 49.72 | 0.1851 | **0.3767** |
| qwen2-vl-2b → cadrille-rl | 11 | **26.04** | 131.97 | **0.2096** | 0.0142 |

**On prompts both can build, Llama-3-8B-Instruct produces better geometry than CADFusion**
(F1 0.233 vs 0.140, CD 32.5 vs 55.2). CADFusion's entire measured advantage is reliability:
IR 82.2% → 16.2%. cadrille is the opposite — it improves conditional quality (F1 0.185 → 0.377)
*and* reliability. CADmium improves both modestly.

Caveat to state: n is 36–80, and the matched subset is by construction the easy tail of the
distribution, so these are descriptive, not inferential.

A second, sharper statistic from the same data — **which prompts each base model could build at all**:

| Base model | OK at L0 | L1 | L2 | **L3** |
|---|---|---|---|---|
| qwen25-coder-7b | 51 | 56 | 40 | 14 |
| llama3-8b-instruct | 32 | 37 | 20 | **0** |
| qwen2-vl-2b | 1 | 11 | 10 | 17 |

**Llama-3-8B-Instruct produced zero valid solids across all 125 L3 prompts.**

### 12.5 Contamination probe — Split B, with a negative control (RQ5)

n_clean = 26 rows (13 uids × 2 variants), n_contaminated = 374 rows (187 × 2).

| Model | family | Score clean | Score contam. | gap | CD med clean | CD med contam. |
|---|---|---|---|---|---|---|
| t2cq-qwen-3b | spec | 0.1757 | 0.2212 | **+0.0455** | 58.10 | 37.11 |
| cadrille | spec | 0.0404 | 0.0847 | **+0.0444** | 158.19 | 115.89 |
| t2cq-mistral-7b | spec | 0.1821 | 0.2201 | **+0.0380** | 57.84 | 35.68 |
| **llama3-8b-instruct** | **gen** | 0.0776 | 0.1109 | **+0.0333** | 85.75 | **18.15** |
| **deepseek-coder-6.7b** | **gen** | 0.1200 | 0.1312 | **+0.0112** | 88.12 | **32.10** |
| **qwen2-vl-2b** | **gen** | 0.0000 | 0.0074 | +0.0074 | — | 0.15 |
| cadrille-rl | spec | 0.0089 | 0.0107 | +0.0018 | 149.13 | 134.53 |
| qwen25-coder-32b | gen | 0.1971 | 0.1946 | −0.0025 | 79.80 | 30.37 |
| cadmium-7b | spec | 0.1436 | 0.1408 | −0.0028 | 50.24 | 35.96 |
| mistral-7b-instruct | gen | 0.0467 | 0.0363 | −0.0104 | 71.20 | **10.31** |
| qwen25-coder-7b | gen | 0.1216 | 0.1082 | −0.0134 | 12.25 | 31.98 |
| cadfusion-v1.1 | spec | 0.1578 | 0.1398 | −0.0180 | 82.35 | 47.37 |
| text2cad | spec | 0.1898 | 0.1537 | −0.0360 | 79.90 | 64.92 |

Mean gap, **specialised** (trained on DeepCAD): **+0.0104**.
Mean gap, **general LLMs** (never trained on DeepCAD): **+0.0043**.

See §13.6 — this table refutes the naive memorisation reading rather than supporting it.

### 12.6 Complexity table — Split A

Score by CAD-sequence complexity bin (n_extrusions × n_curves), specialised systems:

| Model | simple (128) | moderate (176) | complex (100) | very_complex (96) |
|---|---|---|---|---|
| text2cad | 0.3060 | 0.2026 | 0.1624 | 0.1026 |
| cadrille | 0.2304 | 0.2281 | 0.1717 | 0.1293 |
| t2cq-qwen-3b | 0.1782 | 0.1360 | 0.1747 | 0.1063 |
| t2cq-mistral-7b | 0.1591 | 0.1456 | 0.1836 | 0.1170 |
| cadfusion-v1.1 | 0.1091 | 0.0949 | 0.1298 | 0.1081 |
| cadrille-rl | 0.0074 | 0.0059 | 0.0050 | 0.0023 |

### 12.7 Failure modes — Split A, % of 500 (RQ6)

| Model | family | OK | PARSE_FAIL | EXEC_FAIL | EMPTY | INVALID | NON_MANIF | TIMEOUT |
|---|---|---|---|---|---|---|---|---|
| t2cq-qwen-3b | spec | 92.0 | 0.6 | 4.4 | 0.0 | 0.0 | 2.8 | 0.2 |
| t2cq-mistral-7b | spec | 89.4 | 2.6 | 4.8 | 0.2 | 0.0 | 3.0 | 0.0 |
| text2cad | spec | 89.2 | **0.0** | 1.4 | **3.2** | **1.8** | 4.4 | 0.0 |
| cadrille | spec | 86.6 | 0.2 | 7.2 | 0.4 | 0.0 | 5.6 | 0.0 |
| cadfusion-v1.1 | spec | 83.8 | 5.0 | 9.4 | 0.2 | 0.6 | 1.0 | 0.0 |
| cadrille-rl | spec | 48.0 | **52.0** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| cadmium-7b | spec | 47.4 | 4.8 | **33.2** | 0.8 | 5.0 | 8.8 | 0.0 |
| deepseek-coder-6.7b | gen | 44.2 | 2.4 | **50.6** | 1.6 | 0.0 | 1.2 | 0.0 |
| qwen25-coder-32b | gen | 40.8 | 1.2 | **56.6** | 0.4 | 0.0 | 1.0 | 0.0 |
| qwen25-coder-7b | gen | 32.2 | 0.4 | **66.2** | 0.4 | 0.0 | 0.8 | 0.0 |
| llama3-8b-instruct | gen | 17.8 | 3.0 | **78.6** | 0.6 | 0.0 | 0.0 | 0.0 |
| qwen2-vl-2b | gen | 7.8 | **50.0** | 41.8 | 0.4 | 0.0 | 0.0 | 0.0 |
| mistral-7b-instruct | gen | 6.6 | 5.6 | **86.8** | 1.0 | 0.0 | 0.0 | 0.0 |

Scale is not the cure: `qwen25-coder-32b` (32B, 4-bit) still fails to execute on **56.6%** of prompts,
against 66.2% for its 7B sibling. A 4.6× parameter increase buys 10 points of executability.

### 12.8 Figures produced

1. `fig1_failure_modes.png` — stacked validity-code bars per model.
2. `fig2_level_sensitivity.png` — Score vs L0→L3, one line per model. **Carries the two crossings in
   §12.2 and is the single most important figure in the paper.**
3. `fig3_qualitative.png` / `parts_full_splitA.png` — render grid, models × prompts, GT on top.

---

## 13. Findings and causal reasoning

Each finding: what was measured, then the mechanism, then what it does and does not license.

### 13.1 Prompt specificity dominates model identity (RQ3) — the headline finding

**Measured.** Within a model, Score varies up to **41×** across L0→L3 (cadrille 0.0145 → 0.5963).
CD medians move three orders of magnitude within a model (CADmium L0 47.7 → L3 0.116). Across models
*at a fixed level* the spread is far smaller: at L0 every one of the thirteen scores between 0.0015
and 0.1290.

**Mechanism.** L3 prompts are near-literal transcriptions of the CAD program — coordinate system,
each line's endpoints, sketch scale, extrusion depth. At L3 a model does not design, it transcribes,
and transcription is a grammar-learning problem a 2B model solves well. At L0 it gets a one-line
caption and must infer geometry, which none of the thirteen do well.

**What it licenses.** Published single-number text-to-CAD results are dominated by prompt specificity
and are not measuring design ability. Any paper reporting one averaged number over a corpus whose
prompts skew detailed is reporting a transcription score.

### 13.2 The top three are tied, and the general-LLM gap is not

**Measured.** text2cad 0.2018 [0.1786, 0.2263], cadmium-7b 0.1991 [0.1687, 0.2302], cadrille 0.1984
[0.1691, 0.2265] — 0.0034 apart with near-total CI overlap. By contrast every base-vs-fine-tuned
delta in §12.3 has a CI clear of zero.

**Implication.** Report a three-way tie at the top. The benchmark has the resolution to separate
fine-tuned from general models (Δ ≈ 0.07–0.18) but not to separate the three leaders (Δ ≈ 0.003).
Saying so is a contribution: prior work in this area reports point estimates with no intervals, and
at these sample sizes many published "improvements" are within noise.

### 13.3 General LLMs win at vague prompts and score exactly zero at precise ones

**Measured.** Two crossings in §12.2. At L1, deepseek-coder-6.7b (0.1247) and qwen25-coder-32b
(0.1276) beat cadmium-7b (0.0583) and cadrille (0.0246) by 2–5× and are within noise of text2cad
(0.1300). At L3, llama3-8b-instruct and mistral-7b-instruct score **exactly 0.0000**, qwen25-coder-32b
0.0017, while cadrille reaches 0.5963 — a **350×** gap on the same 125 prompts.
Llama-3-8B-Instruct produced **zero valid solids across all 125 L3 prompts** (§12.4).

**Mechanism.** L3 prompts are long (median 539 BERT tokens, max 2,258) and numerically dense. General
LLMs asked to transcribe hundreds of coordinates into CadQuery degenerate into repetition loops and
run out of budget mid-expression — visible directly in qwen2-vl-2b's outputs, where `.cut(...)` calls
repeat verbatim until the cap. The fine-tuned models learned a compact grammar for exactly this and
emit it in a few hundred tokens.

**What it licenses.** The strongest claim in the study: **CAD fine-tuning on DeepCAD buys almost
nothing when the prompt is vague, and buys everything when the prompt is a transcription.** A
benchmark whose prompt corpus skews to either end reaches the opposite conclusion from one skewed to
the other — which is precisely why the field's numbers disagree.

**What it does not license.** It does not show general LLMs are better designers at L0; every model
is poor there in absolute terms (best 0.1290 of a possible 1.0). The honest statement is that at L0
the specialised systems have not earned their fine-tuning.

### 13.4 Fine-tuning buys reliability, not always geometry

**Measured** (§12.4, matched subsets). On prompts both models built: CADFusion is **worse** than its
own base — F1 0.140 vs Llama-3's 0.233, CD 55.2 vs 32.5 — while its IR falls 82.2% → 16.2%. cadrille
improves both (F1 0.185 → 0.377, IR 92.2% → 13.4%). CADmium improves both modestly (F1 0.208 → 0.283).

**Mechanism.** A base LLM emitting CadQuery succeeds only on the small easy tail it can express in a
few lines; conditional on succeeding it does reasonably. Fine-tuning trades some conditional fidelity
for a vastly larger success set.

**What it licenses.** `Score = P(OK) × F1` is doing real work here. A leaderboard on median-CD-over-
valid-only would rank **Llama-3-8B-Instruct above CADFusion** (35.58 vs 51.24 pooled, 32.47 vs 55.19
matched) — an inversion produced entirely by survivorship. State this explicitly; it is the clearest
available argument for the composite metric.

**Caveat.** Matched n is 36–80 and is by construction the easy tail. Descriptive, not inferential.

### 13.5 Failure mode is predicted by representation, not by quality (RQ6)

**Measured.** Failure signatures cluster by what a model emits, not by how good it is.
CadQuery emitters fail at `EXEC_FAIL` — 4.4% (t2cq-qwen) to **86.8%** (mistral-7b-instruct) — with
`PARSE_FAIL` ≈ 0 and `EMPTY_SOLID` ≈ 0. Sequence emitters fail at the geometry stage: text2cad has
`EMPTY_SOLID` 3.2% and `INVALID_SOLID` 1.8% with `EXEC_FAIL` only 1.4%. **text2cad's `PARSE_FAIL` is
0.0%, structurally** — a fixed-length integer vector cannot be syntactically malformed.

**Mechanism.** Each representation places the failure boundary somewhere different. Free-form code
can fail to run but rarely fails to parse; a constrained integer vector always parses but can decode
to a degenerate solid; verbose JSON can be truncated into unparseability.

**What it licenses.** A scalar invalidity ratio aggregates failures occurring at different stages of
different pipelines and is not a comparable quantity across representations. The 8-way decomposition
is the contribution; argue that IR should never again be reported as a scalar in this field.

### 13.6 The contamination probe is refuted by its own control group (RQ5)

**Measured.** Mean clean→contaminated gap: **+0.0104** for the seven specialised systems trained on
DeepCAD, **+0.0043** for the six general LLMs that were not. Individually, llama3-8b-instruct shows
**+0.0333** — larger than four of the seven specialised systems — and its CD median improves from
85.75 on clean to **18.15** on contaminated. mistral-7b-instruct improves from 71.20 to **10.31**.

**Mechanism.** The general LLMs are a **negative control**: they have no DeepCAD training, so a gap
driven by memorisation should be zero for them. It is not — it is the same sign and comparable size.
Therefore the gap is measuring a **difficulty difference between the two slices**, not memorisation.
The 13 clean uids are simply harder than the 187 contaminated ones.

**What to write.** Do **not** claim a memorisation effect was measured — the control refutes it, and
n_clean = 13 uids could not have supported the claim anyway. Instead report this as a methodological
result: *a clean-vs-contaminated score gap is not by itself evidence of memorisation, and running
never-trained models through the same split is a cheap control that can falsify it.* That is a
transferable contribution, and it is more interesting than the finding it replaces.

Then use CADPrompt for the claim that needs no statistical power (§4.5): **93.5% of it is training
data for the fine-tuned systems, so it cannot serve as a headline benchmark comparing them against
general LLMs.** That is a fact about uid lists, and it stands regardless of this table.

### 13.7 cadrille-RL: RL fine-tuning made the model worse than no fine-tuning at all

**Measured.** cadrille-rl scores 0.0054 against SFT's 0.1984 (37× worse from the same base) and —
critically — against its **own un-fine-tuned base model's 0.0153**: Δ = **−0.0099, CI [−0.0170,
−0.0034], significant**. 52.0% PARSE_FAIL, all token soup; voxel IoU 8.5 × 10⁻⁷. The collapse is
monotone in prompt length: IR 0.8% at L0, 15.2% at L1, **93.6% at L2, 98.4% at L3**.

**Mechanism.** At L0 it is *more* valid than the SFT model (IR 0.8% vs 8.0%) — so this is not uniform
degradation, it is a policy narrowed onto short in-distribution inputs. The reward was computed on
cadrille's own short single-sentence descriptions, and the resulting policy does not survive a
539-token L3 prompt.

**What it licenses.** A clean controlled demonstration — same base, same architecture, same prompts,
same metrics, SFT vs RL the only difference — that **RL on a narrow prompt distribution can destroy
out-of-distribution instruction-following while improving in-distribution validity, to the point of
scoring below the un-fine-tuned base model.** The L0 improvement is what makes this a finding rather
than "the checkpoint is broken".

### 13.8 Benchmark design choices moved the ranking more than the models differ

The methodological contribution, and it is an audit of our *own* harness. Three defects were found
after the first full run; each changed the answer.

**(a) The Score definition inverted the top two.** The design doc specified `Score = P(OK) × F1`; the
implementation summed F1 over all *scored* rows including `NON_MANIFOLD`, which is measurable but not
a closed solid. Implemented: cadrille 0.2094 vs text2cad 0.2092. As documented: text2cad 0.2018 vs
cadrille 0.1984. **A one-line spec/code disagreement flipped first place.**

**(b) The token budget cost CADmium 22% of its samples — measured before and after.** A flat 1024
budget is not equal treatment: CADmium's pretty-printed JSON costs ~3.1 chars/token against ~2.6 for
CadQuery. Its longest *complete* output was 1,019 tokens against a 1,024 cap. Raising it to 2048 and
regenerating:

| | 1024 tokens | 2048 tokens |
|---|---|---|
| PARSE_FAIL | 23.6% | **4.8%** |
| P(OK) | 0.384 | **0.474** |
| CD median | 27.06 | 26.31 |
| F1@0.02 (conditional) | 0.3939 | **0.3938** |
| **Score** | 0.1633 | **0.1991** |
| rank | 3rd, clear margin | **tied 1st** |

Conditional geometry is unchanged to four decimal places — the budget never affected what CADmium
could *build*, only how many outputs survived to be built. **A benchmark parameter moved a model from
a clear third place into a three-way tie for first**, while the model itself did not change: greedy
decoding made 370 of its complete outputs byte-identical across the two runs.

**(c) An extraction fallback was worth 80% of a model.** llama3-8b-instruct leaves **79.8%** of its
Split A code fences unclosed (81.0% on Split B) — it simply omits the trailing fence. Its code is
complete and compiles 97.0% of the time. The adapter's unclosed-fence fallback recovers all of it
(0 empty extractions across 4,800 general-LLM outputs). Without that one regex llama3 would have
scored ~80% `PARSE_FAIL` and **CADFusion's ablation would have been meaningless**.

**The general rule for the paper.** *A token budget shared across systems that emit different
representations is not a fair budget, and an output-format convention is not a capability.* Both
checks are cheap: count outputs that do not end in the format's own terminator, and count how often
each leniency rule fires per model. (When we first ran the terminator check our predicate returned
`True` unconditionally for three models, making their "0.0% truncation" vacuous — so the check needs
checking too.)

**Additional harness defects found by audit**, each of which would have produced a plausible-looking
but wrong row:

| Defect | Consequence if unfixed |
|---|---|
| CADFusion import root `src` instead of `src/rendering_utils` | 100% `EXEC_FAIL` — reads as "CADFusion cannot produce geometry" |
| Text-to-CadQuery Mistral given the `### Instruction:` scaffold | echoes the prompt, emits no code — reads as "cannot do CAD" |
| `run_all.sh` roster drift | 5 of 13 models silently absent; CADFusion loading `v1_0` not `v1_1`; wrong Mistral base |
| Split `gt_mesh` paths not portable | 500/500 Split A rows pointed at a placeholder; would surface as model invalidity |
| Qwen2-VL-2B not registered for `AutoModelForCausalLM` | cadrille's base-model ablation impossible to run |
| `tables.py` empty-slice crash | report generation dies *after* all GPU work is done |

**Framing.** Report these as a benchmark-engineering section, not an apology. The defensible claim:
**we audited the harness against recorded artefacts and found that protocol defects moved results by
more than the between-model differences we were trying to measure — evidence that this field's
published comparisons, none of which report such an audit, are not reliable.**

### 13.9 Scale is not the cure

**Measured.** qwen25-coder-32b (32B, 4-bit) still fails to execute on **56.6%** of Split A prompts,
against 66.2% for qwen25-coder-7b — a 4.6× parameter increase buying ten points of executability and
a Score of 0.0813 vs 0.0714 (overlapping CIs). Within Text-to-CadQuery, the 7B LoRA and the 3B full
fine-tune are statistically indistinguishable: Δ = +0.0023, CI [−0.0109, +0.0140].

**What it licenses.** On this task, within the ranges tested, parameter count is not the binding
constraint — representation and fine-tuning are. A 2B fine-tune (cadrille, 0.1984) beats a 32B
general model (0.0813) by 2.4×.

---

## 14. Fairness — designed, verified, corrected

### 14.1 Corrections to earlier records

Two numbers recorded earlier in this project are wrong and are corrected here. Use the values in this
file.

**(a) Text2CAD's prompt truncation is larger than previously stated.** `docs/02_benchmark_design.md`
§8.8 says "10% of Split A prompts — all L3 — exceed" the 512-token BERT encoder limit. Re-measured
with `google-bert/bert-large-uncased` against the actual Split A file:

| Level | median tokens | p90 | max | **> 512** |
|---|---|---|---|---|
| L0 | 31 | 43 | 59 | 0 |
| L1 | 52 | 74 | 90 | 0 |
| L2 | 135 | 205 | 327 | 0 |
| L3 | **539** | 945 | 2258 | **70 / 125 = 56.0%** |

**70 of 500 Split A prompts (14.0%) exceed the cap, and that is 56% of all L3 prompts — the *median*
L3 prompt is already over the limit.** `Cad_VLM/models/layers/text_embed.py` calls the tokenizer with
`truncation=True, max_length=512`, so it truncates **silently**. Text2CAD's L3 column must be read as
"the first 512 tokens of the prompt". This cuts both ways and both should be said: it is a caveat on
its L3 number, and it is also a point in its favour — it reaches Score 0.4209 at L3 while seeing
roughly half of the median L3 prompt.

**(b) CADFusion truncation, stated precisely.** 4.6% of Split A outputs do not *end* with
`<extrude_end>`; 2.4% contain no `<extrude_end>` at all. Both figures are correct under their
respective definitions; quote the definition alongside the number.

### 14.2 Fairness properties that were verified, not assumed

Checked against the recorded artefacts of the 6,300-generation run:

| Property | Evidence |
|---|---|
| Every model saw the same prompts | identical 500 Split A `sample_id`s across all 7 models; the split's task text appears verbatim inside every recorded `prompt_sent` |
| No input truncation at the runner | longest prompt ~2,900 tokens against the runner's 4,096 cap (Text2CAD's own 512-token encoder is separate — §14.1a) |
| Decoding identical | `do_sample=False` for all; each model's own bundled sampling config is explicitly overridden |
| Native scaffolds are the checkpoints', not ours | each transcribed from that repo's inference script and cross-checked against `adapter_config.json` |
| Adapter leniency not tilted | §14.3 |
| No model refused or emitted prose only | 0 refusal-pattern hits and 0 empty extractions across all 4,800 general-LLM outputs, both splits |
| Output format never charged as incapacity | every general LLM's extracted code was non-empty; compile rates 94.4–100% except where the model genuinely looped (§13.8c) |
| Machine load absent from results | 1 `TIMEOUT` in 6,300 samples (0.02%), against a 2% warning threshold |
| Scoring path agrees with itself | `evaluate.py` and `score_export.py` match to printed precision on identical inputs |
| Ground truth identical for all | one `gt_mesh` per uid, resolved from the uid, shared across every model's scoring |
| Canonicalisation symmetric | GT and prediction go through the same function; no per-model convention |

### 14.3 Adapter leniency is symmetric — measured

The three leniency rules (§6.2) could each be a thumb on the scale, so the *size* of each rescue was
measured per model:

- Fenced-code / output extraction changed **0 of 500** outputs for cadrille, **0 of 500** for
  t2cq-qwen-3b and **0 of 500** for t2cq-mistral-7b. Those models emit clean output; the lenient path
  exists for them but never fires.
- The same class of repair **rescued 46 CADmium samples** (null-part dropping).
- The multi-body assembly rule moved CADmium from 1/3 to 3/3 valid on a smoke sample.
- The **unclosed-fence** fallback fired on **399/500** llama3-8b-instruct outputs (Split A; 324/400 on
  Split B), **253/500** for qwen2-vl-2b, and **2/500** for qwen25-coder-7b. It is worth 80% of
  llama3's samples and nothing at all to the model that formats cleanly — leniency available to
  everyone, realised only where each model needs it.

**The generous path exists for every representation and is used only where needed.** That is the
fairness argument: leniency is available symmetrically and its realised effect is asymmetric only
because the models' output hygiene is asymmetric — which is a property of the models, and is exactly
what `PARSE_FAIL` is supposed to measure.

### 14.4 The fairness argument in one paragraph, for the paper

*Every system is re-run from its own published checkpoint, given its own native prompt scaffold
transcribed from its own inference code, with its own architectural token limit, decoded greedily at
the same batch size, on an identical set of 900 prompts whose ground truth is resolved once and
shared. Every output is reduced to a mesh — the only representation all six formats agree on — by an
adapter that is counted as part of the system, so a model that emits unexecutable code is charged
with a failure rather than dropped. Metrics are computed by one implementation, with canonicalisation
applied identically to prediction and reference, a voxel back-end forced to match on both sides, and
ground truth that cannot support a given metric excluded from that metric only and flagged per row.
The ranking statistic requires a closed solid for credit and counts every failure as zero, so it
cannot be gamed by abstention, and it is reported with a bootstrap confidence interval so that ties
are visible as ties.*

---

## 15. Threats to validity — state these in the paper

A reviewer will find each of these. Stating them first is cheaper than being caught by them.

1. **CADFusion's split is unverified.** It trains on SkexGen, whose train/test boundary is not
   DeepCAD's. Until the SkexGen split is obtained (it is only on Google Drive), some Split A uids may
   be in its training data. Flagged per row in the results.
2. **Prompt-distribution mismatch is not equalised.** CADmium and cadrille were trained on their own
   paraphrases of DeepCAD sequences, not on Text2CAD L0–L3, so their Split A numbers include a
   distribution-shift penalty that Text2CAD — the only system trained on all four levels — does not
   pay. §13.7 argues this is a *finding* rather than only a threat, but both readings must be offered.
3. **Ground truth is DeepCAD**, which is sketch-extrude only. Nothing here measures fillets, chamfers,
   revolves, lofts or assemblies. **Do not claim general CAD ability.**
4. **Text2CAD reads at most 512 tokens** — 56% of L3 prompts are silently truncated (§14.1a).
   Architectural to the published model, so reported rather than fixed.
5. **CADFusion truncates its own output** at 512 tokens: 4.6% of Split A outputs end mid-sequence.
   Kept as native, but a few points of its invalidity are budget, not ability.
6. **n = 13 clean CADPrompt uids** is far too small for ranking or for estimating memorisation
   (§13.6). It is a probe, and an underpowered one.
7. **Tessellation tolerance** (`linear_deflection = 0.001`) moves CD in the 4th decimal. Fixed across
   all systems, but it is a free parameter.
8. **Batch-size determinism is assumed, not proven.** All models generate at batch size 8 with left
   padding, so any batching effect applies equally — a reproducibility caveat rather than a fairness
   one — but greedy decoding under left padding is not guaranteed bit-identical to batch size 1, and
   that has not been verified on a GPU.
9. **The 1-shot example** in the general-LLM prompt is itself a design choice that advantages general
   LLMs relative to 0-shot. Both variants are supported; both should be reported.
10. **The dedup threshold ε = 0.08 is a choice.** It is calibrated against the measured
    nearest-neighbour distribution and errs toward keeping shapes, but a different ε gives a
    different Split A. The calibration table is published so the choice is auditable.
11. **`Score = P(OK) × F1@0.02` is a composite** and the τ = 0.02 threshold is a choice. F1@0.05 is
    reported alongside; the ranking is not sensitive to that swap, but it should be shown rather than
    asserted.
12. **Evaluation is single-run.** pass@1 with greedy decoding is deterministic given the same
    software stack, but no seed-variance study across GPU/driver versions was done.

---

## 16. Status and what remains

### 16.1 Complete

- Both splits built, 900 prompts, reproducible from `seed=0`.
- **All 13 systems generated and scored: 11,700 samples, 0 timeouts, 0 harness errors.**
- CADmium regenerated at the corrected 2048-token budget; the fix verified by byte-comparison against
  run 1 (370/370 complete outputs identical, 130/130 truncated outputs extended).
- Llama-3-8B-Instruct licence cleared; all four ablation pairs populated with paired CIs.
- Six tables + `RESULTS.md`; qualitative parts-grid figure rendered.
- Full fairness audit of the harness against recorded artefacts (§13.8, §14).

### 16.2 Outstanding

- **best-of-5 table** — not yet run. This is what makes our numbers comparable to the published
  best-of-N results, and its absence is the largest remaining gap in the comparison-with-prior-work
  story. `02_best_of_k_splitA.csv` is empty by construction until then.
- Sequence-F1 table for the three sequence-emitting systems.
- Optional `--zero-shot` general-LLM variant (12 jobs) — quantifies format compliance vs geometry.
- Level-sensitivity and failure-mode figures at final data.
- The back-fill path in `evaluate.py` has still never fired (every model returned exactly 500/400)
  and remains untested.

### 16.3 Reproducibility checklist

| Item | Where |
|---|---|
| Every model's exact weights + revision | `configs/models.yaml` |
| Every prompt template, verbatim | `configs/prompts.yaml` |
| Split construction, seeded | `t2cbench/data/build_splits.py`, `seed=0` |
| Dedup threshold + calibration | `t2cbench/data/dedup.py`, ε = 0.08 |
| All metric constants | `t2cbench/metrics/geometry.py` |
| Generation commands | `scripts/plan_full_run.py` (single source: `build_gen_cmd`) |
| Scoring environment | `scripts/setup_scoring_env.sh` |
| Raw model outputs | `results/raw/*.jsonl` — retained, one row per prompt, with `prompt_sent` |
| Per-sample scores | `results/scored/*.jsonl` — validity code + every metric per sample |

Raw outputs and per-sample scores are both retained, which means every table in the paper can be
recomputed from the artefacts without re-running a GPU.

---

## 17. Suggested paper structure, and which numbers go where

| Section | Content | Source |
|---|---|---|
| **1. Introduction** | The field cannot compare its own results. Three concrete reasons: 3 different CDs, all-best-of-N, 93.5% contamination. | §4 |
| **2. Related work** | The 5 systems + CADPrompt + DeepCAD. Table of what each emits. | §3, §4.1 |
| **3. Why prior numbers are incomparable** | The strongest early section. CD normalisation table, IR definitions, sampling protocols, contamination counts. | §4.2–4.5 |
| **4. T2C-Bench design** | Universal interface figure; 8-code validity taxonomy; metric protocol; the ranking statistic and why. | §6, §7 |
| **5. Evaluation set** | Split A construction incl. dedup calibration; Split B as a probe not a leaderboard. | §5 |
| **6. Experimental setup** | Models, prompts, token budgets, sampling, hardware. | §3, §8, §10 |
| **7. Results** | Main table w/ CIs; per-level table; complexity; failure modes; contamination; ablation. | §12 |
| **8. Analysis** | Prompt specificity dominates (41×); the three-way tie; general LLMs win at L0/L1 and score 0.0000 at L3; fine-tuning buys reliability not geometry; the RL model losing to its own base; representation-structured failure; CADFusion's inversion; the contamination control. | §13.1–13.7, §13.9 |
| **9. Benchmark engineering** | The self-audit: protocol defects moved the ranking by more than the models differ. | §13.8, §14 |
| **10. Threats to validity** | All twelve, up front. | §15 |
| **11. Conclusion** | These systems are positioned, not ranked. IR should never again be a scalar. Report CIs. | §13 |

### 17.1 The figures to lead with

1. **Level-sensitivity** (`fig2`) — one line per model, Score vs L0→L3. It carries the paper's main
   finding in a single image, including CADFusion's inversion and cadrille-RL's collapse.
2. **Failure-mode stacked bars** (`fig1`) — makes the representation-structured failure argument
   visually, and justifies the 8-code taxonomy in one glance.
3. **Qualitative grid** (`fig3` / `parts_full_splitA.png`) — models × prompts with GT on top and
   CD/IoU per cell. Reviewers in this field expect to see the parts.
4. **Ablation bars** (§12.3) — base vs fine-tuned Score with paired CIs, four pairs. One bar goes the
   wrong way (cadrille-rl), which is the point.

### 17.2 Three sentences that should appear somewhere verbatim

> No ranking claim is made between models whose Score confidence intervals overlap.

> A token budget shared across systems that emit different representations is not a fair budget.

> A scalar invalidity ratio aggregates failures occurring at different stages of different
> pipelines, and is not a comparable quantity across representations.

> A clean-vs-contaminated score gap is not by itself evidence of memorisation: models that never saw
> the training data show the same gap.

> CAD fine-tuning on DeepCAD buys almost nothing when the prompt is vague and everything when the
> prompt is a transcription.

---

## 18. Citation targets

Text2CAD ([arXiv:2409.17106](https://arxiv.org/abs/2409.17106), NeurIPS'24) ·
CADmium ([arXiv:2507.09792](https://arxiv.org/abs/2507.09792), TMLR) ·
CADFusion ([arXiv:2501.19054](https://arxiv.org/abs/2501.19054), ICML'25) ·
cadrille ([arXiv:2505.22914](https://arxiv.org/abs/2505.22914)) ·
Text-to-CadQuery ([arXiv:2505.06507](https://arxiv.org/abs/2505.06507)) ·
CADPrompt (`Kamel773/CAD_Code_Generation`, ICLR'25) ·
DeepCAD (`ChrisWu1997/DeepCAD`) · SkexGen · CAD-Recode.
