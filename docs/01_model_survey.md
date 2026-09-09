# Model & Data Survey

Everything below was verified by cloning each repo and querying the HuggingFace API on 2026-09-09,
not from the papers alone. Where a paper and its code disagree, the code wins.

---

## 1. The five text-to-CAD systems

### 1.1 Text2CAD (NeurIPS'24 Spotlight)

| | |
|---|---|
| Repo | `SadilKhan/Text2CAD` |
| Paper | [arXiv:2409.17106](https://arxiv.org/abs/2409.17106) |
| Architecture | Custom transformer decoder (not an LLM). BERT text encoder → adaptive layer → CAD-sequence decoder |
| Weights | `SadilKhan/Text2CAD` HF **dataset** repo, `text2cad_v1.0/Text2CAD_1.0.pth`, **91.6 MB** |
| **Gated** | **YES** — you must accept the license on the HF page and use an `HF_TOKEN`. This is the only gated asset in the whole benchmark |
| Output | **CAD vector sequence**: `(N, 2)` int array of `(token_type, param)` pairs, 8-bit quantised (256 bins), max length 272 |
| Output → geometry | `CADSequence.from_vec(vec, bit=8, post_processing=True).create_cad_model()` → OCC solid → `create_mesh()` |
| Training data | DeepCAD train split + Text2CAD annotations (~170K models, ~660K prompts) |
| Test set | DeepCAD test split (8,046 uids) |
| Inference | `Cad_VLM/test.py --config_path config/inference.yaml`, top-k decode, **5 samples/prompt** by default |

**Prompt levels** (this is where L0–L3 comes from — confirmed against the annotation CSV columns):

| Level | CSV column | Content |
|---|---|---|
| L0 | `abstract` | One-line shape description from a VLM. *"A rectangular block with a flat top and bottom, and straight edges."* |
| L1 | `beginner` | Layperson design steps, no measurements or jargon |
| L2 | `intermediate` | Generalised geometric description, some detail abstracted away |
| L3 | `expert` | Full geometric description with relative values — coordinate system, every line's start/end point, sketch scale, extrusion depth |

L3 is nearly a literal transcription of the CAD program. L0 is a caption. This spread is the whole
point of the benchmark: **a model can win at L3 by learning a transcription grammar and still be
useless at L0.**

---

### 1.2 CADmium (TMLR 2026)

| | |
|---|---|
| Repo | `chandar-lab/CADmium` |
| Paper | [arXiv:2507.09792](https://arxiv.org/abs/2507.09792) |
| Architecture | **LoRA adapter** (r=64, α=16, all 7 proj modules) on **`Qwen/Qwen2.5-Coder-7B-Instruct`** |
| Weights | `chandar-lab/CADmium-7B` — `adapter_model.safetensors`, **646 MB** (adapter only; base model pulled separately). Also 1.5B / 3B / 14B |
| Gated | No |
| Output | **Minimal JSON** — `{"parts": {"part_1": {"coordinate_system": {...}, "sketch": {"face_1": {"loop_1": {"line_1": {...}}}}, "extrusion": {...}}}}`, coordinates in DeepCAD normalised units |
| Output → geometry | `CADSequence.from_minimal_json(json)` (vendored Text2CAD `CadSeqProc`) → OCC solid → mesh |
| Training data | `chandar-lab/CADmium-ds` — 176,017 GPT-4.1 descriptions of DeepCAD sequences (from minimal JSON + up to 10 Blender renders) |
| Test set | 8,046 uids = **the DeepCAD test split**, one description per uid (not 4 levels) |
| Inference | `torchrun cadmium/src/predict.py --config-name predict` |

Note CADmium's prompts are its **own** GPT-4.1 annotations, not Text2CAD's. Roughly L3-like in
detail. Running it on Text2CAD L0–L3 prompts is a genuine distribution shift — which is exactly
what we want to measure, but it must be labelled as such in the results.

---

### 1.3 CADFusion (ICML 2025)

| | |
|---|---|
| Repo | `microsoft/CADFusion` |
| Paper | [arXiv:2501.19054](https://arxiv.org/abs/2501.19054) |
| Architecture | **LoRA adapter** (r=32, α=32, q/v only) on **`meta-llama/Meta-Llama-3-8B`** |
| Weights | `microsoft/CADFusion` — `v1_0/` (5 rounds) and `v1_1/` (9 rounds, better), **4.26 GB each** |
| Gated | No (but the Llama-3-8B base is gated on HF — accept Meta's license) |
| Output | **SkexGen command-sequence text**: `line,x,y <curve_end> ... <loop_end> <face_end> <sketch_end> add,ext_v,ext_T,ext_R,scale,offset <extrude_end>` — space-separated quantised ints |
| Output → geometry | `src/rendering_utils/parser.py` (`CADparser`) → `.obj` → `parser_visual.py` → STEP/STL |
| Training data | **SkexGen**, *not* Text2CAD. LLM-captioned descriptions of 952-test / ~180K-train SkexGen shapes |
| Test set | 952 shapes (`data/sl_data/test.json`, `serial_num` + `description`) |
| Inference | `./scripts/generate_samples.sh <run> test --full`, **5 samples/prompt**, temp 0.3 (test) / 0.9 |

⚠️ **CADFusion is the odd one out.** Different training corpus (SkexGen), different split, different
tokenisation, and its native prompts are single-sentence visual captions ("a twelve-sided prism base
with a large central circular cutout") — much closer to **L0** than to L3. Expect it to look
comparatively strong at L0 and weak at L3, and say so in the analysis rather than pretending the
comparison is like-for-like.

---

### 1.4 cadrille (2025)

| | |
|---|---|
| Repo | `col14m/cadrille` |
| Paper | [arXiv:2505.22914](https://arxiv.org/abs/2505.22914) |
| Architecture | **Qwen2-VL-2B-Instruct** + point-cloud encoder, full fine-tune. Multi-modal: point cloud / image / **text** |
| Weights | `maksimko123/cadrille` (SFT) and `maksimko123/cadrille-rl` (RL), **4.42 GB each**, full model |
| Gated | No |
| Output | **CadQuery Python code**, CAD-Recode style with integer coords on a ~100 scale: `w0=cq.Workplane('XY',origin=(-100,0,-14)); r=w0.sketch().face(...).finalize().extrude(28)` — result is always in variable `r` |
| Output → geometry | `exec(code)` → `r.val()` → `.tessellate(0.001, 0.1)` → trimesh |
| Training data | Text2CAD (train/val) + CAD-Recode v1.5. **Ships CadQuery ground truth for 171,184 DeepCAD uids** (`maksimko123/text2cad`) — a very useful asset |
| Test set | `test.pkl`, 8,035 uids, **all ⊂ DeepCAD test**. One (shortened) description per uid |
| Inference | `test.py --mode text`, batch 32, `max_new_tokens=768` |

For text mode cadrille uses only **one** prompt per uid — its published text-to-CAD numbers are not
per-level. It also reports **best-of-N** (min CD / max IoU over N samples). See §3.

---

### 1.5 Text-to-CadQuery (IEEE T-ASE submission)

| | |
|---|---|
| Repo | `Text-to-CadQuery/Text-to-CadQuery` |
| Paper | [arXiv:2505.06507](https://arxiv.org/abs/2505.06507) |
| Architecture | Six **fully fine-tuned** open LLMs, no shared architecture |
| Weights | `ricemonster/qwen2.5-3B-SFT`, `ricemonster/Mistral-7B-lora`, `ricemonster/gemma-1B-SFT`, `ricemonster/gpt2-large-sft`, `ricemonster/gpt2-medium-sft`, `ricemonster/codegpt-small-sft` |
| Gated | No |
| Output | **CadQuery Python code** — full standalone scripts with `import cadquery as cq`, named variables, and an `exporters.export(...)` call |
| Output → geometry | `exec(code)` → STL written by the script itself |
| Training data | 170K CadQuery programs auto-annotated with Gemini 2.0 Flash on top of Text2CAD |
| Test set | `inference/test_filtered.jsonl`, **5,198** examples. Input text is the Text2CAD **`description`** column style (numbers spelled out as words: *"zero point one one four nine meters"*) |

**Bonus find:** `ricemonster/NeurIPS11092` (a *model* repo) contains an **ungated mirror of
`text2cad_v1.1.csv`** (1.3 GB) with columns
`uid, abstract, beginner, expert, description, keywords, intermediate, all_level_data, nli_data`.
This is our escape hatch from the gated `SadilKhan/Text2CAD` repo for the L0–L3 prompts. Only the
Text2CAD *checkpoint* still requires accepting the gate.

---

## 2. Output formats side by side

The single most important design fact: **the five systems emit five incompatible representations.**

| System | Native output | Intermediate | Failure modes |
|---|---|---|---|
| Text2CAD | int vector seq `(N,2)`, 8-bit quantised | `CADSequence` → OCC | invalid token seq, degenerate loop, empty solid |
| CADmium | minimal JSON | `CADSequence` → OCC | malformed JSON, unclosed loop, empty solid |
| CADFusion | SkexGen token string | `CADparser` → OBJ → OCC | unparseable seq, non-manifold, empty |
| cadrille | CadQuery code (var `r`) | `exec` → OCC compound | syntax error, runtime error, timeout, empty |
| Text-to-CadQuery | CadQuery script | `exec` → STL | syntax error, runtime error, timeout, empty |
| General LLMs | CadQuery code (prompted) | `exec` → STL | all of the above + refusal/prose |

**They converge on exactly one thing: a boundary-representation solid, hence a mesh.**
That is the only fair place to put the measurement plane. See `02_benchmark_design.md`.

---

## 3. Why the published numbers cannot be put in one table

I read the four evaluation scripts. They compute *three different Chamfer distances*:

| Source | Normalisation | N points | CD formula | Reported |
|---|---|---|---|---|
| Text2CAD `utils.py:1291` | `pts / (max(pts) - min(pts))` — global scalar over the **flattened** array, **no centering** | 8192 | `mean(d²)_a→b + mean(d²)_b→a` | ×1000, median + mean |
| cadrille `evaluate.py:31` | center bbox, scale max extent → 1, translate to (0.5,0.5,0.5) | 8192 | `mean(d²)_a→b + mean(d²)_b→a` | ×1000, **median** |
| CADFusion `chamfer_dist.py` | `pts / max(abs(pts))` — **no centering** | **2000** | `mean(d²)+mean(d²)` | mean + median |
| CADmium | delegates to Text2CAD `eval_seq.py` + CAD-MLLM metrics | 8192 | as Text2CAD | ×1000 |

Text2CAD's `normalize_pc` divides by the range of *all* coordinates flattened together and never
recentres, so a translated copy of the same shape gets a different CD. CADFusion's divides by max
absolute value, which is translation-sensitive too. cadrille's is the only one that is
translation-invariant. **These are not the same metric.** Numbers copied across papers are not
comparable, and neither are the invalidity ratios:

- Text2CAD IR = fraction of samples where `cd < 0`, i.e. the OCC solid failed to build.
- cadrille IR = fraction where no valid mesh file appeared (includes 3-second exec timeouts).
- cadrille additionally reports IR at "skip 0..4", trimming the worst k outliers from mean CD.

And on sampling:

- Text2CAD: 5 samples, `choose_best_index` over CD.
- CADFusion: 5 samples, temp 0.3.
- cadrille: `n_samples` copies, then `argmin(cd)` / `max(iou)` per uid.

**All three headline numbers are best-of-N oracle scores.** A single-sample general LLM compared
against them is being cheated. Our benchmark fixes N across every system and reports pass@1 and
best-of-5 as separate tables.

---

## 4. Data survey

### 4.1 CADPrompt (`Kamel773/CAD_Code_Generation`, ICLR 2025)

200 directories, each named with an **8-digit DeepCAD uid**, each containing:

```
00986814/
├── 00986814.json                                            # original DeepCAD/Onshape entity JSON
├── Ground_Truth.json                                        # face/edge/vertex counts, bbox, volume, area
├── Ground_Truth.stl / .obj                                  # reference mesh
├── Python_Code.py                                           # expert-written CadQuery
├── Natural_Language_Descriptions_Prompt.txt                 # no measurements
└── Natural_Language_Descriptions_Prompt_with_specific_measurements.txt
```

Two prompt variants per object (with / without measurements) — a useful second axis, roughly L1 vs L3.
Units are DeepCAD-normalised (0.25, 0.75, …) despite the `_mm` field names in `Ground_Truth.json`.

### 4.2 🚨 Contamination — the finding that decides the benchmark's credibility

CADPrompt's uids are DeepCAD uids, and I checked them against the DeepCAD test split
(from `maksimko123/deepcad_test_mesh`, 8,046 stl files):

```
CADPrompt ids in DeepCAD TEST split :  13 / 200   (6.5%)
CADPrompt ids NOT in test           : 187 / 200   (93.5%)
  ├─ confirmed in Text2CAD TRAIN    : 132
  ├─ confirmed in Text2CAD VAL      :   6
  └─ in DeepCAD train, outside cadrille's 90,737-uid subset : 49
```

**187 of the 200 CADPrompt objects were in the training set of Text2CAD, CADmium, cadrille and
(probably) CADFusion.** They were *not* in the training set of Qwen / Llama / Mistral / DeepSeek in
any comparable way.

Running "all 200 CADPrompt" as one headline number hands the fine-tuned CAD models a memorisation
advantage over the general LLMs and calls it a win. Any reviewer who checks this sinks the paper.
The design in `02_benchmark_design.md` splits CADPrompt into a **clean** slice (13) and a
**contaminated** slice (187) and reports them separately — the gap between the two slices is itself
one of the more interesting results the benchmark can produce.

### 4.3 Text2CAD test split

8,046 DeepCAD test uids (cadrille's mirror has 8,035 after dropping 11 unbuildable). Four prompts
each from the L0–L3 columns. This is genuinely held out for Text2CAD, CADmium, cadrille and
Text-to-CadQuery. **Caveat:** CADFusion trains on the SkexGen split, whose train/test boundary does
not coincide with DeepCAD's — some DeepCAD test uids may sit in SkexGen train. The SkexGen split is
only on Google Drive; until it is checked, CADFusion carries a "split unverified" flag.

---

## 5. Asset checklist

| Asset | Source | Size | Gated |
|---|---|---|---|
| Text2CAD checkpoint | `SadilKhan/Text2CAD` `text2cad_v1.0/Text2CAD_1.0.pth` | 92 MB | **YES** |
| Text2CAD L0–L3 prompts | `ricemonster/NeurIPS11092` `text2cad_v1.1.csv` | 1.3 GB | no |
| CADmium-7B adapter | `chandar-lab/CADmium-7B` | 646 MB | no |
| Qwen2.5-Coder-7B-Instruct base | `Qwen/Qwen2.5-Coder-7B-Instruct` | 15 GB | no |
| CADFusion v1.1 adapter | `microsoft/CADFusion` `v1_1/` | 4.3 GB | no |
| Meta-Llama-3-8B base | `meta-llama/Meta-Llama-3-8B` | 16 GB | **YES** (Meta license) |
| cadrille SFT / RL | `maksimko123/cadrille`, `-rl` | 4.4 GB each | no |
| Text-to-CadQuery models | `ricemonster/*` | 0.5–15 GB | no |
| DeepCAD test meshes | `maksimko123/deepcad_test_mesh` | 260 MB | no |
| CadQuery GT for 171K uids | `maksimko123/text2cad` | 83 MB | no |
| CADPrompt | `Kamel773/CAD_Code_Generation` (git) | ~1 GB | no |

Two gates to clear by hand: the Text2CAD dataset page and Meta's Llama-3 license.

---

## 6. A100 40 GB feasibility

| Model | Load | Fits |
|---|---|---|
| Text2CAD | 92 MB custom net | trivially |
| cadrille (Qwen2-VL-2B) | 4.4 GB bf16 | yes, batch 32 |
| CADmium-7B (Qwen2.5-Coder-7B + LoRA) | ~16 GB bf16 | yes |
| CADFusion (Llama-3-8B + LoRA) | ~17 GB bf16 | yes |
| Text-to-CadQuery Mistral-7B / Qwen-3B | ~15 GB / ~7 GB | yes |
| General LLMs @ 7–8B | ~16 GB bf16 | yes |
| General LLMs @ 32B | ~65 GB bf16 | needs 4-bit (~20 GB) |

Everything runs sequentially on one A100 40 GB. Run one model per Colab session, write predictions
to Drive, and do all metric computation on CPU afterwards — the geometry kernel, not the GPU, is the
bottleneck for evaluation.
