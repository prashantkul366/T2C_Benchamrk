# =============================================================================
#  T2C-Bench - SMOKE TEST, ALL SPECIALISED MODELS, BOTH SPLITS
#  Paste this whole thing into one Colab cell on the A100 and run it.
#
#  Generates N prompts per (model, split) for all 7 specialised systems, then
#  prints one JSON blob to paste back. Scoring (CD / IoU / IR) happens where
#  pythonocc is, not here.
#
#  VERSION PINS ARE LOAD-BEARING. Colab currently ships transformers 5.x and a
#  torchao that peft rejects; on that stack 6 of the 7 models fail to load:
#    * transformers 5.x moved Qwen2-VL's hidden_size under config.text_config,
#      so cadrille's `config.hidden_size` raises AttributeError. cadrille pins
#      transformers==4.50.3.
#    * peft's torchao dispatcher raises ImportError on Colab's torchao, killing
#      every LoRA model (CADmium, CADFusion, T2CQ-Mistral). We never use torchao
#      quantisation, so it is removed rather than upgraded.
#  The cell verifies the pins took effect before spending any GPU time.
# =============================================================================

N_PROMPTS   = 3          # per model per split
HF_TOKEN    = ""         # only for text2cad + cadfusion
BRANCH      = "claude/t2c-run-specialised"
SKIP_GATED  = False      # True => skip text2cad + cadfusion, no token needed

import os, subprocess, sys

WORK = "/content/drive/MyDrive/t2c_bench"
if not os.path.isdir(WORK):
    from google.colab import drive
    drive.mount("/content/drive")
os.environ["T2C_WORK"] = WORK
for f in ("split_a.jsonl", "split_b.jsonl"):
    assert os.path.exists(f"{WORK}/data/{f}"), f"run notebook 00 first (missing {f})"


def sh(cmd, check=False):
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check)


# --- harness -----------------------------------------------------------------
REPO = "/content/t2cbench_repo"
if not os.path.isdir(REPO):
    sh(f"git clone -q https://github.com/prashantkul366/T2C_Benchamrk {REPO}")
sh(f"cd {REPO} && git fetch -q origin && git checkout -q {BRANCH} && git pull -q origin {BRANCH}")
os.chdir(REPO)

# --- pinned stack ------------------------------------------------------------
print("\n=== installing the pinned stack (a few minutes) ===", flush=True)
sh("pip uninstall -q -y torchao 2>/dev/null; "
   "pip install -q 'transformers==4.50.3' 'peft==0.15.2' 'accelerate>=1.0,<2' "
   "'tokenizers>=0.21,<0.22' sentencepiece qwen-vl-utils==0.0.10 "
   # Text2CAD's model modules import all of these at module scope. Each one was
   # found by a separate failed run, so the list is empirical, not defensive.
   "plyfile prettytable trimesh scikit-learn joblib seaborn loguru rich "
   "pyyaml tqdm 2>&1 | tail -3")

print("\n=== verifying pins BEFORE spending GPU time ===")
ok = True
import importlib
for mod, want in (("transformers", "4.50.3"), ("peft", "0.15.2")):
    try:
        importlib.invalidate_caches()
        m = importlib.import_module(mod); importlib.reload(m)
        got = m.__version__
        good = got.startswith(want)
        ok &= good
        print(f"  {mod:14s} {got:12s} {'OK' if good else f'EXPECTED {want}'}")
    except Exception as e:
        ok = False
        print(f"  {mod:14s} IMPORT FAILED: {type(e).__name__}: {e}")
try:
    import torchao  # noqa: F401
    print("  torchao        STILL PRESENT -- peft LoRA models will fail")
    ok = False
except ImportError:
    print("  torchao        removed  OK")
if not ok:
    print("\n!! RESTART THE RUNTIME (Runtime > Restart session) and re-run this cell.")
    print("   Colab keeps the already-imported modules until a restart.")
    raise SystemExit("pins not in effect")

import torch
print(f"\nGPU: {torch.cuda.get_device_name(0)} "
      f"({torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB)\n")

if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN)

# --- model source repos ------------------------------------------------------
if not os.path.isdir("/content/cadrille"):
    sh("git clone -q --depth 1 https://github.com/col14m/cadrille /content/cadrille")

extra = ""
if not SKIP_GATED:
    if not os.path.isdir("/content/Text2CAD"):
        sh("git clone -q --depth 1 https://github.com/SadilKhan/Text2CAD /content/Text2CAD")
    try:
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download("SadilKhan/Text2CAD",
                               "text2cad_v1.0/Text2CAD_1.0.pth", repo_type="dataset")
        extra = f"--text2cad-repo /content/Text2CAD --text2cad-ckpt {ckpt}"
        print(f"Text2CAD checkpoint: {ckpt}")
    except Exception as e:
        # Text2CAD's model code imports pythonocc at module scope even though the
        # decode path never calls it; the runner installs a poison-pill stub so
        # generation works on a plain GPU runtime. A gated checkpoint we cannot
        # reach is a skipped model, not a failed run.
        print(f"\n!! Text2CAD checkpoint unavailable ({type(e).__name__}); it will be SKIPped."
              f"\n   Accept the licence and set HF_TOKEN to include it.\n")

MODELS = ("text2cad,cadmium-7b,cadfusion-v1.1,cadrille,cadrille-rl,"
          "t2cq-qwen-3b,t2cq-mistral-7b")
if SKIP_GATED:
    MODELS = "cadmium-7b,cadrille,cadrille-rl,t2cq-qwen-3b,t2cq-mistral-7b"

# --- generate, then export ---------------------------------------------------
sh(f"python scripts/smoke_test.py --phase generate --work {WORK} "
   f"-n {N_PROMPTS} --models {MODELS} --cadrille-repo /content/cadrille {extra}")
sh(f"python scripts/smoke_test.py --phase export --work {WORK} --models {MODELS}")

print("\n" + "=" * 78)
print("Copy everything between <<<T2C_SMOKE_EXPORT_BEGIN>>> and <<<...END>>> above.")
print(f"If the paste is too big, download {WORK}/smoke/export.json and attach it.")
print("=" * 78)
