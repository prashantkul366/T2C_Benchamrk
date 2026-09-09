# =============================================================================
#  T2C-Bench - SMOKE TEST, ALL SPECIALISED MODELS, BOTH SPLITS
#  Paste this whole thing into one Colab cell on the A100 and run it.
#
#  It generates N prompts per (model, split) for all 7 specialised systems, then
#  prints one JSON blob. Copy the blob between the markers and paste it back —
#  scoring (CD / IoU / IR) happens on the machine that has pythonocc, not here.
#
#  Every model runs in its own subprocess, so one model failing to load cannot
#  take the run down, and its VRAM is released before the next one starts.
#  Expect ~25-40 min total, dominated by weight downloads.
# =============================================================================

N_PROMPTS   = 3          # per model per split. 3 is enough to see the format.
HF_TOKEN    = ""         # needed ONLY for text2cad + cadfusion (see below)
BRANCH      = "claude/t2c-run-specialised"
SKIP_GATED  = False      # True => skip text2cad + cadfusion, no token needed

# Gated assets, if SKIP_GATED is False:
#   * SadilKhan/Text2CAD        - accept at https://huggingface.co/datasets/SadilKhan/Text2CAD
#   * meta-llama/Meta-Llama-3-8B - accept at https://huggingface.co/meta-llama/Meta-Llama-3-8B
# Everything else is open.
# -----------------------------------------------------------------------------

import os, subprocess, sys

WORK = "/content/drive/MyDrive/t2c_bench"
if not os.path.isdir(WORK):
    from google.colab import drive
    drive.mount("/content/drive")
os.environ["T2C_WORK"] = WORK
assert os.path.exists(f"{WORK}/data/split_a.jsonl"), "run notebook 00 first"
assert os.path.exists(f"{WORK}/data/split_b.jsonl"), "run notebook 00 first"


def sh(cmd, check=False):
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check)


# --- harness -----------------------------------------------------------------
REPO = "/content/t2cbench_repo"
if not os.path.isdir(REPO):
    sh(f"git clone -q https://github.com/prashantkul366/T2C_Benchamrk {REPO}")
sh(f"cd {REPO} && git fetch -q origin && git checkout -q {BRANCH} && git pull -q origin {BRANCH}")
os.chdir(REPO)

sh("pip install -q transformers accelerate peft bitsandbytes sentencepiece "
   "qwen-vl-utils pyyaml tqdm 2>&1 | tail -2")

import torch
print(f"\nGPU: {torch.cuda.get_device_name(0)} "
      f"({torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB)\n")

if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN)

# --- model source repos ------------------------------------------------------
if not os.path.isdir("/content/cadrille"):
    sh("git clone -q --depth 1 https://github.com/col14m/cadrille /content/cadrille")

extra, t2c_ckpt = "", None
if not SKIP_GATED:
    if not os.path.isdir("/content/Text2CAD"):
        sh("git clone -q --depth 1 https://github.com/SadilKhan/Text2CAD /content/Text2CAD")
    try:
        from huggingface_hub import hf_hub_download
        t2c_ckpt = hf_hub_download("SadilKhan/Text2CAD",
                                   "text2cad_v1.0/Text2CAD_1.0.pth", repo_type="dataset")
        print(f"Text2CAD checkpoint: {t2c_ckpt}")
        extra = f"--text2cad-repo /content/Text2CAD --text2cad-ckpt {t2c_ckpt}"
    except Exception as e:
        # A gated asset we cannot reach is a skipped model, not a failed run.
        print(f"\n!! Text2CAD checkpoint unavailable ({type(e).__name__}); "
              f"text2cad will be reported as SKIP.\n   Accept the licence and set HF_TOKEN "
              f"to include it.\n")

MODELS = ("text2cad,cadmium-7b,cadfusion-v1.1,cadrille,cadrille-rl,"
          "t2cq-qwen-3b,t2cq-mistral-7b")
if SKIP_GATED:
    MODELS = "cadmium-7b,cadrille,cadrille-rl,t2cq-qwen-3b,t2cq-mistral-7b"

# --- generate ----------------------------------------------------------------
sh(f"python scripts/smoke_test.py --phase generate --work {WORK} "
   f"-n {N_PROMPTS} --models {MODELS} --cadrille-repo /content/cadrille {extra}")

# --- export the blob to paste back ------------------------------------------
sh(f"python scripts/smoke_test.py --phase export --work {WORK} --models {MODELS}")

print("\n" + "=" * 78)
print("Copy everything between <<<T2C_SMOKE_EXPORT_BEGIN>>> and <<<...END>>> above.")
print(f"If the paste is too big, download {WORK}/smoke/export.json and attach it.")
print("=" * 78)
