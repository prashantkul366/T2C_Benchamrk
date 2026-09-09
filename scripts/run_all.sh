#!/usr/bin/env bash
# Full benchmark sweep. Generation needs a GPU; scoring and reporting do not.
#
# Run one model per session in Colab instead of this script if your GPU time is
# capped -- every runner is resumable, so partial progress is never lost.
set -euo pipefail

WORK="${WORK:-./results}"
DATA="${DATA:-./data}"
MODE="${MODE:-pass_at_1}"
SPLIT_A="$DATA/split_a.jsonl"
SPLIT_B="$DATA/split_b.jsonl"

mkdir -p "$WORK"/{raw,scored,meshes,tables,figures}

gen_hf () {  # name  weights  template  [extra flags...]
  local name="$1" weights="$2" template="$3"; shift 3
  for split in A B; do
    local sf="$SPLIT_A"; [ "$split" = B ] && sf="$SPLIT_B"
    python -m t2cbench.runners.run_hf \
      --model "$weights" --name "$name" --template "$template" "$@" \
      --split "$sf" --out "$WORK/raw/${name}_split${split}_${MODE}.jsonl" --mode "$MODE"
  done
}

echo "=== specialised systems ==="
gen_hf cadmium-7b      chandar-lab/CADmium-7B  native.cadmium \
       --base Qwen/Qwen2.5-Coder-7B-Instruct
gen_hf cadfusion-v1.1  microsoft/CADFusion     native.cadfusion \
       --base meta-llama/Meta-Llama-3-8B --no-chat-template
gen_hf t2cq-qwen-3b    ricemonster/qwen2.5-3B-SFT native.text2cadquery --no-chat-template

echo "=== general LLMs (base models of the three above come first) ==="
for m in "qwen25-coder-7b:Qwen/Qwen2.5-Coder-7B-Instruct" \
         "llama3-8b-instruct:meta-llama/Meta-Llama-3-8B-Instruct" \
         "qwen2-vl-2b:Qwen/Qwen2-VL-2B-Instruct" \
         "mistral-7b-instruct:mistralai/Mistral-7B-Instruct-v0.3" \
         "deepseek-coder-6.7b:deepseek-ai/deepseek-coder-6.7b-instruct"; do
  gen_hf "${m%%:*}" "${m##*:}" general_one_shot
done

echo "=== scoring (CPU) ==="
python scripts/score_all.py --work "$WORK" --data "$DATA"

echo "=== report ==="
python -m t2cbench.report.tables  --scored "$WORK/scored/*.jsonl" --out "$WORK/tables" \
    --ablation-pairs configs/ablation_pairs.json
python -m t2cbench.report.figures --scored "$WORK/scored/*.jsonl" --out "$WORK/figures" \
    --split "$SPLIT_A" --mesh-root "$WORK/meshes" --families configs/model_families.json

echo "done -> $WORK/tables/RESULTS.md"
