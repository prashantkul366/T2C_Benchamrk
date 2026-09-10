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

# The roster is configs/models.yaml, and nothing else. This script used to keep
# its own copy of it, which drifted: by the time anyone checked it was missing
# five of the thirteen models (text2cad, both cadrille checkpoints,
# t2cq-mistral-7b, qwen25-coder-32b), had lost CADFusion's `--subfolder v1_1`,
# and named a Mistral base the checkpoint's own adapter_config.json contradicts.
# Each of those costs A100 hours and yields a row that looks like a model result.
python scripts/plan_full_run.py --work "$WORK" --data "$DATA" --run \
  ${CADRILLE_REPO:+--cadrille-repo "$CADRILLE_REPO"} \
  ${TEXT2CAD_REPO:+--text2cad-repo "$TEXT2CAD_REPO"} \
  ${TEXT2CAD_CKPT:+--text2cad-ckpt "$TEXT2CAD_CKPT"}

echo "=== scoring (CPU) ==="
python scripts/score_all.py --work "$WORK" --data "$DATA"

echo "=== report ==="
python -m t2cbench.report.tables  --scored "$WORK/scored/*.jsonl" --out "$WORK/tables" \
    --ablation-pairs configs/ablation_pairs.json
python -m t2cbench.report.figures --scored "$WORK/scored/*.jsonl" --out "$WORK/figures" \
    --split "$SPLIT_A" --mesh-root "$WORK/meshes" --families configs/model_families.json

echo "done -> $WORK/tables/RESULTS.md"
