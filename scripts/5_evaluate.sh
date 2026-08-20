#!/bin/bash
# Stage 5 — evaluate an unlearned checkpoint on forget vs retain sets.
# Reports forget-success (100 - forget_acc), retention-success (retain_acc),
# their harmonic mean, and ROUGE-L (metrics are defined inside grail_run_eval_lora.py).
#
# Requires the per-domain *validation* tokenized datasets at
#   $GRAIL_TOKENIZED_DIR/<VANILLA_MODEL>/<domain>/<domain>_{retention,unlearn}/normal/tokenized_dataset_val.pt
# (same format as the KnowUnDo evaluation splits; see README "Data preparation").
set -e
source "$(dirname "$0")/config.sh"
cd "$GRAIL_ROOT/llm_evaluation"

# Point this at a checkpoint produced by stage 4.
CKPT="${CKPT:?set CKPT to an unlearned checkpoint dir, e.g. ../llm_unlearn/grail_output_ep3/$VANILLA_MODEL/$LOCALIZED_TYPE/UR_10_RR_20/checkpoint-100}"

for DOMAIN in copyright privacy; do
    echo "[GRAIL] evaluating $CKPT on $DOMAIN"
    python grail_run_eval_lora.py \
        --model_name_or_path "$CKPT" \
        --tokenizer_name "$BASE_MODEL_ID" \
        --config_name "$BASE_MODEL_ID" \
        --per_device_eval_batch_size 1 \
        --do_eval \
        --domain "$DOMAIN" \
        --output_dir "$GRAIL_EVAL_OUTPUT/$DOMAIN" \
        --overwrite_output_dir \
        --overwrite_cache
done
