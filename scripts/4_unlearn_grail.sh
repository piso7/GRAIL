#!/bin/bash
# Stage 4 — GRAIL unlearning with domain-conditioned gradient masking.
# Loads the UR/RR masks (stage 2) and the tokenized combined dataset (stage 3),
# then unlearns. Checkpoints are written to  llm_unlearn/grail_output_ep3/
#   <VANILLA_MODEL>/<LOCALIZED_TYPE>/UR_<ur>_RR_<rr>/   (path set inside the script).
set -e
source "$(dirname "$0")/config.sh"
cd "$GRAIL_ROOT/llm_unlearn"

echo "[GRAIL] unlearning: UR=$UR_PERCENT RR=$RR_PERCENT type=$LOCALIZED_TYPE"
python grail_run_adaptive_unlearn.py \
    --model_id "$BASE_MODEL_ID" \
    --model_name_or_path "$LORA_MODULE" \
    --localization_path "$GRAIL_LOCALIZATION_DIR" \
    --UR_percent "$UR_PERCENT" \
    --RR_percent "$RR_PERCENT" \
    --localized_type "$LOCALIZED_TYPE" \
    --unlearn_method grail \
    --domain grail \
    --do_unlearn \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --model_max_length 256 \
    --num_train_epochs 3 \
    --learning_rate 3e-4 \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --lr_scheduler_type cosine \
    --save_strategy steps \
    --save_steps 20 \
    --save_total_limit 15 \
    --logging_steps 1 \
    --output_dir "$GRAIL_UNLEARN_OUTPUT" \
    --overwrite_output_dir \
    --overwrite_cache
