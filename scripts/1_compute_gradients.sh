#!/bin/bash
# Stage 1 — per-parameter gradient accumulation (MemFlex-style) for each domain.
# Produces grad_info_{domain}_{split}_{num_copies}.pt under
#   llm_localization/outputs/<VANILLA_MODEL>/<LOCALIZED_TYPE>/<BASE_MODEL_NAME>/
set -e
source "$(dirname "$0")/config.sh"
cd "$GRAIL_ROOT/llm_localization"

for cfg in "copyright $COPYRIGHT_SIM $COPYRIGHT_GRAD" "privacy $PRIVACY_SIM $PRIVACY_GRAD"; do
    set -- $cfg
    DOMAIN=$1; SIM=$2; GRAD=$3
    echo "[GRAIL] gradient accumulation: $DOMAIN (sim=$SIM grad=$GRAD)"
    python localization.py \
        --model_name_or_path "$LORA_MODULE" \
        --model_id "$BASE_MODEL_ID" \
        --data_type "$DOMAIN" \
        --sim_thresh "$SIM" \
        --grad_thresh "$GRAD" \
        --splits "unlearn,retention" \
        --save_tag "$LOCALIZED_TYPE"
done
