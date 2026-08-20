#!/bin/bash
# Stage 2 — GRAIL parameter-wise localization.
# Reads the grad_info_*.pt from stage 1 and writes element-wise masks
#   UR/UR_top<UR>.pt  and  RR/RR_top<RR>.pt
# under the same <VANILLA_MODEL>/<LOCALIZED_TYPE>/<BASE_MODEL_NAME>/ directory.
set -e
source "$(dirname "$0")/config.sh"
cd "$GRAIL_ROOT/llm_localization"

echo "[GRAIL] parameter-wise localization: UR=$UR_PERCENT RR=$RR_PERCENT type=$LOCALIZED_TYPE"
python grail_parameter_wise_localization.py \
    --UR "$UR_PERCENT" \
    --RR "$RR_PERCENT" \
    --localized_type "$LOCALIZED_TYPE" \
    --vanilla_model "$VANILLA_MODEL" \
    --base_model "$BASE_MODEL_NAME" \
    --output_dir "$GRAIL_LOCALIZATION_DIR"
