#!/bin/bash
# Stage 3 — tokenization.
#   (a) the combined (privacy + copyright) TRAINING set with GRAIL's 4-factor
#       (domain x segment) tagging, consumed by the unlearning stage:
#         <GRAIL_TOKENIZED_DIR>/<VANILLA_MODEL>/grail/combined/combined_unlearn/
#             adaptive_ascent_plus_descent/tokenized_dataset.pt
#   (b) the per-domain forget/retain VALIDATION sets, consumed by evaluation:
#         <GRAIL_TOKENIZED_DIR>/<VANILLA_MODEL>/<domain>/<domain>_{unlearn,retention}/
#             normal/tokenized_dataset_val.pt
set -e
source "$(dirname "$0")/config.sh"
cd "$GRAIL_ROOT/llm_unlearn"

# (a) GRAIL 4-factor training set
OUT_DIR="$GRAIL_TOKENIZED_DIR/$VANILLA_MODEL/grail/combined/combined_unlearn"
echo "[GRAIL] tokenizing combined training dataset -> $OUT_DIR"
python utils/grail_tokenizer.py \
    --tokenizer_name_or_path "$BASE_MODEL_ID" \
    --max_length 256 \
    --output_dir "$OUT_DIR"

# (b) per-domain validation sets for evaluation (stage 5)
echo "[GRAIL] tokenizing per-domain validation sets -> $GRAIL_TOKENIZED_DIR"
python utils/save_tokenized_dataset.py \
    --tokenizer_name_or_path "$BASE_MODEL_ID" \
    --val
