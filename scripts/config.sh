#!/bin/bash
# Shared configuration for the GRAIL pipeline.
# Every path here is relative to the repo root by default; override any of them
# by exporting the variable before running a stage script.

# Repo root (directory that contains this scripts/ folder)
export GRAIL_ROOT="${GRAIL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# --- Models -----------------------------------------------------------------
# Base (vanilla) model and its short name used in artifact paths.
export BASE_MODEL_ID="${BASE_MODEL_ID:-$GRAIL_ROOT/models/Llama-2-7b-chat-hf}"
export VANILLA_MODEL="${VANILLA_MODEL:-Llama-2-7b-chat-hf}"

# Fine-tuned LoRA module (produced in stage 0, see README) and its folder name.
export LORA_MODULE="${LORA_MODULE:-$GRAIL_ROOT/pretrain/paper_models/final_combined_ft_LORA_20_epochs_inst_lr0.0003_llama2-7b_full_both}"
export BASE_MODEL_NAME="${BASE_MODEL_NAME:-final_combined_ft_LORA_20_epochs_inst_lr0.0003_llama2-7b_full_both}"

# --- Artifact directories (created by the pipeline) -------------------------
export GRAIL_LOCALIZATION_DIR="${GRAIL_LOCALIZATION_DIR:-$GRAIL_ROOT/llm_localization/outputs}"
export GRAIL_TOKENIZED_DIR="${GRAIL_TOKENIZED_DIR:-$GRAIL_ROOT/llm_unlearn/tokenized_dataset}"
export GRAIL_UNLEARN_OUTPUT="${GRAIL_UNLEARN_OUTPUT:-$GRAIL_ROOT/llm_unlearn/grail_output}"
export GRAIL_EVAL_OUTPUT="${GRAIL_EVAL_OUTPUT:-$GRAIL_ROOT/llm_evaluation/grail_eval_output}"

# --- Localization thresholds (MemFlex-style gradient accumulation) -----------
# LLaMA-2-chat:  copyright mu=0.92 sigma=6e-4 ; privacy mu=0.96 sigma=4e-4
export COPYRIGHT_SIM=0.92
export COPYRIGHT_GRAD=6e-4
export PRIVACY_SIM=0.96
export PRIVACY_GRAD=4e-4

# GRAIL parameter-wise localization ratios (OP-UR / OP-RR, in [0,1])
export UR_PERCENT="${UR_PERCENT:-0.1}"
export RR_PERCENT="${RR_PERCENT:-0.2}"
export LOCALIZED_TYPE="${LOCALIZED_TYPE:-zscore}"   # zscore or fim

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
