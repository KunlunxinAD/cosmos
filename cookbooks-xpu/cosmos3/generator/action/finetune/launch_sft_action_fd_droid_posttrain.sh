#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# ============================================================================
# Cosmos3-Nano forward-dynamics (FD) DROID post-training on a single 8-GPU
# Kunlun P800 node. Trains FD on the Cosmos3-DROID success + failure splits.
#
# Self-contained recipe: stages the small inputs (VAE, base DCP, Qwen3-VL
# tokenizer), checks for the manually staged DROID dataset (both splits), then
# trains. Paths are fixed under this cookbook folder; override via env vars.
#
# PREREQUISITE: the P800 fused kernels only exist on the Kunlun branch of
# cosmos-framework (KunlunxinAD/cosmos-framework@kunlun_dev). Install that
# checkout into packages/cosmos3 first — see ../../../README.md.
#
# Run with the cosmos-framework venv active, from this folder:
#   bash launch_sft_action_fd_droid_posttrain.sh
#
# Env vars (override for your filesystem):
#   DATASET_PATH           Cosmos3-DROID parent dir (holds success/ AND failure/);
#                          the dir name MUST be droid_plus_lerobot_640x360_20260412
#   BASE_CHECKPOINT_PATH   base DCP checkpoint dir
#   WAN_VAE_PATH           Wan2.2 VAE .pth
#   TOKENIZER_PATH         local Qwen3-VL-8B-Instruct dir
#   COSMOS_FRAMEWORK_ROOT  override the framework checkout root
#   NPROC_PER_NODE         torchrun --nproc_per_node (default 8)
#   NNODES/NODE_RANK/MASTER_ADDR   multi-node torchrun topology
#   EXTRA_TAIL_OVERRIDES   space-separated extra Hydra overrides
# ============================================================================

set -euo pipefail

COOKBOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOML_FILE="$COOKBOOK_DIR/toml/sft_config/action_fd_droid_posttrain.toml"

: "${DATASET_PATH:=$COOKBOOK_DIR/data/Cosmos3-DROID/droid_plus_lerobot_640x360_20260412}"
: "${BASE_CHECKPOINT_PATH:=$COOKBOOK_DIR/checkpoints/Cosmos3-Nano}"
: "${WAN_VAE_PATH:=$COOKBOOK_DIR/checkpoints/wan22_vae/Wan2.2_VAE.pth}"
: "${TOKENIZER_PATH:=$COOKBOOK_DIR/checkpoints/Qwen3-VL-8B-Instruct}"

for _v in DATASET_PATH BASE_CHECKPOINT_PATH WAN_VAE_PATH TOKENIZER_PATH; do
    [[ "${!_v}" = /* ]] || printf -v "$_v" '%s/%s' "$COOKBOOK_DIR" "${!_v}"
done

# -- 1. Inputs. FD trains on both DROID splits; require them staged explicitly (~TB-scale).
for _split in success failure; do
    if [[ ! -f "$DATASET_PATH/$_split/meta/info.json" ]] \
       && ! compgen -G "$DATASET_PATH/$_split/*/meta/info.json" >/dev/null; then
        cat >&2 <<EOF
ERROR: missing Cosmos3-DROID '$_split' split under:
  $DATASET_PATH

FD post-training needs BOTH success/ and failure/. Note the parent directory MUST
be named droid_plus_lerobot_640x360_20260412 (the loader resolves the dataset
schema from the directory name). Stage them with:
  hf download nvidia/Cosmos3-DROID --repo-type dataset \\
      --local-dir data/Cosmos3-DROID/droid_plus_lerobot_640x360_20260412
or export DATASET_PATH=/path/to/droid_plus_lerobot_640x360_20260412.
EOF
        exit 1
    fi
done

# Wan2.2 VAE (skipped if present).
if [[ ! -f "$WAN_VAE_PATH" ]]; then
    uvx hf@latest download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --local-dir "$(dirname "$WAN_VAE_PATH")"
fi

# Qwen3-VL tokenizer (skipped if present); pinned locally so training runs offline.
if [[ ! -f "$TOKENIZER_PATH/config.json" ]]; then
    uvx hf@latest download Qwen/Qwen3-VL-8B-Instruct --local-dir "$TOKENIZER_PATH"
fi

# Base checkpoint -> DCP (skipped if present).
if [[ ! -d "$BASE_CHECKPOINT_PATH" ]]; then
    python -m cosmos_framework.scripts.convert_model_to_dcp -o "$BASE_CHECKPOINT_PATH" --checkpoint-path Cosmos3-Nano
fi

# -- 2. Resolve the framework checkout root (packaged model JSONs resolve from there).
if [[ -n "${COSMOS_FRAMEWORK_ROOT:-}" ]]; then
    FRAMEWORK_ROOT="$(cd "$COSMOS_FRAMEWORK_ROOT" && pwd)"
else
    FRAMEWORK_ROOT="$(python -c 'import pathlib, cosmos_framework; print(pathlib.Path(cosmos_framework.__file__).resolve().parents[1])')"
fi
[[ -d "$FRAMEWORK_ROOT/cosmos_framework" ]] || {
    echo "ERROR: cannot resolve cosmos-framework root at: $FRAMEWORK_ROOT" >&2
    echo "Activate the cosmos-framework venv, or export COSMOS_FRAMEWORK_ROOT=/path/to/cosmos-framework." >&2
    exit 1
}

export DATASET_PATH BASE_CHECKPOINT_PATH WAN_VAE_PATH

# -- 3. P800 runtime switches (all default-on; override to 0 to A/B a kernel). See ../../../README.md.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export COSMOS_TRAINING=1
export I4_ATTN_BACKENDS="${I4_ATTN_BACKENDS:-flash2}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"
export XDNN_USE_FAST_SWISH="${XDNN_USE_FAST_SWISH:-1}"
export COSMOS_FUSED_RMSNORM="${COSMOS_FUSED_RMSNORM:-1}"
export COSMOS_FUSED_ROPE="${COSMOS_FUSED_ROPE:-1}"
export COSMOS_FUSED_CAUSAL_CONV_PAD="${COSMOS_FUSED_CAUSAL_CONV_PAD:-1}"
export COSMOS_FUSED_VAE_NORM="${COSMOS_FUSED_VAE_NORM:-1}"

# -- 4. Overrides + launch.
TAIL_OVERRIDES=(
    "model.config.vlm_config.tokenizer.pretrained_model_name=$TOKENIZER_PATH"
    "model.config.compile.enabled=false"
    "trainer.straggler_detection.enabled=false"
    "trainer.callbacks.compile_tokenizer.enabled=false"
    "model.config.tokenizer.dtype=float16"
    "trainer.seed=42"
    "trainer.callbacks.device_monitor.every_n=0"
    "trainer.callbacks.ofu.every_n=0"
)
if [[ -n "${EXTRA_TAIL_OVERRIDES:-}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_OVERRIDES_ARRAY=(${EXTRA_TAIL_OVERRIDES})
    TAIL_OVERRIDES+=("${EXTRA_OVERRIDES_ARRAY[@]}")
fi

TORCHRUN_ARGS=(--nproc_per_node="${NPROC_PER_NODE:-8}" --master_port="${MASTER_PORT:-50012}")
[[ -n "${NNODES:-}" ]]      && TORCHRUN_ARGS+=(--nnodes="$NNODES")
[[ -n "${NODE_RANK:-}" ]]   && TORCHRUN_ARGS+=(--node_rank="$NODE_RANK")
[[ -n "${MASTER_ADDR:-}" ]] && TORCHRUN_ARGS+=(--master_addr="$MASTER_ADDR")

OUTPUT_ROOT="${OUTPUT_ROOT:-$COOKBOOK_DIR/outputs/train}"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${LOG_FILENAME:-action_fd_droid_posttrain_sft.log}"

echo ">>> $(date '+%H:%M:%S') framework:  $FRAMEWORK_ROOT"
echo ">>> $(date '+%H:%M:%S') dataset:    $DATASET_PATH"
echo ">>> $(date '+%H:%M:%S') checkpoint: $BASE_CHECKPOINT_PATH"
echo ">>> $(date '+%H:%M:%S') log:        $LOG_FILE"

cd "$FRAMEWORK_ROOT"
IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$OUTPUT_ROOT}" \
PYTHONPATH="${FRAMEWORK_ROOT}${PYTHONPATH:+:$PYTHONPATH}" \
    torchrun "${TORCHRUN_ARGS[@]}" -m cosmos_framework.scripts.train \
    --sft-toml="$TOML_FILE" \
    -- "${TAIL_OVERRIDES[@]}" \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo ">>> $(date '+%H:%M:%S') Done (exit $EXIT_CODE)"
exit $EXIT_CODE
