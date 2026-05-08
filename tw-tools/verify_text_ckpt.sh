#!/bin/bash
# Smoke-test launcher for the text-only Qwen3.5 dense Megatron checkpoint.
#
# Run this *immediately after* import_hf_to_megatron_tw_text.sh and before
# launching a multi-hour pretrain_gpt.py CPT job. The script verifies that:
#   1. AutoBridge dispatches Qwen35DenseTextBridge (and not the VL bridge),
#   2. every parameter of a freshly-built GPTModel has an HF mapping with a
#      present source key (no silent-warning random-init failures), and
#   3. a representative set of saved tensors numerically matches what the
#      bridge mappings would produce from the HF source.
#
# Usage:
#   sbatch tw-tools/verify_text_ckpt.sh <hf_model> <megatron_path>
#
# Example:
#   sbatch tw-tools/verify_text_ckpt.sh \
#       Qwen/Qwen3.5-4B-Base \
#       /shared_silo/scratch/rluukkon/oellm/Megatron-Bridge/megatron_ckpt/Qwen3.5-4B-text

#SBATCH --job-name=verify-text-ckpt
#SBATCH --account=amd-tw-verification
#SBATCH --ntasks=1
#SBATCH --gres=gpu:mi325:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs-convert/verify-text-ckpt-%j.out
#SBATCH --error=logs-convert/verify-text-ckpt-%j.err

set -euo pipefail

HF_MODEL="${1:?Usage: sbatch $0 <hf_model> <megatron_path>}"
MEGATRON_MODEL_PATH="${2:?Usage: sbatch $0 <hf_model> <megatron_path>}"

BRIDGE_ROOT="./"
CONTAINER="/shared_silo/scratch/containers/build-rocm_primus_v25.11_transformers-5.5.4_linear_FA/rocm_primus_v25.11_transformers-5.5.4_linear_FA.sif"
BIND_PATH="${BIND_PATH:-/shared_silo/scratch}"

mkdir -p "${BRIDGE_ROOT}/logs-convert"

echo "========================================"
echo "Verifying Megatron text-only checkpoint"
echo "  hf_model     : $HF_MODEL"
echo "  megatron_path: $MEGATRON_MODEL_PATH"
echo "  bridge mode  : text-only (BRIDGE_QWEN35_USE_VL=0)"
echo "========================================"

MEGATRON_PATH="/shared_silo/scratch/rluukkon/oellm/oellm-autoexp/submodules/Megatron-LM"

apptainer exec --rocm \
    -B "${BIND_PATH}:${BIND_PATH}:rw" \
    --env PYTHONPATH=$MEGATRON_PATH:${BRIDGE_ROOT}/src \
    --env HSA_FORCE_FINE_GRAIN_PCIE=1 \
    --env TRITON_CACHE_DIR=/tmp/triton-${SLURM_JOB_ID} \
    --env TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-${SLURM_JOB_ID} \
    --env MIOPEN_USER_DB_PATH=/tmp/miopen-${SLURM_JOB_ID} \
    --env MIOPEN_CACHE_DIR=/tmp/miopen-cache-${SLURM_JOB_ID} \
    --env RCCL_MSCCL_ENABLE=0 \
    --env BRIDGE_QWEN35_USE_VL=0 \
    "$CONTAINER" \
    python "${BRIDGE_ROOT}/tw-tools/verify_text_ckpt.py" \
        --hf-model "$HF_MODEL" \
        --megatron-path "$MEGATRON_MODEL_PATH"

echo "Verification done."
