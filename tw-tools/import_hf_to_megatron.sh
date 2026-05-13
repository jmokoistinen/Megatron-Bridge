#!/bin/bash
# HF -> Megatron checkpoint conversion for TensorWave MI325X.
# Converts to a text-only GPTModel (vision tower stripped).
#
#   # Dense model (1 GPU, default resources):
#   mkdir -p logs-convert
#   sbatch tw-tools/import_hf_to_megatron.sh \
#       --hf-model Qwen/Qwen3.5-4B-Base \
#       --megatron-path megatron_ckpt/qwen3.5-4b-base
#

# All flags are forwarded verbatim to
# examples/conversion/convert_checkpoints.py import.
#
# NOTE: create logs-convert/ before submitting, e.g. mkdir -p logs-convert
#
#SBATCH --job-name=hf-import
#SBATCH --account=amd-tw-verification
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:mi325:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs-convert/hf-import-%j.out
#SBATCH --error=logs-convert/hf-import-%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse arguments. All args are forwarded unchanged to the Python script.
# ---------------------------------------------------------------------------
HF_MODEL="$1"
OUTPUT_CKPT_PATH="$2"

# Resolve BRIDGE_ROOT from the submit directory so relative paths work correctly.
BRIDGE_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"

CONTAINER="/shared_silo/scratch/containers/build-rocm_primus_v25.11_transformers-5.5.4_linear_FA/rocm_primus_v25.11_transformers-5.5.4_linear_FA.sif"
BIND_PATH="${BIND_PATH:-/shared_silo/scratch}"
MEGATRON_PATH="${BRIDGE_ROOT}/3rdparty/Megatron-LM"
echo "========================================"
echo "HF -> Megatron import (text-only)"
echo "  hf_model     : $HF_MODEL"
echo "  output_ckpt_path: $OUTPUT_CKPT_PATH"
echo "  job_id       : ${SLURM_JOB_ID:-local}"
echo "========================================"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT="${MASTER_PORT:-29500}"

srun apptainer exec --rocm \
    -B "${BIND_PATH}:${BIND_PATH}:rw" \
    --env PYTHONPATH="${MEGATRON_PATH}:${BRIDGE_ROOT}/src" \
    --env HSA_FORCE_FINE_GRAIN_PCIE=1 \
    --env TRITON_CACHE_DIR="/tmp/triton-${SLURM_JOB_ID}" \
    --env TORCHINDUCTOR_CACHE_DIR="/tmp/inductor-${SLURM_JOB_ID}" \
    --env MIOPEN_USER_DB_PATH="/tmp/miopen-${SLURM_JOB_ID}" \
    --env MIOPEN_CACHE_DIR="/tmp/miopen-cache-${SLURM_JOB_ID}" \
    --env RCCL_MSCCL_ENABLE=0 \
    --env MASTER_ADDR="$MASTER_ADDR" \
    --env MASTER_PORT="$MASTER_PORT" \
    --env WORLD_SIZE="$SLURM_NTASKS" \
    --env RANK="$SLURM_PROCID" \
    "$CONTAINER" \
    python "${BRIDGE_ROOT}/examples/conversion/convert_checkpoints.py" import \
        --hf-model "$HF_MODEL" \
        --megatron-path "$OUTPUT_CKPT_PATH"

echo "Done. Checkpoint stored at: $MEGATRON_PATH"