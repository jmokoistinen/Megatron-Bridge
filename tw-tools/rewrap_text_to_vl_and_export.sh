#!/bin/bash
# Re-wrap a Megatron-LM text-only Qwen3.5 checkpoint into a Bridge VL
# checkpoint and export it to HuggingFace, on TensorWave MI325X.
#
# Vision-tower weights are sourced directly from the original HF model
# (--hf-model), so no separate original VL Megatron checkpoint is needed.
#
# Usage:
#   sbatch tw-tools/rewrap_text_to_vl_and_export.sh \
#       <trained_text_ckpt> <hf_model> <out>
#
# Example:
#   sbatch tw-tools/rewrap_text_to_vl_and_export.sh \
#       /shared_silo/scratch/rluukkon/oellm/Megatron-Bridge/megatron_ckpt/Qwen3.5-4B-Base_fix3 \
#       Qwen/Qwen3.5-4B-Base \
#       /shared_silo/scratch/rluukkon/oellm/Megatron-Bridge/roundtrip_conversion

#SBATCH --job-name=qwen35-rewrap-export
#SBATCH --account=amd-tw-verification
#SBATCH --ntasks=1
#SBATCH --gres=gpu:mi325:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=4:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs-convert/rewrap-export-%j.out
#SBATCH --error=logs-convert/rewrap-export-%j.err

set -euo pipefail

TRAINED_TEXT_CKPT="${1:?Usage: sbatch $0 <trained_text_ckpt> <hf_model> <out>}"
HF_MODEL="${2:?missing hf_model}"
OUT="${3:?missing out}"

BRIDGE_ROOT="./"
CONTAINER="/shared_silo/scratch/containers/build-rocm_primus_v25.11_transformers-5.5.4_linear_FA/rocm_primus_v25.11_transformers-5.5.4_linear_FA.sif"
BIND_PATH="${BIND_PATH:-/shared_silo/scratch}"

mkdir -p "${BRIDGE_ROOT}/logs-convert"
mkdir -p "$(dirname "$OUT")"

echo "========================================"
echo "Qwen3.5 re-wrap (text → VL) and HF export"
echo "  trained text ckpt : $TRAINED_TEXT_CKPT"
echo "  HF reference      : $HF_MODEL"
echo "  out (HF VL)       : $OUT"
echo "  out (Megatron VL) : ${OUT}_megatron_vl  [intermediate, can delete after]"
echo "  bridge_root       : $BRIDGE_ROOT"
echo "  job_id            : $SLURM_JOB_ID"
echo "  bridge mode       : VL (BRIDGE_QWEN35_USE_VL=1)"
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
    --env BRIDGE_QWEN35_USE_VL=1 \
    "$CONTAINER" \
    python "${BRIDGE_ROOT}/tw-tools/rewrap_text_to_vl_and_export.py" \
        --trained-text-ckpt "$TRAINED_TEXT_CKPT" \
        --hf-model          "$HF_MODEL" \
        --out               "$OUT"

echo "Done."
echo "  HuggingFace VL   : $OUT"
echo "  Megatron VL (intermediate): ${OUT}_megatron_vl"
