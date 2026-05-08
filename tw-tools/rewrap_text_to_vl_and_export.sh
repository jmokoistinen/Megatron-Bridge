#!/bin/bash
# Re-wrap a Megatron-LM text-only Qwen3.5 checkpoint into a Bridge VL
# checkpoint and export it to HuggingFace, on TensorWave MI325X.
#
# Usage:
#   sbatch tw-tools/rewrap_text_to_vl_and_export.sh \
#       <trained_text_ckpt> <original_vl_ckpt> <hf_model> <out_megatron_vl> <out_hf>
#
# Example:
#   sbatch tw-tools/rewrap_text_to_vl_and_export.sh \
#       /shared_silo/scratch/rluukkon/oellm/oellm-autoexp/output/qwen3_5_4B_tw_test/iter_0001000 \
#       /shared_silo/scratch/rluukkon/oellm/Megatron-Bridge/megatron_ckpt/Qwen3.5-4B \
#       Qwen/Qwen3.5-4B-Base \
#       /shared_silo/scratch/rluukkon/oellm/Megatron-Bridge/megatron_ckpt/Qwen3.5-4B-trained-vl \
#       /shared_silo/scratch/rluukkon/oellm/Megatron-Bridge/hf_exports/Qwen3.5-4B-trained

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

TRAINED_TEXT_CKPT="${1:?Usage: sbatch $0 <trained_text_ckpt> <original_vl_ckpt> <hf_model> <out_megatron_vl> <out_hf>}"
ORIGINAL_VL_CKPT="${2:?missing original_vl_ckpt}"
HF_MODEL="${3:?missing hf_model}"
OUT_MEGATRON_VL="${4:?missing out_megatron_vl}"
OUT_HF="${5:?missing out_hf}"

BRIDGE_ROOT="./"
CONTAINER="/shared_silo/scratch/containers/build-rocm_primus_v25.11_transformers-5.5.4_linear_FA/rocm_primus_v25.11_transformers-5.5.4_linear_FA.sif"
BIND_PATH="${BIND_PATH:-/shared_silo/scratch}"

mkdir -p "${BRIDGE_ROOT}/logs-convert"
mkdir -p "$(dirname "$OUT_MEGATRON_VL")"
mkdir -p "$(dirname "$OUT_HF")"

echo "========================================"
echo "Qwen3.5 re-wrap (text → VL) and HF export"
echo "  trained text ckpt : $TRAINED_TEXT_CKPT"
echo "  original VL ckpt  : $ORIGINAL_VL_CKPT"
echo "  HF reference      : $HF_MODEL"
echo "  out (Megatron VL) : $OUT_MEGATRON_VL"
echo "  out (HF VL)       : $OUT_HF"
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
        --original-vl-ckpt  "$ORIGINAL_VL_CKPT" \
        --hf-model          "$HF_MODEL" \
        --out-megatron-vl   "$OUT_MEGATRON_VL" \
        --out-hf            "$OUT_HF"

echo "Done."
echo "  Megatron VL ckpt : $OUT_MEGATRON_VL"
echo "  HuggingFace VL   : $OUT_HF"
