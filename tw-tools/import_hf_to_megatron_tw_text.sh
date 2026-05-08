#!/bin/bash
# HF → Megatron text-only checkpoint conversion for Qwen3.5 dense models on TensorWave MI325X.
#
# Produces a Megatron `torch_dist` checkpoint whose state-dict matches a plain
# `GPTModel` (no `language_model.` prefix, no vision keys). Use this output
# with Megatron-LM's `pretrain_gpt.py` for text-only continued pretraining.
#
# Vision tower weights from the HuggingFace checkpoint are silently ignored
# during this conversion. They are reattached at export time via
# `tw-tools/rewrap_text_to_vl_and_export.sh`, which expects the matching
# *VL* Megatron checkpoint to also be present (produced by the original
# `tw-tools/import_hf_to_megatron_tw.sh`).
#
# Usage:
#   sbatch tw-tools/import_hf_to_megatron_tw_text.sh <hf_model> [megatron_output_path]
#
# Examples:
#   sbatch tw-tools/import_hf_to_megatron_tw_text.sh Qwen/Qwen3.5-4B-Base
#     → ./megatron_ckpt/Qwen3.5-4B-Base-text
#
#   sbatch tw-tools/import_hf_to_megatron_tw_text.sh \
#       Qwen/Qwen3.5-4B-Base \
#       /shared_silo/scratch/rluukkon/oellm/Megatron-Bridge/megatron_ckpt/Qwen3.5-4B-text

#SBATCH --job-name=hf-import-text
#SBATCH --account=amd-tw-verification
#SBATCH --ntasks=1
#SBATCH --gres=gpu:mi325:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs-convert/hf-import-text-%j.out
#SBATCH --error=logs-convert/hf-import-text-%j.err

set -euo pipefail

HF_MODEL="${1:?Usage: sbatch $0 <hf_model> [megatron_output_path]}"
MEGATRON_MODEL_PATH="${2:-./megatron_ckpt/$(basename "$HF_MODEL")-text}"

BRIDGE_ROOT="./"
CONTAINER="/shared_silo/scratch/containers/build-rocm_primus_v25.11_transformers-5.5.4_linear_FA/rocm_primus_v25.11_transformers-5.5.4_linear_FA.sif"
BIND_PATH="${BIND_PATH:-/shared_silo/scratch}"

mkdir -p "${BRIDGE_ROOT}/logs-convert"

echo "========================================"
echo "HF → Megatron text-only import"
echo "  hf_model     : $HF_MODEL"
echo "  megatron_path: $MEGATRON_MODEL_PATH"
echo "  bridge_root  : $BRIDGE_ROOT"
echo "  job_id       : $SLURM_JOB_ID"
echo "  bridge mode  : text-only (BRIDGE_QWEN35_USE_VL=0)"
echo "========================================"

MEGATRON_PATH="/shared_silo/scratch/rluukkon/oellm/oellm-autoexp/submodules/Megatron-LM"

srun apptainer exec --rocm \
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
    python "${BRIDGE_ROOT}/examples/conversion/convert_checkpoints.py" import \
        --hf-model "$HF_MODEL" \
        --megatron-path "$MEGATRON_MODEL_PATH"

echo "Done. Text-only Megatron checkpoint stored at: $MEGATRON_MODEL_PATH"
