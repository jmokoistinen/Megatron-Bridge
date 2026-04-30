#!/bin/bash
# HF → Megatron checkpoint conversion for TensorWave MI325X

# Usage:
#   sbatch import_hf_to_megatron_tw.sh <hf_model> [megatron_output_path]

# Examples and where output goes:
#   sbatch import_hf_to_megatron_tw.sh Qwen/Qwen3.5-0.6B
#     → /shared_silo/scratch/mika/models/megatron_ckpt/Qwen3.5-0.6B

#   sbatch import_hf_to_megatron_tw.sh Qwen/Qwen3.5-35B-A3B
#     → /shared_silo/scratch/mika/models/megatron_ckpt/Qwen3.5-35B-A3B

#   sbatch import_hf_to_megatron_tw.sh /shared_silo/scratch/mika/models/hf_models/qwen3.5-0.8b
#     → /shared_silo/scratch/mika/models/megatron_ckpt/qwen3.5-0.8b-test

#   sbatch import_hf_to_megatron_tw.sh Qwen/Qwen3.5-0.8B /custom/output/path

#run tested at tw using:
#sbatch import_hf_to_megatron_tw.sh Qwen/Qwen3.5-0.8B  /shared_silo/scratch/mika/models/megatron_ckpt/Qwen3.5-0.8B-test

#SBATCH --job-name=hf-import
#SBATCH --account=amd-tw-verification
#SBATCH --ntasks=1
#SBATCH --gres=gpu:mi325:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs-convert/hf-import-%j.out
#SBATCH --error=logs-convert/hf-import-%j.err

set -euo pipefail

source ~/environment.sh 
HF_MODEL="${1:?Usage: sbatch $0 <hf_model> [megatron_output_path]}"

# Outputs next to wherever you run sbatch from. Always overridable with the second argument. 
MEGATRON_MODEL_PATH="${2:-./megatron_ckpt/$(basename "$HF_MODEL")}"

#set your own path here
#MEGATRON_PATH="${2:-/shared_silo/scratch/mika/models/megatron_ckpt/$(basename "$HF_MODEL")}"

#set your own path here
BRIDGE_ROOT=/shared_silo/scratch/mika/experiments/Megatron-Bridge 

CONTAINER="/shared_silo/scratch/containers/build-rocm_primus_v25.11_transformers-5.5.4_linear_FA/rocm_primus_v25.11_transformers-5.5.4_linear_FA.sif"
BIND_PATH="${BIND_PATH:-/shared_silo/scratch}"

mkdir -p "${BRIDGE_ROOT}/logs-convert"

echo "========================================"
echo "HF → Megatron import"
echo "  hf_model     : $HF_MODEL"
echo "  megatron_model_store_path: $MEGATRON_MODEL_PATH"
echo "  bridge_root  : $BRIDGE_ROOT"
echo "  job_id       : $SLURM_JOB_ID"
echo "========================================"

apptainer exec --rocm \
    -B "${BIND_PATH}:${BIND_PATH}:rw" \
    --env PYTHONPATH="${BRIDGE_ROOT}/python-packages:${BRIDGE_ROOT}/3rdparty/Megatron-LM:${BRIDGE_ROOT}/src" \
    --env HSA_FORCE_FINE_GRAIN_PCIE=1 \
    --env TRITON_CACHE_DIR=/tmp/triton-${SLURM_JOB_ID} \
    --env TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-${SLURM_JOB_ID} \
    --env MIOPEN_USER_DB_PATH=/tmp/miopen-${SLURM_JOB_ID} \
    --env MIOPEN_CACHE_DIR=/tmp/miopen-cache-${SLURM_JOB_ID} \
    --env RCCL_MSCCL_ENABLE=0 \
    "$CONTAINER" \
    python "${BRIDGE_ROOT}/examples/conversion/convert_checkpoints.py" import \
        --hf-model "$HF_MODEL" \
        --megatron-path "$MEGATRON_MODEL_PATH"

echo "Done. Checkpoint stored at: $MEGATRON_MODEL_PATH"