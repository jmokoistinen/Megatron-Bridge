#!/bin/bash
# Compare a roundtrip-exported HuggingFace model against the original.
# CPU-only — no GPU required.
#
# Usage:
#   sbatch tw-tools/compare_roundtrip.sh <orig_hf_dir> <roundtrip_hf_dir>
#
# Example:
#   sbatch tw-tools/compare_roundtrip.sh \
#       /shared_silo/scratch/rluukkon/hf_home/hub/models--Qwen--Qwen3.5-4B-Base/snapshots/1001bb4d826a52d1f399e183466143f4da7b741b \
#       /shared_silo/scratch/rluukkon/oellm/Megatron-Bridge/roundtrip_conversion

#SBATCH --job-name=roundtrip-cmp
#SBATCH --account=amd-tw-verification
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --open-mode=append
#SBATCH --output=logs-convert/roundtrip-cmp-%j.out
#SBATCH --error=logs-convert/roundtrip-cmp-%j.err

set -euo pipefail

ORIG="${1:?Usage: sbatch $0 <orig_hf_dir> <roundtrip_hf_dir>}"
RT="${2:?missing roundtrip_hf_dir}"

BRIDGE_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONTAINER="/shared_silo/scratch/containers/build-rocm_primus_v25.11_transformers-5.5.4_linear_FA/rocm_primus_v25.11_transformers-5.5.4_linear_FA.sif"
BIND_PATH="${BIND_PATH:-/shared_silo/scratch}"

echo "========================================"
echo "Roundtrip comparison"
echo "  orig      : $ORIG"
echo "  roundtrip : $RT"
echo "  job_id    : ${SLURM_JOB_ID:-local}"
echo "========================================"

srun apptainer exec \
    -B "${BIND_PATH}:${BIND_PATH}:rw" \
    --env PYTHONPATH="${BRIDGE_ROOT}/src" \
    --env TRITON_CACHE_DIR="/tmp/triton-${SLURM_JOB_ID}" \
    "$CONTAINER" \
    python "${BRIDGE_ROOT}/tw-tools/compare_roundtrip.py" \
        --orig "$ORIG" \
        --rt   "$RT"
