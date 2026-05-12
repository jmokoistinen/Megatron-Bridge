#!/bin/bash
# HF -> Megatron checkpoint conversion for TensorWave MI325X.
#
# Handles dense, text-only dense, and MoE models with a single script.
# SLURM resources (GPU count, memory, tasks) are computed automatically
# from --expert-model-parallel-size and submitted via self-resubmission.
#
# Usage (run directly, not via sbatch):
#
#   # Dense model (default, keeps VL weights if the bridge supports them)
#   ./tw-tools/import_hf_to_megatron.sh --hf-model Qwen/Qwen3.5-4B-Base
#     -> ./megatron_ckpt/Qwen3.5-4B-Base
#
#   # Dense text-only (strips vision tower)
#   ./tw-tools/import_hf_to_megatron.sh \
#       --hf-model Qwen/Qwen3.5-4B-Base \
#       --text-only
#     -> ./megatron_ckpt/Qwen3.5-4B-Base-text
#
#   # MoE text-only, EP=8 (8 GPUs, avoids EP resharding deadlock at training)
#   ./tw-tools/import_hf_to_megatron.sh \
#       --hf-model Qwen/Qwen3.5-35B-A3B-Base \
#       --text-only \
#       --expert-model-parallel-size 8
#     -> ./megatron_ckpt/Qwen3.5-35B-A3B-Base
#
#   # Override output path
#   ./tw-tools/import_hf_to_megatron.sh \
#       --hf-model Qwen/Qwen3.5-35B-A3B-Base \
#       --megatron-path /shared_silo/scratch/rluukkon/oellm/Megatron-Bridge/megatron_ckpt/Qwen3.5-35B-A3B-Base-ep8 \
#       --text-only \
#       --expert-model-parallel-size 8
#
# All flags after the model name are forwarded verbatim to
# examples/conversion/convert_checkpoints.py import.

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse arguments needed for resource computation.  All args are also passed
# through to the Python script unchanged.
# ---------------------------------------------------------------------------
HF_MODEL=""
MEGATRON_PATH=""
EP=1
TEXT_ONLY=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hf-model)         HF_MODEL="$2";     EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
        --megatron-path)    MEGATRON_PATH="$2"; EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
        --text-only)        TEXT_ONLY=1;        EXTRA_ARGS+=("$1");      shift   ;;
        --expert-model-parallel-size)
                            EP="$2";            EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
        *)                  EXTRA_ARGS+=("$1");                          shift   ;;
    esac
done

if [[ -z "$HF_MODEL" ]]; then
    echo "Error: --hf-model is required" >&2
    echo "Usage: $0 --hf-model <model_id_or_path> [--megatron-path <path>] [--text-only] [--expert-model-parallel-size N] [...]" >&2
    exit 1
fi

# Default output path based on EP and text-only flag.
if [[ -z "$MEGATRON_PATH" ]]; then
    SUFFIX=""
    [[ "$TEXT_ONLY" -eq 1 ]] && SUFFIX="-text"
    MEGATRON_PATH="./megatron_ckpt/$(basename "$HF_MODEL")${SUFFIX}"
    EXTRA_ARGS+=("--megatron-path" "$MEGATRON_PATH")
fi

BRIDGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER="/shared_silo/scratch/containers/build-rocm_primus_v25.11_transformers-5.5.4_linear_FA/rocm_primus_v25.11_transformers-5.5.4_linear_FA.sif"
BIND_PATH="${BIND_PATH:-/shared_silo/scratch}"
MEGATRON_PATH_ABS="/shared_silo/scratch/rluukkon/oellm/oellm-autoexp/submodules/Megatron-LM"

# ---------------------------------------------------------------------------
# Self-submit: when not already inside a SLURM job, compute required resources
# from EP size and submit this script back to sbatch.
# ---------------------------------------------------------------------------
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    MEM_GB=$(( EP * 64 ))
    LOG_DIR="${BRIDGE_ROOT}/logs-convert"
    mkdir -p "$LOG_DIR"

    echo "Submitting SLURM job: EP=${EP}, GPUs=${EP}, MEM=${MEM_GB}G"
    echo "  hf_model     : $HF_MODEL"
    echo "  megatron_path: $MEGATRON_PATH"

    sbatch \
        --job-name="hf-import" \
        --account="amd-tw-verification" \
        --ntasks="$EP" \
        --ntasks-per-node="$EP" \
        --gres="gpu:mi325:${EP}" \
        --cpus-per-task=8 \
        --mem="${MEM_GB}G" \
        --time="3:00:00" \
        --open-mode=append \
        --output="${LOG_DIR}/hf-import-%j.out" \
        --error="${LOG_DIR}/hf-import-%j.err" \
        "${BASH_SOURCE[0]}" "${EXTRA_ARGS[@]}"
    exit $?
fi

# ---------------------------------------------------------------------------
# Inside the SLURM job: run the conversion.
# ---------------------------------------------------------------------------
echo "========================================"
echo "HF -> Megatron import"
echo "  hf_model     : $HF_MODEL"
echo "  megatron_path: $MEGATRON_PATH"
echo "  EP           : $EP"
echo "  text_only    : $TEXT_ONLY"
echo "  job_id       : $SLURM_JOB_ID"
echo "========================================"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT="${MASTER_PORT:-29500}"

srun apptainer exec --rocm \
    -B "${BIND_PATH}:${BIND_PATH}:rw" \
    --env PYTHONPATH="${MEGATRON_PATH_ABS}:${BRIDGE_ROOT}/src" \
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
        "${EXTRA_ARGS[@]}"

echo "Done. Checkpoint stored at: $MEGATRON_PATH"
