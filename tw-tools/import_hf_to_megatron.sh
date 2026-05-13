#!/bin/bash
# HF -> Megatron checkpoint conversion for TensorWave MI325X.
#
# Submit via sbatch (not bash directly):
#
#   # Dense text-only model (1 GPU, default resources):
#   mkdir -p logs-convert
#   sbatch tw-tools/import_hf_to_megatron.sh \
#       --hf-model Qwen/Qwen3.5-4B-Base \
#       --megatron-path megatron_ckpt/qwen3.5-4b-base
#
#   # MoE text-only, EP=8 — override GPU/task/mem directives on the sbatch line:
#   sbatch --ntasks=8 --ntasks-per-node=8 --gres=gpu:mi325:8 --mem=512G \
#       tw-tools/import_hf_to_megatron.sh \
#       --hf-model Qwen/Qwen3.5-35B-A3B-Base \
#       --expert-model-parallel-size 8 \
#       --megatron-path megatron_ckpt/Qwen3.5-35B-A3B-Base-ep8
#
#   # Keep vision tower (opt out of text-only default):
#   sbatch tw-tools/import_hf_to_megatron.sh \
#       --hf-model Qwen/Qwen3.5-4B-Base \
#       --vl
#
# --text-only is the default. Pass --vl to retain vision weights.
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
HF_MODEL=""
OUTPUT_CKPT_PATH=""
EP=1
TEXT_ONLY=1  # default: strip vision tower
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hf-model)         HF_MODEL="$2";      EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
        --output-ckpt-path)    OUTPUT_CKPT_PATH="$2";  EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
        --text-only)        TEXT_ONLY=1;         shift   ;;
        --vl)               TEXT_ONLY=0;         shift   ;;
        --expert-model-parallel-size)
                            EP="$2";             EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
        *)                  EXTRA_ARGS+=("$1");                           shift   ;;
    esac
done

# Inject --text-only into the args forwarded to the Python script only when active.
[[ "$TEXT_ONLY" -eq 1 ]] && EXTRA_ARGS+=("--text-only")

if [[ -z "$HF_MODEL" ]]; then
    echo "Error: --hf-model is required" >&2
    echo "Usage: sbatch $0 --hf-model <model_id_or_path> [--output-ckpt-path <path>] [--vl] [--expert-model-parallel-size N] [...]" >&2
    exit 1
fi

# Resolve BRIDGE_ROOT from the submit directory so relative paths work correctly.
BRIDGE_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"

# Default output path; relative paths are resolved against BRIDGE_ROOT.
# Append -vl suffix only when keeping vision weights (non-default path).
if [[ -z "$OUTPUT_CKPT_PATH" ]]; then
    SUFFIX=""
    [[ "$TEXT_ONLY" -eq 0 ]] && SUFFIX="-vl"
    OUTPUT_CKPT_PATH="${BRIDGE_ROOT}/megatron_ckpt/$(basename "$HF_MODEL")${SUFFIX}"
    EXTRA_ARGS+=("--output-ckpt-path" "$OUTPUT_CKPT_PATH")
elif [[ "$OUTPUT_CKPT_PATH" != /* ]]; then
    OUTPUT_CKPT_PATH="${BRIDGE_ROOT}/${OUTPUT_CKPT_PATH}"
    for i in "${!EXTRA_ARGS[@]}"; do
        if [[ "${EXTRA_ARGS[$i]}" == "--output-ckpt-path" ]]; then
            EXTRA_ARGS[$(( i + 1 ))]="$OUTPUT_CKPT_PATH"
            break
        fi
    done
fi

CONTAINER="/shared_silo/scratch/containers/build-rocm_primus_v25.11_transformers-5.5.4_linear_FA/rocm_primus_v25.11_transformers-5.5.4_linear_FA.sif"
BIND_PATH="${BIND_PATH:-/shared_silo/scratch}"
OUTPUT_CKPT_PATH_ABS="/shared_silo/scratch/rluukkon/oellm/oellm-autoexp/submodules/Megatron-LM"

echo "========================================"
echo "HF -> Megatron import"
echo "  hf_model     : $HF_MODEL"
echo "  megatron_path: $OUTPUT_CKPT_PATH"
echo "  EP           : $EP"
echo "  text_only    : $TEXT_ONLY"
echo "  job_id       : ${SLURM_JOB_ID:-local}"
echo "========================================"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT="${MASTER_PORT:-29500}"

srun apptainer exec --rocm \
    -B "${BIND_PATH}:${BIND_PATH}:rw" \
    --env PYTHONPATH="${OUTPUT_CKPT_PATH_ABS}:${BRIDGE_ROOT}/src" \
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

echo "Done. Checkpoint stored at: $OUTPUT_CKPT_PATH"
