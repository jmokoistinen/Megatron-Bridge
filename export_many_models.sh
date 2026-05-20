#!/bin/bash
# export_many_models.sh — batch launcher for rewrap_text_to_vl_and_export.sh
#
# Discovers iter_XXXXXXX checkpoints in one or more Megatron-LM training run
# directories and submits one sbatch job per checkpoint via the single-checkpoint
# script tw-tools/rewrap_text_to_vl_and_export.sh.
#
# Usage:
#   bash export_many_models.sh [OPTIONS] --ckpt-root ROOT [ROOT ...] --hf-model HF --out-base DIR
#
# Required:
#   --ckpt-root PATH [PATH ...]   Training run root(s) containing iter_* subdirs.
#                                 Accepts multiple paths; keep them before any
#                                 subsequent flag so the parser knows where the
#                                 list ends. Alternatively repeat --ckpt-root
#                                 for each path.
#   --hf-model  PATH              HuggingFace model id or local snapshot path
#                                 (forwarded as-is to rewrap_text_to_vl_and_export.sh).
#   --out-base  DIR               Parent directory under which all per-checkpoint
#                                 HF export subdirs are created.
#
# Optional:
#   --latest-only                 Export only the latest checkpoint per run
#                                 (resolved via latest_checkpointed_iteration.txt
#                                 or the lexically-largest iter_* dir).
#   --iters N [N ...]             Explicit iteration numbers to export, e.g.
#                                 --iters 150 300 450.  Keeps only matching
#                                 iter_XXXXXXX dirs (zero-padded to 7 digits).
#   --dry-run                     Print the sbatch commands without submitting.
#   --sequential                  Chain jobs within each run with
#                                 --dependency=afterok so they do not run in
#                                 parallel (useful when GPU quota is tight).
#   --keep-megatron-vl            Forward --keep-megatron-vl to the Python script.
#   --strict-export               Forward --strict-export to the Python script.
#
# Output layout:
#   <out-base>/
#     <run_name>_iter_0000150/              HF safetensors
#     <run_name>_iter_0000150_megatron_vl/ intermediate (deleted by default)
#     <run_name>_iter_0000300/
#     ...
#
# Examples:
#   # All checkpoints from one run:
#   bash export_many_models.sh \
#       --ckpt-root ../oellm-autoexp/output/qwen3_5_35B_A3B_tw_test_cpt \
#       --hf-model  /shared_silo/scratch/rluukkon/oellm/hf_home/hub/models--Qwen--Qwen3.5-35B-A3B-Base/snapshots/0f0813072d2358973511097385626f21fcb6d422 \
#       --out-base  /shared_silo/scratch/rluukkon/oellm/Megatron-Bridge/exports
#
#   # Latest checkpoint only, dry-run first:
#   bash export_many_models.sh --latest-only --dry-run \
#       --ckpt-root ../oellm-autoexp/output/qwen3_5_35B_A3B_tw_test_cpt \
#       --hf-model  Qwen/Qwen3.5-35B-A3B-Base \
#       --out-base  /shared_silo/scratch/rluukkon/oellm/Megatron-Bridge/exports
#
#   # Multiple runs, sequential within each, specific iters only:
#   bash export_many_models.sh --sequential --iters 150 300 \
#       --ckpt-root ../oellm-autoexp/output/run_A ../oellm-autoexp/output/run_B \
#       --hf-model  Qwen/Qwen3.5-35B-A3B-Base \
#       --out-base  /shared_silo/scratch/rluukkon/oellm/Megatron-Bridge/exports

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SINGLE_CKPT_SCRIPT="${SCRIPT_DIR}/tw-tools/rewrap_text_to_vl_and_export.sh"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
CKPT_ROOTS=()
HF_MODEL=""
OUT_BASE=""
LATEST_ONLY=0
EXPLICIT_ITERS=()
DRY_RUN=0
SEQUENTIAL=0
EXTRA_PY_ARGS=()

_expect_value() {
    # Usage: _expect_value FLAG "$@"  — exits if the next token looks like a flag
    local flag="$1" val="$2"
    if [[ -z "$val" || "$val" == --* ]]; then
        echo "ERROR: $flag requires an argument." >&2
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt-root)
            shift
            # Consume all following non-flag tokens as roots
            while [[ $# -gt 0 && "$1" != --* ]]; do
                CKPT_ROOTS+=("$1")
                shift
            done
            ;;
        --hf-model)
            shift
            _expect_value "--hf-model" "${1:-}"
            HF_MODEL="$1"
            shift
            ;;
        --out-base)
            shift
            _expect_value "--out-base" "${1:-}"
            OUT_BASE="$1"
            shift
            ;;
        --latest-only)
            LATEST_ONLY=1
            shift
            ;;
        --iters)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                EXPLICIT_ITERS+=("$1")
                shift
            done
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --sequential)
            SEQUENTIAL=1
            shift
            ;;
        --keep-megatron-vl)
            EXTRA_PY_ARGS+=(--keep-megatron-vl)
            shift
            ;;
        --strict-export)
            EXTRA_PY_ARGS+=(--strict-export)
            shift
            ;;
        -h|--help)
            sed -n '2,/^set -euo/{ /^set -euo/d; s/^# \{0,1\}//; p }' "$0"
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Validate required arguments
# ---------------------------------------------------------------------------
if [[ ${#CKPT_ROOTS[@]} -eq 0 ]]; then
    echo "ERROR: --ckpt-root is required." >&2; exit 1
fi
if [[ -z "$HF_MODEL" ]]; then
    echo "ERROR: --hf-model is required." >&2; exit 1
fi
if [[ -z "$OUT_BASE" ]]; then
    echo "ERROR: --out-base is required." >&2; exit 1
fi
if [[ "$LATEST_ONLY" -eq 1 && ${#EXPLICIT_ITERS[@]} -gt 0 ]]; then
    echo "ERROR: --latest-only and --iters are mutually exclusive." >&2; exit 1
fi

# ---------------------------------------------------------------------------
# Helper: resolve a single latest iter_* dir from a checkpoint root
# ---------------------------------------------------------------------------
_resolve_latest_iter_dir() {
    local root="$1"
    local tracker="${root}/latest_checkpointed_iteration.txt"
    if [[ -f "$tracker" ]]; then
        local it
        it="$(cat "$tracker" | tr -d '[:space:]')"
        if [[ "$it" =~ ^[0-9]+$ ]]; then
            local cand
            cand="${root}/$(printf 'iter_%07d' "$it")"
            if [[ -d "$cand" ]]; then
                echo "$cand"
                return
            fi
        fi
    fi
    # Fall back: lexically largest iter_* dir
    local last_dir
    last_dir="$(ls -d "${root}"/iter_* 2>/dev/null | sort | tail -n 1)"
    if [[ -z "$last_dir" ]]; then
        echo "ERROR: No iter_* directory found under ${root}" >&2
        exit 1
    fi
    echo "$last_dir"
}

# ---------------------------------------------------------------------------
# Helper: collect iter_* dirs to export for one root
# ---------------------------------------------------------------------------
_collect_iter_dirs() {
    local root="$1"

    if [[ "$LATEST_ONLY" -eq 1 ]]; then
        _resolve_latest_iter_dir "$root"
        return
    fi

    local all_iter_dirs
    mapfile -t all_iter_dirs < <(ls -d "${root}"/iter_* 2>/dev/null | sort)

    if [[ ${#all_iter_dirs[@]} -eq 0 ]]; then
        echo "WARNING: No iter_* directories found under ${root} — skipping." >&2
        return
    fi

    if [[ ${#EXPLICIT_ITERS[@]} -gt 0 ]]; then
        # Filter to the explicitly requested iteration numbers only
        for iter_num in "${EXPLICIT_ITERS[@]}"; do
            local tag
            tag="$(printf 'iter_%07d' "$iter_num")"
            local cand="${root}/${tag}"
            if [[ -d "$cand" ]]; then
                echo "$cand"
            else
                echo "WARNING: Requested iter ${iter_num} (${tag}) not found under ${root} — skipping." >&2
            fi
        done
    else
        printf '%s\n' "${all_iter_dirs[@]}"
    fi
}

# ---------------------------------------------------------------------------
# Summary header
# ---------------------------------------------------------------------------
echo "========================================"
echo "export_many_models.sh — batch checkpoint export"
echo "  HF reference : $HF_MODEL"
echo "  out-base     : $OUT_BASE"
echo "  latest-only  : $LATEST_ONLY"
echo "  dry-run      : $DRY_RUN"
echo "  sequential   : $SEQUENTIAL"
[[ ${#EXPLICIT_ITERS[@]} -gt 0 ]] && echo "  iters filter : ${EXPLICIT_ITERS[*]}"
[[ ${#EXTRA_PY_ARGS[@]} -gt 0 ]] && echo "  extra args   : ${EXTRA_PY_ARGS[*]}"
echo "  ckpt roots   :"
for ROOT in "${CKPT_ROOTS[@]}"; do echo "    $ROOT"; done
echo "========================================"

mkdir -p "$OUT_BASE"

# ---------------------------------------------------------------------------
# Main loop: one sbatch job per checkpoint
# ---------------------------------------------------------------------------
TOTAL_SUBMITTED=0
TOTAL_SKIPPED=0

for ROOT in "${CKPT_ROOTS[@]}"; do
    if [[ ! -d "$ROOT" ]]; then
        echo "ERROR: Checkpoint root does not exist: ${ROOT}" >&2
        exit 1
    fi

    RUN_NAME="$(basename "${ROOT%/}")"
    echo ""
    echo "--- Run: ${RUN_NAME} (${ROOT})"

    # Collect iter dirs for this root
    mapfile -t ITER_DIRS < <(_collect_iter_dirs "$ROOT")

    if [[ ${#ITER_DIRS[@]} -eq 0 ]]; then
        echo "  (no iterations to export)"
        continue
    fi

    echo "  Found ${#ITER_DIRS[@]} iteration(s) to consider:"
    for d in "${ITER_DIRS[@]}"; do echo "    $(basename "$d")"; done

    # Per-run dependency tracking for --sequential
    DEP=""

    for ITER_DIR in "${ITER_DIRS[@]}"; do
        ITER_TAG="$(basename "$ITER_DIR")"   # e.g. iter_0000150
        OUT="${OUT_BASE}/${RUN_NAME}_${ITER_TAG}"

        # Skip already-completed exports
        if [[ -d "$OUT" && -n "$(ls -A "$OUT" 2>/dev/null)" ]]; then
            echo "  [skip] ${OUT} already exists and is non-empty"
            (( TOTAL_SKIPPED++ )) || true
            continue
        fi

        # Build the sbatch command array
        CMD=(sbatch)
        if [[ "$SEQUENTIAL" -eq 1 && -n "$DEP" ]]; then
            CMD+=(--dependency="afterok:${DEP}")
        fi
        CMD+=("$SINGLE_CKPT_SCRIPT" "$ITER_DIR" "$HF_MODEL" "$OUT")
        # Extra Python-level flags are appended as positional args 4+; the
        # single-checkpoint script forwards them to the Python call (see $@).
        CMD+=("${EXTRA_PY_ARGS[@]+"${EXTRA_PY_ARGS[@]}"}")

        echo "  [submit] ${CMD[*]}"

        if [[ "$DRY_RUN" -eq 0 ]]; then
            SBATCH_OUT="$("${CMD[@]}")"
            echo "           -> ${SBATCH_OUT}"
            JOB_ID="$(echo "$SBATCH_OUT" | awk '{print $NF}')"
            DEP="$JOB_ID"
            (( TOTAL_SUBMITTED++ )) || true
        fi
    done
done

echo ""
echo "========================================"
if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Dry-run complete. No jobs were submitted."
else
    echo "Done. Submitted: ${TOTAL_SUBMITTED}  Skipped: ${TOTAL_SKIPPED}"
fi
echo "========================================"
