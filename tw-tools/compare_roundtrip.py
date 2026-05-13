#!/usr/bin/env python3
"""Compare a roundtrip-exported HuggingFace model against the original.

Usage (no GPU / SLURM needed):

    python compare_roundtrip.py \\
        --orig  /shared_silo/scratch/rluukkon/hf_home/hub/models--Qwen--Qwen3.5-4B-Base/snapshots/1001bb4d826a52d1f399e183466143f4da7b741b \\
        --rt    /shared_silo/scratch/rluukkon/oellm/Megatron-Bridge/roundtrip_conversion

Loads both models directly from their safetensors shards (no model class
instantiation). For each shared key it reports max/mean absolute difference
and cosine similarity, flagging clear mismatches.  A summary by key group is
printed at the end.

Expected results for a lossless roundtrip (HF -> Megatron -> HF, no training):
  - vision keys  : byte-identical  (max_abs_diff == 0)
  - LM keys      : near-identical  (max_abs_diff < ~2e-3, cos_sim > 0.9999)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_state_dict(model_dir: str | Path) -> dict[str, torch.Tensor]:
    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        shards = sorted(set(index["weight_map"].values()))
    else:
        # Single-shard model
        shards = ["model.safetensors"]

    try:
        from safetensors.torch import load_file
    except ImportError:
        sys.exit("ERROR: safetensors not installed. Run: pip install safetensors")

    sd: dict[str, torch.Tensor] = {}
    for shard in shards:
        sd.update(load_file(str(model_dir / shard), device="cpu"))
    return sd


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    denom = a.norm() * b.norm()
    if denom == 0:
        return float("nan")
    return (a @ b / denom).item()


def key_group(key: str) -> str:
    if key.startswith("visual"):
        return "vision"
    if key.startswith("merger"):
        return "merger"
    if key.startswith("model."):
        return "lm"
    return "other"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--orig", required=True, help="Path to the original HF model directory")
    parser.add_argument("--rt",   required=True, help="Path to the roundtrip HF model directory")
    parser.add_argument("--cos-thresh",  type=float, default=0.9999,
                        help="Cosine similarity below this is flagged MISMATCH (default: 0.9999)")
    parser.add_argument("--diff-thresh", type=float, default=0.01,
                        help="max_abs_diff above this is flagged MISMATCH (default: 0.01)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-key results for all keys, not just mismatches")
    args = parser.parse_args()

    print(f"Loading original : {args.orig}")
    orig = load_state_dict(args.orig)
    print(f"  {len(orig)} tensors loaded")

    print(f"Loading roundtrip: {args.rt}")
    rt = load_state_dict(args.rt)
    print(f"  {len(rt)} tensors loaded\n")

    orig_keys = set(orig.keys())
    rt_keys   = set(rt.keys())

    only_orig = sorted(orig_keys - rt_keys)
    only_rt   = sorted(rt_keys   - orig_keys)
    shared    = sorted(orig_keys & rt_keys)

    if only_orig:
        print(f"[ORPHAN] Keys only in ORIGINAL ({len(only_orig)}):")
        for k in only_orig:
            print(f"  {k}  {tuple(orig[k].shape)}")
        print()

    if only_rt:
        print(f"[ORPHAN] Keys only in ROUNDTRIP ({len(only_rt)}):")
        for k in only_rt:
            print(f"  {k}  {tuple(rt[k].shape)}")
        print()

    # Per-group accumulators
    groups: dict[str, dict] = {}
    for g in ("vision", "merger", "lm", "other"):
        groups[g] = {"pass": 0, "fail": 0, "skip": 0, "max_diff": 0.0, "min_cos": 1.0}

    header = f"{'KEY':<70} {'SHAPE':>18}  {'MAX_ABS':>9}  {'MEAN_ABS':>9}  {'COS_SIM':>9}  STATUS"
    print(header)
    print("-" * len(header))

    mismatches: list[str] = []

    for key in shared:
        t_orig = orig[key].float()
        t_rt   = rt[key].float()
        g      = key_group(key)
        acc    = groups[g]

        if not t_orig.is_floating_point():
            acc["skip"] += 1
            if args.verbose:
                print(f"{key:<70} {str(tuple(t_orig.shape)):>18}  {'n/a':>9}  {'n/a':>9}  {'n/a':>9}  SKIP(int)")
            continue

        if t_orig.shape != t_rt.shape:
            acc["fail"] += 1
            mismatches.append(key)
            print(f"{key:<70} {str(tuple(t_orig.shape)):>18}  {'SHAPE MISMATCH':>31}  vs {tuple(t_rt.shape)}  FAIL")
            continue

        diff = (t_orig - t_rt).abs()
        max_diff  = diff.max().item()
        mean_diff = diff.mean().item()
        cos       = cosine_sim(t_orig, t_rt)

        acc["max_diff"] = max(acc["max_diff"], max_diff)
        acc["min_cos"]  = min(acc["min_cos"],  cos if not (cos != cos) else 1.0)

        mismatch = (cos < args.cos_thresh) or (max_diff > args.diff_thresh)
        status   = "FAIL" if mismatch else "ok"

        if mismatch:
            acc["fail"] += 1
            mismatches.append(key)
        else:
            acc["pass"] += 1

        if args.verbose or mismatch:
            print(
                f"{key:<70} {str(tuple(t_orig.shape)):>18}  "
                f"{max_diff:9.4e}  {mean_diff:9.4e}  {cos:9.6f}  {status}"
            )

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY BY GROUP")
    print(f"{'='*80}")
    print(f"  {'Group':<10} {'Pass':>6} {'Fail':>6} {'Skip':>6}  {'MaxAbsDiff':>12}  {'MinCosSim':>12}")
    print(f"  {'-'*10} {'-'*6} {'-'*6} {'-'*6}  {'-'*12}  {'-'*12}")
    total_fail = 0
    for g in ("vision", "merger", "lm", "other"):
        acc = groups[g]
        total_fail += acc["fail"]
        min_cos_str = f"{acc['min_cos']:.7f}" if acc["pass"] + acc["fail"] > 0 else "n/a"
        print(
            f"  {g:<10} {acc['pass']:>6} {acc['fail']:>6} {acc['skip']:>6}  "
            f"{acc['max_diff']:12.4e}  {min_cos_str:>12}"
        )

    print(f"\nShared keys compared : {len(shared)}")
    print(f"Orphan (orig only)   : {len(only_orig)}")
    print(f"Orphan (rt only)     : {len(only_rt)}")
    print(f"Total FAIL           : {total_fail}")

    if total_fail == 0:
        print("\nRESULT: PASS -- roundtrip is numerically equivalent to the original.")
    else:
        print(f"\nRESULT: FAIL -- {total_fail} key(s) exceeded thresholds. See above for details.")
        if not args.verbose:
            print("Re-run with --verbose to see all per-key statistics.")

    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
