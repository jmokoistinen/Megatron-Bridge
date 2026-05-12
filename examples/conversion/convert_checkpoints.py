#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Megatron-HuggingFace Checkpoint Conversion

Converts models between HuggingFace and Megatron formats.  Supports dense
models, text-only variants (VL weights stripped), and MoE models that require
multi-GPU expert-parallel conversion.

Import examples:

  # Dense model (default, VL-capable if the bridge supports it)
  python examples/conversion/convert_checkpoints.py import \\
    --hf-model Qwen/Qwen3.5-4B-Base \\
    --megatron-path ./megatron_ckpt/Qwen3.5-4B-Base

  # Dense model, text-only (vision tower weights stripped)
  python examples/conversion/convert_checkpoints.py import \\
    --hf-model Qwen/Qwen3.5-4B-Base \\
    --megatron-path ./megatron_ckpt/Qwen3.5-4B-Base-text \\
    --text-only

  # MoE model, expert-parallel (8 GPUs, EP=8)
  #   Launch with: srun --ntasks=8 python ... --expert-model-parallel-size 8
  python examples/conversion/convert_checkpoints.py import \\
    --hf-model Qwen/Qwen3.5-35B-A3B-Base \\
    --megatron-path ./megatron_ckpt/Qwen3.5-35B-A3B-Base \\
    --text-only --expert-model-parallel-size 8

Export example:

  python examples/conversion/convert_checkpoints.py export \\
    --hf-model Qwen/Qwen3.5-4B-Base \\
    --megatron-path ./megatron_ckpt/Qwen3.5-4B-Base \\
    --hf-path ./exports/Qwen3.5-4B-Base-hf
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional


def validate_path(path: str, must_exist: bool = False) -> Path:
    path_obj = Path(path)
    if must_exist and not path_obj.exists():
        raise ValueError(f"Path does not exist: {path}")
    return path_obj


def get_torch_dtype(dtype_str: str):
    import torch
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Supported: {list(dtype_map.keys())}")
    return dtype_map[dtype_str]


def _setup_distributed_env(expert_model_parallel_size: int) -> tuple[int, int]:
    """Seed distributed env vars from SLURM vars before any Bridge/Megatron import."""
    os.environ.setdefault("RANK", os.environ.get("SLURM_PROCID", "0"))
    os.environ.setdefault("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0"))
    os.environ.setdefault(
        "WORLD_SIZE",
        os.environ.get("SLURM_NTASKS", str(expert_model_parallel_size)),
    )
    os.environ.setdefault("MASTER_ADDR", os.environ.get("MASTER_ADDR", "localhost"))
    os.environ.setdefault("MASTER_PORT", os.environ.get("MASTER_PORT", "29500"))

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    return rank, world_size


def import_hf_to_megatron(
    hf_model: str,
    megatron_path: str,
    text_only: bool = False,
    expert_model_parallel_size: int = 1,
    tensor_model_parallel_size: int = 1,
    torch_dtype: Optional[str] = None,
    device_map: Optional[str] = None,
    trust_remote_code: bool = False,
) -> None:
    """Import a HuggingFace model and save it as a Megatron checkpoint."""

    # Set env vars that Bridge/Megatron read during initialisation BEFORE
    # any import of megatron.bridge so that parallel groups are configured
    # correctly.
    if text_only:
        os.environ["BRIDGE_QWEN35_USE_VL"] = "0"

    distributed = expert_model_parallel_size > 1
    if distributed:
        rank, world_size = _setup_distributed_env(expert_model_parallel_size)
    else:
        rank, world_size = 0, 1

    # Deferred imports: must happen after env vars are set.
    import torch
    from megatron.bridge.models.conversion.auto_bridge import AutoBridge

    if rank == 0:
        mode = "text-only" if text_only else "default (VL)"
        print("=" * 48)
        print("HF -> Megatron import")
        print(f"  hf_model  : {hf_model}")
        print(f"  output    : {megatron_path}")
        print(f"  mode      : {mode}")
        if distributed:
            print(f"  EP        : {expert_model_parallel_size}  TP={tensor_model_parallel_size}")
            print(f"  rank/world: {rank}/{world_size}")
        print("=" * 48)

    if distributed:
        # Multi-GPU path: builds a distributed Megatron model so expert weights
        # are sharded at the target EP size and no cross-EP resharding is needed
        # when loading the checkpoint for training.
        if rank == 0:
            print("Loading HuggingFace model (rank 0) ...")

        bridge = AutoBridge.from_hf_pretrained(
            hf_model,
            torch_dtype=torch.bfloat16,
        )

        provider = bridge.to_megatron_provider(load_weights=True)
        provider.expert_model_parallel_size = expert_model_parallel_size
        provider.tensor_model_parallel_size = tensor_model_parallel_size

        if rank == 0:
            print(
                f"Building Megatron model "
                f"(EP={expert_model_parallel_size}, TP={tensor_model_parallel_size}) ..."
            )

        megatron_model = provider.provide_distributed_model(
            wrap_with_ddp=False,
            use_cpu_initialization=True,
        )

        if rank == 0:
            print(f"Saving checkpoint to {megatron_path} ...")

        hf_tokenizer_kwargs = {}
        if hasattr(bridge._model_bridge, "get_hf_tokenizer_kwargs"):
            hf_tokenizer_kwargs = bridge._model_bridge.get_hf_tokenizer_kwargs() or {}

        bridge.save_megatron_model(
            megatron_model,
            megatron_path,
            hf_tokenizer_path=hf_model,
            hf_tokenizer_kwargs=hf_tokenizer_kwargs,
            low_memory_save=True,
        )

        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            if rank == 0:
                print(f"Done. Checkpoint at {megatron_path}")
            torch.distributed.destroy_process_group()

    else:
        # Single-GPU path.
        kwargs = {}
        if torch_dtype:
            kwargs["torch_dtype"] = get_torch_dtype(torch_dtype)
        if device_map:
            kwargs["device_map"] = device_map
        if trust_remote_code:
            kwargs["trust_remote_code"] = trust_remote_code

        AutoBridge.import_ckpt(
            hf_model_id=hf_model,
            megatron_path=megatron_path,
            **kwargs,
        )

        print(f"Done. Checkpoint stored at: {megatron_path}")

        checkpoint_path = Path(megatron_path)
        if checkpoint_path.exists():
            print("Checkpoint structure:")
            for item in sorted(checkpoint_path.iterdir()):
                print(f"  {'[dir] ' if item.is_dir() else '      '}{item.name}")

        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()


def export_megatron_to_hf(
    hf_model: str,
    megatron_path: str,
    hf_path: str,
    show_progress: bool = True,
    strict: bool = True,
    trust_remote_code: bool = False,
) -> None:
    """Export a Megatron checkpoint to HuggingFace format."""

    # Deferred import so env vars can be set before megatron.bridge loads.
    from megatron.bridge import AutoBridge
    import torch

    checkpoint_path = validate_path(megatron_path, must_exist=True)
    print(f"Exporting: {megatron_path} -> {hf_path}")
    print(f"Found Megatron checkpoint: {checkpoint_path}")

    config_files = list(checkpoint_path.glob("**/run_config.yaml"))
    if not config_files:
        iter_dirs = [d for d in checkpoint_path.iterdir() if d.is_dir() and d.name.startswith("iter_")]
        if iter_dirs:
            latest_iter = max(iter_dirs, key=lambda d: int(d.name.replace("iter_", "")))
            config_files = list(latest_iter.glob("run_config.yaml"))

    if not config_files:
        raise FileNotFoundError(
            f"Could not find run_config.yaml in {checkpoint_path}. "
            "Please ensure this is a valid Megatron checkpoint."
        )

    print(f"Found configuration: {config_files[0]}")

    bridge = AutoBridge.from_auto_config(megatron_path, hf_model, trust_remote_code=trust_remote_code)
    bridge.export_ckpt(
        megatron_path=megatron_path,
        hf_path=hf_path,
        show_progress=show_progress,
        strict=strict,
    )

    print(f"Done. Exported to: {hf_path}")

    export_path = Path(hf_path)
    if export_path.exists():
        print("Export structure:")
        for item in sorted(export_path.iterdir()):
            print(f"  {'[dir] ' if item.is_dir() else '      '}{item.name}")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(
        description="Convert models between HuggingFace and Megatron formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Conversion direction")

    # -------------------------------------------------------------------------
    # import subcommand
    # -------------------------------------------------------------------------
    import_parser = subparsers.add_parser(
        "import", help="Import HuggingFace model to Megatron checkpoint format"
    )
    import_parser.add_argument(
        "--hf-model", required=True, help="HuggingFace model ID or path to model directory"
    )
    import_parser.add_argument(
        "--megatron-path", required=True, help="Output directory for the Megatron checkpoint"
    )
    import_parser.add_argument(
        "--text-only",
        action="store_true",
        help="Strip vision tower weights (sets BRIDGE_QWEN35_USE_VL=0). "
             "Required for text-only continued pretraining with pretrain_gpt.py.",
    )
    import_parser.add_argument(
        "--expert-model-parallel-size",
        type=int,
        default=1,
        metavar="N",
        help="Expert parallelism degree. When N>1 the script uses a multi-GPU "
             "code path that shards expert weights across N ranks, avoiding the "
             "EP resharding issue at training load time. "
             "Launch with 'srun --ntasks=N' (default: 1).",
    )
    import_parser.add_argument(
        "--tensor-model-parallel-size",
        type=int,
        default=1,
        metavar="N",
        help="Tensor parallelism degree (default: 1, only used when --expert-model-parallel-size > 1)",
    )
    import_parser.add_argument(
        "--torch-dtype",
        choices=["float32", "float16", "bfloat16"],
        help="Model precision (single-GPU path only)",
    )
    import_parser.add_argument(
        "--device-map",
        help='Device placement strategy, e.g. "auto" (single-GPU path only)',
    )
    import_parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model code execution",
    )

    # -------------------------------------------------------------------------
    # export subcommand
    # -------------------------------------------------------------------------
    export_parser = subparsers.add_parser(
        "export", help="Export Megatron checkpoint to HuggingFace format"
    )
    export_parser.add_argument(
        "--hf-model",
        required=True,
        help="HuggingFace model ID or path used as reference for config synthesis",
    )
    export_parser.add_argument(
        "--megatron-path", required=True, help="Directory containing the Megatron checkpoint"
    )
    export_parser.add_argument(
        "--hf-path", required=True, help="Output directory for the HuggingFace model"
    )
    export_parser.add_argument(
        "--no-progress", action="store_true", help="Disable progress bar during export"
    )
    export_parser.add_argument(
        "--not-strict",
        action="store_true",
        help="Allow source and target checkpoint to have different keys",
    )
    export_parser.add_argument(
        "--trust-remote-code", action="store_true", help="Allow custom model code execution"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "import":
        import_hf_to_megatron(
            hf_model=args.hf_model,
            megatron_path=args.megatron_path,
            text_only=args.text_only,
            expert_model_parallel_size=args.expert_model_parallel_size,
            tensor_model_parallel_size=args.tensor_model_parallel_size,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
        )

    elif args.command == "export":
        export_megatron_to_hf(
            hf_model=args.hf_model,
            megatron_path=args.megatron_path,
            hf_path=args.hf_path,
            show_progress=not args.no_progress,
            strict=not args.not_strict,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        raise RuntimeError(f"Unknown command: {args.command}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
