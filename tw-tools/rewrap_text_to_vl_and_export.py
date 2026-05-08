#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Re-wrap a text-only Megatron-LM Qwen3.5 checkpoint as a VL checkpoint and export to HF.

This is stage 3 of the text-only CPT pipeline:

    stage 1: ``tw-tools/import_hf_to_megatron_tw_text.sh``
             HF Qwen3.5-XB-Base  -->  Megatron text-only ckpt (GPTModel keys)
    stage 2: oellm-autoexp Megatron-LM ``pretrain_gpt.py``
             continued pretraining on text data only
    stage 3: this script
             1. load the *original* VL Megatron ckpt (vision tower preserved)
             2. load the *trained* text Megatron ckpt (LM weights updated)
             3. splice the trained LM weights into the VL model under the
                ``language_model.`` prefix
             4. ``bridge.save_megatron_model`` -> a fresh Bridge VL ckpt
             5. ``bridge.export_ckpt`` -> HuggingFace VL safetensors

The actual Megatron->HF parameter remapping (QKV split, GDN ``in_proj`` split
into ``in_proj_qkv/z/b/a``, RMSNorm zero-centering reverse, etc.) is handled
entirely by the existing ``Qwen35VLBridge.mapping_registry()``; this script
just stages the inputs.

Vision parameters in the rewrapped checkpoint are byte-identical to the
original HuggingFace release (they were never on a GPU during stage 2). The
language model has CPT-updated weights. As a result the exported HF model
will exhibit "plausibly degraded" VL capabilities: vision-language alignment
will have drifted because the LM moved, but vision encoding itself is
unchanged.

Example (single-GPU)::

    BRIDGE_QWEN35_USE_VL=1 \\
    python rewrap_text_to_vl_and_export.py \\
        --trained-text-ckpt /shared_silo/.../qwen3_5_4B_tw_test/iter_0001000 \\
        --original-vl-ckpt  /shared_silo/.../megatron_ckpt/Qwen3.5-4B \\
        --hf-model          Qwen/Qwen3.5-4B-Base \\
        --out-megatron-vl   /shared_silo/.../megatron_ckpt/Qwen3.5-4B-trained-vl \\
        --out-hf            /shared_silo/.../hf_exports/Qwen3.5-4B-trained
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Iterable

# IMPORTANT: must be set BEFORE any megatron.bridge imports so the VL bridge
# wins the dispatch slot for ``Qwen3_5ForConditionalGeneration``.
os.environ.setdefault("BRIDGE_QWEN35_USE_VL", "1")

import torch  # noqa: E402

from megatron.bridge import AutoBridge  # noqa: E402
from megatron.core import dist_checkpointing  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("rewrap-text-to-vl")


_ITERATION_DIR_PREFIX = "iter_"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--trained-text-ckpt",
        required=True,
        type=str,
        help=(
            "Path to the Megatron-LM text-only training output. May be either the "
            "training root (auto-discovers the latest ``iter_*`` directory) or a "
            "specific ``iter_*`` directory."
        ),
    )
    parser.add_argument(
        "--original-vl-ckpt",
        required=True,
        type=str,
        help=(
            "Path to the *original* Megatron Bridge VL checkpoint produced by "
            "``tw-tools/import_hf_to_megatron_tw.sh`` (i.e. the one whose "
            "``run_config.yaml`` references ``Qwen35VLModelProvider``). Used as "
            "the source of vision-tower weights and config."
        ),
    )
    parser.add_argument(
        "--hf-model",
        required=True,
        type=str,
        help="HuggingFace model id (e.g. ``Qwen/Qwen3.5-4B-Base``) used for tokenizer + auto-config.",
    )
    parser.add_argument(
        "--out-megatron-vl",
        required=True,
        type=str,
        help="Where to write the rewrapped Megatron Bridge VL checkpoint.",
    )
    parser.add_argument(
        "--out-hf",
        required=True,
        type=str,
        help="Where to write the final HuggingFace VL safetensors directory.",
    )
    parser.add_argument(
        "--strict-export",
        action="store_true",
        help="Pass ``strict=True`` to ``export_ckpt`` (disabled by default to "
        "tolerate harmless key drops such as MTP).",
    )
    parser.add_argument(
        "--keep-megatron-vl",
        action="store_true",
        help="Keep the intermediate rewrapped Megatron VL checkpoint after a successful HF export.",
    )
    return parser.parse_args()


def _resolve_iter_dir(path: str | Path) -> Path:
    """Resolve a checkpoint root to a specific ``iter_*`` subdirectory.

    If ``path`` itself is an ``iter_*`` directory, it is returned unchanged.
    Otherwise the ``latest_checkpointed_iteration.txt`` tracker (or, failing
    that, the lexically-largest ``iter_*`` subdirectory) is used.
    """
    path = Path(path)
    if path.name.startswith(_ITERATION_DIR_PREFIX):
        return path

    tracker = path / "latest_checkpointed_iteration.txt"
    if tracker.exists():
        try:
            it = int(tracker.read_text().strip())
            cand = path / f"{_ITERATION_DIR_PREFIX}{it:07d}"
            if cand.is_dir():
                return cand
        except (ValueError, OSError) as exc:
            logger.warning("Could not parse %s: %s -- falling back to directory scan", tracker, exc)

    iter_dirs = sorted(p for p in path.iterdir() if p.is_dir() and p.name.startswith(_ITERATION_DIR_PREFIX))
    if not iter_dirs:
        raise FileNotFoundError(
            f"No iter_* checkpoint directory found under {path}. "
            "Pass either the iter_NNNNNNN directory directly, or a checkpoint root "
            "containing latest_checkpointed_iteration.txt."
        )
    return iter_dirs[-1]


def _build_text_provider(vl_bridge: AutoBridge):
    """Construct the matching :class:`Qwen35DenseTextProvider` for the LM portion.

    Re-uses the same :class:`PreTrainedCausalLM` instance that the VL bridge
    is operating on (so the language settings are guaranteed to match the
    HF reference used by the VL provider).
    """
    # Local import: env var is already set, so this is safe even when the text
    # bridge registration is suppressed (we only need the classes themselves).
    from megatron.bridge.models.qwen.qwen3_5_dense_bridge import (
        Qwen35DenseTextBridge,
        Qwen35DenseTextProvider,
    )

    text_bridge = Qwen35DenseTextBridge()
    provider: Qwen35DenseTextProvider = text_bridge.provider_bridge(vl_bridge.hf_pretrained)
    # Match the parallelism config of the freshly built VL model so the
    # sharded_state_dict shapes line up with what the trained ckpt stores.
    return provider


def _iter_lm_pairs(gpt_state: dict, vl_state: dict) -> Iterable[tuple[str, str]]:
    """Yield ``(gpt_key, vl_key)`` pairs to copy.

    A simple prefix transform: every key in the GPT state-dict maps to
    ``language_model.<key>`` in the VL state-dict, *if* that target exists.
    Missing targets (e.g. trained-only artefacts) are surfaced to the caller.
    """
    for gpt_key in gpt_state.keys():
        target = f"language_model.{gpt_key}"
        if target in vl_state:
            yield gpt_key, target


def _is_gdn_out_norm_key(gpt_key: str) -> bool:
    """Return True for the GDN ``out_norm.weight`` keys that need a semantic shift.

    The text bridge (``Qwen35DenseTextProvider``) sets
    ``layernorm_zero_centered_gamma=False`` and stores HF's RMSNorm weight
    untransformed, while the VL bridge (``Qwen35VLModelProvider``) keeps the
    field at ``True`` *and* registers ``RMSNorm2ZeroCenteredRMSNormMapping``
    only for ``out_norm.weight`` -- meaning the VL Megatron checkpoint slot
    holds ``hf_value - 1.0``. Splicing the trained text value as-is would
    leave the VL bridge's export adding 1 again, double-shifting the tensor
    by +1 in the final HF safetensors. The ``-1`` we apply at splice time
    cancels the bridge's re-add and preserves byte-identical math through
    the round-trip.
    """
    if not gpt_key.endswith(".out_norm.weight"):
        return False
    return ".self_attention." in gpt_key


def main() -> int:
    args = parse_args()

    if os.environ.get("BRIDGE_QWEN35_USE_VL", "0") != "1":
        # We set the default at module load. If the user explicitly overrode it
        # to 0, abort early with a clear diagnostic rather than failing later.
        raise RuntimeError(
            "rewrap_text_to_vl_and_export.py requires BRIDGE_QWEN35_USE_VL=1 so the "
            "Qwen3.5 dense VL bridge is registered. Refusing to continue with VL=0."
        )

    trained_text_iter_dir = _resolve_iter_dir(args.trained_text_ckpt)
    logger.info("Trained text checkpoint:   %s", trained_text_iter_dir)
    logger.info("Original VL checkpoint:    %s", args.original_vl_ckpt)
    logger.info("HuggingFace reference:     %s", args.hf_model)
    logger.info("Output (Megatron VL):      %s", args.out_megatron_vl)
    logger.info("Output (HuggingFace VL):   %s", args.out_hf)

    from megatron.bridge.training.model_load_save import temporary_distributed_context

    # ------------------------------------------------------------------
    # Phase 1: load + splice + save the rewrapped VL checkpoint.
    # ------------------------------------------------------------------
    with temporary_distributed_context(backend="gloo"):
        # Build the VL bridge against the HF reference (config-only is enough
        # since we'll overwrite with the original VL ckpt weights anyway).
        vl_bridge = AutoBridge.from_hf_pretrained(args.hf_model)
        logger.info("Loaded HF reference into AutoBridge: %s", type(vl_bridge._model_bridge).__name__)

        # Load original VL Megatron checkpoint -> populates language_model.*,
        # vision_model.*, merger.*, patch_embed.*, MTP, etc.
        vl_models = vl_bridge.load_megatron_model(args.original_vl_ckpt, wrap_with_ddp=False)
        if not isinstance(vl_models, list):
            vl_models = [vl_models]
        logger.info("Loaded VL Megatron checkpoint: %d model shard(s)", len(vl_models))

        # Build a text-only GPTModel matching the trained checkpoint and load it.
        text_provider = _build_text_provider(vl_bridge)
        gpt_model = text_provider.provide(pre_process=True, post_process=True)
        gpt_sharded = gpt_model.sharded_state_dict()
        logger.info("Bare GPTModel built; %d sharded entries", len(gpt_sharded))

        loaded_gpt = dist_checkpointing.load(gpt_sharded, str(trained_text_iter_dir))
        gpt_model.load_state_dict(loaded_gpt, strict=True)
        logger.info("Trained text checkpoint loaded into GPTModel (%d tensors).", len(loaded_gpt))

        # ------------------------------------------------------------------
        # Splice trained LM weights into VL model.language_model.*
        # ------------------------------------------------------------------
        gpt_state = dict(gpt_model.state_dict())
        # Strip any "module." prefix that Megatron's DDP/wrapper might add.
        gpt_state = {
            (k[len("module.") :] if k.startswith("module.") else k): v for k, v in gpt_state.items()
        }

        n_copied = 0
        n_lm_keys_in_vl = 0
        n_skipped = 0
        n_zero_centered_shifts = 0
        for vl_model in vl_models:
            vl_state = dict(vl_model.state_dict())
            n_lm_keys_in_vl = sum(1 for k in vl_state if k.startswith("language_model."))
            for gpt_key, vl_key in _iter_lm_pairs(gpt_state, vl_state):
                src = gpt_state[gpt_key]
                dst = vl_state[vl_key]
                if dst.shape != src.shape:
                    raise ValueError(
                        f"Shape mismatch on rewrap: gpt[{gpt_key}]={tuple(src.shape)} vs "
                        f"vl[{vl_key}]={tuple(dst.shape)}. Did the text bridge and VL bridge "
                        f"produce mismatched architectures (head_dim, num_query_groups, ...)?"
                    )
                value = src.to(dst.dtype)
                if _is_gdn_out_norm_key(gpt_key):
                    # See _is_gdn_out_norm_key for the math: text uses
                    # zero-centered=False (HF passthrough), VL uses
                    # zero-centered=True with a -1 import shift on this key.
                    value = value - 1.0
                    n_zero_centered_shifts += 1
                dst.copy_(value)
                n_copied += 1

            untouched = [
                k
                for k in vl_state
                if k.startswith("language_model.") and k.removeprefix("language_model.") not in gpt_state
            ]
            for k in untouched:
                logger.info("[keep-original] %s", k)
                n_skipped += 1
            untouched_gpt = [k for k in gpt_state if f"language_model.{k}" not in vl_state]
            for k in untouched_gpt:
                logger.warning("[orphan-trained] %s (no matching language_model.* in VL)", k)

        logger.info(
            "Spliced %d trained LM tensors into VL model "
            "(%d total LM keys in VL; %d retained from original VL ckpt; "
            "%d zero-centered-gamma shifts applied to GDN out_norm.weight).",
            n_copied,
            n_lm_keys_in_vl,
            n_skipped,
            n_zero_centered_shifts,
        )

        # Free the GPTModel before save to reduce peak memory.
        del gpt_model
        del gpt_state
        del loaded_gpt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # Save the rewrapped VL checkpoint in Megatron Bridge format.
        # ------------------------------------------------------------------
        vl_bridge.save_megatron_model(
            vl_models,
            args.out_megatron_vl,
            hf_tokenizer_path=args.hf_model,
            low_memory_save=True,
        )
        logger.info("Wrote rewrapped Megatron VL checkpoint to %s", args.out_megatron_vl)

    # ------------------------------------------------------------------
    # Phase 2: HuggingFace export (opens its own dist context).
    # ------------------------------------------------------------------
    export_bridge = AutoBridge.from_hf_pretrained(args.hf_model)
    export_bridge.export_ckpt(
        args.out_megatron_vl,
        args.out_hf,
        show_progress=True,
        strict=args.strict_export,
    )
    logger.info("Wrote HuggingFace VL safetensors to %s", args.out_hf)

    if not args.keep_megatron_vl:
        # Optional: leave a breadcrumb so it is obvious the user can delete this.
        breadcrumb = Path(args.out_megatron_vl) / "_REWRAP_INTERMEDIATE.md"
        try:
            breadcrumb.write_text(
                "This checkpoint was produced by rewrap_text_to_vl_and_export.py as an\n"
                "intermediate before HF export. It can be deleted unless you want to\n"
                "re-export with different settings without rerunning the rewrap step.\n"
            )
        except OSError:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
