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
             1. load the *original* HF VL model (vision tower preserved)
             2. convert it to a Megatron VL model in memory
             3. overwrite the language_model.* weights with the trained
                Megatron text checkpoint
             4. ``bridge.save_megatron_model`` -> a fresh Bridge VL ckpt
             5. ``bridge.export_ckpt`` -> HuggingFace VL safetensors

The actual Megatron->HF parameter remapping (QKV split, GDN ``in_proj`` split
into ``in_proj_qkv/z/b/a``, RMSNorm zero-centering reverse, etc.) is handled
entirely by the existing ``Qwen35VLBridge.mapping_registry()``; this script
just stages the inputs.

Vision parameters in the output are byte-identical to the original HuggingFace
release (they were never on a GPU during stage 2). The language model has
CPT-updated weights. As a result the exported HF model will exhibit "plausibly
degraded" VL capabilities: vision-language alignment will have drifted because
the LM moved, but vision encoding itself is unchanged.

Example (single-GPU)::

    BRIDGE_QWEN35_USE_VL=1 \\
    python rewrap_text_to_vl_and_export.py \\
        --trained-text-ckpt /shared_silo/.../megatron_ckpt/Qwen3.5-4B-Base_fix3 \\
        --hf-model          Qwen/Qwen3.5-4B-Base \\
        --out               /shared_silo/.../roundtrip_conversion
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
from megatron.core.utils import init_method_normal, scaled_init_method_normal, unwrap_model  # noqa: E402


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
        "--hf-model",
        required=True,
        type=str,
        help=(
            "HuggingFace model id or local path for the original VL model "
            "(e.g. ``Qwen/Qwen3.5-4B-Base``). Used for both the VL bridge config "
            "and as the source of vision-tower weights."
        ),
    )
    parser.add_argument(
        "--out",
        required=True,
        type=str,
        help=(
            "Output directory for the final HuggingFace VL safetensors. "
            "An intermediate Megatron Bridge VL checkpoint is saved alongside "
            "at ``<out>_megatron_vl`` and can be deleted after a successful export."
        ),
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
    """Construct the text-only provider matching the VL bridge's architecture.

    Automatically selects between the dense and MoE text bridges by inspecting
    the HuggingFace model class name. Re-uses the same
    :class:`PreTrainedCausalLM` instance as the VL bridge so language settings
    are guaranteed to match.
    """
    # Derive the architecture name from the config (no weight load) rather than
    # from the instantiated model, to avoid a modelopt monkey-patch bug that
    # crashes AutoModelForCausalLM.from_pretrained on this container image.
    architectures = getattr(vl_bridge.hf_pretrained.config, "architectures", [])
    hf_class_name = architectures[0] if architectures else ""

    if "Moe" in hf_class_name or "MoE" in hf_class_name:
        from megatron.bridge.models.qwen.qwen3_5_dense_bridge import (
            Qwen35MoETextBridge,
        )
        text_bridge = Qwen35MoETextBridge()
    else:
        from megatron.bridge.models.qwen.qwen3_5_dense_bridge import (
            Qwen35DenseTextBridge,
        )
        text_bridge = Qwen35DenseTextBridge()

    provider = text_bridge.provider_bridge(vl_bridge.hf_pretrained)
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
    out_hf = args.out
    out_megatron_vl = str(Path(args.out).with_name(Path(args.out).name + "_megatron_vl"))

    logger.info("Trained text checkpoint:        %s", trained_text_iter_dir)
    logger.info("HuggingFace reference:          %s", args.hf_model)
    logger.info("Output (HuggingFace VL):        %s", out_hf)
    logger.info("Output (Megatron VL, intermediate): %s", out_megatron_vl)

    from megatron.bridge.training.model_load_save import temporary_distributed_context

    # ------------------------------------------------------------------
    # Phase 1: build VL model from HF, overwrite LM weights, save.
    # ------------------------------------------------------------------
    with temporary_distributed_context(backend="gloo"):
        vl_bridge = AutoBridge.from_hf_pretrained(args.hf_model)
        logger.info("Loaded HF reference into AutoBridge: %s", type(vl_bridge._model_bridge).__name__)

        # Build the text-only GPTModel FIRST, before to_megatron_model().
        # provide() is a low-level call that hits _initialize_affine_weight_gpu
        # with init_method=None when no Megatron CLI args have been parsed.
        # provide_distributed_model(use_cpu_initialization=True) routes through
        # CPU init which doesn't require init_method, matching what to_megatron_model()
        # does internally.
        text_provider = _build_text_provider(vl_bridge)
        # TransformerConfig.__post_init__ should set init_method from init_method_std,
        # but in the bridge dataclass inheritance chain it may be skipped. Set it
        # explicitly so module constructors (MLP, attention) that call not_none(config.init_method)
        # don't crash. The actual values are irrelevant: perform_initialization=False (below)
        # ensures no random init is applied -- weights are loaded from the checkpoint.
        if text_provider.init_method is None:
            _std = text_provider.init_method_std or 0.02
            text_provider.init_method = init_method_normal(_std)
        if text_provider.output_layer_init_method is None:
            _std = text_provider.init_method_std or 0.02
            text_provider.output_layer_init_method = scaled_init_method_normal(
                _std, text_provider.num_layers
            )
        # Skip random weight initialization: weights are loaded from the trained
        # checkpoint immediately after. This mirrors what to_megatron_model(load_weights=True)
        # does internally (auto_bridge.py line 1330) and avoids TypeError when init_method
        # is None (no Megatron CLI args have been parsed to supply it).
        text_provider.perform_initialization = False
        gpt_model_list = text_provider.provide_distributed_model(
            wrap_with_ddp=False,
            use_cpu_initialization=True,
        )
        gpt_model = unwrap_model(
            gpt_model_list if isinstance(gpt_model_list, list) else [gpt_model_list]
        )[0]
        gpt_sharded = gpt_model.sharded_state_dict()
        logger.info("Bare GPTModel built; %d sharded entries", len(gpt_sharded))

        loaded_gpt = dist_checkpointing.load(gpt_sharded, str(trained_text_iter_dir))
        # dist_checkpointing.load may return checkpoint-level metadata keys
        # (e.g. "checkpoint_version", "iteration") alongside model weights.
        model_param_keys = set(gpt_model.state_dict().keys())
        gpt_model.load_state_dict(
            {k: v for k, v in loaded_gpt.items() if k in model_param_keys},
            strict=True,
        )
        logger.info("Trained text checkpoint loaded into GPTModel (%d tensors).", len(loaded_gpt))

        # Now build the full VL model from the original HF weights. Vision-tower
        # weights come from HF directly; LM weights will be overwritten below.
        vl_models = vl_bridge.to_megatron_model(wrap_with_ddp=False, use_cpu_initialization=True)
        if not isinstance(vl_models, list):
            vl_models = [vl_models]
        logger.info(
            "Built VL model from HF weights: %d shard(s). "
            "Vision-tower weights are populated; LM weights will be overwritten.",
            len(vl_models),
        )

        # ------------------------------------------------------------------
        # Overwrite language_model.* in the VL model with the trained weights.
        # ------------------------------------------------------------------
        gpt_state = dict(gpt_model.state_dict())
        # Strip any "module." prefix that Megatron's DDP/wrapper might add.
        gpt_state = {
            (k[len("module.") :] if k.startswith("module.") else k): v for k, v in gpt_state.items()
        }

        n_copied = 0
        n_lm_keys_in_vl = 0
        n_skipped = 0
        for vl_model in vl_models:
            vl_state = dict(vl_model.state_dict())
            n_lm_keys_in_vl = sum(1 for k in vl_state if k.startswith("language_model."))
            for gpt_key, vl_key in _iter_lm_pairs(gpt_state, vl_state):
                src = gpt_state[gpt_key]
                dst = vl_state[vl_key]
                # TransformerEngine stores opaque _extra_state buffers as None in
                # state_dict(); skip them (they are not trainable weights).
                if src is None or dst is None:
                    continue
                if dst.shape != src.shape:
                    raise ValueError(
                        f"Shape mismatch on rewrap: gpt[{gpt_key}]={tuple(src.shape)} vs "
                        f"vl[{vl_key}]={tuple(dst.shape)}. Did the text bridge and VL bridge "
                        f"produce mismatched architectures (head_dim, num_query_groups, ...)?"
                    )
                # Both bridges use layernorm_zero_centered_gamma=True with
                # RMSNorm2ZeroCenteredRMSNormMapping for out_norm.weight, so
                # both store w_hf - 1. No per-key semantic adjustment needed.
                dst.copy_(src.to(dst.dtype))
                n_copied += 1

            untouched = [
                k
                for k in vl_state
                if k.startswith("language_model.") and k.removeprefix("language_model.") not in gpt_state
            ]
            for k in untouched:
                logger.info("[keep-hf] %s", k)
                n_skipped += 1
            untouched_gpt = [k for k in gpt_state if f"language_model.{k}" not in vl_state]
            for k in untouched_gpt:
                logger.warning("[orphan-trained] %s (no matching language_model.* in VL)", k)

        logger.info(
            "Overwrote %d LM tensors in VL model "
            "(%d total LM keys in VL; %d retained from original HF).",
            n_copied,
            n_lm_keys_in_vl,
            n_skipped,
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
            out_megatron_vl,
            hf_tokenizer_path=args.hf_model,
            low_memory_save=True,
        )
        logger.info("Wrote rewrapped Megatron VL checkpoint to %s", out_megatron_vl)

    # ------------------------------------------------------------------
    # Phase 2: HuggingFace export (opens its own dist context).
    # ------------------------------------------------------------------
    export_bridge = AutoBridge.from_hf_pretrained(args.hf_model)
    export_bridge.export_ckpt(
        out_megatron_vl,
        out_hf,
        show_progress=True,
        strict=args.strict_export,
    )
    logger.info("Wrote HuggingFace VL safetensors to %s", out_hf)

    if not args.keep_megatron_vl:
        breadcrumb = Path(out_megatron_vl) / "_REWRAP_INTERMEDIATE.md"
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
