#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Smoke-test for the text-only Qwen3.5 dense Megatron checkpoint.

Run this *immediately after* ``import_hf_to_megatron_tw_text.sh`` and before
starting a multi-hour CPT job. It pinpoints silent conversion failures by
checking three independent sources of evidence:

1. Bridge selection: confirm ``AutoBridge.from_hf_pretrained(...)`` selects
   ``Qwen35DenseTextBridge`` (and not the VL bridge) under the current env.
2. Coverage: every parameter / persistent buffer of a freshly-built GPTModel
   has a mapping registered, and every mapped HF source key exists in the HF
   safetensors index. Without this check the framework only emits a
   ``logger.warning`` and silently leaves random init values in the saved
   checkpoint -- which manifests downstream as a CPT loss starting near
   ``log(vocab_size)`` instead of ~2-3.
3. Numerics: load a few critical tensors back from the saved
   ``torch_dist`` checkpoint and compare against the reference HF tensors
   (after applying the bridge transform). Any wrong layer-type mapping
   (GDN vs full-attention misalignment), wrong QKV interleaving, missing
   ``attention_output_gate`` slice, etc. shows up here as a large delta
   rather than as a correctly-shaped but numerically-garbage tensor.

The script does **not** require any GPU and runs in a few minutes for the 4B.

Example::

    apptainer exec --rocm \\
        -B /shared_silo/scratch:/shared_silo/scratch:rw \\
        --env PYTHONPATH=$MEGATRON_LM_PATH:$BRIDGE_ROOT/src \\
        --env BRIDGE_QWEN35_USE_VL=0 \\
        $CONTAINER \\
        python tw-tools/verify_text_ckpt.py \\
            --hf-model       Qwen/Qwen3.5-4B-Base \\
            --megatron-path  /shared_silo/.../megatron_ckpt/Qwen3.5-4B-text \\
            --num-tensors-to-spotcheck 8
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import sys
from pathlib import Path
from typing import Iterable

# Must be set before any megatron.bridge imports so the text-only bridge is
# selected by ``AutoBridge.from_hf_pretrained``.
os.environ.setdefault("BRIDGE_QWEN35_USE_VL", "0")

import torch  # noqa: E402

from megatron.bridge import AutoBridge  # noqa: E402
from megatron.bridge.models.conversion.utils import persistent_buffers  # noqa: E402
from megatron.core import dist_checkpointing  # noqa: E402
from megatron.core.utils import unwrap_model  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("verify-text-ckpt")


_ITER_DIR_PREFIX = "iter_"

# A handful of representative LM keys that tile across attention types and
# MLP. If any of these regress numerically, the conversion is broken even
# if the coverage check passes.
_DEFAULT_SPOTCHECK_KEYS = (
    # Embedding (no transform on import).
    "embedding.word_embeddings.weight",
    # Final layernorm (zero-centered RMSNorm => HF stores normal weights, no transform here).
    "decoder.final_layernorm.weight",
    # Layer 0 (GDN) MLP gate+up fused: tests GatedMLPMapping.
    "decoder.layers.0.mlp.linear_fc1.weight",
    # Layer 0 (GDN) MLP down: simple AutoMapping passthrough.
    "decoder.layers.0.mlp.linear_fc2.weight",
    # Layer 0 (GDN) GDN out_norm: tests RMSNorm2ZeroCenteredRMSNormMapping (subtract 1).
    "decoder.layers.0.self_attention.out_norm.weight",
    # Layer 0 (GDN) conv1d: tests GDNConv1dMapping.
    "decoder.layers.0.self_attention.conv1d.weight",
    # Layer 3 (full attention) QKVG fused: tests QKVMapping with attention_output_gate=True.
    "decoder.layers.3.self_attention.linear_qkv.weight",
    # Layer 3 (full attention) o_proj: simple AutoMapping passthrough.
    "decoder.layers.3.self_attention.linear_proj.weight",
    # Layer 3 (full attention) q_layernorm: tests AutoMapping for qk_layernorm.
    "decoder.layers.3.self_attention.q_layernorm.weight",
)


def _resolve_iter_dir(megatron_path: Path) -> Path:
    """Pick the highest ``iter_*`` directory under ``megatron_path`` if present."""
    if (megatron_path / "common.pt").exists() or (megatron_path / "metadata.json").exists():
        # User already pointed at an iteration directory.
        return megatron_path
    candidates = [p for p in megatron_path.iterdir() if p.is_dir() and p.name.startswith(_ITER_DIR_PREFIX)]
    if not candidates:
        raise FileNotFoundError(
            f"No 'iter_*' subdirectory found under {megatron_path} and the path "
            f"itself does not look like a Megatron checkpoint dir."
        )
    return max(candidates, key=lambda p: int(p.name.replace(_ITER_DIR_PREFIX, "")))


def _check_bridge_selection(hf_model: str) -> AutoBridge:
    """Section 1: verify the right bridge is dispatched."""
    logger.info("=== [1/3] Bridge selection check ===")
    bridge = AutoBridge.from_hf_pretrained(hf_model)
    bridge_cls_name = type(bridge._model_bridge).__name__
    logger.info("Selected bridge class: %s", bridge_cls_name)
    logger.info(
        "Selected provider class: %s",
        type(bridge._model_bridge.provider_bridge(bridge._provider_bridge_input)).__name__,
    )
    if bridge_cls_name != "Qwen35DenseTextBridge":
        raise SystemExit(
            f"Expected Qwen35DenseTextBridge but AutoBridge selected {bridge_cls_name}. "
            f"Is BRIDGE_QWEN35_USE_VL set to 1? Current value: "
            f"'{os.environ.get('BRIDGE_QWEN35_USE_VL', '<unset>')}'"
        )
    return bridge


def _iter_lm_param_names(model) -> Iterable[str]:
    for name, _ in itertools.chain(model.named_parameters(), persistent_buffers(model)):
        if "_extra_state" in name:
            continue
        yield name


def _check_mapping_coverage(bridge: AutoBridge) -> None:
    """Section 2: every model param must have a mapping AND a present HF source.

    This is exactly what the bridge's hard fail-fast wrapper (added to
    ``Qwen35DenseTextBridge.load_weights_hf_to_megatron``) enforces at import
    time, but we also assert it here so the user can detect missing mappings
    *without* re-running the multi-minute import.
    """
    logger.info("=== [2/3] Coverage check (no random-init silent failures) ===")

    # Build the model fresh, no weight loading -- we only need its parameter set.
    provider = bridge.to_megatron_provider(load_weights=False)
    if hasattr(provider, "finalize"):
        provider.finalize()
    models = provider.provide_distributed_model(wrap_with_ddp=False, use_cpu_initialization=True)

    unwrapped = unwrap_model(models)
    registry = bridge._model_bridge.mapping_registry()
    share_emb = bridge._model_bridge._share_embeddings_and_output_weights(unwrapped[0].config)

    hf_state_keys = None
    src = getattr(bridge.hf_pretrained, "state", None)
    src = getattr(src, "source", None) if src is not None else None
    if src is not None:
        try:
            hf_state_keys = set(src.get_all_keys())
        except Exception as exc:  # pragma: no cover -- best-effort check
            logger.warning("Could not enumerate HF state keys: %s", exc)
            hf_state_keys = None

    unmapped: list[str] = []
    missing_hf: list[tuple[str, str]] = []
    n_total = 0
    for model in unwrapped:
        for name in _iter_lm_param_names(model):
            n_total += 1
            if share_emb and "output_layer" in name:
                continue  # tied output projection -- intentional skip
            clean = bridge._model_bridge._unwrap_name(name)
            mapping = registry.megatron_to_hf_lookup(
                bridge._model_bridge._get_lora_unwrapped_name(clean)
            )
            if mapping is None:
                unmapped.append(clean)
                continue
            if hf_state_keys is not None and not getattr(mapping, "allow_hf_name_mismatch", False):
                hf_param = mapping.hf_param
                if isinstance(hf_param, str):
                    if hf_param not in hf_state_keys:
                        missing_hf.append((clean, hf_param))
                else:
                    for hp in hf_param.values():
                        if hp not in hf_state_keys:
                            missing_hf.append((clean, hp))

    logger.info(
        "Inspected %d parameters/buffers (skipped *_extra_state*; tied=%s).",
        n_total,
        share_emb,
    )

    if unmapped:
        logger.error("Megatron parameters with NO mapping (would be left at random init):")
        for n in sorted(set(unmapped)):
            logger.error("  - %s", n)
    if missing_hf:
        logger.error("Megatron parameters whose HF source key is missing:")
        for m, h in sorted(set(missing_hf)):
            logger.error("  - %s  <-  %s", m, h)
    if unmapped or missing_hf:
        raise SystemExit(
            "Coverage check FAILED: the saved checkpoint will have random-init "
            "parameters. Re-import after fixing the bridge mappings."
        )
    logger.info("Coverage check PASSED.")


def _spotcheck_numerics(
    bridge: AutoBridge,
    iter_dir: Path,
    spotcheck_keys: tuple[str, ...],
) -> None:
    """Section 3: load critical tensors and verify numerical equivalence with HF.

    For each Megatron key in ``spotcheck_keys`` we:
      * read the saved tensor from the ``torch_dist`` checkpoint;
      * recompute the expected tensor by running the bridge's
        ``hf_to_megatron`` transform on the corresponding HF tensors;
      * assert max-abs-diff is within ``1e-4`` (bf16 round-trip tolerance).
    """
    logger.info("=== [3/3] Numerical spot-check on %d critical tensors ===", len(spotcheck_keys))

    # Build a fresh GPTModel and populate it with HF weights via the bridge.
    # This mirrors what import_ckpt does internally, except we keep the model
    # in CPU memory and never call save_megatron_model.
    fresh = bridge.to_megatron_model(wrap_with_ddp=False, use_cpu_initialization=True)
    fresh_unwrapped = unwrap_model(fresh if isinstance(fresh, list) else [fresh])[0]
    # TransformerEngine _extra_state entries can be None before the first forward;
    # skip them — the spotcheck keys never reference _extra_state.
    expected_state = {
        k: v.detach().cpu().clone()
        for k, v in fresh_unwrapped.state_dict().items()
        if v is not None
    }

    # Reuse the same model instance to load the on-disk checkpoint.
    # expected_state is already captured above; loading overwrites the model
    # in-place so we can then read saved_state from it.  Avoids a second
    # model build (to_megatron_provider(load_weights=False) passes None as
    # init_method which triggers TypeError in CPU-init mode).
    sharded = fresh_unwrapped.sharded_state_dict()
    loaded = dist_checkpointing.load(sharded, str(iter_dir))
    # dist_checkpointing.load may include checkpoint-level metadata keys
    # (e.g. "checkpoint_version", "iteration", "content_metadata") that are
    # not model parameters -- strip them before calling load_state_dict.
    model_param_keys = set(fresh_unwrapped.state_dict().keys())
    loaded_model_only = {k: v for k, v in loaded.items() if k in model_param_keys}
    fresh_unwrapped.load_state_dict(loaded_model_only, strict=True)
    saved_state = {
        k: v.detach().cpu().clone()
        for k, v in fresh_unwrapped.state_dict().items()
        if v is not None
    }

    failures = 0
    tol = 1e-4
    for key in spotcheck_keys:
        if key not in saved_state:
            logger.warning("[SKIP] %s missing from saved checkpoint", key)
            continue
        if key not in expected_state:
            logger.warning("[SKIP] %s missing from freshly-converted reference", key)
            continue
        a = saved_state[key].float()
        b = expected_state[key].float()
        if a.shape != b.shape:
            logger.error("[FAIL] %s shape mismatch saved=%s vs expected=%s", key, a.shape, b.shape)
            failures += 1
            continue
        diff = (a - b).abs().max().item()
        if diff > tol:
            logger.error(
                "[FAIL] %s max-abs-diff=%.3e (saved mean=%.3e, expected mean=%.3e)",
                key,
                diff,
                a.mean().item(),
                b.mean().item(),
            )
            failures += 1
        else:
            logger.info("[PASS] %s max-abs-diff=%.3e", key, diff)

    # Also assert that the saved tensors are not at the init distribution
    # (mean ~0, std ~init_method_std). For CPT we expect non-trivial stats
    # from the pretrained model.
    init_std = float(getattr(fresh_unwrapped.config, "init_method_std", 0.02))
    sus_keys: list[str] = []
    for key in spotcheck_keys:
        if key not in saved_state:
            continue
        t = saved_state[key].float()
        # Layer-norm/scale tensors trivially fail this; skip them.
        if "layer_norm_weight" in key or key.endswith(("layernorm.weight", "norm.weight", "out_norm.weight")):
            continue
        std = t.std().item()
        # Random init has std ~= init_std; pretrained should be different.
        if abs(std - init_std) < 1e-3:
            sus_keys.append(f"{key} (std={std:.5e} ~= init_std={init_std})")
    if sus_keys:
        logger.warning(
            "Suspicious tensors that look like random init (std too close to init_method_std=%s):",
            init_std,
        )
        for k in sus_keys:
            logger.warning("  - %s", k)

    if failures:
        raise SystemExit(
            f"Numerical spot-check FAILED on {failures}/{len(spotcheck_keys)} keys. "
            f"The bridge mappings produced different values than the saved checkpoint. "
            f"Re-investigate the conversion before training."
        )

    logger.info("Numerical spot-check PASSED.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--hf-model",
        required=True,
        help="HuggingFace model id (e.g. Qwen/Qwen3.5-4B-Base) used for the conversion.",
    )
    parser.add_argument(
        "--megatron-path",
        required=True,
        help="Output directory of import_hf_to_megatron_tw_text.sh (the dir containing iter_*).",
    )
    parser.add_argument(
        "--spotcheck-keys",
        nargs="*",
        default=None,
        help="Optional override list of Megatron keys to spot-check. Default is a small "
        "set covering all transform types (QKVG, GatedMLP, GDN conv1d, GDN out_norm).",
    )
    parser.add_argument(
        "--skip-numeric-spotcheck",
        action="store_true",
        help="Only run the bridge selection + coverage checks (cheap, CPU-only).",
    )
    args = parser.parse_args()

    megatron_path = Path(args.megatron_path).resolve()
    if not megatron_path.exists():
        raise SystemExit(f"--megatron-path does not exist: {megatron_path}")
    iter_dir = _resolve_iter_dir(megatron_path)
    logger.info("Using checkpoint iteration dir: %s", iter_dir)

    bridge = _check_bridge_selection(args.hf_model)
    _check_mapping_coverage(bridge)

    if not args.skip_numeric_spotcheck:
        spotcheck_keys = tuple(args.spotcheck_keys) if args.spotcheck_keys else _DEFAULT_SPOTCHECK_KEYS
        _spotcheck_numerics(bridge, iter_dir, spotcheck_keys)
    else:
        logger.info("Skipping numerical spot-check (per --skip-numeric-spotcheck).")

    logger.info("All checks PASSED. Text-only checkpoint at %s looks good.", megatron_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
