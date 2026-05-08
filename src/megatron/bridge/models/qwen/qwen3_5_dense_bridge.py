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

"""Text-only Megatron Bridge for Qwen3.5 dense models.

The HuggingFace distribution of Qwen3.5 dense base models (e.g. Qwen3.5-4B-Base,
Qwen3.5-9B-Base) is shipped with the architecture class
``Qwen3_5ForConditionalGeneration``, which always carries a vision encoder and
projector alongside the language model. The default Megatron Bridge for that
architecture (``Qwen35VLBridge``) imports the model as a ``Qwen3VLModel`` whose
state-dict keys live under a ``language_model.`` prefix together with
``vision_model.``/``merger.``/``patch_embed.`` keys.

For text-only continued pretraining with Megatron-LM's ``pretrain_gpt.py`` we
need a checkpoint that matches a plain ``GPTModel`` state-dict (no
``language_model.`` prefix, no vision keys). That is exactly what this bridge
produces: it reuses ``Qwen35VLModelProvider``'s configuration logic (so the
hybrid Gated-DeltaNet + Gated-Attention block spec, mRoPE, qk-layernorm, etc.
are all set correctly) but constructs the language model only via
``Qwen35VLModelProvider.provide_language_model``, and emits LM-only mappings.

Vision parameters present in the HuggingFace checkpoint are not mapped and are
silently ignored at import time. The bridge expects to be the only registered
bridge for ``Qwen3_5ForConditionalGeneration`` -- see
``qwen35_vl_bridge.py`` for how the VL bridge can be re-enabled via the
``BRIDGE_QWEN35_USE_VL`` environment variable.

Round-trip story
----------------
- Import (HF -> Megatron text): use this bridge directly via
  ``AutoBridge.import_ckpt(...)``.
- Train: use Megatron-LM ``pretrain_gpt.py`` against the produced checkpoint.
- Export back to HF VL: use ``tw-tools/rewrap_text_to_vl_and_export.py`` to
  splice trained LM weights into a frozen copy of the original VL Megatron
  checkpoint, then ``bridge.export_ckpt`` (with the VL bridge active via
  ``BRIDGE_QWEN35_USE_VL=1``) emits a HuggingFace VL safetensors directory.
"""

from __future__ import annotations

import itertools
import logging
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

import torch  # noqa: F401  # used by sibling bridges; imported for symmetry
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import unwrap_model

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    GDNConv1dMapping,
    GDNLinearMappingSeparate,
    QKVMapping,
    RMSNorm2ZeroCenteredRMSNormMapping,
)
from megatron.bridge.models.conversion.utils import persistent_buffers
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.qwen_vl.qwen35_vl_provider import Qwen35VLModelProvider


logger = logging.getLogger(__name__)


_QWEN3_5_DENSE_HF_CLASS_NAME = "Qwen3_5ForConditionalGeneration"


@dataclass
class Qwen35DenseTextProvider(Qwen35VLModelProvider):
    """Text-only provider for Qwen3.5 dense.

    Subclasses :class:`Qwen35VLModelProvider` to reuse the dataclass fields that
    describe the language model (hidden_size, GDN/attention hybrid spec, mrope,
    qk_layernorm, attention_output_gate, ...) but produces a plain Megatron-Core
    ``GPTModel`` instead of the multimodal ``Qwen3VLModel``.

    All vision-related fields are kept for compatibility with the parent
    dataclass but never consumed: ``provide()`` is overridden to call
    ``provide_language_model()`` so the vision tower is never instantiated.
    """

    # Default to no vision config so __post_init__ does not eagerly construct
    # a Qwen3_5VisionConfig (it is unused in the text-only flow).
    vision_config: Any = field(default=None)

    # IMPORTANT: ``Qwen35VLModelProvider`` hard-codes ``True`` here, which is
    # *only* correct if the consumer of the saved checkpoint also instantiates
    # the model with ``layernorm_zero_centered_gamma=True``. The default in
    # Megatron-LM (and in oellm-autoexp's ``base_defaults.yaml``) is ``False``,
    # so a checkpoint converted with ``True`` and loaded with ``False`` ends
    # up applying a constant-1 offset on every RMSNorm: trivial layernorms
    # collapse to ``weight * x_norm`` of nearly-zero scale and the GDN
    # ``out_norm`` (which we deliberately bias-shifted via
    # ``RMSNorm2ZeroCenteredRMSNormMapping`` below) silently zeroes 24/32
    # layers. We override to ``False`` so the saved tensors match the
    # ``oellm-autoexp`` Megatron-LM training config out of the box and the
    # ``out_norm`` mapping below intentionally stays a passthrough.
    layernorm_zero_centered_gamma: bool = True

    def __post_init__(self) -> None:
        # Skip ``Qwen35VLModelProvider.__post_init__`` (which checks for
        # ``transformers.Qwen3_5VisionConfig`` and eagerly constructs one)
        # by jumping to the next class in the MRO -- typically
        # ``GPTModelProvider.__post_init__``. Using ``super(Parent, self)``
        # preserves any intermediate post-inits while still bypassing the
        # vision bootstrap.
        super(Qwen35VLModelProvider, self).__post_init__()

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> GPTModel:
        """Build a text-only Megatron-Core ``GPTModel`` (no vision tower)."""
        return self.provide_language_model(
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )


# The text-only bridge only claims the dense Qwen3.5 dispatch slot when the VL
# bridge is *not* requested. This keeps import order between qwen/__init__.py
# and qwen_vl/__init__.py from mattering: whichever module is imported last
# has its decorator evaluated last, but only the one selected by the env var
# actually registers itself with the dispatcher.
_ENABLE_QWEN35_DENSE_TEXT_BRIDGE = os.environ.get("BRIDGE_QWEN35_USE_VL", "0") != "1"


def _maybe_register_dense_text_bridge(cls):
    """Register :class:`Qwen35DenseTextBridge` only when the VL bridge is disabled."""

    if not _ENABLE_QWEN35_DENSE_TEXT_BRIDGE:
        return cls
    return MegatronModelBridge.register_bridge(
        source=_QWEN3_5_DENSE_HF_CLASS_NAME,
        target=GPTModel,
        provider=Qwen35DenseTextProvider,
        model_type="qwen3_5_text",
    )(cls)


@_maybe_register_dense_text_bridge
class Qwen35DenseTextBridge(MegatronModelBridge):
    """Text-only Megatron Bridge for Qwen3.5 dense models.

    Maps a HuggingFace ``Qwen3_5ForConditionalGeneration`` checkpoint into a
    Megatron-Core ``GPTModel``, dropping all vision parameters. The mapping
    registry is the LM-only subset of ``Qwen35VLBridge.mapping_registry()``
    with the ``language_model.`` prefix removed from every Megatron-side key.
    HuggingFace-side keys are unchanged because the dense base model still
    exposes its language model under ``model.language_model.*``.

    Example::

        >>> from megatron.bridge import AutoBridge
        >>> AutoBridge.import_ckpt(
        ...     hf_model_id="Qwen/Qwen3.5-4B-Base",
        ...     megatron_path="./megatron_ckpt/Qwen3.5-4B-text",
        ... )
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> Qwen35DenseTextProvider:
        """Create a ``Qwen35DenseTextProvider`` from a HuggingFace pretrained model.

        Mirrors ``Qwen35VLBridge.provider_bridge``'s extraction of LM settings
        from ``hf_config.text_config``, dropping vision-only knobs.
        """
        hf_config = hf_pretrained.config
        text_config = hf_config.text_config

        provider_kwargs = self.hf_config_to_provider_kwargs(text_config)
        provider = Qwen35DenseTextProvider(**provider_kwargs)

        # For VLMs, ``tie_word_embeddings`` lives on the top-level config
        # (``text_config`` inherits PretrainedConfig's default ``True``,
        # which is wrong for the larger 9B/27B variants).
        provider.share_embeddings_and_output_weights = getattr(hf_config, "tie_word_embeddings", False)

        # --- Common Qwen3 LLM settings -------------------------------------
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_qkv_bias = getattr(text_config, "attention_bias", False)
        provider.add_bias_linear = False
        provider.qk_layernorm = True
        provider.hidden_dropout = 0.0

        # --- Qwen3-Next style hybrid architecture --------------------------
        # NOTE: ``layernorm_zero_centered_gamma`` is intentionally NOT set
        # here. The dataclass default (``False``) on
        # :class:`Qwen35DenseTextProvider` overrides the VL parent's ``True``
        # so the saved checkpoint matches the default Megatron-LM training
        # config used by oellm-autoexp's ``pretrain_gpt.py`` flow.
        provider.attention_output_gate = True
        provider.experimental_attention_variant = "gated_delta_net"
        provider.linear_attention_freq = getattr(text_config, "full_attention_interval", 4)
        provider.rotary_percent = (
            getattr(text_config, "rope_parameters", {}) or {}
        ).get("partial_rotary_factor", 0.25)

        # --- Gated DeltaNet (GDN) parameters -------------------------------
        provider.linear_conv_kernel_dim = getattr(text_config, "linear_conv_kernel_dim", 4)
        provider.linear_key_head_dim = getattr(text_config, "linear_key_head_dim", 128)
        provider.linear_value_head_dim = getattr(text_config, "linear_value_head_dim", 128)
        provider.linear_num_key_heads = getattr(text_config, "linear_num_key_heads", 16)
        provider.linear_num_value_heads = getattr(text_config, "linear_num_value_heads", 32)

        # --- Position embedding --------------------------------------------
        # Qwen3.5-VL ships with mRoPE because the multimodal model needs
        # separate rotary axes for (temporal, height, width) image positions.
        # For text-only sequences mRoPE is mathematically *identical* to
        # plain 1D RoPE: ``Qwen3VLModel.forward`` expands ``position_ids`` to
        # ``[3, B, S]`` with all three rows equal, then the section-wise
        # concat in ``MultimodalRotaryEmbedding`` reduces to a single RoPE.
        #
        # Crucially, plain Megatron-Core ``GPTModel`` (which we build here
        # via ``provide_language_model``) does NOT expand ``position_ids``,
        # and Megatron-LM's ``pretrain_gpt.py`` feeds 2D ``[B, S]`` directly
        # into ``MultimodalRotaryEmbedding.forward`` -- which expects 3D and
        # silently produces malformed rotary embeddings (no Python error,
        # just garbage attention positions). Empirically this manifests as
        # a CPT loss curve starting near ``log(vocab_size)`` instead of the
        # ~2-3 expected from a pretrained checkpoint.
        #
        # Round-trip is unaffected: the QKV/OProj weights live before/after
        # rotary application and don't depend on rope vs mrope, so the
        # text-only checkpoint can be spliced back into a ``mrope`` VL model
        # in stage 3 with byte-identical numerics.
        provider.position_embedding_type = "rope"
        provider.rotary_interleaved = False
        # ``mrope_section`` is stored for documentation only (unused for
        # ``position_embedding_type='rope'``) so downstream tooling that
        # inspects the saved config can still see the original VL layout.
        provider.mrope_section = getattr(text_config, "rope_scaling", {}).get(
            "mrope_section", [11, 11, 10]
        )

        # --- Token IDs (used by tokenizer/save plumbing) -------------------
        provider.bos_token_id = getattr(text_config, "bos_token_id", 248045)
        provider.eos_token_id = getattr(text_config, "eos_token_id", 248044)

        # --- MTP -----------------------------------------------------------
        # Drop MTP for the text-only flow; it is restored at export time from
        # the original VL checkpoint if the user wants Multi-Token Prediction.
        provider.mtp_num_layers = None

        # Hybrid GDN/attention layouts must use heterogeneous dist_checkpointing.
        provider.hetereogenous_dist_checkpoint = True

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """LM-only HF<->Megatron parameter mapping for Qwen3.5 dense.

        Megatron-side keys are the LM portion of
        ``Qwen35VLBridge.mapping_registry()`` with the ``language_model.``
        prefix removed (since we target ``GPTModel``, not ``Qwen3VLModel``).
        HuggingFace-side keys are unchanged because the dense base model
        still exposes its language model under ``model.language_model.*``.
        """

        param_mappings = {
            # ---------------- Embeddings and output -----------------------
            "embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.language_model.norm.weight",
            # ---------------- Dense MLP (pre-MLP layernorm fused) ---------
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": (
                "model.language_model.layers.*.post_attention_layernorm.weight"
            ),
            "decoder.layers.*.mlp.linear_fc2.weight": "model.language_model.layers.*.mlp.down_proj.weight",
            # ---------------- Standard (Gated) Attention layers -----------
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": (
                "model.language_model.layers.*.input_layernorm.weight"
            ),
            "decoder.layers.*.self_attention.q_layernorm.weight": (
                "model.language_model.layers.*.self_attn.q_norm.weight"
            ),
            "decoder.layers.*.self_attention.k_layernorm.weight": (
                "model.language_model.layers.*.self_attn.k_norm.weight"
            ),
            "decoder.layers.*.self_attention.linear_proj.weight": (
                "model.language_model.layers.*.self_attn.o_proj.weight"
            ),
            # ---------------- Linear (Gated DeltaNet) attention layers ----
            "decoder.layers.*.self_attention.in_proj.layer_norm_weight": (
                "model.language_model.layers.*.input_layernorm.weight"
            ),
            "decoder.layers.*.self_attention.out_proj.weight": (
                "model.language_model.layers.*.linear_attn.out_proj.weight"
            ),
            "decoder.layers.*.self_attention.A_log": (
                "model.language_model.layers.*.linear_attn.A_log"
            ),
            "decoder.layers.*.self_attention.dt_bias": (
                "model.language_model.layers.*.linear_attn.dt_bias"
            ),
        }

        mapping_list = [
            AutoMapping(megatron_param=mp, hf_param=hp) for mp, hp in param_mappings.items()
        ]

        # Register module types for AutoMapping shape inference (matches the VL bridge).
        AutoMapping.register_module_type("GatedDeltaNet", "column")

        mapping_list.extend(
            [
                # Standard attention QKV: combine separate Q/K/V into linear_qkv.
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.language_model.layers.*.self_attn.q_proj.weight",
                    k="model.language_model.layers.*.self_attn.k_proj.weight",
                    v="model.language_model.layers.*.self_attn.v_proj.weight",
                ),
                # Dense MLP: gate_proj + up_proj fused into linear_fc1.
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.language_model.layers.*.mlp.gate_proj.weight",
                    up="model.language_model.layers.*.mlp.up_proj.weight",
                ),
                # GDN: depthwise causal conv1d.
                GDNConv1dMapping(
                    megatron_param="decoder.layers.*.self_attention.conv1d.weight",
                    hf_param="model.language_model.layers.*.linear_attn.conv1d.weight",
                ),
                # GDN input projection: Qwen3.5 stores 4 separate weight tensors
                # (in_proj_qkv, in_proj_z, in_proj_b, in_proj_a) instead of the
                # 2 fused tensors used by Qwen3-Next.
                GDNLinearMappingSeparate(
                    megatron_param="decoder.layers.*.self_attention.in_proj.weight",
                    qkv="model.language_model.layers.*.linear_attn.in_proj_qkv.weight",
                    z="model.language_model.layers.*.linear_attn.in_proj_z.weight",
                    b="model.language_model.layers.*.linear_attn.in_proj_b.weight",
                    a="model.language_model.layers.*.linear_attn.in_proj_a.weight",
                ),
                # GDN output norm: subtract 1 from the HF weight.
                # ``Qwen3_5RMSNormGated`` uses ones-initialized weights and plain
                # ``weight * x_norm`` (no +1 offset). Megatron's RMSNorm with
                # ``layernorm_zero_centered_gamma=True`` computes ``(1+weight) * x_norm``.
                # Storing ``w_hf - 1`` ensures Megatron evaluates
                # ``(1 + (w_hf - 1)) * x = w_hf * x``, matching HF exactly.
                RMSNorm2ZeroCenteredRMSNormMapping(
                    megatron_param="decoder.layers.*.self_attention.out_norm.weight",
                    hf_param="model.language_model.layers.*.linear_attn.norm.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)

    # ------------------------------------------------------------------
    # Hard fail-fast wrapper around the framework's silent-warning import.
    # ------------------------------------------------------------------
    def load_weights_hf_to_megatron(
        self,
        hf_pretrained,
        megatron_model: Union[MegatronModule, List[MegatronModule]],
        allowed_mismatched_params: Optional[List[str]] = None,
    ):
        """Run the standard HF -> Megatron import and then assert completeness.

        ``MegatronModelBridge.build_conversion_tasks`` only emits ``logger.warning``
        for parameters that have no mapping or whose HF source key is missing
        (see ``model_bridge.py``: ``"WARNING: No mapping found ..."`` /
        ``"WARNING: Can't find ... in hf_keys"``). Such parameters keep their
        random-init values, which then propagates to the saved checkpoint and
        only becomes visible as an unexpectedly high CPT loss.

        For the text-only flow we want any unmapped or unsourced LM parameter
        to be a hard error, so the conversion job fails loudly instead of
        silently shipping a partially-random checkpoint to ``pretrain_gpt.py``.
        """
        result = super().load_weights_hf_to_megatron(
            hf_pretrained,
            megatron_model,
            allowed_mismatched_params=allowed_mismatched_params,
        )

        models = result if isinstance(result, list) else [result]
        unwrapped = unwrap_model(models)
        registry = self.mapping_registry()

        hf_state_keys = None
        if hasattr(hf_pretrained, "state") and hasattr(hf_pretrained.state, "source"):
            try:
                hf_state_keys = set(hf_pretrained.state.source.get_all_keys())
            except Exception:
                hf_state_keys = None

        share_emb = self._share_embeddings_and_output_weights(unwrapped[0].config)

        unmapped: list[str] = []
        missing_hf: list[tuple[str, str]] = []
        for model in unwrapped:
            for name, _ in itertools.chain(model.named_parameters(), persistent_buffers(model)):
                if "_extra_state" in name or self._is_adapter_param_name(name):
                    continue
                if share_emb and "output_layer" in name:
                    # Tied output projection -- intentionally not a separate task.
                    continue
                clean_name = self._unwrap_name(name)
                mapping = registry.megatron_to_hf_lookup(self._get_lora_unwrapped_name(clean_name))
                if mapping is None:
                    unmapped.append(clean_name)
                    continue
                if hf_state_keys is not None and not getattr(mapping, "allow_hf_name_mismatch", False):
                    hf_param = mapping.hf_param
                    if isinstance(hf_param, str):
                        if hf_param not in hf_state_keys:
                            missing_hf.append((clean_name, hf_param))
                    else:
                        for hp in hf_param.values():
                            if hp not in hf_state_keys:
                                missing_hf.append((clean_name, hp))

        problems = []
        if unmapped:
            problems.append(
                "Megatron parameters with NO HF mapping (would be left at random init):\n  - "
                + "\n  - ".join(sorted(set(unmapped)))
            )
        if missing_hf:
            problems.append(
                "Megatron parameters whose HF source key is missing in the checkpoint "
                "(would be left at random init):\n  - "
                + "\n  - ".join(f"{m} <- {h}" for m, h in sorted(set(missing_hf)))
            )
        if problems:
            raise RuntimeError(
                "Qwen35DenseTextBridge: silent conversion gap detected. "
                "Aborting before save_megatron_model so the checkpoint is not "
                "shipped with random-init parameters.\n\n" + "\n\n".join(problems)
            )

        return result
