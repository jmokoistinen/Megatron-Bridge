#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""
Export Megatron checkpoint to HuggingFace with custom tokenizer.

This script manually exports the checkpoint to ensure the correct tokenizer is used.
"""

import argparse
from pathlib import Path
from transformers import AutoTokenizer

from megatron.bridge import AutoBridge
from megatron.bridge.training.model_load_save import temporary_distributed_context


def export_with_custom_tokenizer(
    hf_model: str,
    megatron_path: str,
    hf_path: str,
    tokenizer_path: str,
    show_progress: bool = True,
):
    """
    Export a Megatron checkpoint to HuggingFace format with custom tokenizer.

    Args:
        hf_model: HuggingFace model ID for architecture reference (e.g., "Qwen/Qwen3-0.6B")
        megatron_path: Directory path where the Megatron checkpoint is stored
        hf_path: Directory path where the HuggingFace model will be saved
        tokenizer_path: HuggingFace tokenizer ID or path (e.g., "openai/gpt-oss-120b")
        show_progress: Display progress bar during weight export
    """
    print(f"🔄 Starting export with custom tokenizer")
    print(f"   Model architecture: {hf_model}")
    print(f"   Tokenizer: {tokenizer_path}")
    print(f"   Megatron checkpoint: {megatron_path}")
    print(f"   Output path: {hf_path}")

    hf_path = Path(hf_path)
    hf_path.mkdir(parents=True, exist_ok=True)

    # Create bridge from HF model (for architecture and config)
    print(f"\n📥 Loading reference model architecture from: {hf_model}")
    bridge = AutoBridge.from_hf_pretrained(hf_model, trust_remote_code=True)

    # Load custom tokenizer
    print(f"\n📝 Loading custom tokenizer from: {tokenizer_path}")
    custom_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    print(f"   ✓ Tokenizer loaded")
    print(f"   ✓ Tokenizer vocab size: {len(custom_tokenizer)}")
    print(f"   ✓ Tokenizer type: {type(custom_tokenizer).__name__}")

    # Export using CPU context
    with temporary_distributed_context(backend="gloo"):
        # Load the Megatron model
        print(f"\n📦 Loading Megatron checkpoint from: {megatron_path}")
        megatron_model = bridge.load_megatron_model(megatron_path, wrap_with_ddp=False)
        print(f"   ✓ Megatron model loaded")

        # Save config (from the reference model)
        print(f"\n💾 Saving model configuration")
        bridge.hf_pretrained.config.save_pretrained(hf_path)
        print(f"   ✓ Config saved to {hf_path}/config.json")

        # Save the CUSTOM tokenizer (not the one from the bridge)
        print(f"\n💾 Saving custom tokenizer")
        custom_tokenizer.save_pretrained(hf_path)
        print(f"   ✓ Tokenizer saved to {hf_path}/")

        # Verify tokenizer files were saved correctly
        tokenizer_files = list(hf_path.glob("tokenizer*")) + list(hf_path.glob("vocab*")) + list(hf_path.glob("merges.txt"))
        if tokenizer_files:
            print(f"   ✓ Tokenizer files created:")
            for tf in sorted(tokenizer_files):
                print(f"      - {tf.name}")

        # Update config.json with custom tokenizer's special token IDs and vocab size
        print(f"\n🔧 Updating config.json to match custom tokenizer")
        config = bridge.hf_pretrained.config

        # Update vocab size
        custom_vocab_size = len(custom_tokenizer)
        if config.vocab_size != custom_vocab_size:
            print(f"   - Updating vocab_size: {config.vocab_size} → {custom_vocab_size}")
            config.vocab_size = custom_vocab_size

        # Update special token IDs if they exist in the tokenizer
        if custom_tokenizer.bos_token_id is not None and hasattr(config, 'bos_token_id'):
            if config.bos_token_id != custom_tokenizer.bos_token_id:
                print(f"   - Updating bos_token_id: {config.bos_token_id} → {custom_tokenizer.bos_token_id}")
                config.bos_token_id = custom_tokenizer.bos_token_id

        if custom_tokenizer.eos_token_id is not None and hasattr(config, 'eos_token_id'):
            if config.eos_token_id != custom_tokenizer.eos_token_id:
                print(f"   - Updating eos_token_id: {config.eos_token_id} → {custom_tokenizer.eos_token_id}")
                config.eos_token_id = custom_tokenizer.eos_token_id

        if custom_tokenizer.pad_token_id is not None and hasattr(config, 'pad_token_id'):
            if not hasattr(config, 'pad_token_id') or config.pad_token_id != custom_tokenizer.pad_token_id:
                print(f"   - Updating pad_token_id: {getattr(config, 'pad_token_id', None)} → {custom_tokenizer.pad_token_id}")
                config.pad_token_id = custom_tokenizer.pad_token_id

        # Save the updated config
        config.save_pretrained(hf_path)
        print(f"   ✓ Config updated and saved")

        # Save generation config if it exists
        if hasattr(bridge.hf_pretrained, '_generation_config') and bridge.hf_pretrained._generation_config is not None:
            print(f"\n💾 Saving generation config")
            bridge.hf_pretrained.generation_config.save_pretrained(hf_path)
            print(f"   ✓ Generation config saved")

        # Save model weights
        print(f"\n💾 Saving model weights (this may take a while...)")
        bridge.save_hf_weights(
            megatron_model,
            hf_path,
            show_progress=show_progress,
            strict=False,
            merge_adapter_weights=True,
        )
        print(f"   ✓ Model weights saved")

    print(f"\n✅ Successfully exported model to: {hf_path}")
    print(f"   - Model weights: from {megatron_path}")
    print(f"   - Model config: from {hf_model}")
    print(f"   - Tokenizer: from {tokenizer_path}")

    # Verify the export
    if hf_path.exists():
        print("\n📁 Export structure:")
        for item in sorted(hf_path.iterdir()):
            if item.is_dir():
                print(f"   📂 {item.name}/")
            else:
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"   📄 {item.name} ({size_mb:.1f} MB)")

    print("\n🔍 Verify the tokenizer:")
    print("   from transformers import AutoTokenizer")
    print(f"   tokenizer = AutoTokenizer.from_pretrained('{hf_path}')")
    print(f"   print(tokenizer.name_or_path)  # Should be: {tokenizer_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export Megatron checkpoint to HuggingFace with custom tokenizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--hf-model",
        required=True,
        help="HuggingFace model ID for architecture reference (e.g., 'Qwen/Qwen3-0.6B')",
    )
    parser.add_argument(
        "--megatron-path",
        required=True,
        help="Directory path where the Megatron checkpoint is stored",
    )
    parser.add_argument(
        "--hf-path",
        required=True,
        help="Directory path where the HuggingFace model will be saved",
    )
    parser.add_argument(
        "--tokenizer-path",
        required=True,
        help="HuggingFace tokenizer ID or path (e.g., 'openai/gpt-oss-120b')",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar during export",
    )

    args = parser.parse_args()

    export_with_custom_tokenizer(
        hf_model=args.hf_model,
        megatron_path=args.megatron_path,
        hf_path=args.hf_path,
        tokenizer_path=args.tokenizer_path,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
