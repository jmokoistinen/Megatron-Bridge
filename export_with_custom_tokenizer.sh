#!/bin/bash
# Export Megatron checkpoint to HuggingFace format with GPT-OSS tokenizer

export PYTHONPATH=$PWD/3rdparty/Megatron-LM:$PYTHONPATH
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

HF_MODEL="Qwen/Qwen3-0.6B"
TOKENIZER="openai/gpt-oss-120b"

# Use v2 which manually exports each component to ensure correct tokenizer
python export_custom_tokenizer_v2.py \
    --hf-model $HF_MODEL \
    --tokenizer-path $TOKENIZER \
    --megatron-path $1 \
    --hf-path $2
