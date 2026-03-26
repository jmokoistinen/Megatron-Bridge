export PYTHONPATH=$PWD/3rdparty/Megatron-LM:$PYTHONPATH
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
#HF_MODEL="Qwen/Qwen3-30B-A3B-Base"
HF_MODEL="Qwen/Qwen3-0.6B"
python examples/conversion/convert_checkpoints.py import --hf-model $HF_MODEL --megatron-path $2 --hf-path $1
