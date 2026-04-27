
#srun --account=amd-tw-verification --ntasks=1 --gres=gpu:mi325:1 --time=3:00:00 --mem=64G --pty bash
 
cd /shared_silo/scratch/mika/experiments/Megatron-Bridge

source ~/environment.sh
export CONTAINER=/shared_silo/scratch/containers/build-rocm_primus_v25.11_transformers-5.5.4_linear_FA/rocm_primus_v25.11_transformers-5.5.4_linear_FA.sif

export PYTHONPATH=$(pwd)/python-packages:$(pwd)/3rdparty/Megatron-LM:$(pwd)/src:$PYTHONPATH #

#download model (might work without, using hf folder directly)
#hf download Qwen/Qwen3.5-35B-A3B --local-dir /shared_silo/scratch/mika/models/hf_models/qwen3.5-35B-A3B

apptainer shell $CONTAINER

# python 
# Force partial import to cache submodules before the __init__ fails
try:
    import modelopt.torch.quantization.utils
except ImportError:
    pass

from megatron.bridge.models.conversion import AutoBridge

AutoBridge.import_ckpt(
    hf_model_id="/shared_silo/scratch/mika/models/hf_models/qwen3.5-0.8b",
    megatron_path="/shared_silo/scratch/mika/models/megatron_ckpt/qwen3.5-0.8b",
    tp=1, pp=1,
)


# AutoBridge.import_ckpt(
#     hf_model_id="/shared_silo/scratch/mika/models/hf_models/qwen3-0.6b",
#     megatron_path="/shared_silo/scratch/mika/models/megatron_ckpt/qwen3-0.6b",
#     tp=1, 
#     pp=1, 
# )

#tokenize 

#training