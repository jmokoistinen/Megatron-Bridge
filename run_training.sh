#ls ../Meg_brid/

cd /shared_silo/scratch/mika/experiments/Megatron-Bridge

source ~/environment.sh
export CONTAINER=/shared_silo/scratch/mika/containers/build/rocm_primus_v25.11_transformers-5.5.4_linear_FA2.sif

export PYTHONPATH=$(pwd)/python-packages:$(pwd)/3rdparty/Megatron-LM:$(pwd)/src:$PYTHONPATH #

apptainer exec $CONTAINER python convert.py



# apptainer shell $CONTAINER


# # Force partial import to cache submodules before the __init__ fails
# try:
#     import modelopt.torch.quantization.utils
# except ImportError:
#     pass

# from megatron.bridge.models.conversion import AutoBridge

# AutoBridge.import_ckpt(
#     hf_model_id="/shared_silo/scratch/mika/models/hf_models/qwen3.5-0.8b",
#     megatron_path="/shared_silo/scratch/mika/models/megatron_ckpt/qwen3.5-0.8b",
#     tp=1, pp=1,
# )






# AutoBridge.import_ckpt(
#     hf_model_id="/shared_silo/scratch/mika/models/hf_models/qwen3-0.6b",
#     megatron_path="/shared_silo/scratch/mika/models/megatron_ckpt/qwen3-0.6b",
#     tp=1, 
#     pp=1, 
# )


# #export CONTAINER=/shared_silo/scratch/containers/rocm_primus_v25.11_transformers-5.5.0.sif
# #export CONTAINER=/shared_silo/scratch/containers/rocm_primus_v25.11_transformers-4.5.7_linear_FA.sif

# #export PYTHONPATH=$(pwd)/3rdparty/Megatron-LM:$(pwd)/src:$PYTHONPATH #

# #singularity exec -B $PWD $CONTAINER pip install --target python-packages onnx 
# #singularity exec -B $PWD $CONTAINER pip install --target python-packages omegaconf pulp 
# #singularity exec -B $PWD $CONTAINER pip install --target python-packages flash-linear-attention #--no-deps
# #singularity exec -B $PWD $CONTAINER pip install --target python-packages transformers



# #done at start.py
# import sys, os
# #sys.path.insert(0, '/shared_silo/scratch/mika/experiments/Megatron-Bridge/src')
# #sys.path.insert(0, '/shared_silo/scratch/mika/experiments/Megatron-Bridge/3rdparty/Megatron-LM')
# sys.path.insert(0, os.getcwd()+'/src')
# sys.path.insert(0, os.getcwd()+'/3rdparty/Megatron-LM')
# import sys
