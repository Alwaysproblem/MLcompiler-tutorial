# env preparation
# conda create -n pytorch-build python=3.10  cuda-tools=12.2.2 cuda-toolkit=12.2.2 cuda-nvcc_linux-64=12.2.2 cuda-libraries-dev=12.2.2 cuda-driver-dev cuda=12.2.2 cuda-compiler=12.2.2 cudnn cuda-gdb cuda-cudart-dev cuda-cudart-static -c nvidia -y 

export LIBRARY_PATH=/root/miniconda3/envs/pytorch-build/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/root/miniconda3/envs/pytorch-build/lib:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

DEBUG_CUDA=1 DEBUG=1 CUDA_NVCC_EXECUTABLE=/root/miniconda3/envs/pytorch-build/bin/nvcc python3 setup.py develop

