#!/bin/bash

env_base_dir=$(conda info env --base)

_target=${1:-'all'}
_workspaceFolder=$(pwd)

rm -rf ${_workspaceFolder}/build
mkdir ${_workspaceFolder}/build
cd ${_workspaceFolder}/build

# For conda users
cmake ..  -G Ninja --no-warn-unused-cli \
  -Wno-dev \
  -DCMAKE_MODULE_PATH="${env_base_dir}/envs/mhlo/lib/cmake/mlir;${env_base_dir}/envs/mhlo/lib/cmake/llvm;${_workspaceFolder}/third_party/mhlo/lib/cmake/mlir-hlo" \
  -DMLIR_DIR=${env_base_dir}/envs/mhlo/lib/cmake/mlir \
  -DLLVM_DIR=${env_base_dir}/envs/mhlo/lib/cmake/llvm \
  -DMLIR_TABLEGEN_EXEUTABLE:FILEPATH=${env_base_dir}/envs/mhlo/bin/mlir-tblgen \
  -DMHLO_DIR=${_workspaceFolder}/third_party/mhlo/lib/cmake/mlir-hlo \
  -DPOPLAR_SDK_PATH:PATH="${POPLAR_SDK_ENABLED}" \
  -DCMAKE_BUILD_TYPE:STRING=Debug \
  -DCMAKE_C_COMPILER:FILEPATH=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/g++

# ninja
cmake \
  --build ${_workspaceFolder}/build \
  --config Debug --target ${_target}
