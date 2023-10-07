#!/bin/bash

_target=${1:-'all'}
_workspaceFolder=$(pwd)

rm -rf ${_workspaceFolder}/build
mkdir ${_workspaceFolder}/build
cd ${_workspaceFolder}/build

# For conda users
# cmake ..  -G Ninja --no-warn-unused-cli \
#   -Wno-dev \
#   -DCMAKE_MODULE_PATH="/root/anaconda3/envs/mhlo/lib/cmake/mlir;/root/anaconda3/envs/mhlo/lib/cmake/llvm;${_workspaceFolder}/third_party/mhlo/lib/cmake/mlir-hlo" \
#   -DMLIR_DIR=/root/anaconda3/envs/mhlo/lib/cmake/mlir \
#   -DLLVM_DIR=/root/anaconda3/envs/mhlo/lib/cmake/llvm \
#   -DMLIR_TABLEGEN_EXEUTABLE:FILEPATH=/root/anaconda3/envs/mhlo/bin/mlir-tblgen \
#   -DMHLO_DIR=${_workspaceFolder}/third_party/mhlo/lib/cmake/mlir-hlo \
#   -DPOPLAR_SDK_PATH:PATH="${POPLAR_SDK_ENABLED}" \
#   -DCMAKE_BUILD_TYPE:STRING=Debug \
#   -DCMAKE_C_COMPILER:FILEPATH=/usr/bin/gcc \
#   -DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/g++

# for non-conda users.
cmake ..  -G Ninja --no-warn-unused-cli \
  -Wno-dev \
  -DPOPLAR_SDK_PATH:PATH="${POPLAR_SDK_ENABLED}" \
  -DCMAKE_MODULE_PATH="${_workspaceFolder}/third_party/llvm/lib/cmake/mlir;${_workspaceFolder}/third_party/llvm/lib/cmake/llvm;${_workspaceFolder}/third_party/mhlo/lib/cmake/mlir-hlo" \
  -DMLIR_DIR=${_workspaceFolder}/third_party/llvm/lib/cmake/mlir \
  -DLLVM_DIR=${_workspaceFolder}/third_party/llvm/lib/cmake/llvm \
  -DMHLO_DIR=${_workspaceFolder}/third_party/mhlo/lib/cmake/mlir-hlo \
  -DMLIR_TABLEGEN_EXE=${_workspaceFolder}/third_party/llvm/bin/mlir-tblgen \
  -DCMAKE_BUILD_TYPE:STRING=Debug \
  -DCMAKE_C_COMPILER:FILEPATH=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/g++
  # -DLLVM_ENABLE_LLD=ON \
  # -DCMAKE_MODULE_PATH="${_workspaceFolder}/external/llvm-project/build/lib/cmake/mlir;${_workspaceFolder}/external/llvm-project/build/lib/cmake/llvm;${_workspaceFolder}/third_party/mhlo/lib/cmake/mlir-hlo" \

# ninja
cmake \
  --build ${_workspaceFolder}/build \
  --config Debug --target ${_target}
