#!/bin/bash

_target=${1:-'all'}
_workspaceFolder=$(pwd)

rm -rf ${_workspaceFolder}/build
mkdir ${_workspaceFolder}/build
cd ${_workspaceFolder}/build

# For conda users
cmake ..  -G Ninja --no-warn-unused-cli \
  -Wno-dev \
  -DCMAKE_MODULE_PATH="/usr/lib/llvm-16/lib/cmake/mlir;/usr/lib/llvm-16/lib/cmake/llvm;${_workspaceFolder}/third_party/mhlo/lib/cmake/mlir-hlo" \
  -DMLIR_DIR=/usr/lib/llvm-16/lib/cmake/mlir \
  -DLLVM_DIR=/usr/lib/llvm-16/lib/cmake/llvm \
  -DMLIR_TABLEGEN_EXEUTABLE:FILEPATH=/usr/lib/llvm-16/bin/mlir-tblgen \
  -DMHLO_DIR=${_workspaceFolder}/third_party/mhlo/lib/cmake/mlir-hlo \
  -DPOPLAR_SDK_PATH:PATH="${POPLAR_SDK_ENABLED}" \
  -DCMAKE_BUILD_TYPE:STRING=Debug \
  -DCMAKE_C_COMPILER:FILEPATH=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/g++

# for non-conda users.
# cmake ..  -G Ninja --no-warn-unused-cli \
#   -Wno-dev \
#   -DLLVM_ENABLE_LLD=ON \
#   -DPOPLAR_SDK_PATH:PATH="${POPLAR_SDK_ENABLED}" \
#   -DMLIR_DIR=${_workspaceFolder}/third_party/llvm/lib/cmake/mlir \
#   -DLLVM_DIR=${_workspaceFolder}/third_party/llvm/lib/cmake/llvm \
#   -DMHLO_DIR=${_workspaceFolder}/third_party/mhlo/lib/cmake/mlir-hlo \
#   -DCMAKE_MODULE_PATH="${_workspaceFolder}/third_party/llvm/lib/cmake/mlir;${_workspaceFolder}/third_party/llvm/lib/cmake/llvm;${_workspaceFolder}/third_party/mhlo/lib/cmake/mlir-hlo" \
#   -DMLIR_TABLEGEN_EXE=${_workspaceFolder}/third_party/llvm/bin/mlir-tblgen \
#   -DCMAKE_BUILD_TYPE:STRING=Debug \
#   -DCMAKE_C_COMPILER:FILEPATH=/usr/bin/gcc \
#   -DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/g++

# ninja
cmake \
  --build ${_workspaceFolder}/build \
  --config Debug --target ${_target}
