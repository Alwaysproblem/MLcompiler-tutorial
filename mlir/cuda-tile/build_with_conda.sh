#!/bin/bash

_target=${1:-'all'}

rm -rf build
mkdir build

_workspaceFolder=$(pwd)

cd build


cmake ..  -G Ninja --no-warn-unused-cli \
  -Wno-dev \
  -DCMAKE_MODULE_PATH="/root/miniconda3/envs/mlir/lib/cmake/mlir;/root/miniconda3/envs/mlir/lib/cmake/llvm" \
  -DMLIR_TABLEGEN_EXE:FILEPATH=/root/miniconda3/envs/mlir/bin/mlir-tblgen \
  -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE \
  -DCMAKE_BUILD_TYPE:STRING=Debug \
  -DCMAKE_C_COMPILER:FILEPATH=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/g++

# ninja
cmake \
  --build ${_workspaceFolder}/build \
  --config Debug --target ${_target}
