#!/bin/bash

_target=${1:-'all'}

rm -rf build
mkdir build

_workspaceFolder=$(pwd)

cd build

# For non-conda users:
cmake .. -Wno-dev -G Ninja \
  -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE \
  -DCMAKE_BUILD_TYPE:STRING=Debug \
  -DCMAKE_C_COMPILER:FILEPATH=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/g++ \
  -DMLIR_DIR=${_workspaceFolder}/third_party/llvm/lib/cmake/mlir \
  -DLLVM_DIR=${_workspaceFolder}/third_party/llvm/lib/cmake/llvm \
  -DCMAKE_MODULE_PATH="${_workspaceFolder}/third_party/llvm/lib/cmake/mlir;${_workspaceFolder}/third_party/llvm/lib/cmake/llvm" \
  -DMLIR_TABLEGEN_EXE=${_workspaceFolder}/third_party/llvm/bin/mlir-tblgen

# ninja
cmake \
  --build ${_workspaceFolder}/build \
  --config Debug --target ${_target}
