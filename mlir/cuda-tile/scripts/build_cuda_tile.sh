#!/bin/bash

if [[ -f "/usr/bin/git" ]]; then
  WORKSPACEROOT=$(git rev-parse --show-toplevel)/mlir/cuda-tile || WORKSPACEROOT=`pwd`
fi

echo "Building cuda-tile IR in ${WORKSPACEROOT}/third_party/cuda-tile"

cd ${WORKSPACEROOT}/third_party/cuda-tile

git checkout -q -- .

rm -rf build

cmake -G Ninja -S ${WORKSPACEROOT}/third_party/cuda-tile -B build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DCUDA_TILE_ENABLE_BINDINGS_PYTHON=OFF \
  -DCUDA_TILE_ENABLE_TESTING=OFF \
  -DCMAKE_INSTALL_PREFIX=${WORKSPACEROOT}/third_party/cuda \
  -DCUDA_TILE_USE_LLVM_INSTALL_DIR=${WORKSPACEROOT}/third_party/llvm

cmake --build build 

cd build
cmake --install .
