#!/bin/bash
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

if [[ $# -ne 2 ]] ; then
  echo "Usage: $0 <path/to/llvm> <build_dir>"
  exit 1
fi

# LLVM source
LLVM_SRC_DIR="$1"
build_dir="${LLVM_SRC_DIR}/build"
install_dir="$2/llvm"

if ! [ -f "$LLVM_SRC_DIR/llvm/CMakeLists.txt" ]; then
  echo "Expected the path to LLVM to be set correctly (got '$LLVM_SRC_DIR'): can't find CMakeLists.txt"
  exit 1
fi
echo "Using LLVM source dir: $LLVM_SRC_DIR"

# Setup directories.
echo "Building MLIR in $build_dir"
rm -rf "$build_dir"
mkdir -p "$build_dir"

echo "Installing MLIR in $install_dir"
rm -rf ${install_dir}
mkdir -p ${install_dir}

echo "Beginning build (commands will echo)"
set -x

cmake -GNinja \
  "-H$LLVM_SRC_DIR/llvm" \
  "-B$build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
  -DLLVM_ENABLE_LLD=ON \
  -DLLVM_ENABLE_BACKTRACES=OFF \
  -DLLVM_INCLUDE_UTILS=ON \
  -DCMAKE_INSTALL_PREFIX=${install_dir} \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_BUILD_UTILS=ON \
  -DLLVM_INCLUDE_TOOLS=ON \
  -DLLVM_BUILD_TOOLS=ON \
  -DLLVM_BUILD_LLVM_DYLIB=ON \
  -DLLVM_LINK_LLVM_DYLIB=ON

  # -DLLVM_ENABLE_RTTI=ON \
  # -DLLVM_ENABLE_LIBEDIT=OFF \
  # -DLLVM_ENABLE_BINDINGS=OFF \
  # -DLLVM_INCLUDE_DOCS=OFF \
  # -DLLVM_INCLUDE_TESTS=ON \
  # -DLLVM_INCLUDE_BENCHMARKS=OFF \
  # -DLLVM_ENABLE_BACKTRACES=ON \
  # -DLLVM_ENABLE_RTTI=ON \
  # -DLLVM_INCLUDE_EXAMPLES=OFF \
  # -DLLVM_ENABLE_ASSERTIONS=On
  # -DCMAKE_C_COMPILER=clang \
  # -DCMAKE_CXX_COMPILER=clang++ \
  # -DBUILD_SHARED_LIBS=ON \

cmake --build "$build_dir"
# cmake --build "$build_dir" --target check-mlir

pushd "$build_dir"
ninja install
popd
