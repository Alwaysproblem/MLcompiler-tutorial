#!/bin/bash

if [[ $# -ne 2 ]] ; then
  echo "Usage: $0 <path/to/mhlo> <build_dir>"
  exit 1
fi

_WORKSPACE=`pwd`

# mkdir -p ${_WORKSPACE}/third_party
# build_tools/build_mlir.sh ${_WORKSPACE}/external/llvm-project ${_WORKSPACE}/third_party/ || exit 1

cp ${_WORKSPACE}/external/llvm-project/build/bin/llvm-lit
   ${_WORKSPACE}/external/llvm-project/build/bin/not \
   ${_WORKSPACE}/external/llvm-project/build/bin/FileCheck \
   ${_WORKSPACE}/external/llvm-project/build/bin/count \
   /usr/local/bin

cp ${_WORKSPACE}/external/llvm-project/build/bin/llvm-lit \
   /usr/lib/llvm-16/bin/llvm-lit

MHLO_SRC_DIR="$1"
build_dir="$2/mhlo"

rm -rf "$build_dir"

# Setup directories.
echo "Building mhlo in $build_dir"
mkdir -p "$build_dir"

cmake -GNinja \
    "-H$MHLO_SRC_DIR" \
    "-B$build_dir" \
    -DLLVM_ENABLE_LLD=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=On \
    -DMLIR_DIR=/usr/lib/llvm-16/lib/cmake/mlir
    # -DBUILD_SHARED_LIBS=ON \

cmake --build "$build_dir" --target check-mlir-hlo

#  need to cp `FileCheck` `not` `count` to conda env bin dir.
# cp third_party/llvm/bin/not third_party/llvm/bin/FileCheck third_party/llvm/bin/count /root/anaconda3/envs/mhlo/bin
# cp ./third_party/llvm/bin/mlir-cpu-runner /root/anaconda3/envs/mhlo/bin

mkdir -p ${_WORKSPACE}/third_party/mhlo/include
