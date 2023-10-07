#!/bin/bash

if [[ $# -ne 3 ]] ; then
  echo "Usage: $0 <path/to/mhlo> <path/to/llvm/build> <build_dir>"
  exit 1
fi

MHLO_SRC_DIR="$1"
LLVM_BUILD_DIR="$2"
build_dir="$3/mhlo"
# build_dir="${MHLO_SRC_DIR}/build"
# install_dir="$3/mhlo"

# Setup directories.
echo "Building mhlo in $build_dir"
rm -rf "$build_dir"
mkdir -p "$build_dir"

echo "Installing mhlo in $install_dir"
rm -rf ${install_dir}
mkdir -p ${install_dir}

cmake -GNinja \
    "-H$MHLO_SRC_DIR" \
    "-B$build_dir" \
    -DLLVM_ENABLE_LLD=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=On \
    -DLLVM_EXTERNAL_LIT=${LLVM_BUILD_DIR}/../../external/llvm-project/build/bin/llvm-lit \
    -DMLIR_DIR=${LLVM_BUILD_DIR}/lib/cmake/mlir
    # -DCMAKE_INSTALL_PREFIX=${install_dir} \

cmake --build "$build_dir" --target check-mlir-hlo

# cd "$build_dir"
# ninja install
