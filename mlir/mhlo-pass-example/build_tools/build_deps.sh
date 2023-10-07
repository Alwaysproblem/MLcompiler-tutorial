#!/bin/bash

_WORKSPACE=`pwd`

mkdir -p ${_WORKSPACE}/third_party

build_tools/build_mlir.sh ${_WORKSPACE}/external/llvm-project ${_WORKSPACE}/third_party/ || exit 1

build_tools/build_mhlo.sh ${_WORKSPACE}/external/mlir-hlo ${_WORKSPACE}/third_party/llvm ${_WORKSPACE}/third_party || exit 1

# fix cmake can not file include dir (mhlo)
mkdir -p ${_WORKSPACE}/third_party/mhlo/include
