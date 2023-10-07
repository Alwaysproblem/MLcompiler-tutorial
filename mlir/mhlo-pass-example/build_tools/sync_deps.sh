#!/bin/bash

_WORKSPACE=`pwd`

mkdir -p ${_WORKSPACE}/external

cd ${_WORKSPACE}/external

git clone -b $(cat ${_WORKSPACE}/build_tools/llvm_version.txt) https://github.com/llvm/llvm-project.git
git clone https://github.com/tensorflow/mlir-hlo

cd mlir-hlo
git checkout $(cat ${_WORKSPACE}/build_tools/mhlo_version.txt)
