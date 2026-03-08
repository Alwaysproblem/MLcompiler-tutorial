#!/bin/bash

./third_party/llvm/bin/mlir-opt sample/test.mlir \
  -canonicalize -cse \
  -lower-affine \
  -convert-scf-to-cf \
  -convert-arith-to-llvm \
  -convert-math-to-llvm \
  -finalize-memref-to-llvm \
  -convert-func-to-llvm \
  -reconcile-unrealized-casts \
  -o lowered-llvm-dialect.mlir

./third_party/llvm/bin/mlir-translate lowered-llvm-dialect.mlir --mlir-to-llvmir -o lowered.ll

clang++ -g -O0 lowered.ll cuda_shim/cuda_shim.cc \
  -I/usr/local/cuda/include \
  -L/usr/lib/x86_64-linux-gnu \
  -lcuda -ldl -lpthread -o cuda_shim/a.out
