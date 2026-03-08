export MLIR_RUNNER_UTILS=`pwd`/../third_party/llvm/lib/libmlir_runner_utils.so
export MLIR_CUDA_RUNTIME=`pwd`/../third_party/llvm/lib/libmlir_cuda_runtime.so

# Set this to your GPU arch, e.g. sm_120 for RTX 50xx (if your toolchain supports it).
export CUDA_ARCH=${CUDA_ARCH:-sm_120}

rm -rf example-nvvm.mlir example.ll


../third_party/llvm/bin/mlir-opt gpu.mlir -cse \
  -gpu-lower-to-nvvm-pipeline="cubin-chip=sm_120 opt-level=3" \
  --reconcile-unrealized-casts -cse -o example-nvvm.mlir


# ../third_party/llvm/bin/mlir-opt gpu.mlir \
#   --pass-pipeline="builtin.module(
#     nvvm-attach-target{chip=sm_80 O=3},
#     gpu.module(convert-gpu-to-nvvm),
#     gpu-module-to-binary,
#     lower-host-to-llvm
#   )" \
#   -o example-nvvm.mlir


# ../third_party/llvm/bin/mlir-opt gpu.mlir \
#   -gpu-lower-to-nvvm-pipeline="cubin-chip=sm_80 opt-level=3" \
#   -reconcile-unrealized-casts \
#   -canonicalize -cse \
#   -o example-nvvm.mlir

# ../third_party/llvm/bin/mlir-opt gpu.mlir \
#   --convert-scf-to-cf \
#   --convert-index-to-llvm \
#   --convert-arith-to-llvm \
#   --finalize-memref-to-llvm \
#   --convert-cf-to-llvm \
#   --convert-func-to-llvm \
#   --convert-to-llvm \
#   --reconcile-unrealized-casts \
#   -canonicalize -cse \
#   -o example-nvvm.mlir


# ../third_party/llvm/bin/mlir-opt gpu.mlir \
#   -gpu-lower-to-nvvm-pipeline="cubin-chip=sm_120 opt-level=3" \
#   -reconcile-unrealized-casts \
#   -o example-nvvm.mlir

# --gpu-to-llvm="use-bare-pointers-for-kernels=1 intersperse-sizes-for-kernels=1"
# ../third_party/llvm/bin/mlir-opt gpu.mlir \
#   --pass-pipeline="builtin.module(
#     nvvm-attach-target{chip=sm_89 O=3},
#     gpu.module(convert-gpu-to-nvvm),
#     convert-scf-to-cf,
#     convert-index-to-llvm,
#     convert-arith-to-llvm,
#     convert-math-to-llvm,
#     convert-func-to-llvm,
#     gpu-to-llvm,
#     convert-cf-to-llvm,
#     finalize-memref-to-llvm,
#     gpu-module-to-binary,
#     reconcile-unrealized-casts
#   )" -o example-nvvm.mlir


# ../third_party/llvm/bin/mlir-opt gpu.mlir \
#   --gpu-to-llvm

# ../third_party/llvm/bin/mlir-opt gpu.mlir \
#   --pass-pipeline="builtin.module(
#     gpu-kernel-outlining,
#     nvvm-attach-target{chip=sm_80 O=3},
#     gpu.module(convert-gpu-to-nvvm),
#     gpu-module-to-binary,
#     convert-scf-to-cf,
#     convert-cf-to-llvm,
#     lower-host-to-llvm,
#     reconcile-unrealized-casts
#   )" \
#   -o example-nvvm.mlir

# ../third_party/llvm/bin/mlir-opt gpu.mlir \
#   --pass-pipeline="builtin.module(
#     gpu-kernel-outlining,
#     nvvm-attach-target{chip=sm_80 O=3},
#     gpu.module(convert-gpu-to-nvvm),
#     gpu-module-to-binary,

#     gpu-to-llvm,

#     convert-scf-to-cf,
#     convert-index-to-llvm,
#     convert-arith-to-llvm,
#     convert-memref-to-llvm,
#     finalize-memref-to-llvm,
#     convert-cf-to-llvm,
#     convert-func-to-llvm,

#     reconcile-unrealized-casts
#   )" \
#   -o example-nvvm.mlir


../third_party/llvm/bin/mlir-translate example-nvvm.mlir        \
  --mlir-to-llvmir                      \
  -o example.ll

  # -gpu-lower-to-nvvm-pipeline="cubin-format=bin,cubin-chip=${CUDA_ARCH}"

  # | ../third_party/llvm/bin/mlir-runner \
  #     --shared-libs=$MLIR_CUDA_RUNTIME \
  #     --shared-libs=$MLIR_RUNNER_UTILS \
  #     --entry-point=_mlir_ciface_main \
  #     --entry-point-result=void

# ../third_party/llvm/bin/mlir-runner example-nvvm.mlir \
#   --entry-point-result=void \
#   --shared-libs=${MLIR_RUNNER_UTILS} \
#   --shared-libs=${MLIR_CUDA_RUNTIME}

