# Standalone environment for MLIR tutorial

**NB: The code of this tutorial is from the [mlir-Toy-Example-tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-1/) and [mlir-transform-tutorial](https://mlir.llvm.org/docs/Tutorials/transform/).
This repo only provide a simple way to setting up the environment. The toy file used in mlir-example all be in [example directory](../example/) and `Ch1-Ch7` is the Toy tutorial example code `Ch8` is an naive example to add `toy.matmul` operation and `transform_Ch2-H` is for transform dialect tutorials**

## Environment Setup

### Environment Preparation with conda (Optional)

- OS must be higher than ubuntu 22.04.
- install gcc-13 and g++-13

```bash
apt update -y && \
apt install -yq gcc-13 g++-13
# apt install -yq software-properties-common \
# add-apt-repository -y ppa:ubuntu-toolchain-r/test \
# apt update -y
# apt install -yq gcc-11 g++-11
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 20
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 20
```

- install cmake and ninja you can choose one way you like. conda is best for me.

```bash
conda create -n mlir -y
conda activate mlir
# conda install cmake ninja clang-format clang lld ncurses mlir llvm -c conda-forge
conda install cmake ninja clang-format clang clang-tools mlir zlib spdlog fmt lit llvm=19.* -c conda-forge -y
# create -n mlir cmake ninja clang-format clang mlir zlib spdlog fmt lit llvm -c conda-forge -y
```

- build example with conda

```bash
cd example
bash build_with_conda.sh all
```

### Environment Preparation with dev containers

Please choose the `Dev Containers: Open Folder in Container...`

- build example with dev containers

```bash
cd example
bash scripts/sync_deps.sh
bash scripts/build_deps.sh
bash scripts/build_cuda_tile.sh
bash build.sh all
```

## Configure the Clangd

```bash
cd example
# after you configure the project with cmake, you can configure the clangd by run the following command
compdb -p build list > compile_commands.json
```

## Run These code and understand mlir

### Toy Cuda Examples

> Note: if you want to run the toy-cuda example on the cuda < 13.x, you need to write you own cuda kernel.
> The cuda tile dialect is only supported on cuda 13.x and above, and the generated kernel is only compatible with cuda 13.x and above.
>
> So, I do the testing with the source code `cuda_shim/outlined_gpu_kernel.cu`
> compiled with `nvcc` with `-arch=sm_80` and `--cubin` flag to generate the cubin file under the cuda 12.x,
> and then load the cubin file in the runtime.
> please mv the generated `cuda_tile.bin` to `/tmp/cuda_tile-94d280.bin` before you run the example.
>
> `cp cuda_shim/cuda_tile.cubin /tmp/cuda_tile-94d280.bin`
>
> warning: if you are not using the `-use-cache` option, it will delete the `cuda_tile.bin` after the execution,
> so please backup it before you run the example.

- Show the outline gpu kernel IR

```bash
./build/Toy/toy-cuda sample/matmul.toy -emit=gpu-ir --grid 8,1,1 
# The GPU related actions will be used
# Grid dimensions: 8,1,1
# module {
#   toy.func @main() {
#     %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00, 9.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00, 1.000000e+01]]> : tensor<2x4xf32>
#     %1 = toy.constant dense<[[1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01], [1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01]]> : tensor<2x4xf32>
#     %2 = toy.constant dense<[[7.000000e+00, 8.000000e+00, 9.000000e+00, 1.300000e+01], [1.000000e+01, 1.100000e+01, 1.200000e+01, 1.400000e+01]]> : tensor<2x4xf32>
#     %3 = toy.launch_gpu @outlined_gpu_kernel_0(%0, %2, %1) {grid = array<i64: 8, 1, 1>} : (tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
#     toy.print %3 : tensor<2x4xf32>
#     toy.return
#   }
#   toy.gpu_func @outlined_gpu_kernel_0(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>, %arg2: tensor<2x4xf32>) -> tensor<2x4xf32> {
#     %0 = toy.mul %arg0, %arg1 : tensor<2x4xf32>
#     %1 = toy.mul %arg2, %arg1 : tensor<2x4xf32>
#     %2 = toy.add %0, %1 : tensor<2x4xf32>
#     toy.return %2 : tensor<2x4xf32>
#   }
# }
```

- Show the cuda tile IR

```bash
./build/Toy/toy-cuda sample/matmul.toy -emit=cuda-tile-ir --grid 8,1,1
# The GPU related actions will be used
# Grid dimensions: 8,1,1
# module {
#   toy.func @main() {
#     %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00, 9.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00, 1.000000e+01]]> : tensor<2x4xf32>
#     %1 = toy.constant dense<[[1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01], [1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01]]> : tensor<2x4xf32>
#     %2 = toy.constant dense<[[7.000000e+00, 8.000000e+00, 9.000000e+00, 1.300000e+01], [1.000000e+01, 1.100000e+01, 1.200000e+01, 1.400000e+01]]> : tensor<2x4xf32>
#     %3 = toy.launch_gpu @outlined_gpu_kernel_0(%0, %2, %1) {grid = array<i64: 8, 1, 1>} : (tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
#     toy.print %3 : tensor<2x4xf32>
#     toy.return
#   }
#   cuda_tile.module @cuda_tile_module {
#     entry @outlined_gpu_kernel_0(%arg0: tile<ptr<f32>>, %arg1: tile<ptr<f32>>, %arg2: tile<ptr<f32>>, %arg3: tile<ptr<f32>>) {
#       %tview = make_tensor_view %arg0, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xf32, strides=[4,1]>
#       %pview = make_partition_view %tview : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>
#       %cst_0_i32 = constant <i32: 0> : tile<i32>
#       %tile, %result_token = load_view_tko weak %pview[%cst_0_i32, %cst_0_i32] : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>, tile<i32> -> tile<2x4xf32>, token
#       %tview_0 = make_tensor_view %arg1, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xf32, strides=[4,1]>
#       %pview_1 = make_partition_view %tview_0 : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>
#       %tile_2, %result_token_3 = load_view_tko weak %pview_1[%cst_0_i32, %cst_0_i32] : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>, tile<i32> -> tile<2x4xf32>, token
#       %0 = mulf %tile, %tile_2  : tile<2x4xf32>
#       %tview_4 = make_tensor_view %arg2, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xf32, strides=[4,1]>
#       %pview_5 = make_partition_view %tview_4 : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>
#       %tile_6, %result_token_7 = load_view_tko weak %pview_5[%cst_0_i32, %cst_0_i32] : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>, tile<i32> -> tile<2x4xf32>, token
#       %tile_8, %result_token_9 = load_view_tko weak %pview_1[%cst_0_i32, %cst_0_i32] : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>, tile<i32> -> tile<2x4xf32>, token
#       %1 = mulf %tile_6, %tile_8  : tile<2x4xf32>
#       %2 = addf %0, %1  : tile<2x4xf32>
#       %tview_10 = make_tensor_view %arg3, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xf32, strides=[4,1]>
#       %pview_11 = make_partition_view %tview_10 : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>
#       %3 = store_view_tko weak %2, %pview_11[%cst_0_i32, %cst_0_i32] : tile<2x4xf32>, partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>, tile<i32> -> token
#       return
#     }
#   }
# }
```

- Show the affine and gpu dialect IR

```bash
./build/Toy/toy-cuda sample/matmul.toy -emit=gpu-affine --grid 8,1,1 -opt -use-cache
# The GPU related actions will be used
# Grid dimensions: 8,1,1
# module {
#   func.func private @cuda_shim_unload_module(i64)
#   func.func private @cuda_shim_stream_destroy(i64)
#   func.func private @cuda_shim_free(i64, i64)
#   func.func private @cuda_shim_stream_synchronize(i64)
#   func.func private @cuda_shim_launch_grid_packed(i64, i64, i32, i32, i32, i64, i64, i64, i32)
#   func.func private @cuda_shim_memcpy_d2h(i64, i64, i64)
#   func.func private @cuda_shim_memcpy_h2d(i64, i64, i64)
#   func.func private @cuda_shim_malloc(i64, i64, i1) -> i64
#   func.func private @cuda_shim_stream_create() -> i64
#   func.func private @cuda_shim_load_module_from_file(i64, i64) -> i64
#   memref.global "private" constant @kname : memref<22xi8> = dense<[111, 117, 116, 108, 105, 110, 101, 100, 95, 103, 112, 117, 95, 107, 101, 114, 110, 101, 108, 95, 48, 0]>
#   memref.global "private" constant @cuda_blob : memref<26xi8> = dense<[47, 116, 109, 112, 47, 99, 117, 100, 97, 95, 116, 105, 108, 101, 45, 57, 52, 100, 50, 56, 48, 46, 98, 105, 110, 0]>
#   func.func @main() {
#     %c4_i32 = arith.constant 4 : i32
#     %c1_i32 = arith.constant 1 : i32
#     %c8_i32 = arith.constant 8 : i32
#     %c8_i64 = arith.constant 8 : i64
#     %false = arith.constant false
#     %c32_i64 = arith.constant 32 : i64
#     %c26_i64 = arith.constant 26 : i64
#     %cst = arith.constant 8.000000e+00 : f32
#     %cst_0 = arith.constant 7.000000e+00 : f32
#     %cst_1 = arith.constant 1.800000e+01 : f32
#     %cst_2 = arith.constant 1.700000e+01 : f32
#     %cst_3 = arith.constant 1.600000e+01 : f32
#     %cst_4 = arith.constant 1.500000e+01 : f32
#     %cst_5 = arith.constant 1.400000e+01 : f32
#     %cst_6 = arith.constant 1.300000e+01 : f32
#     %cst_7 = arith.constant 1.200000e+01 : f32
#     %cst_8 = arith.constant 1.100000e+01 : f32
#     %cst_9 = arith.constant 1.000000e+01 : f32
#     %cst_10 = arith.constant 6.000000e+00 : f32
#     %cst_11 = arith.constant 5.000000e+00 : f32
#     %cst_12 = arith.constant 4.000000e+00 : f32
#     %cst_13 = arith.constant 9.000000e+00 : f32
#     %cst_14 = arith.constant 3.000000e+00 : f32
#     %cst_15 = arith.constant 2.000000e+00 : f32
#     %cst_16 = arith.constant 1.000000e+00 : f32
#     %c3 = arith.constant 3 : index
#     %c2 = arith.constant 2 : index
#     %c1 = arith.constant 1 : index
#     %c0 = arith.constant 0 : index
#     %alloc = memref.alloc() : memref<2x4xf32>
#     %alloc_17 = memref.alloc() : memref<2x4xf32>
#     %alloc_18 = memref.alloc() : memref<2x4xf32>
#     %alloc_19 = memref.alloc() : memref<2x4xf32>
#     affine.store %cst_16, %alloc_19[0, 0] : memref<2x4xf32>
#     affine.store %cst_15, %alloc_19[0, 1] : memref<2x4xf32>
#     affine.store %cst_14, %alloc_19[0, 2] : memref<2x4xf32>
#     affine.store %cst_13, %alloc_19[0, 3] : memref<2x4xf32>
#     affine.store %cst_12, %alloc_19[1, 0] : memref<2x4xf32>
#     affine.store %cst_11, %alloc_19[1, 1] : memref<2x4xf32>
#     affine.store %cst_10, %alloc_19[1, 2] : memref<2x4xf32>
#     affine.store %cst_9, %alloc_19[1, 3] : memref<2x4xf32>
#     affine.store %cst_8, %alloc_18[0, 0] : memref<2x4xf32>
#     affine.store %cst_7, %alloc_18[0, 1] : memref<2x4xf32>
#     affine.store %cst_6, %alloc_18[0, 2] : memref<2x4xf32>
#     affine.store %cst_5, %alloc_18[0, 3] : memref<2x4xf32>
#     affine.store %cst_4, %alloc_18[1, 0] : memref<2x4xf32>
#     affine.store %cst_3, %alloc_18[1, 1] : memref<2x4xf32>
#     affine.store %cst_2, %alloc_18[1, 2] : memref<2x4xf32>
#     affine.store %cst_1, %alloc_18[1, 3] : memref<2x4xf32>
#     affine.store %cst_0, %alloc_17[0, 0] : memref<2x4xf32>
#     affine.store %cst, %alloc_17[0, 1] : memref<2x4xf32>
#     affine.store %cst_13, %alloc_17[0, 2] : memref<2x4xf32>
#     affine.store %cst_6, %alloc_17[0, 3] : memref<2x4xf32>
#     affine.store %cst_9, %alloc_17[1, 0] : memref<2x4xf32>
#     affine.store %cst_8, %alloc_17[1, 1] : memref<2x4xf32>
#     affine.store %cst_7, %alloc_17[1, 2] : memref<2x4xf32>
#     affine.store %cst_5, %alloc_17[1, 3] : memref<2x4xf32>
#     %0 = memref.get_global @cuda_blob : memref<26xi8>
#     %intptr = memref.extract_aligned_pointer_as_index %0 : memref<26xi8> -> index
#     %1 = arith.index_cast %intptr : index to i64
#     %2 = memref.get_global @kname : memref<22xi8>
#     %intptr_20 = memref.extract_aligned_pointer_as_index %2 : memref<22xi8> -> index
#     %3 = arith.index_cast %intptr_20 : index to i64
#     %4 = call @cuda_shim_load_module_from_file(%1, %c26_i64) : (i64, i64) -> i64
#     %5 = call @cuda_shim_stream_create() : () -> i64
#     %alloc_21 = memref.alloc() : memref<4xi64>
#     %alloc_22 = memref.alloc() : memref<4xi64>
#     %6 = call @cuda_shim_malloc(%c32_i64, %5, %false) : (i64, i64, i1) -> i64
#     %intptr_23 = memref.extract_aligned_pointer_as_index %alloc_19 : memref<2x4xf32> -> index
#     %7 = arith.index_cast %intptr_23 : index to i64
#     call @cuda_shim_memcpy_h2d(%6, %7, %c32_i64) : (i64, i64, i64) -> ()
#     memref.store %6, %alloc_21[%c0] : memref<4xi64>
#     memref.store %c8_i64, %alloc_22[%c0] : memref<4xi64>
#     %8 = call @cuda_shim_malloc(%c32_i64, %5, %false) : (i64, i64, i1) -> i64
#     %intptr_24 = memref.extract_aligned_pointer_as_index %alloc_17 : memref<2x4xf32> -> index
#     %9 = arith.index_cast %intptr_24 : index to i64
#     call @cuda_shim_memcpy_h2d(%8, %9, %c32_i64) : (i64, i64, i64) -> ()
#     memref.store %8, %alloc_21[%c1] : memref<4xi64>
#     memref.store %c8_i64, %alloc_22[%c1] : memref<4xi64>
#     %10 = call @cuda_shim_malloc(%c32_i64, %5, %false) : (i64, i64, i1) -> i64
#     %intptr_25 = memref.extract_aligned_pointer_as_index %alloc_18 : memref<2x4xf32> -> index
#     %11 = arith.index_cast %intptr_25 : index to i64
#     call @cuda_shim_memcpy_h2d(%10, %11, %c32_i64) : (i64, i64, i64) -> ()
#     memref.store %10, %alloc_21[%c2] : memref<4xi64>
#     memref.store %c8_i64, %alloc_22[%c2] : memref<4xi64>
#     %12 = call @cuda_shim_malloc(%c32_i64, %5, %false) : (i64, i64, i1) -> i64
#     %intptr_26 = memref.extract_aligned_pointer_as_index %alloc : memref<2x4xf32> -> index
#     %13 = arith.index_cast %intptr_26 : index to i64
#     memref.store %12, %alloc_21[%c3] : memref<4xi64>
#     memref.store %c8_i64, %alloc_22[%c3] : memref<4xi64>
#     %intptr_27 = memref.extract_aligned_pointer_as_index %alloc_21 : memref<4xi64> -> index
#     %14 = arith.index_cast %intptr_27 : index to i64
#     %intptr_28 = memref.extract_aligned_pointer_as_index %alloc_22 : memref<4xi64> -> index
#     %15 = arith.index_cast %intptr_28 : index to i64
#     call @cuda_shim_launch_grid_packed(%4, %3, %c8_i32, %c1_i32, %c1_i32, %5, %14, %15, %c4_i32) : (i64, i64, i32, i32, i32, i64, i64, i64, i32) -> ()
#     call @cuda_shim_stream_synchronize(%5) : (i64) -> ()
#     call @cuda_shim_memcpy_d2h(%13, %12, %c32_i64) : (i64, i64, i64) -> ()
#     memref.dealloc %alloc_21 : memref<4xi64>
#     memref.dealloc %alloc_22 : memref<4xi64>
#     call @cuda_shim_free(%12, %5) : (i64, i64) -> ()
#     call @cuda_shim_free(%10, %5) : (i64, i64) -> ()
#     call @cuda_shim_free(%8, %5) : (i64, i64) -> ()
#     call @cuda_shim_free(%6, %5) : (i64, i64) -> ()
#     call @cuda_shim_stream_destroy(%5) : (i64) -> ()
#     call @cuda_shim_unload_module(%4) : (i64) -> ()
#     toy.print %alloc : memref<2x4xf32>
#     memref.dealloc %alloc_19 : memref<2x4xf32>
#     memref.dealloc %alloc_18 : memref<2x4xf32>
#     memref.dealloc %alloc_17 : memref<2x4xf32>
#     memref.dealloc %alloc : memref<2x4xf32>
#     return
#   }
# }
```

- Show the LLVM Dialect IR

```bash
./build/Toy/toy-cuda sample/matmul.toy -emit=gpu-llvm --grid 8,1,1 -opt -use-cache
# The GPU related actions will be used
# Grid dimensions: 8,1,1
# module {
#   llvm.mlir.global internal constant @nl("\0A\00") {addr_space = 0 : i32}
#   llvm.mlir.global internal constant @frmt_spec("%f \00") {addr_space = 0 : i32}
#   llvm.func @printf(!llvm.ptr, ...) -> i32
#   llvm.func @free(!llvm.ptr)
#   llvm.func @malloc(i64) -> !llvm.ptr
#   llvm.func @cuda_shim_unload_module(i64) attributes {sym_visibility = "private"}
#   llvm.func @cuda_shim_stream_destroy(i64) attributes {sym_visibility = "private"}
#   llvm.func @cuda_shim_free(i64, i64) attributes {sym_visibility = "private"}
#   llvm.func @cuda_shim_stream_synchronize(i64) attributes {sym_visibility = "private"}
#   llvm.func @cuda_shim_launch_grid_packed(i64, i64, i32, i32, i32, i64, i64, i64, i32) attributes {sym_visibility = "private"}
#   llvm.func @cuda_shim_memcpy_d2h(i64, i64, i64) attributes {sym_visibility = "private"}
#   llvm.func @cuda_shim_memcpy_h2d(i64, i64, i64) attributes {sym_visibility = "private"}
#   llvm.func @cuda_shim_malloc(i64, i64, i1) -> i64 attributes {sym_visibility = "private"}
#   llvm.func @cuda_shim_stream_create() -> i64 attributes {sym_visibility = "private"}
#   llvm.func @cuda_shim_load_module_from_file(i64, i64) -> i64 attributes {sym_visibility = "private"}
#   llvm.mlir.global private constant @kname(dense<[111, 117, 116, 108, 105, 110, 101, 100, 95, 103, 112, 117, 95, 107, 101, 114, 110, 101, 108, 95, 48, 0]> : tensor<22xi8>) {addr_space = 0 : i32} : !llvm.array<22 x i8>
#   llvm.mlir.global private constant @cuda_blob(dense<[47, 116, 109, 112, 47, 99, 117, 100, 97, 95, 116, 105, 108, 101, 45, 57, 52, 100, 50, 56, 48, 46, 98, 105, 110, 0]> : tensor<26xi8>) {addr_space = 0 : i32} : !llvm.array<26 x i8>
#   llvm.func @main() {
#     %0 = llvm.mlir.constant(4 : i32) : i32
#     %1 = llvm.mlir.constant(1 : i32) : i32
#     %2 = llvm.mlir.constant(8 : i32) : i32
#     %3 = llvm.mlir.constant(8 : i64) : i64
#     %4 = llvm.mlir.constant(false) : i1
#  .......
#     %376 = llvm.mlir.constant(1 : index) : i64
#     llvm.br ^bb3(%374 : i64)
#   ^bb3(%377: i64):  // 2 preds: ^bb2, ^bb4
#     %378 = llvm.icmp "slt" %377, %375 : i64
#     llvm.cond_br %378, ^bb4, ^bb5
#   ^bb4:  // pred: ^bb3
#     %379 = llvm.extractvalue %45[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
#   %71 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double %70), !dbg !24
#   %72 = getelementptr inbounds nuw i8, ptr %0, i64 28, !dbg !24
#   %73 = load float, ptr %72, align 4, !dbg !24
#   %74 = fpext float %73 to double, !dbg !24
#   %75 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double %74), !dbg !24
#   %putchar.1 = tail call i32 @putchar(i32 10), !dbg !24
#   tail call void @free(ptr %3), !dbg !23
#   tail call void @free(ptr %2), !dbg !22
#   tail call void @free(ptr %1), !dbg !21
#   tail call void @free(ptr %0), !dbg !20
#   ret void, !dbg !20
# }

# ; Function Attrs: nofree nounwind
# declare noundef i32 @putchar(i32 noundef) local_unnamed_addr #0

# attributes #0 = { nofree nounwind }
# attributes #1 = { mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" }
# attributes #2 = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" }

# !llvm.dbg.cu = !{!0}
# !llvm.module.flags = !{!2}

# !0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "MLIR", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
# !1 = !DIFile(filename: "<unknown>", directory: "")
# !2 = !{i32 2, !"Debug Info Version", i32 3}
# !3 = !DISubprogram(name: "printf", linkageName: "printf", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagOptimized)
# !4 = !DISubroutineType(cc: DW_CC_normal, types: !5)
# !5 = !{}
# !6 = !DISubprogram(name: "free", linkageName: "free", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagOptimized)
# !7 = !DISubprogram(name: "malloc", linkageName: "malloc", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagOptimized)
# !8 = !DISubprogram(name: "cuda_shim_unload_module", linkageName: "cuda_shim_unload_module", scope: !9, file: !9, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagOptimized)
# !9 = !DIFile(filename: "matmul.toy", directory: "sample")
# !10 = !DISubprogram(name: "cuda_shim_stream_destroy", linkageName: "cuda_shim_stream_destroy", scope: !9, file: !9, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagOptimized)
# !11 = !DISubprogram(name: "cuda_shim_free", linkageName: "cuda_shim_free", scope: !9, file: !9, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagOptimized)
# !12 = !DISubprogram(name: "cuda_shim_stream_synchronize", linkageName: "cuda_shim_stream_synchronize", scope: !9, file: !9, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagOptimized)
# !13 = !DISubprogram(name: "cuda_shim_launch_grid_packed", linkageName: "cuda_shim_launch_grid_packed", scope: !9, file: !9, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagOptimized)
# !14 = !DISubprogram(name: "cuda_shim_memcpy_d2h", linkageName: "cuda_shim_memcpy_d2h", scope: !9, file: !9, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagOptimized)
# !15 = !DISubprogram(name: "cuda_shim_memcpy_h2d", linkageName: "cuda_shim_memcpy_h2d", scope: !9, file: !9, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagOptimized)
# !16 = !DISubprogram(name: "cuda_shim_malloc", linkageName: "cuda_shim_malloc", scope: !9, file: !9, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagOptimized)
# !17 = !DISubprogram(name: "cuda_shim_stream_create", linkageName: "cuda_shim_stream_create", scope: !9, file: !9, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagOptimized)
# !18 = !DISubprogram(name: "cuda_shim_load_module_from_file", linkageName: "cuda_shim_load_module_from_file", scope: !9, file: !9, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagOptimized)
# !19 = distinct !DISubprogram(name: "main", linkageName: "main", scope: !9, file: !9, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
# !20 = !DILocation(line: 1, column: 1, scope: !19)
# !21 = !DILocation(line: 14, scope: !19)
# !22 = !DILocation(line: 8, scope: !19)
# !23 = !DILocation(line: 4, column: 11, scope: !19)
# !24 = !DILocation(line: 16, column: 3, scope: !19)
```

- Run Jit with Cuda

```bash
./build/Toy/toy-cuda sample/matmul.toy -emit=nv-gpu-jit --grid 8,1,1 -opt -use-cache
# The GPU related actions will be used
# Grid dimensions: 8,1,1
# 22.000000 36.000000 52.000000 140.000000
# 75.000000 96.000000 119.000000 198.000000
# 200.000000 260.000000
# 322.000000 422.000000
# 84.000000 112.000000 144.000000 299.000000
# 190.000000 231.000000 276.000000 392.000000
```
