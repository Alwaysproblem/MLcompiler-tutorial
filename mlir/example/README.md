# Standalone environment for MLIR tutorial.

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
bash build.sh all
```

## Configure the Clangd

```bash
cd example
# after you configure the project with cmake, you can configure the clangd by run the following command
compdb -p build list > compile_commands.json
```

## Run These code and understand mlir

### Toy Examples

- Ch1

```bash
$./build/Ch1/mlir-example-ch1 Ch1/example.toy -emit=ast
# Module:
#   Function
#     Proto 'main' @Ch1/example.toy:1:1
#     Params: []
#     Block {
#       VarDecl a<> @Ch1/example.toy:4:3
#         Literal: <2, 3>[ <3>[ 1.000000e+00, 2.000000e+00, 3.000000e+00], <3>[ 4.000000e+00, 5.000000e+00, 6.000000e+00]] @Ch1/example.toy:4:11
#       VarDecl b<2, 3> @Ch1/example.toy:8:3
#         Literal: <6>[ 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00] @Ch1/example.toy:8:17
#       Print [ @Ch1/example.toy:12:3
#         BinOp: * @Ch1/example.toy:12:24
#           Call 'transpose' [ @Ch1/example.toy:12:9
#             var: a @Ch1/example.toy:12:19
#           ]
#           Call 'transpose' [ @Ch1/example.toy:12:24
#             var: b @Ch1/example.toy:12:34
#           ]
#       ]
#     } // Block
```

- Ch2

```bash
$./build/Ch2/mlir-example-ch2 Ch2/codegen.toy -emit=mlir
# module {
#   toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
#     %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
#     %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64>
#     %2 = toy.mul %0, %1 : tensor<*xf64>
#     toy.return %2 : tensor<*xf64>
#   }
#   toy.func @main() {
#     %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
#     %1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64>
#     %2 = toy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>
#     %3 = toy.reshape(%2 : tensor<6xf64>) to tensor<2x3xf64>
#     %4 = toy.generic_call @multiply_transpose(%1, %3) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
#     %5 = toy.generic_call @multiply_transpose(%3, %1) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
#     toy.print %5 : tensor<*xf64>
#     toy.return
#   }
# }
```

- Ch3

```bash
$./build/Ch3/mlir-example-ch3 Ch3/opt.toy  -emit=mlir
# module {
#   toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
#     %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
#     %1 = toy.transpose(%0 : tensor<*xf64>) to tensor<*xf64>
#     %2 = toy.transpose(%1 : tensor<*xf64>) to tensor<*xf64>
#     %3 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64>
#     %4 = toy.mul %2, %3 : tensor<*xf64>
#     toy.return %4 : tensor<*xf64>
#   }
#   toy.func @main() {
#     %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
#     %1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64>
#     %2 = toy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>
#     %3 = toy.reshape(%2 : tensor<6xf64>) to tensor<2x3xf64>
#     %4 = toy.generic_call @multiply_transpose(%1, %3) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
#     %5 = toy.generic_call @multiply_transpose(%3, %1) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
#     %6 = toy.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>
#     %7 = toy.reshape(%6 : tensor<2xf64>) to tensor<2x1xf64>
#     %8 = toy.reshape(%7 : tensor<2x1xf64>) to tensor<2x1xf64>
#     %9 = toy.reshape(%8 : tensor<2x1xf64>) to tensor<2x1xf64>
#     toy.print %5 : tensor<*xf64>
#     toy.return
#   }
# }
$./build/Ch3/mlir-example-ch3 Ch3/opt.toy  -emit=mlir -opt
# module {
#   toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
#     %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
#     %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64>
#     %2 = toy.mul %0, %1 : tensor<*xf64>
#     toy.return %2 : tensor<*xf64>
#   }
#   toy.func @main() {
#     %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
#     %1 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
#     %2 = toy.generic_call @multiply_transpose(%0, %1) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
#     %3 = toy.generic_call @multiply_transpose(%1, %0) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
#     toy.print %3 : tensor<*xf64>
#     toy.return
#   }
# }
```

- Ch4

```bash
$./build/Ch4/mlir-example-ch4 Ch4/opt.toy  -emit=mlir
# module {
#   toy.func private @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
#     %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
#     %1 = toy.transpose(%0 : tensor<*xf64>) to tensor<*xf64>
#     %2 = toy.transpose(%1 : tensor<*xf64>) to tensor<*xf64>
#     %3 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64>
#     %4 = toy.mul %2, %3 : tensor<*xf64>
#     toy.return %4 : tensor<*xf64>
#   }
#   toy.func @main() {
#     %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
#     %1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64>
#     %2 = toy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>
#     %3 = toy.reshape(%2 : tensor<6xf64>) to tensor<2x3xf64>
#     %4 = toy.generic_call @multiply_transpose(%1, %3) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
#     %5 = toy.generic_call @multiply_transpose(%3, %1) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
#     %6 = toy.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>
#     %7 = toy.reshape(%6 : tensor<2xf64>) to tensor<2x1xf64>
#     %8 = toy.reshape(%7 : tensor<2x1xf64>) to tensor<2x1xf64>
#     %9 = toy.reshape(%8 : tensor<2x1xf64>) to tensor<2x1xf64>
#     toy.print %5 : tensor<*xf64>
#     toy.return
#   }
# }
$./build/Ch4/mlir-example-ch4 Ch4/opt.toy  -emit=mlir -opt
# module {
#   toy.func @main() {
#     %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
#     %1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
#     %2 = toy.mul %1, %1 : tensor<3x2xf64>
#     toy.print %2 : tensor<3x2xf64>
#     toy.return
#   }
# }
```

- Ch5

```bash
$ ./build/Ch5/mlir-example-ch5 Ch5/example.toy -emit=mlir-affine
# module {
#   func.func @main() {
#     %cst = arith.constant 6.000000e+00 : f64
#     %cst_0 = arith.constant 5.000000e+00 : f64
#     %cst_1 = arith.constant 4.000000e+00 : f64
#     %cst_2 = arith.constant 3.000000e+00 : f64
#     %cst_3 = arith.constant 2.000000e+00 : f64
#     %cst_4 = arith.constant 1.000000e+00 : f64
#     %0 = memref.alloc() : memref<3x2xf64>
#     %1 = memref.alloc() : memref<3x2xf64>
#     %2 = memref.alloc() : memref<2x3xf64>
#     affine.store %cst_4, %2[0, 0] : memref<2x3xf64>
#     affine.store %cst_3, %2[0, 1] : memref<2x3xf64>
#     affine.store %cst_2, %2[0, 2] : memref<2x3xf64>
#     affine.store %cst_1, %2[1, 0] : memref<2x3xf64>
#     affine.store %cst_0, %2[1, 1] : memref<2x3xf64>
#     affine.store %cst, %2[1, 2] : memref<2x3xf64>
#     affine.for %arg0 = 0 to 3 {
#       affine.for %arg1 = 0 to 2 {
#         %3 = affine.load %2[%arg1, %arg0] : memref<2x3xf64>
#         affine.store %3, %1[%arg0, %arg1] : memref<3x2xf64>
#       }
#     }
#     affine.for %arg0 = 0 to 3 {
#       affine.for %arg1 = 0 to 2 {
#         %3 = affine.load %1[%arg0, %arg1] : memref<3x2xf64>
#         %4 = arith.mulf %3, %3 : f64
#         affine.store %4, %0[%arg0, %arg1] : memref<3x2xf64>
#       }
#     }
#     toy.print %0 : memref<3x2xf64>
#     memref.dealloc %2 : memref<2x3xf64>
#     memref.dealloc %1 : memref<3x2xf64>
#     memref.dealloc %0 : memref<3x2xf64>
#     return
#   }
# }
$ ./build/Ch5/mlir-example-ch5 Ch5/example.toy -emit=mlir-affine -opt
# module {
#   func.func @main() {
#     %cst = arith.constant 6.000000e+00 : f64
#     %cst_0 = arith.constant 5.000000e+00 : f64
#     %cst_1 = arith.constant 4.000000e+00 : f64
#     %cst_2 = arith.constant 3.000000e+00 : f64
#     %cst_3 = arith.constant 2.000000e+00 : f64
#     %cst_4 = arith.constant 1.000000e+00 : f64
#     %0 = memref.alloc() : memref<3x2xf64>
#     %1 = memref.alloc() : memref<2x3xf64>
#     affine.store %cst_4, %1[0, 0] : memref<2x3xf64>
#     affine.store %cst_3, %1[0, 1] : memref<2x3xf64>
#     affine.store %cst_2, %1[0, 2] : memref<2x3xf64>
#     affine.store %cst_1, %1[1, 0] : memref<2x3xf64>
#     affine.store %cst_0, %1[1, 1] : memref<2x3xf64>
#     affine.store %cst, %1[1, 2] : memref<2x3xf64>
#     affine.for %arg0 = 0 to 3 {
#       affine.for %arg1 = 0 to 2 {
#         %2 = affine.load %1[%arg1, %arg0] : memref<2x3xf64>
#         %3 = arith.mulf %2, %2 : f64
#         affine.store %3, %0[%arg0, %arg1] : memref<3x2xf64>
#       }
#     }
#     toy.print %0 : memref<3x2xf64>
#     memref.dealloc %1 : memref<2x3xf64>
#     memref.dealloc %0 : memref<3x2xf64>
#     return
#   }
# }
```

- Ch6

```bash
$ ./build/Ch6/mlir-example-ch6 Ch6/example.toy -emit=jit
# 1.000000 16.000000
# 4.000000 25.000000
# 9.000000 36.000000

$ ./build/Ch6/mlir-example-ch6 Ch6/example.toy -emit=llvm --mlir-print-ir-after-all
# // -----// IR Dump After Canonicalizer (canonicalize) //----- //
# toy.func @main() {
#   %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
#   %1 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
#   %2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<*xf64>
#   %3 = toy.transpose(%1 : tensor<2x3xf64>) to tensor<*xf64>
#   %4 = toy.mul %2, %3 : tensor<*xf64>
#   toy.print %4 : tensor<*xf64>
#   toy.return
# }

# // -----// IR Dump After Inliner (inline) //----- //
# module {
#   toy.func @main() {
#     %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
#     %1 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
#     %2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<*xf64>
#     %3 = toy.transpose(%1 : tensor<2x3xf64>) to tensor<*xf64>
#     %4 = toy.mul %2, %3 : tensor<*xf64>
#     toy.print %4 : tensor<*xf64>
#     toy.return
#   }
# }


# // -----// IR Dump After {anonymous}::ShapeInferencePass () //----- //
# toy.func @main() {
#   %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
#   %1 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
#   %2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
#   %3 = toy.transpose(%1 : tensor<2x3xf64>) to tensor<3x2xf64>
#   %4 = toy.mul %2, %3 : tensor<3x2xf64>
#   toy.print %4 : tensor<3x2xf64>
#   toy.return
# }

# // -----// IR Dump After Canonicalizer (canonicalize) //----- //
# toy.func @main() {
#   %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
#   %1 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
#   %2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
#   %3 = toy.transpose(%1 : tensor<2x3xf64>) to tensor<3x2xf64>
#   %4 = toy.mul %2, %3 : tensor<3x2xf64>
#   toy.print %4 : tensor<3x2xf64>
#   toy.return
# }

# // -----// IR Dump After CSE (cse) //----- //
# toy.func @main() {
#   %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
#   %1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
#   %2 = toy.mul %1, %1 : tensor<3x2xf64>
#   toy.print %2 : tensor<3x2xf64>
#   toy.return
# }

# // -----// IR Dump After {anonymous}::ToyToAffineLoweringPass () //----- //
# module {
#   func.func @main() {
#     %0 = memref.alloc() : memref<3x2xf64>
#     %1 = memref.alloc() : memref<3x2xf64>
#     %2 = memref.alloc() : memref<2x3xf64>
#     %c0 = arith.constant 0 : index
#     %c1 = arith.constant 1 : index
#     %c2 = arith.constant 2 : index
#     %cst = arith.constant 1.000000e+00 : f64
#     affine.store %cst, %2[%c0, %c0] : memref<2x3xf64>
#     %cst_0 = arith.constant 2.000000e+00 : f64
#     affine.store %cst_0, %2[%c0, %c1] : memref<2x3xf64>
#     %cst_1 = arith.constant 3.000000e+00 : f64
#     affine.store %cst_1, %2[%c0, %c2] : memref<2x3xf64>
#     %cst_2 = arith.constant 4.000000e+00 : f64
#     affine.store %cst_2, %2[%c1, %c0] : memref<2x3xf64>
#     %cst_3 = arith.constant 5.000000e+00 : f64
#     affine.store %cst_3, %2[%c1, %c1] : memref<2x3xf64>
#     %cst_4 = arith.constant 6.000000e+00 : f64
#     affine.store %cst_4, %2[%c1, %c2] : memref<2x3xf64>
#     affine.for %arg0 = 0 to 3 {
#       affine.for %arg1 = 0 to 2 {
#         %3 = affine.load %2[%arg1, %arg0] : memref<2x3xf64>
#         affine.store %3, %1[%arg0, %arg1] : memref<3x2xf64>
#       }
#     }
#     affine.for %arg0 = 0 to 3 {
#       affine.for %arg1 = 0 to 2 {
#         %3 = affine.load %1[%arg0, %arg1] : memref<3x2xf64>
#         %4 = affine.load %1[%arg0, %arg1] : memref<3x2xf64>
#         %5 = arith.mulf %3, %4 : f64
#         affine.store %5, %0[%arg0, %arg1] : memref<3x2xf64>
#       }
#     }
#     toy.print %0 : memref<3x2xf64>
#     memref.dealloc %2 : memref<2x3xf64>
#     memref.dealloc %1 : memref<3x2xf64>
#     memref.dealloc %0 : memref<3x2xf64>
#     return
#   }
# }


# // -----// IR Dump After Canonicalizer (canonicalize) //----- //
# func.func @main() {
#   %cst = arith.constant 6.000000e+00 : f64
#   %cst_0 = arith.constant 5.000000e+00 : f64
#   %cst_1 = arith.constant 4.000000e+00 : f64
#   %cst_2 = arith.constant 3.000000e+00 : f64
#   %cst_3 = arith.constant 2.000000e+00 : f64
#   %cst_4 = arith.constant 1.000000e+00 : f64
#   %0 = memref.alloc() : memref<3x2xf64>
#   %1 = memref.alloc() : memref<3x2xf64>
#   %2 = memref.alloc() : memref<2x3xf64>
#   affine.store %cst_4, %2[0, 0] : memref<2x3xf64>
#   affine.store %cst_3, %2[0, 1] : memref<2x3xf64>
#   affine.store %cst_2, %2[0, 2] : memref<2x3xf64>
#   affine.store %cst_1, %2[1, 0] : memref<2x3xf64>
#   affine.store %cst_0, %2[1, 1] : memref<2x3xf64>
#   affine.store %cst, %2[1, 2] : memref<2x3xf64>
#   affine.for %arg0 = 0 to 3 {
#     affine.for %arg1 = 0 to 2 {
#       %3 = affine.load %2[%arg1, %arg0] : memref<2x3xf64>
#       affine.store %3, %1[%arg0, %arg1] : memref<3x2xf64>
#     }
#   }
#   affine.for %arg0 = 0 to 3 {
#     affine.for %arg1 = 0 to 2 {
#       %3 = affine.load %1[%arg0, %arg1] : memref<3x2xf64>
#       %4 = affine.load %1[%arg0, %arg1] : memref<3x2xf64>
#       %5 = arith.mulf %3, %4 : f64
#       affine.store %5, %0[%arg0, %arg1] : memref<3x2xf64>
#     }
#   }
#   toy.print %0 : memref<3x2xf64>
#   memref.dealloc %2 : memref<2x3xf64>
#   memref.dealloc %1 : memref<3x2xf64>
#   memref.dealloc %0 : memref<3x2xf64>
#   return
# }

# // -----// IR Dump After CSE (cse) //----- //
# func.func @main() {
#   %cst = arith.constant 6.000000e+00 : f64
#   %cst_0 = arith.constant 5.000000e+00 : f64
#   %cst_1 = arith.constant 4.000000e+00 : f64
#   %cst_2 = arith.constant 3.000000e+00 : f64
#   %cst_3 = arith.constant 2.000000e+00 : f64
#   %cst_4 = arith.constant 1.000000e+00 : f64
#   %0 = memref.alloc() : memref<3x2xf64>
#   %1 = memref.alloc() : memref<3x2xf64>
#   %2 = memref.alloc() : memref<2x3xf64>
#   affine.store %cst_4, %2[0, 0] : memref<2x3xf64>
#   affine.store %cst_3, %2[0, 1] : memref<2x3xf64>
#   affine.store %cst_2, %2[0, 2] : memref<2x3xf64>
#   affine.store %cst_1, %2[1, 0] : memref<2x3xf64>
#   affine.store %cst_0, %2[1, 1] : memref<2x3xf64>
#   affine.store %cst, %2[1, 2] : memref<2x3xf64>
#   affine.for %arg0 = 0 to 3 {
#     affine.for %arg1 = 0 to 2 {
#       %3 = affine.load %2[%arg1, %arg0] : memref<2x3xf64>
#       affine.store %3, %1[%arg0, %arg1] : memref<3x2xf64>
#     }
#   }
#   affine.for %arg0 = 0 to 3 {
#     affine.for %arg1 = 0 to 2 {
#       %3 = affine.load %1[%arg0, %arg1] : memref<3x2xf64>
#       %4 = arith.mulf %3, %3 : f64
#       affine.store %4, %0[%arg0, %arg1] : memref<3x2xf64>
#     }
#   }
#   toy.print %0 : memref<3x2xf64>
#   memref.dealloc %2 : memref<2x3xf64>
#   memref.dealloc %1 : memref<3x2xf64>
#   memref.dealloc %0 : memref<3x2xf64>
#   return
# }

# // -----// IR Dump After {anonymous}::ToyToLLVMLoweringPass () //----- //
# module {
#   llvm.func @free(!llvm.ptr<i8>)
#   llvm.mlir.global internal constant @nl("\0A\00")
#   llvm.mlir.global internal constant @frmt_spec("%f \00")
#   llvm.func @printf(!llvm.ptr<i8>, ...) -> i32
#   llvm.func @malloc(i64) -> !llvm.ptr<i8>
#   llvm.func @main() {
# ...
```

- Ch7

```bash
$ ./build/Ch7/mlir-example-ch7 Ch7/struct-codegen.toy -emit=jit
# 1.000000 16.000000
# 4.000000 25.000000
# 9.000000 36.000000
```

- Ch8

```bash
$ ./build/Ch8/mlir-example-ch8 Ch8/matmul.toy.mlir -emit=mlir
# module {
#   toy.func private @matmul_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
#     %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
#     %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64>
#     %2 = toy.matmul(%0 : tensor<*xf64>, %1 : tensor<*xf64>) to tensor<*xf64>
#     toy.return %2 : tensor<*xf64>
#   }
#   toy.func @main() {
#     %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
#     %1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64>
#     %2 = toy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>
#     %3 = toy.reshape(%2 : tensor<6xf64>) to tensor<3x2xf64>
#     %4 = toy.generic_call @matmul_transpose(%1, %3) : (tensor<2x3xf64>, tensor<3x2xf64>) -> tensor<*xf64>
#     toy.print %4 : tensor<*xf64>
#     toy.return
#   }
# }
```

```bash
$ ./build/Ch8/mlir-example-ch8 Ch8/matmul.toy -emit=jit
# 14.000000 32.000000
# 32.000000 77.000000
```

### Transform Dialect

Please flow the [mlir-transform-tutorial](https://mlir.llvm.org/docs/Tutorials/transform/). If you have some questions about the way to run these examples, please check the top lines of each mlir files.

- transform Ch2

```bash
$ ./build/transform_Ch2/transform-opt-ch2 --transform-interpreter transform_Ch2/ops.mlir
# module {
#   func.func private @orig()
#   func.func private @updated()
#   func.func @test() {
#     call @updated() : () -> () # <---- This will be changed to @updated from @orig
#     return
#   }
#   module attributes {transform.with_named_sequence} {
#     transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
#       %0 = transform.structured.match ops{["func.call"]} in %arg0 : (!transform.any_op) -> !transform.any_op
#       transform.my.change_call_target %0, "updated" : !transform.any_op
#       transform.yield
#     }
#   }
# }
```

- transform Ch3

```bash
$ ./build/transform_Ch3/transform-opt-ch3 --transform-interpreter transform_Ch3/ops.mlir --allow-unregistered-dialect --split-input-file
# module {
#   func.func private @orig()
#   func.func private @updated()
#   func.func @test1() {
#     call @updated() : () -> ()
#     return
#   }
#   module attributes {transform.with_named_sequence} {
#     transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
#       %0 = transform.structured.match ops{["func.call"]} in %arg0 : (!transform.any_op) -> !transform.op<"func.call">
#       transform.my.change_call_target %0, "updated" : !transform.op<"func.call">
#       transform.yield 
#     }
#   }
# }

# // -----
# module {
#   func.func private @orig()
#   func.func @test2() {
#     "my.mm4"() : () -> ()
#     return
#   }
#   module attributes {transform.with_named_sequence} {
#     transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
#       %0 = transform.structured.match ops{["func.call"]} in %arg0 : (!transform.any_op) -> !transform.my.call_op_interface
#       %1 = transform.my.call_to_op %0 : (!transform.my.call_op_interface) -> !transform.any_op
#       transform.yield 
#     }
#   }
# }
```
