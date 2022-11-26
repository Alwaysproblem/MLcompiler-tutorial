# MLIR

## Environment Setup

- install cmake and ninja you can choose one way you like. conda is best for me.

```bash
conda create -n mlir
conda activate mlir
conda install cmake ninja
```

- build mlir and llvm basic package from [mlir](https://mlir.llvm.org/getting_started/)

```bash
git clone https://github.com/llvm/llvm-project.git
mkdir llvm-project/build
cd llvm-project/build
cmake -G Ninja ../llvm \
   -DCMAKE_INSTALL_PREFIX=install \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON
# Using clang and lld speeds up the build, we recomment adding:
#  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON
# CCache can drastically speed up further rebuilds, try adding:
#  -DLLVM_CCACHE_BUILD=ON
# Optionally, using ASAN/UBSAN can find bugs early in development, enable with:
# -DLLVM_USE_SANITIZER="Address;Undefined"
# Optionally, enabling integration tests as well
# -DMLIR_INCLUDE_INTEGRATION_TESTS=ON
cmake --build . --target check-mlir
ninja install
```

- move the install directory to the `example` and rename with `third_party`

```bash
mv install ../../example/third_party
# cd ../../ && rm llvm-project
```

## build example

```bash
cd example
bash build all
```
