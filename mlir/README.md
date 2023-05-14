# Standalone environment for MLIR tutorial.

*note: The code of this tutorial is from the [mlir-website-tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-1/). This repo only provide a simple way to setting up the environment.*

## Environment Setup

- install cmake and ninja you can choose one way you like. conda is best for me.

```bash
conda create -n mlir
conda activate mlir
conda install cmake ninja clang lld ncurses -y
```

or

```bash
apt install zlib1g-dev g++-9 gcc-9 cmake ninja-build clang lld
```

- build mlir and llvm basic package from [mlir](https://mlir.llvm.org/getting_started/)

```bash
git clone -b release/15.x https://github.com/llvm/llvm-project.git
# git clone -b 502c246519ec7462450e0b05465063d190cadcb5 https://github.com/llvm/llvm-project.git
mkdir llvm-project/build
cd llvm-project/build
cmake -G Ninja ../llvm \
   -DCMAKE_INSTALL_PREFIX=install \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON
   # -DLLVM_ENABLE_LLD=ON -DLLVM_ENABLE_LIBEDIT=FALSE
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

## conda setup

```bash
# os must be higher than ubuntu 22.04.
# the gcc or g++ version need to be higher than gcc-11
# apt install -yq software-properties-common \
# add-apt-repository -y ppa:ubuntu-toolchain-r/test \
# apt install -yq gcc-11 g++-11
# conda install cmake ninja clang-format clang lld ncurses mlir llvm -c conda-forge
conda install cmake ninja clang-format clang=15.* mlir=15.* llvm=15.0.7 -c conda-forge
```

## build example

```bash
cd example
bash build.sh all
```
