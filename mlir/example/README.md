# Standalone environment for MLIR tutorial.

*note: The code of this tutorial is from the [mlir-website-tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-1/). This repo only provide a simple way to setting up the environment.*

## Environment Setup

- OS must be higher than ubuntu 22.04.
- install gcc-11 and g++-11

```bash
apt update -y && \
apt install -yq gcc-11 g++-11
# apt install -yq software-properties-common \
# add-apt-repository -y ppa:ubuntu-toolchain-r/test \
# apt install -yq gcc-11 g++-11
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 20
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 20
```

- install cmake and ninja you can choose one way you like. conda is best for me.

```bash
conda create -n mlir -y
conda activate mlir
# conda install cmake ninja clang-format clang lld ncurses mlir llvm -c conda-forge
conda install cmake ninja clang-format clang=15.* mlir=15.* llvm=15.0.7 -c conda-forge -y
```

## build example

```bash
cd example
bash build.sh all
```
