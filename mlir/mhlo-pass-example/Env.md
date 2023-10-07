# Env prepare

Ubuntu 22.04

## use popconda to choose the sdk.

```bash
popconda -- conda create -n mhlo python=3.10 pre-commit lld cmake ninja fmt yapf pylint zlib clang-format clang=16.* clang-tools lit spdlog llvm=16.* mlir -c conda-forge -y
conda activate mhlo
```

```
cd mhlo-phlo-prototype
bash build_tools/sync_deps.sh
```

```shell
bash build_tools/build_mhlo_with_conda.sh external/mlir-hlo third_party
```

```shell
bash build_with_conda.sh
```

## non-conda user

- Install dependencies

```shell
apt install -y g++ gcc cmake ninja-build libspdlog-dev python3.10 python3.10-dev python3-pip zlib1g zlib1g-dev

# apt install clang-16 lldb-16 lld-16 cmake ninja-build libspdlog-dev zlib1g zlib1g-dev -y
# apt install -y libllvm-16-ocaml-dev libllvm16 llvm-16 llvm-16-dev llvm-16-runtime
# apt install -y libmlir-16-dev mlir-16-tools
# update-alternatives --install /usr/bin/ld.lld ld.lld /usr/bin/lld-16 20
# python3 -m pip install lit
```

- copy apt keys to `/etc/apt/sources.list`

```shell
cat <<EOF >> /etc/apt/sources.list
deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy main
# deb-src http://apt.llvm.org/jammy/ llvm-toolchain-jammy main
deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-16 main
# deb-src http://apt.llvm.org/jammy/ llvm-toolchain-jammy-16 main
deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-17 main
# deb-src http://apt.llvm.org/jammy/ llvm-toolchain-jammy-17 main
EOF
```

```shell
wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc
```

```shell
apt update -y && apt install lld-16 -y
update-alternatives --install /usr/bin/ld.lld ld.lld /usr/bin/lld-16 20
```

- clone and build llvm and mhlo

```
cd mhlo-phlo-prototype
bash build_tools/sync_deps.sh
bash build_tools/build_deps.sh
```

- Enable SDK 1470

```shell
source <your SDK-1470>
```

- build targets

```shell
bash build.sh
```

## Run

```shell
build/mhlo-popir/mhlo-popir-opt
```
