FROM alwaysproblem/fastdev-u2204:nv13.1.0

ARG UID=1000
ARG GID=1000

RUN echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy main" > /etc/apt/sources.list.d/llvm.list \
    && echo "deb-src http://apt.llvm.org/jammy/ llvm-toolchain-jammy main" >> /etc/apt/sources.list.d/llvm.list \
    && echo "# 20" >> /etc/apt/sources.list.d/llvm.list \
    && echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-20 main" >> /etc/apt/sources.list.d/llvm.list \
    && echo "deb-src http://apt.llvm.org/jammy/ llvm-toolchain-jammy-20 main" >> /etc/apt/sources.list.d/llvm.list \
    && echo "# 21" >> /etc/apt/sources.list.d/llvm.list \
    && echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-21 main" >> /etc/apt/sources.list.d/llvm.list \
    && echo "deb-src http://apt.llvm.org/jammy/ llvm-toolchain-jammy-21 main" >> /etc/apt/sources.list.d/llvm.list \
    && wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc \
    && apt update -y && \
    apt install -y \
    python3 python3-dev python3-setuptools python3-pip \
    libtinfo-dev zlib1g-dev \
    build-essential cmake ninja-build \
    clang-20 clang-tidy-20 clangd-20 cmake-format \
    clang-format-20 lldb-20 lld-20 libfmt-dev libspdlog-dev \
    && apt clean -y && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/clang clang /usr/bin/clang-20 100 \
    && update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-20 100 \
    && update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-20 100 \
    && update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-20 100 \
    && update-alternatives --install /usr/bin/clang-tidy clang-tidy /usr/bin/clang-tidy-20 100 \
    && update-alternatives --install /usr/bin/lld lld /usr/bin/lld-20 100 \
    && update-alternatives --install /usr/bin/lldb lldb /usr/bin/lldb-20 100

RUN apt update -y && apt install -yq software-properties-common \
    && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && apt update -yq \
    && apt install -yq gcc-13 g++-13 gdb \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100 \
    && apt clean -y && rm -rf /var/lib/apt/lists/*

RUN git config --global --add safe.directory '*' && \
    /root/.local/bin/setup_new_user ${UID} ${GID} && \
    python3 -m pip install pre-commit compdb
