FROM alwaysproblem/fastdev-u2204:zsh

ARG UID=1000
ARG GID=1000

RUN echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy main" > /etc/apt/sources.list.d/llvm.list \
    && echo "deb-src http://apt.llvm.org/jammy/ llvm-toolchain-jammy main" >> /etc/apt/sources.list.d/llvm.list \
    && echo "# 18" >> /etc/apt/sources.list.d/llvm.list \
    && echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-18 main" >> /etc/apt/sources.list.d/llvm.list \
    && echo "deb-src http://apt.llvm.org/jammy/ llvm-toolchain-jammy-18 main" >> /etc/apt/sources.list.d/llvm.list \
    && echo "# 19" >> /etc/apt/sources.list.d/llvm.list \
    && echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-19 main" >> /etc/apt/sources.list.d/llvm.list \
    && echo "deb-src http://apt.llvm.org/jammy/ llvm-toolchain-jammy-19 main" >> /etc/apt/sources.list.d/llvm.list \
    && wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc \
    && apt update -y && \
    apt install -y \
    python3 python3-dev python3-setuptools \
    libtinfo-dev zlib1g-dev \
    build-essential cmake ninja-build \
    clang-19 clang-tidy-19 clangd-19 \
    clang-format-19 lldb-19 lld-19 libfmt-dev libspdlog-dev \
    && update-alternatives --install /usr/bin/clang clang /usr/bin/clang-19 100 \
    && update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-19 100 \
    && update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-19 100 \
    && update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-19 100 \
    && update-alternatives --install /usr/bin/clang-tidy clang-tidy /usr/bin/clang-tidy-19 100 \
    && update-alternatives --install /usr/bin/lld lld /usr/bin/lld-19 100 \
    && update-alternatives --install /usr/bin/lldb lldb /usr/bin/lldb-19 100

RUN git config --global --add safe.directory '*' && \
    /root/.local/bin/setup_new_user ${UID} ${GID}
