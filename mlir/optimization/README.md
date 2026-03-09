# Optimization Overview

This tutorial series walks through key optimization techniques in ML compilers using MLIR, ordered by pedagogical progression. Each stage builds on concepts from the previous one.

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

## Plan

### Phase 1: MatMul (Foundation)

**Goal:** Establish core optimization vocabulary and mechanics.

| Topic                | Description                                                                                                                                             |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Structured Op        | Define and lower a matmul via `linalg.generic` / named ops; understand the iteration domain, indexing maps, and payload.                                |
| Tiling               | Apply `scf.forall` / `scf.for` tile-and-fuse to decompose the M×N×K loop nest; explore tile-size trade-offs.                                            |
| Locality             | Demonstrate cache-friendly access via loop permutation (MKN vs MNK), packing, and micro-kernel promotion to registers.                                  |
| Simple Cost Model    | Introduce a basic analytical model (FLOPs, memory traffic, arithmetic intensity) to guide tile-size selection.                                          |
| Pipeline Abstraction | Compose the above into a reusable pass pipeline: tile → promote → vectorize → lower, showing how MLIR pass infrastructure orchestrates transformations. |

**Deliverable:** An end-to-end optimized matmul that is competitive with a naive BLAS call, with clear before/after IR at every stage.

---

### Phase 2: Conv2D + Activation Fusion (Spatial & Fusion)

**Goal:** Extend tiling to spatial dimensions and introduce operator fusion.

| Topic          | Description                                                                                                                                                |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Fusion         | Fuse an elementwise activation (ReLU, GELU) into the convolution producer-consumer pair; understand producer-consumer analysis and the legality of fusion. |
| Spatial Tiling | Tile output height and width dimensions; manage the resulting input tile expansion due to the kernel window (halo).                                        |
| Layout         | Explore NHWC vs NCHW (and packed variants like NCHWc); understand how data layout affects vectorization and memory access patterns.                        |
| Halo / Reuse   | Handle overlapping input regions across tiles; compute the halo size from kernel size, stride, and dilation; demonstrate data reuse.                       |

**Deliverable:** A fused conv2d + activation kernel with explicit spatial tiling, demonstrating measurable speedup from fusion and layout selection.

---

### Phase 3: LayerNorm / Softmax (Reduction Scheduling)

**Goal:** Tackle reduction-heavy operations where numerical stability and scheduling are tightly coupled.

| Topic                      | Description                                                                                                                                    |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Reduction Scheduling       | Implement multi-pass (mean → variance → normalize) vs single-pass (Welford) reduction strategies; tile reductions across threads.              |
| Scratch Buffer             | Allocate and manage intermediate buffers (`memref.alloca` / workspace) for partial results; understand buffer lifetime and placement.          |
| Numerics–Schedule Coupling | Show how the softmax "max-subtract" trick and log-sum-exp rewriting are not just numerical choices but directly constrain the legal schedules. |

**Deliverable:** A numerically stable, tiled LayerNorm/Softmax implementation with clear discussion of how algorithmic rewrites enable (or block) specific schedules.

---

### Phase 4: Subgraph Fusion & Memory Planning (Graph Level)

**Goal:** Move from single-op to multi-op / graph-level optimization.

| Topic                    | Description                                                                                                                                                      |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Graph Scheduling         | Decide fusion groups and execution order across a small subgraph (e.g., matmul → bias → layernorm); model the trade-off between parallelism and memory pressure. |
| Peak Memory Optimization | Apply operator reordering, in-place updates, and buffer sharing (liveness analysis) to minimize peak memory; visualize the memory waterline before/after.        |

**Deliverable:** A small end-to-end subgraph whose peak memory and kernel count are jointly optimized, with tooling to visualize the memory timeline.

---

### Suggested Timeline

| Week | Phase                         | Key Milestone                                |
| ---- | ----------------------------- | -------------------------------------------- |
| 1–3  | Phase 1 – MatMul              | Tiled + vectorized matmul with pass pipeline |
| 4–5  | Phase 2 – Conv2D + Activation | Fused conv2d-relu with spatial tiling        |
| 6–7  | Phase 3 – LayerNorm / Softmax | Numerically stable tiled reduction           |
| 8–9  | Phase 4 – Subgraph Fusion     | Graph-level fusion with memory planning      |
| 10   | Wrap-up                       | Benchmarking, profiling, and write-up        |

