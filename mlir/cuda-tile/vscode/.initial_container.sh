docker run -d --gpus all \
  --privileged -ti \
  --cap-add=SYS_ADMIN --cap-add=SYS_PTRACE \
  --shm-size 4G \
  --ulimit memlock=-1:-1 \
  --security-opt seccomp=unconfined --ipc=host \
  -v $PWD:/work -w /work \
  nvidia/cuda:13.0.0-devel-ubuntu22.04 bash
  # bash -lc 'nsys --version && nsys profile --trace=cuda,nvtx,osrt --stats=true -o sysrep ./matmul'
