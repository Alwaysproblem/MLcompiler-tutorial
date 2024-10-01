# env
# apt install gcc-12 g++-12 (ubuntu 22.04)
# update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 20
# # cd /root/miniconda3/envs/jax-build/lib
# # ln -s /lib/x86_64-linux-gnu/libtinfo.so.6 libtinfo.so.6
# # conda create -n jax-build python=3.10 build numpy libstdcxx-ng=12 gxx=12.* gcc=12.* gdb=12.* gxx_linux-64==12.* cuda-tools=12.2.2 cuda-toolkit=12.2.2 cudnn cuda-nvcc_linux-64=12.2.2 cuda-libraries-dev=12.2.2 cuda-driver-dev cuda=12.2.2 cuda-compiler=12.2.2 cuda-gdb cuda-cudart-dev -c nvidia -y
# conda create -n jax-build python=3.10 build numpy cudatoolkit cuda-tools=12.2.2 cuda-toolkit=12.2.2 cudnn=8.9 cuda-nvcc_linux-64=12.2.2 cuda-libraries-dev=12.2.2 cuda-driver-dev cuda=12.2.2 cuda-compiler=12.2.2 cuda-gdb cuda-cudart-dev -c conda-forge -c nvidia -y
# conda activate jax-build
# python build/build.py --enable_cuda --configure_only --cuda_path /root/miniconda3/envs/jax-build
# bazel run --verbose_failures=true //jaxlib/tools:build_wheel -- --output_path=/root/Desktop/dockerVolumn/MLcompiler-tutorial/xla/jax/dist --cpu=x86_64
