// please run inside `nvidia/cuda:12.4.1-devel-ubuntu22.04` container if you
// want to use the 4090 RTX GPU with 12.4<= cuda <= 13.0.
// pelase compile with the command:
//  nvcc -std=c++17 -arch=sm_89 -cubin outlined_gpu_kernel.cu -o cuda_tile.cubin
#include <cuda_runtime.h>
#include <cstdio>

extern "C" __global__ void outlined_gpu_kernel_0(const float* a0, const float* a1,
                                                 const float* a2, float* out) {
  // 2x4 = 8 elements, row-major with stride (4,1)
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= 8) return;

  // out[tid] = a0[tid]*a1[tid] + a2[tid]*a1[tid]
  float x0 = a0[tid];
  float x1 = a1[tid];
  float x2 = a2[tid];
  out[tid] = x0 * x1 + x2 * x1;
  // 等价：out[tid] = (x0 + x2) * x1;
}
// A = [[1,2,3,9],[4,5,6,10]] B = [[11,12,13,114],[15,16,17,18]]
