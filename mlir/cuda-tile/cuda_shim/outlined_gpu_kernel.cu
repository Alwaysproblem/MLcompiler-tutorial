// please run inside `nvidia/cuda:12.4.1-devel-ubuntu22.04` container if you
// want to use the 4090 RTX GPU with 12.4<= cuda <= 13.0.
// pelase compile with the command:
//  nvcc -std=c++17 -arch=sm_89 -cubin outlined_gpu_kernel.cu -o cuda_tile.cubin
#include <cstdio>
#include <cuda_runtime.h>

extern "C" __global__ void outlined_gpu_kernel_0(const float *a0,
                                                 const float *a1, float *out) {
  // 2x4 = 8 elements, row-major with stride (4,1)
  const int n = 8;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  for (int i = tid; i < n; i += stride) {
    float x0 = a0[i];
    float x1 = a1[i];
    out[i] = x0 * x1 + x1;
  }
}

extern "C" __global__ void outlined_gpu_kernel_1(const float *a0,
                                                 const float *a1, float *out) {
  // Matmul: (2x4) * (4x2) -> (2x2), row-major layout.
  const int n = 4;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  for (int i = tid; i < n; i += stride) {
    const int row = i / 2;
    const int col = i % 2;

    float acc = 0.0f;
    for (int k = 0; k < 4; ++k) {
      acc += a0[row * 4 + k] * a1[k * 2 + col];
    }
    out[i] = acc;
  }
}

extern "C" __global__ void outlined_gpu_kernel_2(const float *a0,
                                                 const float *a1,
                                                 const float *a2, float *out) {
  // 2x4 = 8 elements, row-major with stride (4,1)
  const int n = 8;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  for (int i = tid; i < n; i += stride) {
    // out[i] = a0[i]*a1[i] + a2[i]*a1[i]
    float x0 = a0[i];
    float x1 = a1[i];
    float x2 = a2[i];
    out[i] = x0 * x1 + x2 * x1;
    // 等价：out[i] = (x0 + x2) * x1;
  }
}
// A = [[1,2,3,9],[4,5,6,10]] B = [[11,12,13,114],[15,16,17,18]]
