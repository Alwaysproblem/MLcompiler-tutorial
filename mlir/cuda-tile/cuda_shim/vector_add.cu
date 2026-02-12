// Simple vector add kernel for PTX generation targeting Ada (RTX 4090).
// nvcc -std=c++17 -arch=sm_89 -ptx vector_add.cu -o vector_add.ptx
extern "C" __global__ void vector_add(const float *a, const float *b,
                                       float *out, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = a[idx] + b[idx];
  }
}
