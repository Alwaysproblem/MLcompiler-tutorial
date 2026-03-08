// Minimal demo showing how to load a PTX file and launch a kernel via the cuda_shim API.
// 1) Build PTX for Ada (RTX 4090) for the sample kernel in vector_add.cu:
//    nvcc -std=c++17 -arch=sm_89 -ptx vector_add.cu -o vector_add.ptx
// 2) Build this runner together with the shim (nvcc handles the CUDA driver link flags):
//    nvcc -std=c++17 load_ptx_main.cpp cuda_shim.cc -o load_ptx_demo -lcuda -lcudart
// 3) Run: ./load_ptx_demo vector_add.ptx vector_add 1048576

// nvcc -std=c++17 --cudart static load_ptx_main.cpp cuda_shim.cc -o load_ptx_demo -lcuda -lcudadevrt -lcudart_static -ldl -lrt -pthread
// g++-11 -std=c++17 load_ptx_main.cpp cuda_shim.cc -I/usr/local/cuda/include -L/usr/lib/x86_64-linux-gnu -lcuda -ldl -pthread -o load_ptx_demo
#include <cuda.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// The shim has no public header, so we redeclare the extern "C" hooks we need.
extern "C" uint64_t cuda_shim_load_module_from_image(uint64_t image_ptr,
                                                      uint64_t image_nbytes);
extern "C" uint64_t cuda_shim_load_module_from_file(uint64_t file_path_ptr,
                                                     uint64_t file_path_nbytes);
extern "C" void cuda_shim_unload_module(uint64_t module_handle);
extern "C" uint64_t cuda_shim_malloc(uint64_t nbytes, uint64_t stream,
                                      bool is_host_shared);
extern "C" void cuda_shim_free(uint64_t dptr, uint64_t stream);
extern "C" void cuda_shim_memcpy_h2d(uint64_t dst_dptr, uint64_t src_hptr,
                                      uint64_t nbytes);
extern "C" void cuda_shim_memcpy_d2h(uint64_t dst_hptr, uint64_t src_dptr,
                                      uint64_t nbytes);
extern "C" uint64_t cuda_shim_stream_create(void);
extern "C" void cuda_shim_stream_destroy(uint64_t stream);
extern "C" void cuda_shim_stream_synchronize(uint64_t stream);
extern "C" void cuda_shim_launch_packed(uint64_t module_handle,
                                         uint64_t kernel_name_ptr,
                                         uint32_t gridX, uint32_t gridY,
                                         uint32_t gridZ, uint32_t blockX,
                                         uint32_t blockY, uint32_t blockZ,
                                         uint32_t sharedMemBytes,
                                         uint64_t stream,
                                         uint64_t arg_data_ptr,
                                         uint64_t arg_sizes_ptr,
                                         uint32_t num_args);

namespace {

// Round up to next multiple of 8 to match cuda_shim_launch_packed's alignment.
size_t align8(size_t value) { return (value + 7) & ~static_cast<size_t>(7); }

// Load an entire file into a byte buffer.
bool loadFile(const std::string &path, std::vector<char> &buffer) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Failed to open PTX file: " << path << "\n";
    return false;
  }
  file.seekg(0, std::ios::end);
  const auto size = static_cast<size_t>(file.tellg());
  file.seekg(0, std::ios::beg);
  buffer.resize(size);
  file.read(buffer.data(), buffer.size());
  return true;
}

// Append a trivially copyable argument into the packed arg buffer.
template <typename T>
void appendArg(std::vector<uint8_t> &argData, std::vector<uint64_t> &argSizes,
               const T &value) {
  const size_t aligned = align8(argData.size());
  if (aligned > argData.size()) {
    argData.resize(aligned, 0);
  }
  const uint8_t *ptr = reinterpret_cast<const uint8_t *>(&value);
  argData.insert(argData.end(), ptr, ptr + sizeof(T));
  argSizes.push_back(static_cast<uint64_t>(sizeof(T)));
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr
        << "Usage: " << argv[0]
        << " <path/to/kernel.ptx> [kernel_name=vector_add] [num_elements=1048576]\n";
    return 1;
  }

  const std::string ptxPath = argv[1];
  const std::string kernelName = (argc >= 3) ? argv[2] : std::string("vector_add");
  const int numElems = (argc >= 4) ? std::atoi(argv[3]) : (1 << 20);
  const size_t numBytes = static_cast<size_t>(numElems) * sizeof(float);

  // std::vector<char> ptx;
  // if (!loadFile(ptxPath, ptx)) {
  //   return 1;
  // }

  // Load module from the PTX blob.
  const uint64_t module_handle_for_launch =
      cuda_shim_load_module_from_file(
          reinterpret_cast<uint64_t>(ptxPath.data()),
          static_cast<uint64_t>(ptxPath.size()));
  if (module_handle_for_launch == 0) {
    std::cerr << "Failed to load module from PTX: " << ptxPath << "\n";
    return 1;
  }

  // cuda_shim_launch_packed expects a pointer to a CUmodule stored in host
  // memory. Keep a stack copy and pass its address to satisfy that ABI.
  // CUmodule module = reinterpret_cast<CUmodule>(module_handle_raw);
  // const uint64_t module_handle_for_launch =
  //     reinterpret_cast<uint64_t>(&module);

  const uint64_t stream = cuda_shim_stream_create();

  // Allocate device buffers.
  const uint64_t dOut = cuda_shim_malloc(numBytes, stream, /*is_host_shared=*/false);
  const uint64_t dA = cuda_shim_malloc(numBytes, stream, /*is_host_shared=*/false);
  const uint64_t dB = cuda_shim_malloc(numBytes, stream, /*is_host_shared=*/false);

  std::vector<float> hA(numElems);
  std::vector<float> hB(numElems);
  std::vector<float> hOut(numElems, 0.0f);
  for (int i = 0; i < numElems; ++i) {
    hA[i] = static_cast<float>(i) * 0.5f;
    hB[i] = static_cast<float>(i) * 1.5f;
  }

  cuda_shim_memcpy_h2d(dA, reinterpret_cast<uint64_t>(hA.data()), numBytes);
  cuda_shim_memcpy_h2d(dB, reinterpret_cast<uint64_t>(hB.data()), numBytes);

  // Pack kernel arguments: (float* out, const float* a, const float* b, int n)
  std::vector<uint8_t> argData;
  std::vector<uint64_t> argSizes;
  const uint64_t argOut = dOut;
  const uint64_t argA = dA;
  const uint64_t argB = dB;
  const int argN = numElems;

  appendArg(argData, argSizes, argA);
  appendArg(argData, argSizes, argB);
  appendArg(argData, argSizes, argOut);
  appendArg(argData, argSizes, argN);

  const uint32_t blockX = 256;
  const uint32_t gridX = static_cast<uint32_t>((numElems + blockX - 1) / blockX);

  cuda_shim_launch_packed(
      module_handle_for_launch,
      reinterpret_cast<uint64_t>(kernelName.c_str()),
      gridX, 1, 1,
      blockX, 1, 1,
      /*sharedMemBytes=*/0,
      stream,
      reinterpret_cast<uint64_t>(argData.data()),
      reinterpret_cast<uint64_t>(argSizes.data()),
      static_cast<uint32_t>(argSizes.size()));

  cuda_shim_stream_synchronize(stream);

  cuda_shim_memcpy_d2h(reinterpret_cast<uint64_t>(hOut.data()), dOut, numBytes);

  // Quick correctness check.
  bool ok = true;
  for (int i = 0; i < numElems; ++i) {
    const float expect = hA[i] + hB[i];
    if (std::abs(hOut[i] - expect) > 1e-5f) {
      std::cerr << "Mismatch at index " << i << ": got " << hOut[i]
                << ", expected " << expect << "\n";
      ok = false;
      break;
    }
  }

  std::cout << (ok ? "Success" : "Failure") << " for " << numElems
            << " elements" << std::endl;

  cuda_shim_free(dOut, stream);
  cuda_shim_free(dA, stream);
  cuda_shim_free(dB, stream);
  cuda_shim_stream_destroy(stream);
  cuda_shim_unload_module(module_handle_for_launch);

  return ok ? 0 : 1;
}
