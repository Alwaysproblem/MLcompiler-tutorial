//===- CudaRuntimeWrappers.cpp - MLIR CUDA API wrapper library ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements C wrappers around the CUDA library for easy linking in ORC jit.
// Also adds some debugging helpers that are helpful when writing MLIR code to
// run on GPUs.
//
//===----------------------------------------------------------------------===//

#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>

#include "cuda.h"
#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include <vector>

// We assume the program runs on the linux platform if not on Windows.
// Copy from
// third_party/llvm-project/mlir/lib/ExecutionEngine/CudaRuntimeWrappers.cpp

#if CUDA_VERSION >= 13000

#define MLIR_CUDA_WRAPPERS_EXPORT __attribute__((visibility("default")))

#define CUDA_REPORT_IF_ERROR(expr)                                             \
  [](CUresult result) {                                                        \
    if (!result)                                                               \
      return;                                                                  \
    const char *name = nullptr;                                                \
    cuGetErrorName(result, &name);                                             \
    if (!name)                                                                 \
      name = "<unknown>";                                                      \
    fprintf(stderr, "'%s' failed with '%s'\n", #expr, name);                   \
  }(expr)

thread_local static int32_t defaultDevice = 0;

/// Helper method that checks environment value for debugging.
static bool isDebugEnabled() {
  const char *kDebugEnvironmentVariable = "MLIR_CUDA_DEBUG";
  static bool isEnabled = getenv(kDebugEnvironmentVariable) != nullptr;
  return isEnabled;
}

#define debug_print(fmt, ...)                                                  \
  do {                                                                         \
    if (isDebugEnabled())                                                      \
      fprintf(stderr, "%s:%d:%s(): " fmt, "CudaRuntimeWrappers.cpp", __LINE__, \
              __func__, __VA_ARGS__);                                          \
  } while (0)

// Returns default CUdevice
static CUdevice getDefaultCuDevice() {
  CUdevice device;
  CUDA_REPORT_IF_ERROR(cuDeviceGet(&device, /*ordinal=*/defaultDevice));
  return device;
}

// Make the primary context of the current default device current for the
// duration
//  of the instance and restore the previous context on destruction.
class ScopedContext {
public:
  ScopedContext() {
    // Static reference to CUDA primary context for device ordinal
    // defaultDevice.
    static CUcontext context = [] {
      CUDA_REPORT_IF_ERROR(cuInit(/*flags=*/0));
      CUcontext ctx;
      // Note: this does not affect the current context.
      CUDA_REPORT_IF_ERROR(
          cuDevicePrimaryCtxRetain(&ctx, getDefaultCuDevice()));
      return ctx;
    }();

    CUDA_REPORT_IF_ERROR(cuCtxPushCurrent(context));
  }

  ~ScopedContext() { CUDA_REPORT_IF_ERROR(cuCtxPopCurrent(nullptr)); }
};

extern "C" MLIR_CUDA_WRAPPERS_EXPORT CUmodule
mgpuModuleLoad(void *data, size_t /*gpuBlobSize*/) {
  ScopedContext scopedContext;
  CUmodule module = nullptr;
  CUDA_REPORT_IF_ERROR(cuModuleLoadData(&module, data));
  return module;
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT CUmodule mgpuModuleLoadJIT(void *data,
                                                                int optLevel) {
  ScopedContext scopedContext;
  CUmodule module = nullptr;
  char jitErrorBuffer[4096] = {0};
  CUjit_option jitOptions[] = {CU_JIT_ERROR_LOG_BUFFER,
                               CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                               CU_JIT_OPTIMIZATION_LEVEL};
  void *jitOptionsVals[] = {jitErrorBuffer,
                            reinterpret_cast<void *>(sizeof(jitErrorBuffer)),
                            reinterpret_cast<void *>(optLevel)};

  CUresult result =
      cuModuleLoadDataEx(&module, data, 3, jitOptions, jitOptionsVals);
  if (result) {
    fprintf(stderr, "JIT compilation failed with: '%s'\n", jitErrorBuffer);
    CUDA_REPORT_IF_ERROR(result);
  }
  return module;
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuModuleUnload(CUmodule module) {
  CUDA_REPORT_IF_ERROR(cuModuleUnload(module));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT CUfunction
mgpuModuleGetFunction(CUmodule module, const char *name) {
  CUfunction function = nullptr;
  CUDA_REPORT_IF_ERROR(cuModuleGetFunction(&function, module, name));
  return function;
}

// The wrapper uses intptr_t instead of CUDA's unsigned int to match
// the type of MLIR's index type. This avoids the need for casts in the
// generated MLIR code.
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuLaunchKernel(CUfunction function, intptr_t gridX, intptr_t gridY,
                 intptr_t gridZ, intptr_t blockX, intptr_t blockY,
                 intptr_t blockZ, int32_t smem, CUstream stream, void **params,
                 void **extra, size_t /*paramsCount*/) {
  ScopedContext scopedContext;
  if (smem > 0) {
    // Avoid checking driver as it's more expensive than if statement
    int32_t maxShmem = 0;
    CUdevice device = getDefaultCuDevice();
    CUDA_REPORT_IF_ERROR(cuDeviceGet(&device, /*ordinal=*/defaultDevice));
    CUDA_REPORT_IF_ERROR(cuDeviceGetAttribute(
        &maxShmem, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        device));
    if (maxShmem < smem) {
      fprintf(stderr,
              "Requested shared memory (%dkb) is larger than maximum allowed "
              "shared memory (%dkb) for this device\n",
              smem, maxShmem);
    }
    CUDA_REPORT_IF_ERROR(cuFuncSetAttribute(
        function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem));
  }
  debug_print("Launching kernel, grid=%ld,%ld,%ld, "
              "threads: %ld, %ld, %ld, "
              "smem: %dkb\n",
              gridX, gridY, gridZ, blockX, blockY, blockZ, smem);
  CUDA_REPORT_IF_ERROR(cuLaunchKernel(function, gridX, gridY, gridZ, blockX,
                                      blockY, blockZ, smem, stream, params,
                                      extra));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT CUstream mgpuStreamCreate() {
  ScopedContext scopedContext;
  CUstream stream = nullptr;
  CUDA_REPORT_IF_ERROR(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  return stream;
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuStreamDestroy(CUstream stream) {
  CUDA_REPORT_IF_ERROR(cuStreamDestroy(stream));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuStreamSynchronize(CUstream stream) {
  CUDA_REPORT_IF_ERROR(cuStreamSynchronize(stream));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuStreamWaitEvent(CUstream stream,
                                                              CUevent event) {
  CUDA_REPORT_IF_ERROR(cuStreamWaitEvent(stream, event, /*flags=*/0));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT CUevent mgpuEventCreate() {
  ScopedContext scopedContext;
  CUevent event = nullptr;
  CUDA_REPORT_IF_ERROR(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING));
  return event;
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuEventDestroy(CUevent event) {
  CUDA_REPORT_IF_ERROR(cuEventDestroy(event));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuEventSynchronize(CUevent event) {
  CUDA_REPORT_IF_ERROR(cuEventSynchronize(event));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuEventRecord(CUevent event,
                                                          CUstream stream) {
  CUDA_REPORT_IF_ERROR(cuEventRecord(event, stream));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void *
mgpuMemAlloc(uint64_t sizeBytes, CUstream stream, bool isHostShared) {
  ScopedContext scopedContext;
  CUdeviceptr ptr = 0;
  if (sizeBytes == 0)
    return reinterpret_cast<void *>(ptr);

  if (isHostShared) {
    CUDA_REPORT_IF_ERROR(
        cuMemAllocManaged(&ptr, sizeBytes, CU_MEM_ATTACH_GLOBAL));
    return reinterpret_cast<void *>(ptr);
  }
  CUDA_REPORT_IF_ERROR(cuMemAlloc(&ptr, sizeBytes));
  return reinterpret_cast<void *>(ptr);
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuMemFree(void *ptr,
                                                      CUstream /*stream*/) {
  CUDA_REPORT_IF_ERROR(cuMemFree(reinterpret_cast<CUdeviceptr>(ptr)));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuMemcpy(void *dst, void *src, size_t sizeBytes, CUstream stream) {
  CUDA_REPORT_IF_ERROR(cuMemcpyAsync(reinterpret_cast<CUdeviceptr>(dst),
                                     reinterpret_cast<CUdeviceptr>(src),
                                     sizeBytes, stream));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuMemset32(void *dst, unsigned int value, size_t count, CUstream stream) {
  CUDA_REPORT_IF_ERROR(cuMemsetD32Async(reinterpret_cast<CUdeviceptr>(dst),
                                        value, count, stream));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuMemset16(void *dst, unsigned short value, size_t count, CUstream stream) {
  CUDA_REPORT_IF_ERROR(cuMemsetD16Async(reinterpret_cast<CUdeviceptr>(dst),
                                        value, count, stream));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuSetDefaultDevice(int32_t device) {
  defaultDevice = device;
}

// ===----------------------------------------------------------------------===//

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuCtxSynchronize() {
  ScopedContext scopedContext;
  CUDA_REPORT_IF_ERROR(cuCtxSynchronize());
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuMemcpyHtoD(void *dst, void *src,
                                                         size_t sizeBytes) {
  CUDA_REPORT_IF_ERROR(
      cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(dst), src, sizeBytes));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuMemcpyDtoH(void *dst, void *src,
                                                         size_t sizeBytes) {
  CUDA_REPORT_IF_ERROR(
      cuMemcpyDtoH(dst, reinterpret_cast<CUdeviceptr>(src), sizeBytes));
}

//===----------------------------------------------------------------------===//

static inline CUdeviceptr asDevPtr(uint64_t h) {
  return static_cast<CUdeviceptr>(h);
}
static inline uint64_t asHandle(CUdeviceptr p) {
  return static_cast<uint64_t>(p);
}

static inline CUstream asStream(uint64_t h) {
  return reinterpret_cast<CUstream>(static_cast<uintptr_t>(h));
}
static inline uint64_t asStreamHandle(CUstream s) {
  return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(s));
}

static inline CUevent asEvent(uint64_t h) {
  return reinterpret_cast<CUevent>(static_cast<uintptr_t>(h));
}
static inline uint64_t asEventHandle(CUevent e) {
  return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(e));
}

static inline void *asHostPtr(uint64_t h) {
  return reinterpret_cast<void *>(static_cast<uintptr_t>(h));
}
static inline const void *asHostCPtr(uint64_t h) {
  return reinterpret_cast<const void *>(static_cast<uintptr_t>(h));
}

// Align up helper
static inline uint64_t alignUp(uint64_t x, uint64_t a) {
  return (x + (a - 1)) & ~(a - 1);
}

// Load module from PTX or CUBIN image in memory.
// Driver API supports cuModuleLoadDataEx for both PTX and cubin (it
// auto-detects).
extern "C" uint64_t cuda_shim_load_module_from_image(uint64_t image_ptr,
                                                     uint64_t image_nbytes) {

  (void)image_nbytes;
  auto data = const_cast<void *>(asHostCPtr(image_ptr));
  CUmodule mod = mgpuModuleLoad(data, image_nbytes);
  return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(mod));
}

extern "C" uint64_t cuda_shim_load_module_jit_from_image(uint64_t image_ptr,
                                                         uint64_t image_nbytes,
                                                         int opt_level) {

  (void)image_nbytes;
  auto data = const_cast<void *>(asHostCPtr(image_ptr));
  CUmodule mod = mgpuModuleLoadJIT(data, opt_level);
  return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(mod));
}

extern "C" uint64_t
cuda_shim_load_module_from_file(uint64_t file_path_ptr,
                                uint64_t /*file_path_nbytes*/) {
  auto file_path_cstr =
      reinterpret_cast<const char *>(asHostCPtr(file_path_ptr));
  // fprintf(stdout, "%s", file_path_cstr);
  CUmodule module = nullptr;
  ScopedContext scopedContext;
  CUDA_REPORT_IF_ERROR(cuModuleLoad(&module, file_path_cstr));
  return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(module));
}

extern "C" void cuda_shim_unload_module(uint64_t module_handle) {
  CUmodule module =
      reinterpret_cast<CUmodule>(static_cast<uintptr_t>(module_handle));
  mgpuModuleUnload(module);
}

extern "C" uint64_t cuda_shim_malloc(uint64_t nbytes, uint64_t stream,
                                     bool is_host_shared) {
  CUstream cu_stream = asStream(stream);
  if (stream == 0)
    cu_stream = nullptr;
  void *ptr = mgpuMemAlloc(nbytes, /*stream=*/cu_stream,
                           /*isHostShared=*/is_host_shared);
  return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
}

extern "C" void cuda_shim_free(uint64_t dptr, uint64_t stream) {
  CUstream cu_stream = asStream(stream);
  void *ptr = reinterpret_cast<void *>(static_cast<uintptr_t>(dptr));
  if (stream == 0) {
    cu_stream = nullptr;
  }
  mgpuMemFree(ptr, /*stream=*/cu_stream);
}

extern "C" void cuda_shim_memset32(uint64_t dptr, uint32_t value,
                                   uint64_t count_dwords, uint64_t stream) {
  void *ptr = reinterpret_cast<void *>(static_cast<uintptr_t>(dptr));
  CUstream cu_stream = asStream(stream);
  mgpuMemset32(ptr, value, count_dwords, cu_stream);
}

extern "C" void cuda_shim_memset16(uint64_t dptr, uint32_t value,
                                   uint64_t count_dwords, uint64_t stream) {
  void *ptr = reinterpret_cast<void *>(static_cast<uintptr_t>(dptr));
  CUstream cu_stream = asStream(stream);
  mgpuMemset16(ptr, value, count_dwords, cu_stream);
}

extern "C" uint64_t cuda_shim_stream_create(void) {
  CUstream stream = mgpuStreamCreate();
  return asStreamHandle(stream);
}

extern "C" void cuda_shim_stream_destroy(uint64_t stream) {
  CUstream cu_stream = asStream(stream);
  mgpuStreamDestroy(cu_stream);
}

extern "C" void cuda_shim_stream_synchronize(uint64_t stream) {
  CUstream cu_stream = asStream(stream);
  mgpuStreamSynchronize(cu_stream);
}

extern "C" uint64_t cuda_shim_event_create(void) {
  CUevent event = mgpuEventCreate();
  return asEventHandle(event);
}

extern "C" void cuda_shim_event_destroy(uint64_t ev) {
  CUevent event = asEvent(ev);
  mgpuEventDestroy(event);
}

extern "C" void cuda_shim_event_record(uint64_t ev, uint64_t stream) {
  CUevent event = asEvent(ev);
  CUstream cu_stream = asStream(stream);
  mgpuEventRecord(event, cu_stream);
}

extern "C" void cuda_shim_event_synchronize(uint64_t ev) {
  CUevent event = asEvent(ev);
  mgpuEventSynchronize(event);
}

extern "C" void cuda_shim_stream_wait_event(uint64_t stream, uint64_t ev) {
  CUstream cu_stream = asStream(stream);
  CUevent event = asEvent(ev);
  mgpuStreamWaitEvent(cu_stream, event);
}

// ----------------------------- Memcpy (raw ABI) --------------------------
// Host pointers are passed as uint64_t. This is the key of 2A.

extern "C" void cuda_shim_memcpy_h2d(uint64_t dst_dptr, uint64_t src_hptr,
                                     uint64_t nbytes) {
  ScopedContext scopedContext;
  auto dst = asHostPtr(dst_dptr);
  auto src = asHostPtr(src_hptr);
  mgpuMemcpyHtoD(dst, src, static_cast<size_t>(nbytes));
}

extern "C" void cuda_shim_memcpy_d2h(uint64_t dst_hptr, uint64_t src_dptr,
                                     uint64_t nbytes) {
  ScopedContext scopedContext;
  auto dst = asHostPtr(dst_hptr);
  auto src = asHostPtr(src_dptr);
  mgpuMemcpyDtoH(dst, src, static_cast<size_t>(nbytes));
}

// ----------------------------- Kernel launch -----------------------------
// The hardest part is kernelParams (void**).
// We avoid building it in MLIR. Instead MLIR passes:
// - arg_data_ptr: host pointer to a packed buffer containing raw argument bytes
// - arg_sizes_ptr: host pointer to uint64_t[num_args], each is the byte-size of
// that argument The shim constructs kernelParams[i] = &arg_data[offset_i] with
// 8-byte alignment. This matches typical ABI expectations for scalar/pointer
// args. If you have special alignment requirements, extend this (e.g., per-arg
// alignment array).

extern "C" void cuda_shim_launch_packed(
    uint64_t module_handle, uint64_t kernel_name_ptr, uint32_t gridX,
    uint32_t gridY, uint32_t gridZ, uint32_t blockX, uint32_t blockY,
    uint32_t blockZ, uint32_t sharedMemBytes, uint64_t stream,
    uint64_t arg_data_ptr, uint64_t arg_sizes_ptr, uint32_t num_args) {

  auto mh = reinterpret_cast<CUmodule>(static_cast<uintptr_t>(module_handle));
  if (!mh) {
    fprintf(stderr, "[cuda_shim] launch_packed: invalid module handle\n");
    abort();
  }

  const char *kname =
      reinterpret_cast<const char *>(asHostCPtr(kernel_name_ptr));
  if (!kname) {
    fprintf(stderr, "[cuda_shim] launch_packed: null kernel name\n");
    abort();
  }

  CUfunction fn = mgpuModuleGetFunction(mh, kname);

  auto *argData = reinterpret_cast<uint8_t *>(asHostPtr(arg_data_ptr));
  auto *argSizes =
      reinterpret_cast<const uint64_t *>(asHostCPtr(arg_sizes_ptr));

  if (num_args > 0 && (!argData || !argSizes)) {
    fprintf(stderr, "[cuda_shim] launch_packed: argData/argSizes null\n");
    abort();
  }

  // Build kernelParams array on heap (safe for large num_args).
  std::vector<void *> params;
  params.resize(num_args);

  uint64_t off = 0;
  for (uint32_t i = 0; i < num_args; ++i) {
    // 8-byte align each argument start (common safe default).
    off = alignUp(off, 8);
    params[i] = argData + off;
    off += argSizes[i];
  }

  auto cu_stream = asStream(stream);

  if (stream == 0) {
    cu_stream = nullptr;
  }

  mgpuLaunchKernel(fn, static_cast<intptr_t>(gridX),
                   static_cast<intptr_t>(gridY), static_cast<intptr_t>(gridZ),
                   static_cast<intptr_t>(blockX), static_cast<intptr_t>(blockY),
                   static_cast<intptr_t>(blockZ),
                   static_cast<int32_t>(sharedMemBytes), cu_stream,
                   params.data(), nullptr, static_cast<size_t>(num_args));
}

// Convenience: 1D launch, shared=0, stream optional
extern "C" void
cuda_shim_launch_block_packed(uint64_t module_handle, uint64_t kernel_name_ptr,
                              uint32_t blockX, uint32_t blockY, uint32_t blockZ,
                              uint64_t stream, uint64_t arg_data_ptr,
                              uint64_t arg_sizes_ptr, uint32_t num_args) {
  cuda_shim_launch_packed(module_handle, kernel_name_ptr, 1, 1, 1, blockX,
                          blockY, blockZ, 0, stream, arg_data_ptr,
                          arg_sizes_ptr, num_args);
}

// Optional: global sync (avoid in async pipeline; prefer event/stream sync)
extern "C" void cuda_shim_ctx_synchronize(void) { mgpuCtxSynchronize(); }

// only for debugging
extern "C" void cuda_debug_dump_float(uint64_t dptr, int n) {
  auto *p = reinterpret_cast<const float*>(static_cast<uintptr_t>(dptr));
  for (uint32_t i = 0; i < n; ++i) {
    fprintf(stderr, "i=%u v=%f\n", i, p[i]);
  }
}

#endif
