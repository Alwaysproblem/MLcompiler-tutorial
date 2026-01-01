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

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda.h"
#include "cuda_bf16.h"
#include "cuda_fp16.h"

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

#endif
