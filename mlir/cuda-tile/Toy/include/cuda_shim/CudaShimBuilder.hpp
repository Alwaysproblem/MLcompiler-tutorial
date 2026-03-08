//===-------------------- CudaShimBuilder.hpp -----------------------------===//
//
// CUDA Runtime Shim Layer for MLIR Lowering
//
// This header provides a thin abstraction layer ("shim") between MLIR-generated
// host code and the CUDA Driver API. It is designed to be used during MLIR
// dialect lowering passes (e.g., from a high-level Toy dialect or GPU dialect
// down to function calls that interact with the CUDA runtime).
//
// Key components:
//
//   CudaShimFn (enum class)
//     Enumerates all supported CUDA shim operations, organized into categories:
//       - Module:  Loading/unloading CUDA modules (PTX/CUBIN)
//       - Memory:  Device memory allocation, deallocation, and host↔device
//                  memory transfers (memcpy H2D / D2H)
//       - Stream:  Stream creation, destruction, and synchronization
//       - Event:   (Commented out) Event lifecycle and synchronization
//       - Launch:  Kernel launch with packed arguments (full grid/block config
//                  or simplified block-only variant)
//       - Context: Context-level synchronization
//
//   CudaShimRegistry (class)
//     Manages the declaration and caching of CUDA shim function declarations
//     within an MLIR ModuleOp. Provides:
//       - getOrInsert(): Lazily declares a shim function in the module IR
//       - call():        Emits a func.call operation to a shim function
//     Function signatures are defined internally via specOf() and map each
//     CudaShimFn to its C ABI-compatible MLIR FunctionType (using i64/i32/i1).
//
//   Utility Functions:
//     - createGlobalForStringAttr():  Creates a memref.global for a
//         null-terminated string (e.g., kernel names, file paths)
//     - getIndexFromValue():          Extracts an aligned pointer from a memref
//         value and casts it to i64
//     - getIndexFromGlobalMemref():   Combines memref.get_global + pointer
//         extraction for global memrefs
//     - createCallToCudaShimMalloc(): Convenience wrapper to emit a
//         cuda_shim_malloc call with host-shared flag
//     - getNbytes():                  Computes the byte size of a MemRefType
//
//   C ABI Declarations (extern "C"):
//     Forward declarations of all CUDA shim runtime functions. These are
//     implemented in a companion .cpp/.cu file and wrap CUDA Driver API calls
//     behind a flat C ABI using uint64_t handles for opaque CUDA objects
//     (modules, streams, events, device pointers).
//
//   JIT Symbol Registration:
//     - buildCudaShimSymbolMap():  Builds an ORC JIT symbol map linking shim
//         function names to their native addresses
//     - registerCudaShimSymbols(): Registers all shim symbols with an MLIR
//         ExecutionEngine, enabling JIT execution of lowered MLIR programs
//         that call into the CUDA shim layer
//
// Usage:
//   1. Instantiate CudaShimRegistry with the top-level ModuleOp.
//   2. In lowering patterns, use registry.call() to emit shim invocations.
//   3. For JIT execution, call registerCudaShimSymbols() on the
//      ExecutionEngine before invoking the compiled module.
//
//===----------------------------------------------------------------------===//

#ifndef TOY_CUDA_SHIM_BUILDER_H
#define TOY_CUDA_SHIM_BUILDER_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

enum class CudaShimFn {
  // ----- Module -----
  LoadModuleFromImage,
  LoadModuleFromFile,
  UnloadModule,

  // ----- Memory -----
  Malloc,
  Free,
  // Memset32,
  // Memset16,
  MemcpyH2D,
  MemcpyD2H,

  // ----- Stream -----
  StreamCreate,
  StreamDestroy,
  StreamSynchronize,
  // StreamWaitEvent,

  // ----- Event -----
  // EventCreate,
  // EventDestroy,
  // EventRecord,
  // EventSynchronize,

  // ----- Kernel Launch -----
  LaunchPacked,
  LaunchBlockPacked,

  // ----- Context -----
  CtxSynchronize
};

static llvm::DenseMap<CudaShimFn, llvm::StringRef> CudaShimFnNames = {
    {CudaShimFn::LoadModuleFromImage, "cuda_shim_load_module_from_image"},
    {CudaShimFn::LoadModuleFromFile, "cuda_shim_load_module_from_file"},
    {CudaShimFn::UnloadModule, "cuda_shim_unload_module"},
    {CudaShimFn::Malloc, "cuda_shim_malloc"},
    {CudaShimFn::Free, "cuda_shim_free"},
    {CudaShimFn::MemcpyH2D, "cuda_shim_memcpy_h2d"},
    {CudaShimFn::MemcpyD2H, "cuda_shim_memcpy_d2h"},
    {CudaShimFn::StreamCreate, "cuda_shim_stream_create"},
    {CudaShimFn::StreamDestroy, "cuda_shim_stream_destroy"},
    {CudaShimFn::StreamSynchronize, "cuda_shim_stream_synchronize"},
    {CudaShimFn::LaunchPacked, "cuda_shim_launch_packed"},
    {CudaShimFn::LaunchBlockPacked, "cuda_shim_launch_grid_packed"},
    {CudaShimFn::CtxSynchronize, "cuda_shim_ctx_synchronize"},
};

static llvm::DenseMap<CudaShimFn, llvm::StringRef> CudaShimFnType = {
    {CudaShimFn::LoadModuleFromImage, "i64, i64 -> i64"},
    {CudaShimFn::LoadModuleFromFile, "i64, i64 -> i64"},
    {CudaShimFn::UnloadModule, "i64"},
    {CudaShimFn::Malloc, "i64, i64, i1 -> i64"},
    {CudaShimFn::Free, "i64, i64"},
    {CudaShimFn::MemcpyH2D, "i64, i64, i64"},
    {CudaShimFn::MemcpyD2H, "i64, i64, i64"},
    {CudaShimFn::StreamCreate, " -> i64"},
    {CudaShimFn::StreamDestroy, "i64"},
    {CudaShimFn::StreamSynchronize, "i64"},
    {CudaShimFn::LaunchPacked, "i64, i64, i32, i32, i32, i64, i64, i64, i32"},
    {CudaShimFn::LaunchBlockPacked,
     "i64, i64, i32, i32, i32, i64, i64, i64, i32"},
    {CudaShimFn::CtxSynchronize, "i64"},
};

static llvm::SmallVector<mlir::Type>
parseTypeList(mlir::OpBuilder &rewriter, llvm::StringRef typeListStr) {
  llvm::SmallVector<mlir::Type> types;
  for (auto typeStr : llvm::split(typeListStr, ',')) {
    typeStr = typeStr.trim();
    if (typeStr == "i64") {
      types.push_back(rewriter.getI64Type());
    } else if (typeStr == "i32") {
      types.push_back(rewriter.getI32Type());
    } else if (typeStr == "i1") {
      types.push_back(rewriter.getI1Type());
    } else {
      llvm_unreachable("Unsupported type in CudaShimFnType");
    }
  }
  return types;
}

static mlir::FunctionType
getFunctionTypeForCudaShimFn(mlir::OpBuilder &rewriter,
                             llvm::StringRef description) {
  auto i64 = rewriter.getI64Type();
  auto i32 = rewriter.getI32Type();
  auto i1 = rewriter.getI1Type();
  auto io = description.split("->");
  io.first = io.first.trim();
  io.second = io.second.trim();
  if (io.first.empty() && io.second.empty()) {
    return rewriter.getFunctionType({}, {});
  } else if (io.first.empty()) {
    return rewriter.getFunctionType({}, parseTypeList(rewriter, io.second));
  } else if (io.second.empty()) {
    return rewriter.getFunctionType(parseTypeList(rewriter, io.first), {});
  } else {
    return rewriter.getFunctionType(parseTypeList(rewriter, io.first),
                                    parseTypeList(rewriter, io.second));
  }
}

class CudaShimRegistry {
public:
  explicit CudaShimRegistry(mlir::ModuleOp module) : module(module) {}

  mlir::func::FuncOp getOrInsert(mlir::OpBuilder &rewriter,
                                 mlir::Operation *anchor, CudaShimFn which) {
    auto key = static_cast<unsigned>(which);
    if (auto it = cache.find(key); it != cache.end())
      return it->second;

    auto spec = specOf(which, rewriter);
    auto existing = module.lookupSymbol<mlir::func::FuncOp>(spec.name);
    if (existing) {
      cache[key] = existing;
      return existing;
    }

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());

    auto f = mlir::func::FuncOp::create(rewriter, anchor->getLoc(), spec.name,
                                        spec.ty);
    f.setPrivate();
    cache[key] = f;
    return f;
  }

  mlir::func::CallOp call(mlir::OpBuilder &rewriter, mlir::Operation *anchor,
                          CudaShimFn which, mlir::ValueRange operands = {}) {
    auto f = getOrInsert(rewriter, anchor, which);

    return mlir::func::CallOp::create(rewriter, anchor->getLoc(), f.getName(),
                                      f.getFunctionType().getResults(),
                                      operands);
  }

private:
  struct Spec {
    mlir::StringRef name;
    mlir::FunctionType ty;
  };

  static Spec specOf(CudaShimFn which, mlir::OpBuilder &rewriter) {
    return {CudaShimFnNames[which],
            getFunctionTypeForCudaShimFn(rewriter, CudaShimFnType[which])};
    //   auto i64 = rewriter.getI64Type();
    //   auto i32 = rewriter.getI32Type();
    //   auto i1 = rewriter.getI1Type();

    //   switch (which) {

    //   // ===== Module =====
    //   case CudaShimFn::LoadModuleFromImage:
    //     return {CudaShimFnNames[which],
    //             rewriter.getFunctionType({i64, i64}, {i64})};

    //   case CudaShimFn::LoadModuleFromFile:
    //     return {CudaShimFnNames[which],
    //             rewriter.getFunctionType({i64, i64}, {i64})};

    //   case CudaShimFn::UnloadModule:
    //     return {CudaShimFnNames[which], rewriter.getFunctionType({i64}, {})};

    //   // ===== Memory =====
    //   case CudaShimFn::Malloc:
    //     return {CudaShimFnNames[which],
    //             rewriter.getFunctionType({i64, i64, i1}, {i64})};

    //   case CudaShimFn::Free:
    //     return {CudaShimFnNames[which], rewriter.getFunctionType({i64, i64},
    //     {})};

    //     // case CudaShimFn::Memset32:
    //     //   return {"cuda_shim_memset32",
    //     //           rewriter.getFunctionType({i64, i32, i64, i64}, {})};

    //     // case CudaShimFn::Memset16:
    //     //   return {"cuda_shim_memset16",
    //     //           rewriter.getFunctionType({i64, i32, i64, i64}, {})};

    //   case CudaShimFn::MemcpyH2D:
    //     return {CudaShimFnNames[which],
    //             rewriter.getFunctionType({i64, i64, i64}, {})};

    //   case CudaShimFn::MemcpyD2H:
    //     return {CudaShimFnNames[which],
    //             rewriter.getFunctionType({i64, i64, i64}, {})};

    //   // ===== Stream =====
    //   case CudaShimFn::StreamCreate:
    //     return {CudaShimFnNames[which], rewriter.getFunctionType({}, {i64})};

    //   case CudaShimFn::StreamDestroy:
    //     return {CudaShimFnNames[which], rewriter.getFunctionType({i64}, {})};

    //   case CudaShimFn::StreamSynchronize:
    //     return {CudaShimFnNames[which],
    //             rewriter.getFunctionType({i64}, {})};

    //   // case CudaShimFn::StreamWaitEvent:
    //   //   return {"cuda_shim_stream_wait_event",
    //   //           rewriter.getFunctionType({i64, i64}, {})};

    //   // ===== Event =====
    //   // case CudaShimFn::EventCreate:
    //   //   return {"cuda_shim_event_create", rewriter.getFunctionType({},
    //   {i64})};

    //   // case CudaShimFn::EventDestroy:
    //   //   return {"cuda_shim_event_destroy", rewriter.getFunctionType({i64},
    //   //   {})};

    //   // case CudaShimFn::EventRecord:
    //   //   return {"cuda_shim_event_record",
    //   //           rewriter.getFunctionType({i64, i64}, {})};

    //   // case CudaShimFn::EventSynchronize:
    //   //   return {"cuda_shim_event_synchronize",
    //   //           rewriter.getFunctionType({i64}, {})};

    //   // ===== Launch =====
    //   case CudaShimFn::LaunchPacked:
    //     return {"cuda_shim_launch_packed",
    //             rewriter.getFunctionType(
    //                 {
    //                     i64,           // module_handle
    //                     i64,           // kernel_name_ptr
    //                     i32, i32, i32, // grid
    //                     i32, i32, i32, // block
    //                     i32,           // sharedMemBytes
    //                     i64,           // stream
    //                     i64,           // arg_data_ptr
    //                     i64,           // arg_sizes_ptr
    //                     i32            // num_args
    //                 },
    //                 {})};

    //   case CudaShimFn::LaunchBlockPacked:
    //     return {"cuda_shim_launch_grid_packed",
    //             rewriter.getFunctionType(
    //                 {
    //                     i64,           // module_handle
    //                     i64,           // kernel_name_ptr
    //                     i32, i32, i32, // block
    //                     i64,           // stream
    //                     i64,           // arg_data_ptr
    //                     i64,           // arg_sizes_ptr
    //                     i32            // num_args
    //                 },
    //                 {})};

    //   // ===== Context =====
    //   case CudaShimFn::CtxSynchronize:
    //     return {"cuda_shim_ctx_synchronize", rewriter.getFunctionType({},
    //     {})};
    //   }

    //   llvm_unreachable("Unhandled CudaShimFn");
  }

  mlir::ModuleOp module;
  llvm::DenseMap<unsigned, mlir::func::FuncOp> cache;
};

inline mlir::memref::GlobalOp
createGlobalForStringAttr(mlir::OpBuilder &rewriter, mlir::Operation *op,
                          llvm::StringRef sym_name, mlir::StringAttr attr) {
  auto loc = op->getLoc();
  auto moduleOp = op->getParentOfType<mlir::ModuleOp>();

  if (auto global = moduleOp.lookupSymbol<mlir::memref::GlobalOp>(sym_name);
      global) {
    return global;
  }

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());

  auto str = attr.getValue();
  std::vector<uint8_t> bytes(str.begin(), str.end());
  bytes.push_back(0);

  auto type = mlir::RankedTensorType::get({(int64_t)bytes.size()},
                                          rewriter.getIntegerType(8));

  auto memrefType = mlir::MemRefType::get({(int64_t)bytes.size()},
                                          rewriter.getIntegerType(8));

  auto denseAttr =
      mlir::DenseElementsAttr::get(type, llvm::ArrayRef<uint8_t>(bytes));

  auto global = mlir::memref::GlobalOp::create(
      rewriter, loc, sym_name,
      /*sym_visibility=*/rewriter.getStringAttr("private"), memrefType,
      denseAttr,
      /*constant=*/true,
      /*alignment=*/nullptr);

  return global;
}

inline mlir::arith::IndexCastOp getIndexFromValue(mlir::OpBuilder &rewriter,
                                                  mlir::Location loc,
                                                  mlir::Value value) {
  auto extractOp = mlir::memref::ExtractAlignedPointerAsIndexOp::create(
      rewriter, loc, rewriter.getIndexType(), value);
  auto indexCastOp = mlir::arith::IndexCastOp::create(
      rewriter, loc, rewriter.getI64Type(), extractOp.getResult());
  return indexCastOp;
}

inline mlir::arith::IndexCastOp
getIndexFromGlobalMemref(mlir::OpBuilder &rewriter, mlir::Location loc,
                         mlir::memref::GlobalOp global) {

  auto getGlobalOp = mlir::memref::GetGlobalOp::create(
      rewriter, loc, global.getType(), global.getName());

  return getIndexFromValue(rewriter, loc, getGlobalOp.getResult());
}

inline mlir::func::CallOp createCallToCudaShimMalloc(
    mlir::OpBuilder &rewriter, mlir::Location loc, CudaShimRegistry &registry,
    mlir::func::CallOp stream, mlir::arith::ConstantIntOp nbytesVal,
    bool isHostShared) {
  mlir::arith::ConstantIntOp isHostSharedVal;
  if (isHostShared) {
    isHostSharedVal = mlir::arith::ConstantIntOp::create(rewriter, loc, 1, 1);
  } else {
    isHostSharedVal = mlir::arith::ConstantIntOp::create(rewriter, loc, 0, 1);
  }
  auto sreamVal = stream.getResult(0);
  auto callee =
      registry.call(rewriter, stream, CudaShimFn::Malloc,
                    mlir::ValueRange{nbytesVal, sreamVal, isHostSharedVal});
  return callee;
}

inline unsigned long getNbytes(mlir::Type tensorType) {
  auto ranked_tensor_type = llvm::cast<mlir::MemRefType>(tensorType);
  return llvm::divideCeil(ranked_tensor_type.getNumElements() *
                              ranked_tensor_type.getElementTypeBitWidth(),
                          8);
}

extern "C" {
// Load module from PTX or CUBIN image in memory.
// Driver API supports cuModuleLoadDataEx for both PTX and cubin (it
// auto-detects).
uint64_t cuda_shim_load_module_from_image(uint64_t image_ptr,
                                          uint64_t image_nbytes);
uint64_t cuda_shim_load_module_jit_from_image(uint64_t image_ptr,
                                              uint64_t image_nbytes,
                                              int opt_level);

uint64_t cuda_shim_load_module_from_file(uint64_t file_path_ptr,
                                         uint64_t /*file_path_nbytes*/);

void cuda_shim_unload_module(uint64_t module_handle);

uint64_t cuda_shim_malloc(uint64_t nbytes, uint64_t stream,
                          bool is_host_shared);

void cuda_shim_free(uint64_t dptr, uint64_t stream);

void cuda_shim_memset32(uint64_t dptr, uint32_t value, uint64_t count_dwords,
                        uint64_t stream);
void cuda_shim_memset16(uint64_t dptr, uint32_t value, uint64_t count_dwords,
                        uint64_t stream);

uint64_t cuda_shim_stream_create(void);

void cuda_shim_stream_destroy(uint64_t stream);

void cuda_shim_stream_synchronize(uint64_t stream);

uint64_t cuda_shim_event_create(void);

void cuda_shim_event_destroy(uint64_t ev);

void cuda_shim_event_record(uint64_t ev, uint64_t stream);

void cuda_shim_event_synchronize(uint64_t ev);

void cuda_shim_stream_wait_event(uint64_t stream, uint64_t ev);

// ----------------------------- Memcpy (raw ABI) --------------------------
// Host pointers are passed as uint64_t. This is the key of 2A.

void cuda_shim_memcpy_h2d(uint64_t dst_dptr, uint64_t src_hptr,
                          uint64_t nbytes);

void cuda_shim_memcpy_d2h(uint64_t dst_hptr, uint64_t src_dptr,
                          uint64_t nbytes);

void cuda_shim_launch_packed(uint64_t module_handle, uint64_t kernel_name_ptr,
                             uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                             uint32_t blockX, uint32_t blockY, uint32_t blockZ,
                             uint32_t sharedMemBytes, uint64_t stream,
                             uint64_t arg_data_ptr, uint64_t arg_sizes_ptr,
                             uint32_t num_args);

// Convenience: 1D launch, shared=0, stream optional
void cuda_shim_launch_grid_packed(uint64_t module_handle,
                                  uint64_t kernel_name_ptr, uint32_t blockX,
                                  uint32_t blockY, uint32_t blockZ,
                                  uint64_t stream, uint64_t arg_data_ptr,
                                  uint64_t arg_sizes_ptr, uint32_t num_args);

// Optional: global sync (avoid in async pipeline; prefer event/stream sync)
void cuda_shim_ctx_synchronize(void);

// only for debugging
void cuda_debug_dump_float(uint64_t dptr, int n);
}

static inline llvm::orc::SymbolMap
buildCudaShimSymbolMap(llvm::orc::MangleAndInterner interner) {

  using llvm::JITSymbolFlags;
  using llvm::orc::ExecutorAddr;
  using llvm::orc::ExecutorSymbolDef;
  using llvm::orc::SymbolMap;

  SymbolMap syms;

  auto add = [&](const char *name, void *addr) {
    syms[interner(name)] =
        ExecutorSymbolDef::fromPtr(addr, JITSymbolFlags::Exported);
  };

  // ---- ctx ----
  add("cuda_shim_ctx_synchronize", (void *)&cuda_shim_ctx_synchronize);

  // ---- module ----
  add("cuda_shim_load_module_from_image",
      (void *)&cuda_shim_load_module_from_image);
  add("cuda_shim_load_module_jit_from_image",
      (void *)&cuda_shim_load_module_jit_from_image);
  add("cuda_shim_load_module_from_file",
      (void *)&cuda_shim_load_module_from_file);
  add("cuda_shim_unload_module", (void *)&cuda_shim_unload_module);

  // ---- memory ----
  add("cuda_shim_malloc", (void *)&cuda_shim_malloc);
  add("cuda_shim_free", (void *)&cuda_shim_free);

  // ---- memcpy ----
  add("cuda_shim_memcpy_h2d", (void *)&cuda_shim_memcpy_h2d);
  add("cuda_shim_memcpy_d2h", (void *)&cuda_shim_memcpy_d2h);

  // ---- stream ----
  add("cuda_shim_stream_create", (void *)&cuda_shim_stream_create);
  add("cuda_shim_stream_destroy", (void *)&cuda_shim_stream_destroy);
  add("cuda_shim_stream_synchronize", (void *)&cuda_shim_stream_synchronize);

  // ---- event ----
  add("cuda_shim_event_create", (void *)&cuda_shim_event_create);
  add("cuda_shim_event_destroy", (void *)&cuda_shim_event_destroy);
  add("cuda_shim_event_record", (void *)&cuda_shim_event_record);
  add("cuda_shim_event_synchronize", (void *)&cuda_shim_event_synchronize);
  add("cuda_shim_stream_wait_event", (void *)&cuda_shim_stream_wait_event);

  // ---- launch ----
  add("cuda_shim_launch_packed", (void *)&cuda_shim_launch_packed);
  add("cuda_shim_launch_grid_packed", (void *)&cuda_shim_launch_grid_packed);

  return syms;
}

static inline void registerCudaShimSymbols(mlir::ExecutionEngine &engine) {
  engine.registerSymbols([](llvm::orc::MangleAndInterner interner) {
    return buildCudaShimSymbolMap(interner);
  });
}

#endif // TOY_CUDA_SHIM_BUILDER_H
