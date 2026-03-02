#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

class CudaShimRegistry {
public:
  explicit CudaShimRegistry(mlir::ModuleOp module) : module(module) {}

  mlir::func::FuncOp getOrInsert(mlir::PatternRewriter &rewriter,
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

  mlir::func::CallOp call(mlir::PatternRewriter &rewriter,
                          mlir::Operation *anchor, CudaShimFn which,
                          mlir::ValueRange operands = {}) {
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

  static Spec specOf(CudaShimFn which, mlir::PatternRewriter &rewriter) {
    auto i64 = rewriter.getI64Type();
    auto i32 = rewriter.getI32Type();
    auto i1 = rewriter.getI1Type();

    switch (which) {

    // ===== Module =====
    case CudaShimFn::LoadModuleFromImage:
      return {"cuda_shim_load_module_from_image",
              rewriter.getFunctionType({i64, i64}, {i64})};

    case CudaShimFn::LoadModuleFromFile:
      return {"cuda_shim_load_module_from_file",
              rewriter.getFunctionType({i64, i64}, {i64})};

    case CudaShimFn::UnloadModule:
      return {"cuda_shim_unload_module", rewriter.getFunctionType({i64}, {})};

    // ===== Memory =====
    case CudaShimFn::Malloc:
      return {"cuda_shim_malloc",
              rewriter.getFunctionType({i64, i64, i1}, {i64})};

    case CudaShimFn::Free:
      return {"cuda_shim_free", rewriter.getFunctionType({i64, i64}, {})};

      // case CudaShimFn::Memset32:
      //   return {"cuda_shim_memset32",
      //           rewriter.getFunctionType({i64, i32, i64, i64}, {})};

      // case CudaShimFn::Memset16:
      //   return {"cuda_shim_memset16",
      //           rewriter.getFunctionType({i64, i32, i64, i64}, {})};

    case CudaShimFn::MemcpyH2D:
      return {"cuda_shim_memcpy_h2d",
              rewriter.getFunctionType({i64, i64, i64}, {})};

    case CudaShimFn::MemcpyD2H:
      return {"cuda_shim_memcpy_d2h",
              rewriter.getFunctionType({i64, i64, i64}, {})};

    // ===== Stream =====
    case CudaShimFn::StreamCreate:
      return {"cuda_shim_stream_create", rewriter.getFunctionType({}, {i64})};

    case CudaShimFn::StreamDestroy:
      return {"cuda_shim_stream_destroy", rewriter.getFunctionType({i64}, {})};

    case CudaShimFn::StreamSynchronize:
      return {"cuda_shim_stream_synchronize",
              rewriter.getFunctionType({i64}, {})};

    // case CudaShimFn::StreamWaitEvent:
    //   return {"cuda_shim_stream_wait_event",
    //           rewriter.getFunctionType({i64, i64}, {})};

    // ===== Event =====
    // case CudaShimFn::EventCreate:
    //   return {"cuda_shim_event_create", rewriter.getFunctionType({}, {i64})};

    // case CudaShimFn::EventDestroy:
    //   return {"cuda_shim_event_destroy", rewriter.getFunctionType({i64},
    //   {})};

    // case CudaShimFn::EventRecord:
    //   return {"cuda_shim_event_record",
    //           rewriter.getFunctionType({i64, i64}, {})};

    // case CudaShimFn::EventSynchronize:
    //   return {"cuda_shim_event_synchronize",
    //           rewriter.getFunctionType({i64}, {})};

    // ===== Launch =====
    case CudaShimFn::LaunchPacked:
      return {"cuda_shim_launch_packed",
              rewriter.getFunctionType(
                  {
                      i64,           // module_handle
                      i64,           // kernel_name_ptr
                      i32, i32, i32, // grid
                      i32, i32, i32, // block
                      i32,           // sharedMemBytes
                      i64,           // stream
                      i64,           // arg_data_ptr
                      i64,           // arg_sizes_ptr
                      i32            // num_args
                  },
                  {})};

    case CudaShimFn::LaunchBlockPacked:
      return {"cuda_shim_launch_block_packed",
              rewriter.getFunctionType(
                  {
                      i64,           // module_handle
                      i64,           // kernel_name_ptr
                      i32, i32, i32, // block
                      i64,           // stream
                      i64,           // arg_data_ptr
                      i64,           // arg_sizes_ptr
                      i32            // num_args
                  },
                  {})};

    // ===== Context =====
    case CudaShimFn::CtxSynchronize:
      return {"cuda_shim_ctx_synchronize", rewriter.getFunctionType({}, {})};
    }

    llvm_unreachable("Unhandled CudaShimFn");
  }

  mlir::ModuleOp module;
  llvm::DenseMap<unsigned, mlir::func::FuncOp> cache;
};

inline mlir::memref::GlobalOp
createGlobalForStringAttr(mlir::PatternRewriter &rewriter, mlir::Operation *op,
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

inline mlir::arith::IndexCastOp
getIndexFromValue(mlir::PatternRewriter &rewriter, mlir::Location loc,
                  mlir::Value value) {
  auto extractOp = mlir::memref::ExtractAlignedPointerAsIndexOp::create(
      rewriter, loc, rewriter.getIndexType(), value);
  auto indexCastOp = mlir::arith::IndexCastOp::create(
      rewriter, loc, rewriter.getI64Type(), extractOp.getResult());
  return indexCastOp;
}

inline mlir::arith::IndexCastOp
getIndexFromGlobalMemref(mlir::PatternRewriter &rewriter, mlir::Location loc,
                         mlir::memref::GlobalOp global) {

  auto getGlobalOp = mlir::memref::GetGlobalOp::create(
      rewriter, loc, global.getType(), global.getName());

  return getIndexFromValue(rewriter, loc, getGlobalOp.getResult());
}

inline mlir::func::CallOp createCallToCudaShimMalloc(
    mlir::PatternRewriter &rewriter, mlir::Location loc,
    CudaShimRegistry &registry, mlir::func::CallOp stream,
    mlir::arith::ConstantIntOp nbytesVal, bool isHostShared) {
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
