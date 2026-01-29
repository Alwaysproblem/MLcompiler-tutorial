#include "cuda_tile/Dialect/CudaTile/IR/Types.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"
#include "toy/Dialect.h"
#include "toy/Passes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/LogicalResult.h"

#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#define DEBUG_TYPE "toy-to-cuda-tile"

void debugPrintShape(mlir::ArrayRef<int64_t> shape,
                     llvm::StringRef prefix = "") {
  std::string shapeStr;
  llvm::raw_string_ostream shapeOS(shapeStr);
  shapeOS << "[";
  llvm::interleaveComma(shape, shapeOS);
  shapeOS << "]";
  shapeOS.flush();
  LDBG() << prefix << shapeStr;
}

mlir::cuda_tile::MakeTensorViewOp
makeTensorViewForArg(mlir::OpBuilder &rewriter, mlir::Location loc,
                     mlir::Value arg, mlir::ArrayRef<int64_t> shape) {
  auto resultType = rewriter.getI64ArrayAttr(shape);
  LDBG() << "shape: " << resultType;
  auto ptrElem =
      llvm::dyn_cast<mlir::cuda_tile::TileType>(arg.getType()).getElementType();
  auto eleType =
      llvm::dyn_cast<mlir::cuda_tile::PointerType>(ptrElem).getPointeeType();
  mlir::cuda_tile::TensorViewType tensorViewType =
      mlir::cuda_tile::TensorViewType::get(rewriter.getContext(), eleType,
                                           shape,
                                           /*strides=*/{shape.back(), 1});
  // LDBG() << "Creating TensorViewType: " << tensorViewType;
  auto make_tensor_view = mlir::cuda_tile::MakeTensorViewOp::create(
      rewriter, loc, tensorViewType, arg,
      /*dynamicShape=*/mlir::ValueRange{},
      /*dynamicStrides=*/mlir::ValueRange{});
  return make_tensor_view;
}

int64_t alignPower2(int x) {
  int64_t power = 1;
  while (power < x) {
    power *= 2;
  }
  return power;
}

static bool isFromFuncArg(mlir::Value v) {
  if (auto barg = llvm::dyn_cast<mlir::BlockArgument>(v)) {
    return true;
  }
  return false;
}

static std::optional<mlir::Value> insertUnrealizedConversionCastOp(
    mlir::Value opv, mlir::Value v, mlir::RankedTensorType logical,
    mlir::PatternRewriter &rewriter, mlir::Location loc) {
  auto tileTy = llvm::dyn_cast<mlir::cuda_tile::TileType>(v.getType());
  if (!tileTy)
    return v;

  auto elemTy = tileTy.getElementType();
  auto ptrTy = llvm::dyn_cast<mlir::cuda_tile::PointerType>(elemTy);
  if (!ptrTy)
    return v;

  if (!isFromFuncArg(opv)) {
    auto logicalTy = llvm::cast<mlir::RankedTensorType>(opv.getType());
    auto alignedI32 = llvm::to_vector<4>(
        llvm::map_range(logicalTy.getShape(), [](int64_t dim) {
          return static_cast<int32_t>(alignPower2(dim));
        }));
    llvm::SmallVector<int64_t, 4> alignedShape;
    alignedShape.reserve(alignedI32.size());
    for (int32_t dim : alignedI32)
      alignedShape.push_back(static_cast<int64_t>(dim));
    mlir::cuda_tile::TileType resultTileTy =
        mlir::cuda_tile::TileType::get(alignedShape, ptrTy.getPointeeType());

    // Here the TypeConverter will change the toy.add result type to
    // tile<ptr<f32>>, but we actually need tile<...xf32> to do computation. so
    // we need to insert a cast here. if we don't do this, the
    // `UnrealizedConversionCastOp` will be automatically inserted later during
    // conversion. like: %10 = "builtin.unrealized_conversion_cast"(%9)
    // {__pure_type_conversion__}
    //                      : (!cuda_tile.tile<2x4xf32>) ->
    //                      !cuda_tile.tile<ptr<f32>>
    // since the TypeConverter can not know which input is from function arg or
    // not. so, here we do the cast manually to delete those cast Op since the
    // `cuda_tile.add` can accept tile<...xf32> directly if args is not from the
    // block arguments.
    mlir::UnrealizedConversionCastOp castOp =
        mlir::UnrealizedConversionCastOp::create(rewriter, loc, {resultTileTy},
                                                 v);
    return castOp.getResult(0);
  }

  return std::nullopt; // no need to insert cast
}

static mlir::cuda_tile::MakePartitionViewOp
makePartitionViewForArg(mlir::PatternRewriter &rewriter, mlir::Location loc,
                        mlir::Value v, mlir::RankedTensorType logical) {
  // 1) make_tensor_view from tile<ptr<f32>>
  auto tensorView = makeTensorViewForArg(rewriter, loc, v, logical.getShape());

  // 2) 创建 partition_view，tile 形状使用 2 的幂对齐
  auto alignedI32 =
      llvm::to_vector<4>(llvm::map_range(logical.getShape(), [](int64_t dim) {
        return static_cast<int32_t>(alignPower2(dim));
      }));
  auto partViewTy = mlir::cuda_tile::PartitionViewType::get(
      rewriter.getContext(), rewriter.getDenseI32ArrayAttr(alignedI32),
      llvm::dyn_cast<mlir::cuda_tile::TensorViewType>(
          tensorView->getResult(0).getType()),
      /*partitions=*/{0, 1}, {});
  auto partView = mlir::cuda_tile::MakePartitionViewOp::create(
      rewriter, loc, partViewTy, tensorView);
  return partView;
}

static mlir::Value ensureTileValue(mlir::Value opv, mlir::Value v,
                                   mlir::RankedTensorType logical,
                                   mlir::PatternRewriter &rewriter) {
  auto loc = v.getLoc();
  auto maybeCastV =
      insertUnrealizedConversionCastOp(opv, v, logical, rewriter, loc);
  if (maybeCastV.has_value()) {
    return maybeCastV.value();
  }

  auto alignedI32 =
      llvm::to_vector<4>(llvm::map_range(logical.getShape(), [](int64_t dim) {
        return static_cast<int32_t>(alignPower2(dim));
      }));

  auto partView = makePartitionViewForArg(rewriter, loc, v, logical);

  // 3) 准备索引常量和 load
  auto i32TileTy = mlir::cuda_tile::TileType::get({}, rewriter.getI32Type());
  auto zeroAttr =
      mlir::DenseIntElementsAttr::get(i32TileTy, llvm::ArrayRef<int32_t>{0});
  auto zeroIdx =
      mlir::cuda_tile::ConstantOp::create(rewriter, loc, i32TileTy, zeroAttr);

  auto memOrd = mlir::cuda_tile::MemoryOrderingSemanticsAttr::get(
      rewriter.getContext(), mlir::cuda_tile::MemoryOrderingSemantics::WEAK);
  auto tokenTy = mlir::cuda_tile::TokenType::get(rewriter.getContext());

  // auto memory_ordering_attr =
  //     mlir::cuda_tile::MemoryOrderingSemanticsAttr::get(
  //         rewriter.getContext(),
  //         mlir::cuda_tile::MemoryOrderingSemantics::WEAK);

  auto tensorViewTy = llvm::cast<mlir::cuda_tile::TensorViewType>(
      partView.getTensorView().getType());
  LDBG() << "TensorViewType for LoadViewTkoOp: " << tensorViewTy;
  llvm::SmallVector<int64_t, 4> alignedLoadShape(alignedI32.begin(),
                                                 alignedI32.end());
  debugPrintShape(alignedLoadShape);

  auto resTileTy = mlir::cuda_tile::TileType::get(
      {alignedLoadShape.begin(), alignedLoadShape.end()},
      tensorViewTy.getElementType());
  auto load = mlir::cuda_tile::LoadViewTkoOp::create(
      rewriter, loc, {resTileTy, tokenTy}, memOrd, {}, partView,
      mlir::ValueRange{zeroIdx, zeroIdx}, {}, {});

  return load.getResult(0);
}

static mlir::Value ensureStoreValue(mlir::Value opv, mlir::Value v,
                                    mlir::RankedTensorType logical,
                                    mlir::PatternRewriter &rewriter) {
  auto loc = v.getLoc();
  auto castOp =
      insertUnrealizedConversionCastOp(opv, v, logical, rewriter, loc);
  if (castOp.has_value()) {
    return castOp.value();
  }
  return v;
}

//===----------------------------------------------------------------------===//
// 1) TypeConverter: tensor<...xf32> -> tile<ptr<f32>> (plus we will create
// views)
//===----------------------------------------------------------------------===//
struct ToyToCudaTileTypeConverter : public mlir::TypeConverter {
  ToyToCudaTileTypeConverter(mlir::MLIRContext *ctx) {
    addConversion([](mlir::Type t) { return t; }); // identity for others

    addConversion([&](mlir::RankedTensorType t) -> mlir::Type {
      // Example: only handle f32 ranked tensor for now.
      auto elemTy = llvm::dyn_cast<mlir::FloatType>(t.getElementType());
      if (!elemTy || elemTy.getWidth() != 32)
        return {};

      auto ptrElem = mlir::cuda_tile::PointerType::get(elemTy);
      auto newType = mlir::cuda_tile::TileType::get({}, ptrElem);

      // tile<ptr<f32>> : the exact spelling depends on your cuda_tile dialect
      // types.
      return newType;
    });

    // Important: if you have tensor results too, you need a materialization
    // strategy. e.g. create temporary buffers and store into them, or return
    // ptr to output.
  }
};

//===----------------------------------------------------------------------===//
// 2) Pattern: toy.gpu_func -> create cuda_tile.module entry
//===----------------------------------------------------------------------===//
struct LowerToyGPUFuncToCudaTileEntry
    : public mlir::OpConversionPattern<mlir::toy::GPUFuncOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::toy::GPUFuncOp op,
                  mlir::toy::GPUFuncOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Find / create cuda_tile.module container (you can also create once in
    // pass)
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    mlir::cuda_tile::ModuleOp cudaMod;
    for (auto m : moduleOp.getOps<mlir::cuda_tile::ModuleOp>()) {
      cudaMod = m;
      break;
    }

    if (!cudaMod) {
      rewriter.setInsertionPointToEnd(moduleOp.getBody());
      cudaMod = mlir::cuda_tile::ModuleOp::create(rewriter, op.getLoc(),
                                                  "cuda_tile_module");
    }
    LDBG() << "Found / Created CudaTile Module: \n" << cudaMod;

    llvm::SmallVector<mlir::Type> entryArgTys;
    llvm::SmallVector<llvm::ArrayRef<int64_t>, 4> entryArgShapes;
    for (auto t : op.getFunctionType().getInputs()) {
      auto ct = getTypeConverter()->convertType(t);
      if (!ct)
        return rewriter.notifyMatchFailure(op, "cannot convert arg type");
      LDBG() << "Converted arg type: " << ct;
      entryArgTys.push_back(ct);
      auto rt = llvm::dyn_cast<mlir::RankedTensorType>(t);
      entryArgShapes.push_back(rt.getShape());
    }

    for (auto t : op.getFunctionType().getResults()) {
      auto ct = getTypeConverter()->convertType(t);
      if (!ct)
        return rewriter.notifyMatchFailure(op, "cannot convert result type");
      LDBG() << "Converted result type: " << ct;
      // Optionally, add as extra arg instead of return.
      entryArgTys.push_back(ct);
      auto rt = llvm::dyn_cast<mlir::RankedTensorType>(t);
      entryArgShapes.push_back(rt.getShape());
    }
    auto newFnType = rewriter.getFunctionType(entryArgTys, {});

    mlir::Block &bodyBlock = cudaMod.getBodyRegion().front();
    mlir::OpBuilder::InsertionGuard guard(rewriter);

    rewriter.setInsertionPointToStart(&bodyBlock);

    auto entry = mlir::cuda_tile::EntryOp::create(
        rewriter, op.getLoc(), op.getSymName(), newFnType,
        /*arg_attrs=*/{}, /*res_attrs=*/{}, {});

    LDBG() << "CudaTile Module: \n" << cudaMod;

    auto *bb = entry.addEntryBlock();

    rewriter.setInsertionPointToStart(bb);
    // 1. create a get_tile_block_id op
    auto tileBlockId = mlir::cuda_tile::GetTileBlockIdOp::create(
        rewriter, op->getLoc(),
        {mlir::cuda_tile::TileType::get({}, rewriter.getI32Type()),
         mlir::cuda_tile::TileType::get({}, rewriter.getI32Type()),
         mlir::cuda_tile::TileType::get({}, rewriter.getI32Type())});

    llvm::SmallVector<mlir::Value> tensorViews;

    // for (auto [idx, arg] : llvm::enumerate(bb->getArguments())) {
    //   // 2. create a make_tensor_view op
    //   auto resultType = rewriter.getI64ArrayAttr(entryArgShapes[idx]);
    //   LDBG() << "Argument " << idx << " : " << arg << ", shape: " <<
    //   resultType; auto ptrElem =
    //   llvm::dyn_cast<mlir::cuda_tile::TileType>(arg.getType())
    //                      .getElementType();
    //   auto eleType = llvm::dyn_cast<mlir::cuda_tile::PointerType>(ptrElem)
    //                      .getPointeeType();
    //   mlir::cuda_tile::TensorViewType tensorViewType =
    //       mlir::cuda_tile::TensorViewType::get(
    //           rewriter.getContext(), eleType, entryArgShapes[idx],
    //           /*strides=*/{entryArgShapes[idx].back(), 1});
    //   // LDBG() << "Creating TensorViewType: " << tensorViewType;
    //   auto make_tensor_view = mlir::cuda_tile::MakeTensorViewOp::create(
    //       rewriter, op->getLoc(), tensorViewType, arg,
    //       /*dynamicShape=*/mlir::ValueRange{},
    //       /*dynamicStrides=*/mlir::ValueRange{});
    //   // LDBG() << "Created MakeTensorViewOp: \n" << make_tensor_view  ;
    //   tensorViews.push_back(make_tensor_view.getResult());
    // }
    for (auto [idx, arg] : llvm::enumerate(bb->getArguments())) {
      tensorViews.push_back(arg);
    }

    auto *srcBlock = &op.getBody().front();
    llvm::SmallVector<mlir::Value> argValues;
    argValues.reserve(srcBlock->getNumArguments());
    for (unsigned i = 0; i < srcBlock->getNumArguments(); ++i) {
      argValues.push_back(tensorViews[i]);
    }

    auto *srcTerminator = srcBlock->getTerminator();
    rewriter.mergeBlocks(srcBlock, bb, argValues);

    auto retOp = mlir::cuda_tile::ReturnOp::create(rewriter, op.getLoc());

    LDBG() << "Created CudaTile Entry Op: \n" << entry;

    // Erase old op.
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToCudaTile Conversion Patterns: Binary operations
//===----------------------------------------------------------------------===//

template <typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering : public mlir::OpConversionPattern<BinaryOp> {
  using mlir::OpConversionPattern<BinaryOp>::OpConversionPattern;
  using OpAdaptor = typename mlir::OpConversionPattern<BinaryOp>::OpAdaptor;

  llvm::LogicalResult
  matchAndRewrite(BinaryOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto logical =
        llvm::dyn_cast<mlir::RankedTensorType>(op.getResult().getType());
    if (!logical) {
      return rewriter.notifyMatchFailure(op, "result is not RankedTensorType");
    }
    auto lhsLoaded =
        ensureTileValue(op.getLhs(), adaptor.getLhs(), logical, rewriter);
    auto rhsLoaded =
        ensureTileValue(op.getRhs(), adaptor.getRhs(), logical, rewriter);

    LDBG() << "After ensureTileValue LHS: " << lhsLoaded;
    LDBG() << "After ensureTileValue RHS: " << rhsLoaded;

    auto tileTy = lhsLoaded.getType();
    auto binOp = LoweredBinaryOp::create(rewriter, loc, tileTy, lhsLoaded,
                                         rhsLoaded, {});
    rewriter.replaceOp(op, binOp.getResult());
    return llvm::success();
  }
};
using AddOpLowering =
    BinaryOpLowering<mlir::toy::AddOp, mlir::cuda_tile::AddFOp>;
using MulOpLowering =
    BinaryOpLowering<mlir::toy::MulOp, mlir::cuda_tile::MulFOp>;

//===----------------------------------------------------------------------===//
// ToyToCudaTile Conversion Patterns: Return operations
//===----------------------------------------------------------------------===//

struct ReturnLowering : public mlir::OpConversionPattern<mlir::toy::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::toy::ReturnOp op, mlir::toy::ReturnOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto outputPtr = op->getBlock()->getArguments().back();
    auto logical =
        llvm::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    if (!logical) {
      return rewriter.notifyMatchFailure(op, "result is not RankedTensorType");
    }
    auto retValLoaded = ensureStoreValue(
        op.getOperand(0), adaptor.getOperands().front(), logical, rewriter);
    LDBG() << "After ensureStoreValue RET: " << retValLoaded;

    auto partView = makePartitionViewForArg(rewriter, loc, outputPtr, logical);

    auto tkTy = mlir::cuda_tile::TokenType::get(rewriter.getContext());
    auto memoryOrd = mlir::cuda_tile::MemoryOrderingSemanticsAttr::get(
        rewriter.getContext(), mlir::cuda_tile::MemoryOrderingSemantics::WEAK);

    auto i32TileTy = mlir::cuda_tile::TileType::get({}, rewriter.getI32Type());
    auto zeroAttr =
        mlir::DenseIntElementsAttr::get(i32TileTy, llvm::ArrayRef<int32_t>{0});
    auto zeroIdx =
        mlir::cuda_tile::ConstantOp::create(rewriter, loc, i32TileTy, zeroAttr);

    auto storeOp = mlir::cuda_tile::StoreViewTkoOp::create(
        rewriter, loc, {tkTy}, memoryOrd, {}, retValLoaded, partView,
        mlir::ValueRange{zeroIdx, zeroIdx}, {}, {});

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToCudaTileLoweringPass
//===----------------------------------------------------------------------===//

namespace {
struct ToyToCudaTileLoweringPass
    : public mlir::PassWrapper<ToyToCudaTileLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyToCudaTileLoweringPass)

  llvm::StringRef getArgument() const override { return "toy-to-cuda-tile"; }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::cuda_tile::CudaTileDialect>();
  }

  void runOnOperation() final;
};
}; // namespace

mlir::cuda_tile::ModuleOp createCudaModuleOp(mlir::OpBuilder &builder,
                                             mlir::ModuleOp &moduleOp) {
  mlir::OpBuilder::InsertionGuard guard(builder);

  builder.setInsertionPoint(moduleOp.getBody(), moduleOp.getBody()->end());
  auto cudaTileModuleOp = mlir::cuda_tile::ModuleOp::create(
      builder, moduleOp.getLoc(), "cuda_tile_module");

  LDBG() << "Created CudaTile Module: \n" << cudaTileModuleOp;
  return cudaTileModuleOp;
}

void ToyToCudaTileLoweringPass::runOnOperation() {
  auto moduleOp = getOperation();
  auto *ctx = moduleOp.getContext();
  mlir::ConversionTarget target(*ctx);
  target.addLegalDialect<mlir::cuda_tile::CudaTileDialect>();
  target.addLegalOp<mlir::ModuleOp>();

  // Keep host-side toy.func/main legal (or lower it later).
  target.addLegalOp<mlir::toy::FuncOp, mlir::toy::PrintOp,
                    mlir::toy::ConstantOp, mlir::toy::LaunchGpuOp>();

  target.addIllegalOp<mlir::toy::GPUFuncOp, mlir::toy::AddOp, mlir::toy::MulOp,
                      mlir::toy::ReturnOp>();

  moduleOp.walk([&](mlir::toy::GPUFuncOp gfun) {
    ToyToCudaTileTypeConverter typeConverter(ctx);
    mlir::RewritePatternSet patterns(ctx);

    patterns.add<LowerToyGPUFuncToCudaTileEntry, MulOpLowering, AddOpLowering,
                 ReturnLowering>(typeConverter, ctx);

    if (mlir::failed(
            mlir::applyFullConversion(gfun, target, std::move(patterns))))
      signalPassFailure();
  });

  // -------------------------------
  // auto moduleOp = getOperation();
  // auto *ctx = moduleOp.getContext();
  // // The first thing to define is the conversion target. This will define the
  // // final target for this lowering.
  // mlir::ConversionTarget target(*ctx);

  // target.addLegalDialect<mlir::cuda_tile::CudaTileDialect>();
  // target.addLegalOp<mlir::ModuleOp>();

  // // Keep host-side toy.func/main legal (or lower it later).
  // target.addLegalOp<mlir::toy::FuncOp, mlir::toy::PrintOp,
  //                   mlir::toy::ConstantOp, mlir::toy::LaunchGpuOp>();

  // target
  //     .addIllegalOp<mlir::toy::GPUFuncOp, mlir::toy::AddOp,
  //     mlir::toy::MulOp,mlir::toy::ReturnOp>();

  // ToyToCudaTileTypeConverter typeConv(&*ctx);

  // mlir::RewritePatternSet patterns(&*ctx);
  // patterns.add<LowerToyGPUFuncToCudaTileEntry>(typeConv, &*ctx);

  // // TODO: add patterns for toy.transpose/toy.matmul/toy.add/toy.mul
  // patterns.add<MulOpLowering, AddOpLowering>(typeConv, ctx);

  // if (mlir::failed(mlir::applyPartialConversion(moduleOp, target,
  //                                               std::move(patterns)))) {
  //   signalPassFailure();
  // }
  // -------------------------------
}

namespace mlir::toy {

std::unique_ptr<mlir::Pass> createCudaTileLoweringPass() {
  return std::make_unique<ToyToCudaTileLoweringPass>();
};

}; // namespace mlir::toy
