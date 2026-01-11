#include "cuda_tile/Dialect/CudaTile/IR/Types.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
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
#include <string>

#define DEBUG_TYPE "toy-to-cuda-tile"

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
    srcTerminator->remove();

    rewriter.mergeBlocks(srcBlock, bb, argValues);

    // rewriter.setInsertionPointToEnd(bb);
    auto retOp = mlir::cuda_tile::ReturnOp::create(rewriter, op.getLoc());
    rewriter.eraseOp(srcTerminator);

    LDBG() << "Created CudaTile Entry Op: \n" << entry;

    // Erase old op.
    rewriter.eraseOp(op);
    // rewriter.replaceOp(op, llvm::None);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToCudaTile Conversion Patterns: Binary operations
//===----------------------------------------------------------------------===//

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

template <typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering : public mlir::OpConversionPattern<BinaryOp> {
  using mlir::OpConversionPattern<BinaryOp>::OpConversionPattern;
  using OpAdaptor = typename mlir::OpConversionPattern<BinaryOp>::OpAdaptor;

  llvm::LogicalResult
  matchAndRewrite(BinaryOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto lhsShape =
        llvm::dyn_cast<mlir::RankedTensorType>(op.getLhs().getType())
            .getShape();
    auto rhsShape =
        llvm::dyn_cast<mlir::RankedTensorType>(op.getRhs().getType())
            .getShape();

    debugPrintShape(lhsShape, "LHS shape: ");
    debugPrintShape(rhsShape, "RHS shape: ");

    mlir::cuda_tile::MakeTensorViewOp lhsTensorView =
        makeTensorViewForArg(rewriter, loc, adaptor.getLhs(), lhsShape);
    mlir::cuda_tile::MakeTensorViewOp rhsTensorView =
        makeTensorViewForArg(rewriter, loc, adaptor.getRhs(), rhsShape);

    // LDBG() << "Lhs Type: " << adaptor.getLhs().getType();
    // LDBG() << "Rhs Type: " << adaptor.getRhs().getType();
    auto lhsShape_32 =
        llvm::to_vector<4>(llvm::map_range(lhsShape, [](int64_t dim) {
          return static_cast<int32_t>(alignPower2(dim));
        }));
    auto rhsShape_32 =
        llvm::to_vector<4>(llvm::map_range(rhsShape, [](int64_t dim) {
          return static_cast<int32_t>(alignPower2(dim));
        }));

    LDBG() << "lhsTensorView Type" << lhsTensorView->getResult(0).getType();

    mlir::cuda_tile::PartitionViewType lhsPartViewType =
        mlir::cuda_tile::PartitionViewType::get(
            rewriter.getContext(), rewriter.getDenseI32ArrayAttr(lhsShape_32),
            llvm::dyn_cast<mlir::cuda_tile::TensorViewType>(
                lhsTensorView->getResult(0).getType()),
            /*partitions=*/{0, 1}, {});
    mlir::cuda_tile::PartitionViewType rhsPartViewType =
        mlir::cuda_tile::PartitionViewType::get(
            rewriter.getContext(), rewriter.getDenseI32ArrayAttr(rhsShape_32),
            llvm::dyn_cast<mlir::cuda_tile::TensorViewType>(
                rhsTensorView->getResult(0).getType()),
            /*partitions=*/{0, 1}, {});

    mlir::cuda_tile::MakePartitionViewOp lhsPartitionView =
        mlir::cuda_tile::MakePartitionViewOp::create(
            rewriter, loc, lhsPartViewType, lhsTensorView);
    mlir::cuda_tile::MakePartitionViewOp rhsPartitionView =
        mlir::cuda_tile::MakePartitionViewOp::create(
            rewriter, loc, rhsPartViewType, rhsTensorView);

    auto tileTy = llvm::cast<mlir::cuda_tile::PartitionViewType>(
                      lhsPartitionView.getType())
                      .getViewTileType();
    auto zeroIdx = mlir::cuda_tile::ConstantOp::create(
        rewriter, loc, mlir::cuda_tile::TileType::get({}, rewriter.getI32Type()),
        rewriter.getI32IntegerAttr(0));
    auto lhsLoaded = mlir::cuda_tile::LoadViewTkoOp::create(
        rewriter, loc, tileTy, lhsPartitionView,
        mlir::ValueRange{zeroIdx, zeroIdx});
    auto rhsLoaded = mlir::cuda_tile::LoadViewTkoOp::create(
        rewriter, loc, tileTy, rhsPartitionView,
        mlir::ValueRange{zeroIdx, zeroIdx});

    // if (!tensor_view) {
    //   return rewriter.notifyMatchFailure(op, "lhs is not a TensorViewType");
    // }
    // LDBG() << "LHS view type: " << tensor_view;
    // auto result = mlir::cuda_tile::PartitionViewType::get(
    //     rewriter.getContext(), lhsShape,
    //     llvm::dyn_cast<mlir::cuda_tile::TensorViewType>(
    //         adaptor.getLhs().getType())
    //         .getElementType(),
    //     /*partitions=*/{0, 1});

    // auto lhsPartitionView = mlir::cuda_tile::MakePartitionViewOp::create(
    //     rewriter, loc, result, adaptor.getLhs());

    // auto rhsPartitionView =
    // mlir::cuda_tile::MakePartitionViewOp::create(rewriter, loc,
    //                                                      adaptor.getRhs(),
    //                                                      rhsShape);

    // makeTensorViewForArg(rewriter, loc, adaptor.getLhs(), lhsShape);
    // makeTensorViewForArg(rewriter, loc, adaptor.getRhs(), rhsShape);

    // auto tensorType =
    //     llvm::dyn_cast<mlir::RankedTensorType>(*op->result_type_begin());
    // LDBG() << "Lowering tensorType: " << tensorType;

    // auto lhsShape = tensorType.getShape();

    // auto lhsViewType = llvm::dyn_cast<mlir::cuda_tile::TensorViewType>(
    //     adaptor.getLhs().getType());
    // LDBG() << "LHS view type: " << adaptor.getLhs().getType();
    // if (!lhsViewType) {
    //   return rewriter.notifyMatchFailure(op, "lhs is not a TensorViewType");
    // }
    // auto lhsPartitionView =
    // mlir::cuda_tile::MakePartitionViewOp::create(rewriter, loc,
    //                                                              adaptor.lhs(),
    //                                                              lhsViewType);
    // auto

    return llvm::success();
  }
};
using AddOpLowering =
    BinaryOpLowering<mlir::toy::AddOp, mlir::cuda_tile::AddFOp>;
using MulOpLowering =
    BinaryOpLowering<mlir::toy::MulOp, mlir::cuda_tile::MulFOp>;

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
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  mlir::ConversionTarget target(*ctx);

  target.addLegalDialect<mlir::cuda_tile::CudaTileDialect>();
  target.addLegalOp<mlir::ModuleOp>();

  // Keep host-side toy.func/main legal (or lower it later).
  target.addLegalOp<mlir::toy::FuncOp, mlir::toy::PrintOp, mlir::toy::ReturnOp,
                    mlir::toy::ConstantOp, mlir::toy::LaunchGpuOp>();

  target.addIllegalOp<mlir::toy::GPUFuncOp>();

  ToyToCudaTileTypeConverter typeConv(&*ctx);

  mlir::RewritePatternSet patterns(&*ctx);
  patterns.add<LowerToyGPUFuncToCudaTileEntry>(typeConv, &*ctx);

  // TODO: add patterns for toy.transpose/toy.matmul/toy.add/toy.mul
  patterns.add<MulOpLowering, AddOpLowering>(typeConv, ctx);

  if (mlir::failed(mlir::applyPartialConversion(moduleOp, target,
                                                std::move(patterns)))) {
    signalPassFailure();
  }

  // auto moduleOp = getOperation();

  // // Here we would implement the actual lowering logic from Toy GPUFuncOp
  // // to CudaTile operations. For now, we just log that the pass is running.
  // // LDBG() << "Running Toy to CudaTile lowering on GPUFuncOp: " << moduleOp
  // //        ;

  // mlir::OpBuilder builder(moduleOp.getContext());
  // // 1. Create new cuda_tile.module Op in the last section.
  // auto cudaTileModuleOp = createCudaModuleOp(builder, moduleOp);
  // // mlir::SymbolTable cudaTileSymbolTable(cudaTileModuleOp);

  // moduleOp->walk([&](mlir::toy::GPUFuncOp gfunOp) {
  //   mlir::OpBuilder::InsertionGuard guard(builder);
  //   // setInsertionPointToEnd expects a Block*, so take the address of the
  //   // single block inside the cuda_tile.module region.
  //   builder.setInsertionPointToEnd(&cudaTileModuleOp.getBodyRegion().front());
  //   auto gfunc_name =
  //       gfunOp->getAttrOfType<mlir::StringAttr>("sym_name").getValue();
  //   llvm::SmallVector<mlir::Type, 8> newArgTypes;

  //   LDBG() << "Lowering GPU function: " << gfunc_name;
  //   LDBG() << "Converting input type into cuda tile type";

  //   llvm::SmallVector<llvm::ArrayRef<int64_t>, 4> inputShapes;
  //   // llvm::SmallVector<llvm::ArrayRef<int64_t>, 4> resultShapes;

  //   for (mlir::Type t : gfunOp.getFunctionType().getInputs()) {
  //     LDBG() << "Original arg type: " << t;
  //     auto tt = llvm::dyn_cast<mlir::TensorType>(t);
  //     auto elemType = tt.getElementType();
  //     auto ptrElem = mlir::cuda_tile::PointerType::get(elemType);
  //     auto newType = mlir::cuda_tile::TileType::get({}, ptrElem);
  //     LDBG() << "The new arg type for cuda tile: " << newType;
  //     newArgTypes.push_back(newType);
  //     inputShapes.push_back(tt.getShape());
  //   }

  //   LDBG() << "Converting result type into cuda tile type";
  //   for (mlir::Type t : gfunOp.getFunctionType().getResults()) {
  //     LDBG() << "Original result type: " << t;
  //     auto tt = llvm::dyn_cast<mlir::TensorType>(t);
  //     auto elemType = tt.getElementType();
  //     auto ptrElem = mlir::cuda_tile::PointerType::get(elemType);
  //     auto newType = mlir::cuda_tile::TileType::get({}, ptrElem);
  //     LDBG() << "The new arg type for cuda tile: " << newType;
  //     newArgTypes.push_back(newType);
  //     inputShapes.push_back(tt.getShape());
  //   }

  //   auto newFnType = builder.getFunctionType(newArgTypes, {});
  //   auto fname = builder.getStringAttr(gfunc_name);
  //   auto argTypes = builder.getTypeArrayAttr(newArgTypes);
  //   auto cudaEntryOp = mlir::cuda_tile::EntryOp::create(
  //       builder, gfunOp.getLoc(), fname, newFnType,
  //       /*arg_attrs=*/{}, /*res_attrs=*/{}, {});
  //   auto bb = cudaEntryOp.addEntryBlock();
  //   builder.setInsertionPointToStart(bb);
  //   // 1. create a get_tile_block_id op
  //   auto tileBlockId = mlir::cuda_tile::GetTileBlockIdOp::create(
  //       builder, gfunOp->getLoc(),
  //       {mlir::cuda_tile::TileType::get({}, builder.getI32Type()),
  //        mlir::cuda_tile::TileType::get({}, builder.getI32Type()),
  //        mlir::cuda_tile::TileType::get({}, builder.getI32Type())});
  //   for (auto [idx, arg] : llvm::enumerate(bb->getArguments())) {
  //     // 2. create a make_tensor_view op
  //     auto resultType = builder.getI64ArrayAttr(inputShapes[idx]);
  //     LDBG() << "Argument " << idx << " : " << arg << ", shape: " <<
  //     resultType; auto ptrElem =
  //     llvm::dyn_cast<mlir::cuda_tile::TileType>(arg.getType())
  //                        .getElementType();
  //     auto eleType = llvm::dyn_cast<mlir::cuda_tile::PointerType>(ptrElem)
  //                        .getPointeeType();
  //     mlir::cuda_tile::TensorViewType tensorViewType =
  //         mlir::cuda_tile::TensorViewType::get(
  //             builder.getContext(), eleType, inputShapes[idx],
  //             /*strides=*/{inputShapes[idx].back(), 1});
  //     // LDBG() << "Creating TensorViewType: " << tensorViewType;
  //     auto make_tensor_view = mlir::cuda_tile::MakeTensorViewOp::create(
  //         builder, gfunOp->getLoc(), tensorViewType, arg,
  //         /*dynamicShape=*/mlir::ValueRange{},
  //         /*dynamicStrides=*/mlir::ValueRange{});
  //     // LDBG() << "Created MakeTensorViewOp: \n" << make_tensor_view  ;
  //   }

  //   auto retOp = mlir::cuda_tile::ReturnOp::create(builder, gfunOp.getLoc());

  //   LDBG() << "Created CudaTile Entry Op: \n" << cudaEntryOp;
  // });
}

namespace mlir::toy {

std::unique_ptr<mlir::Pass> createCudaTileLoweringPass() {
  return std::make_unique<ToyToCudaTileLoweringPass>();
};

}; // namespace mlir::toy
