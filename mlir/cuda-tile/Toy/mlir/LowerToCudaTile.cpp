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

#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"

#include <cstdint>
#include <memory>
#include <string>

#define DEBUG_TYPE "toy-to-cuda-tile"

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

  // Here we would implement the actual lowering logic from Toy GPUFuncOp
  // to CudaTile operations. For now, we just log that the pass is running.
  // LDBG() << "Running Toy to CudaTile lowering on GPUFuncOp: " << moduleOp
  //        ;

  mlir::OpBuilder builder(moduleOp.getContext());
  // 1. Create new cuda_tile.module Op in the last section.
  auto cudaTileModuleOp = createCudaModuleOp(builder, moduleOp);
  // mlir::SymbolTable cudaTileSymbolTable(cudaTileModuleOp);

  moduleOp->walk([&](mlir::toy::GPUFuncOp gfunOp) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    // setInsertionPointToEnd expects a Block*, so take the address of the
    // single block inside the cuda_tile.module region.
    builder.setInsertionPointToEnd(&cudaTileModuleOp.getBodyRegion().front());
    auto gfunc_name =
        gfunOp->getAttrOfType<mlir::StringAttr>("sym_name").getValue();
    llvm::SmallVector<mlir::Type, 8> newArgTypes;

    LDBG() << "Lowering GPU function: " << gfunc_name;
    LDBG() << "Converting input type into cuda tile type";

    llvm::SmallVector<llvm::ArrayRef<int64_t>, 4> inputShapes;
    // llvm::SmallVector<llvm::ArrayRef<int64_t>, 4> resultShapes;

    for (mlir::Type t : gfunOp.getFunctionType().getInputs()) {
      LDBG() << "Original arg type: " << t;
      auto tt = llvm::dyn_cast<mlir::TensorType>(t);
      auto elemType = tt.getElementType();
      auto ptrElem = mlir::cuda_tile::PointerType::get(elemType);
      auto newType = mlir::cuda_tile::TileType::get({}, ptrElem);
      LDBG() << "The new arg type for cuda tile: " << newType;
      newArgTypes.push_back(newType);
      inputShapes.push_back(tt.getShape());
    }

    LDBG() << "Converting result type into cuda tile type";
    for (mlir::Type t : gfunOp.getFunctionType().getResults()) {
      LDBG() << "Original result type: " << t;
      auto tt = llvm::dyn_cast<mlir::TensorType>(t);
      auto elemType = tt.getElementType();
      auto ptrElem = mlir::cuda_tile::PointerType::get(elemType);
      auto newType = mlir::cuda_tile::TileType::get({}, ptrElem);
      LDBG() << "The new arg type for cuda tile: " << newType;
      newArgTypes.push_back(newType);
      inputShapes.push_back(tt.getShape());
    }

    auto newFnType = builder.getFunctionType(newArgTypes, {});
    auto fname = builder.getStringAttr(gfunc_name);
    auto argTypes = builder.getTypeArrayAttr(newArgTypes);
    auto cudaEntryOp = mlir::cuda_tile::EntryOp::create(
        builder, gfunOp.getLoc(), fname, newFnType,
        /*arg_attrs=*/{}, /*res_attrs=*/{}, {});
    auto bb = cudaEntryOp.addEntryBlock();
    builder.setInsertionPointToStart(bb);
    // 1. create a get_tile_block_id op
    auto tileBlockId = mlir::cuda_tile::GetTileBlockIdOp::create(
        builder, gfunOp->getLoc(),
        {mlir::cuda_tile::TileType::get({}, builder.getI32Type()),
         mlir::cuda_tile::TileType::get({}, builder.getI32Type()),
         mlir::cuda_tile::TileType::get({}, builder.getI32Type())});
    for (auto [idx, arg] : llvm::enumerate(bb->getArguments())) {
      // 2. create a make_tensor_view op
      auto resultType = builder.getI64ArrayAttr(inputShapes[idx]);
      LDBG() << "Argument " << idx << " : " << arg << ", shape: " << resultType;
      auto ptrElem = llvm::dyn_cast<mlir::cuda_tile::TileType>(arg.getType())
                         .getElementType();
      auto eleType = llvm::dyn_cast<mlir::cuda_tile::PointerType>(ptrElem)
                         .getPointeeType();
      mlir::cuda_tile::TensorViewType tensorViewType =
          mlir::cuda_tile::TensorViewType::get(
              builder.getContext(), eleType, inputShapes[idx],
              /*strides=*/{inputShapes[idx].back(), 1});
      // LDBG() << "Creating TensorViewType: " << tensorViewType;
      auto make_tensor_view = mlir::cuda_tile::MakeTensorViewOp::create(
          builder, gfunOp->getLoc(), tensorViewType, arg,
          /*dynamicShape=*/mlir::ValueRange{},
          /*dynamicStrides=*/mlir::ValueRange{});
      // LDBG() << "Created MakeTensorViewOp: \n" << make_tensor_view  ;
    }

    auto retOp = mlir::cuda_tile::ReturnOp::create(builder, gfunOp.getLoc());

    LDBG() << "Created CudaTile Entry Op: \n" << cudaEntryOp;
  });
}

namespace mlir::toy {

std::unique_ptr<mlir::Pass> createCudaTileLoweringPass() {
  return std::make_unique<ToyToCudaTileLoweringPass>();
};

}; // namespace mlir::toy
