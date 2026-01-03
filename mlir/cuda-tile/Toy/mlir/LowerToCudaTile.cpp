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

  LDBG() << "Created CudaTile Module: \n" << cudaTileModuleOp << "\n";
  return cudaTileModuleOp;
}

void ToyToCudaTileLoweringPass::runOnOperation() {
  auto moduleOp = getOperation();

  // Here we would implement the actual lowering logic from Toy GPUFuncOp
  // to CudaTile operations. For now, we just log that the pass is running.
  // LDBG() << "Running Toy to CudaTile lowering on GPUFuncOp: " << moduleOp
  //        << "\n";

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

    LDBG() << "Lowering GPU function: " << gfunc_name << "\n";
    LDBG() << "Converting input type into cuda tile type" << "\n";

    for (mlir::Type t : gfunOp.getFunctionType().getInputs()) {
      LDBG() << "Original arg type: " << t << "\n";
      auto tt = llvm::dyn_cast<mlir::TensorType>(t);
      auto elemType = tt.getElementType();
      auto ptrElem = mlir::cuda_tile::PointerType::get(elemType);
      auto newType = mlir::cuda_tile::TileType::get({}, ptrElem);
      LDBG() << "The new arg type for cuda tile: " << newType << "\n";
      newArgTypes.push_back(newType);
    }

    LDBG() << "Converting result type into cuda tile type" << "\n";
    for (mlir::Type t : gfunOp.getFunctionType().getResults()) {
      LDBG() << "Original result type: " << t << "\n";
      auto tt = llvm::dyn_cast<mlir::TensorType>(t);
      auto elemType = tt.getElementType();
      auto ptrElem = mlir::cuda_tile::PointerType::get(elemType);
      auto newType = mlir::cuda_tile::TileType::get({}, ptrElem);
      LDBG() << "The new arg type for cuda tile: " << newType << "\n";
      newArgTypes.push_back(newType);
    }

    auto newFnType = builder.getFunctionType(newArgTypes, {});
    auto fname = builder.getStringAttr(gfunc_name);
    auto argTypes = builder.getTypeArrayAttr(newArgTypes);
    auto cudaEntryOp = mlir::cuda_tile::EntryOp::create(
        builder, gfunOp.getLoc(), fname, newFnType,
        /*arg_attrs=*/{}, /*res_attrs=*/{}, {});
    auto bb = cudaEntryOp.addEntryBlock();
    builder.setInsertionPointToStart(bb);
    auto retOp = mlir::cuda_tile::ReturnOp::create(builder, gfunOp.getLoc());

    LDBG() << "Created CudaTile Entry Op: \n" << cudaEntryOp << "\n";
  });
}

namespace mlir::toy {

std::unique_ptr<mlir::Pass> createCudaTileLoweringPass() {
  return std::make_unique<ToyToCudaTileLoweringPass>();
};

}; // namespace mlir::toy
