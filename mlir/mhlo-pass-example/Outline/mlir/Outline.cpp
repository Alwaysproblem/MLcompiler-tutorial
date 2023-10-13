#include <mhlo/IR/hlo_ops.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>

#include "passes/Outline.h"

#include <memory>
#include <numeric>

using namespace mlir;

namespace {

mlir::Operation *createFunction(mlir::PatternRewriter &rewriter,
                                mlir::Value root, mlir::Value input) {
  mlir::ModuleOp module_op = llvm::dyn_cast<mlir::ModuleOp>(
      input.getParentRegion()->getParentOp()->getParentOp());
  mlir::Block *module_block = module_op.getBody();
  mlir::OpBuilder builder(input.getContext());
  builder.setInsertionPointToStart(module_block);
  mlir::FunctionType func_type =
      builder.getFunctionType(input.getType(), root.getType());
  func::FuncOp func_op =
      builder.create<func::FuncOp>(root.getLoc(), "tanh_function", func_type);

  builder.setInsertionPointToStart(func_op.addEntryBlock());
  mlir::Value argument = func_op.getArgument(0);
  mlir::Value exp_res = builder.create<mhlo::ExpOp>(root.getLoc(), argument);
  mlir::Value neg_res = builder.create<mhlo::NegOp>(root.getLoc(), argument);
  mlir::Value exp_neg_res = builder.create<mhlo::ExpOp>(root.getLoc(), neg_res);
  mlir::Value sub_res =
      builder.create<mhlo::SubtractOp>(root.getLoc(), exp_res, exp_neg_res);
  mlir::Value add_res =
      builder.create<mhlo::AddOp>(root.getLoc(), exp_res, exp_neg_res);
  mlir::Value root_res =
      builder.create<mhlo::DivOp>(root.getLoc(), sub_res, add_res);

  builder.create<func::ReturnOp>(root.getLoc(), root_res);
  // rewriter.create<func::CallOp>(root.getLoc(), "tanh_function",
  // root.getType(),
  //                               input);
  // module_op.print(llvm::outs());
  // return rewriter.create<mhlo::TanhOp>(root.getLoc(), input);
  return rewriter.create<func::CallOp>(root.getLoc(), func_op, input);
}

} // namespace

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "Outline.inc"
#include "OutlinePdll.inc"
} // namespace

namespace {
struct OutlinePass
    : public PassWrapper<OutlinePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OutlinePass)

  void runOnOperation() final;
};
} // namespace

void OutlinePass::runOnOperation() {
  auto op = getOperation();
  RewritePatternSet patterns(&getContext());
  patterns.add<OutlinePattern>(&getContext());
  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
    signalPassFailure();
}

namespace {
struct OutlinePdllPass
    : public PassWrapper<OutlinePdllPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OutlinePdllPass)

  void runOnOperation() final;
};
} // namespace

void OutlinePdllPass::runOnOperation() {
  auto op = getOperation();
  RewritePatternSet patterns(&getContext());
  patterns.add<OutlinePdllOptPattern>(&getContext());
  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> mhlo::createOutlinePass() {
  // There are 2 methods to achieve the same goal:
  // 1. use the tddr rules to rewrite the IR
  // return std::make_unique<OutlinePass>();
  // 2. use the pdll to rewrite the IR
  return std::make_unique<OutlinePdllPass>();
}
