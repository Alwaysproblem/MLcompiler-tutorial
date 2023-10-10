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
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>

#include "passes/Tanh.h"

#include <memory>
#include <numeric>

using namespace mlir;

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "Tanh.inc"
} // namespace

namespace {
struct PopulateTanhPass
    : public PassWrapper<PopulateTanhPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PopulateTanhPass)

  void runOnOperation() final;
};
} // namespace

void PopulateTanhPass::runOnOperation() {
  auto op = getOperation();
  RewritePatternSet patterns(&getContext());
  patterns.add<PopulateTanh>(&getContext());
  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> mhlo::createPopulateTanhPass() {
  return std::make_unique<PopulateTanhPass>();
}
