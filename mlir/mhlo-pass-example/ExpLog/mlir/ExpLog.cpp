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

#include "passes/ExpLog.h"

#include <memory>
#include <numeric>

using namespace mlir;

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "ExpLog.inc"
} // namespace

namespace {
struct ExpLogEmitPassPass
    : public PassWrapper<ExpLogEmitPassPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExpLogEmitPassPass)

  void runOnOperation() final;
};
} // namespace

void ExpLogEmitPassPass::runOnOperation() {
  auto op = getOperation();
  RewritePatternSet patterns(&getContext());
  patterns.add<ExpLogEmit>(&getContext());
  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> mhlo::createExpLogEmitPass() {
  return std::make_unique<ExpLogEmitPassPass>();
}
