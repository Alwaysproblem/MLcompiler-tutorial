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

#include "passes/Pow2.h"

#include <memory>
#include <numeric>

using namespace mlir;

namespace {
bool ValueEql2(Value operand) {
  FloatAttr::ValueType FValue = FloatAttr::ValueType(2.0);
  if (matchPattern(operand, m_ConstantFloat(&FValue))) {
    if (FValue.convertToFloat() == 2.0) {
      return true;
    }
  }
  llvm::outs() << "FValue: " << FValue.convertToFloat() << "\n";
  return false;
}
} // namespace

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "Pow2.inc"
} // namespace

namespace {
struct SubstitutePow2Pass
    : public PassWrapper<SubstitutePow2Pass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SubstitutePow2Pass)

  void runOnOperation() final;
};
} // namespace

void SubstitutePow2Pass::runOnOperation() {
  auto op = getOperation();
  RewritePatternSet patterns(&getContext());
  patterns.add<Pow2OptPattern>(&getContext());
  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> mhlo::createSubstitutePow2Pass() {
  return std::make_unique<SubstitutePow2Pass>();
}
