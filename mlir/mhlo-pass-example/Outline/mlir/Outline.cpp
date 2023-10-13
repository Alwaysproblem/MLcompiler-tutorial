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
bool ValueEql2(Value operand) {
  FloatAttr::ValueType FValue = FloatAttr::ValueType(2.0);
  if (matchPattern(operand, m_ConstantFloat(&FValue))) {
    if (FValue.convertToFloat() == 2.0) {
      return true;
    }
  }
  return false;
}

static LogicalResult Eqn2Impl(PatternRewriter &rewriter, Value value) {
  return success(ValueEql2(value));
}

// static Operation *FuncOpImpl(PDLResultList &results, Value value) {
//   // insert special rewrite logic here.
//   Operation *resultOp = ;
//   return resultOp;
// }

} // namespace

void registerNativeConstraints(RewritePatternSet &patterns) {
  patterns.getPDLPatterns().registerConstraintFunction("Eqn2", Eqn2Impl);
}

// void registerNativeRewrite(RewritePatternSet &patterns) {
//   patterns.getPDLPatterns().registerRewriteFunction("FuncOp", FuncOpImpl);
// }

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
  // --- insert the native constraints ---
  // registerNativeConstraints(patterns);
  // --- insert the native constraints ---
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
