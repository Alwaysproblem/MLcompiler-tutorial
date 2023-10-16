#include <mhlo/IR/hlo_ops.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Analysis/CallGraph.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Parser/Parser.h>
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
  return false;
}

static LogicalResult Eqn2Impl(PatternRewriter &rewriter, Value value) {
  return success(ValueEql2(value));
}

} // namespace

void registerNativeConstraints(RewritePatternSet &patterns) {
  patterns.getPDLPatterns().registerConstraintFunction("Eqn2", Eqn2Impl);
}

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "Pow2.inc"
#include "Pow2Pdll.inc"
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

namespace {
struct SubstitutePow2PdllPass
    : public PassWrapper<SubstitutePow2PdllPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SubstitutePow2PdllPass)

  void runOnOperation() final;
};
} // namespace

void SubstitutePow2PdllPass::runOnOperation() {
  auto op = getOperation();
  RewritePatternSet patterns(&getContext());
  // --- insert the native constraints ---
  registerNativeConstraints(patterns);
  // --- insert the native constraints ---
  patterns.add<Pow2PdllOptPattern>(&getContext());
  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
    signalPassFailure();
}

namespace {
#define GEN_PASS_DEF_POW2PASS
#include "Pow2Pass.inc"
} // namespace

namespace {
struct SubstitutePow2PdllGenPass
    : impl::Pow2PassBase<SubstitutePow2PdllGenPass> {
  void runOnOperation() final {
    auto op = getOperation();
    RewritePatternSet patterns(&getContext());
    // --- insert the native constraints ---
    registerNativeConstraints(patterns);
    // --- insert the native constraints ---
    patterns.add<Pow2PdllOptPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
      signalPassFailure();
  };
};
} // namespace

std::unique_ptr<mlir::Pass> mhlo::createSubstitutePow2Pass() {
  // There are 2 methods to achieve the same goal:
  // 1. use the tddr rules to rewrite the IR
  // return std::make_unique<SubstitutePow2Pass>();
  // 2. use the pdll to rewrite the IR
  // return std::make_unique<SubstitutePow2PdllPass>();
  // 3. use tddr to generate pass declaration.
  return std::make_unique<SubstitutePow2PdllPass>();
}

/// An interesting analysis.
struct StaticOpCounterAnalysis {
  llvm::StringMap<int> opCount;
  // Compute this analysis with the provided operation.
  StaticOpCounterAnalysis(Operation *op) : opCount({}){};

  void add(Operation *op) {
    auto opName = op->getName().getStringRef();
    opCount.find(opName) == opCount.end()
        ? opCount[opName] = 1
        : opCount[opName] = opCount[opName] + 1;
  }

  llvm::StringMap<int> getOpCount() const { return opCount; };
};

struct StaticOpCounterAnalysisWithDependency {
  StaticOpCounterAnalysisWithDependency(Operation *op, AnalysisManager &am) {
    // Request other analysis as dependency
    StaticOpCounterAnalysis &otherAnalysis =
        am.getAnalysis<StaticOpCounterAnalysis>();
  }

  bool isInvalidated(const AnalysisManager::PreservedAnalyses &pa) {
    // Check if analysis or its dependency were invalidated
    return !pa.isPreserved<StaticOpCounterAnalysisWithDependency>() ||
           !pa.isPreserved<StaticOpCounterAnalysis>();
  }
};

namespace {
struct StaticOpCounter
    : public PassWrapper<StaticOpCounter, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StaticOpCounter)

  void runOnOperation() final;
};
} // namespace

void StaticOpCounter::runOnOperation() {
  StaticOpCounterAnalysis &myAnalysis = getAnalysis<StaticOpCounterAnalysis>();
  auto module_op = getOperation();
  for (auto &op : module_op.getOps()) {
    myAnalysis.add(&op);
  }

  llvm::outs() << "================================================="
               << "\n";
  llvm::outs() << "MLIR-PASS-TUTOR: static analysis results\n";
  llvm::outs() << "=================================================\n";
  const char *str1 = "NAME";
  const char *str2 = "#N DIRECT CALLS";
  llvm::outs() << llvm::format("%-20s %-10s\n", str1, str2);
  llvm::outs() << "-------------------------------------------------"
               << "\n";
  for (auto &CallCount : myAnalysis.getOpCount()) {
    llvm::outs() << llvm::format("%-20s %-10lu\n",
                                 CallCount.first().str().c_str(),
                                 CallCount.getValue());
  }

  llvm::outs() << "-------------------------------------------------"
               << "\n\n";
}

std::unique_ptr<mlir::Pass> mhlo::createStaticOpCounter() {
  // There are 2 methods to achieve the same goal:
  // 1. use the tddr rules to rewrite the IR
  // return std::make_unique<SubstitutePow2Pass>();
  // 2. use the pdll to rewrite the IR
  // return std::make_unique<SubstitutePow2PdllPass>();
  // 3. use tddr to generate pass declaration.
  return std::make_unique<StaticOpCounter>();
}
