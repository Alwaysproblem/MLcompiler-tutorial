#include <mhlo/IR/hlo_ops.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include "mlir/IR/Dominance.h"
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Analysis/CallGraph.h>
#include <mlir/IR/Action.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>

#include "passes/Pow2.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

#include <memory>
#include <numeric>

using namespace mlir;

namespace {
// We add MLIR actions here as an example.
/// A custom Action can be defined minimally by deriving from
/// `tracing::ActionImpl`. The action is same as the pass declaration with tddr
/// rules. only for `xxx-opt` binary. and run with `--log-actions-to=-` to dump
/// the actions.
class EchoAction : public tracing::ActionImpl<EchoAction> {
public:
  using Base = tracing::ActionImpl<EchoAction>;
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EchoAction)

  /// Actions are initialized with an array of IRUnit (that is either Operation,
  /// Block, or Region) that provide context for the IR affected by a
  /// transformation.
  EchoAction(ArrayRef<IRUnit> irUnits, int iteration)
      : Base(irUnits), iteration(iteration) {}
  /// This tag should uniquely identify this action, it can be matched for
  /// filtering during processing.
  static constexpr StringLiteral tag = "echo-action";
  static constexpr StringLiteral desc = "Just echo the iteration";

  void print(raw_ostream &os) const override {
    os << "EchoAction: " << iteration << "\n";
  }

private:
  int iteration;
};
} // namespace

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
  // Here we can add the statistic (the statistic is only works on debug mode)
  // Since the statistic has the atomic value, we can't use the default copy
  // constructor and assignment operator.
  // SubstitutePow2Pass() = default;
  // SubstitutePow2Pass(const SubstitutePow2Pass &other) {
  //   this->statistic = other.statistic.getValue();
  // };
  // mlir::Pass::Statistic statistic{this, "example-statistic", "An example
  // statistic"};

  void runOnOperation() final;
};
} // namespace

void SubstitutePow2Pass::runOnOperation() {
  auto op = getOperation();
  RewritePatternSet patterns(&getContext());
  patterns.add<Pow2OptPattern>(&getContext());
  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
    signalPassFailure();
  // Here we can add the statistic (the statistic is only works on debug mode)
  // statistic++;
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
struct Pow2PassOptions {
  bool Pow2Pass = true;
};

#define GEN_PASS_DEF_POW2PASS
#include "Pow2Pass.inc"
} // namespace

namespace {
struct SubstitutePow2PdllGenPass
    : impl::Pow2PassBase<SubstitutePow2PdllGenPass> {
  void runOnOperation() final {
    auto op = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(&*context);
    // --- insert the native constraints ---
    registerNativeConstraints(patterns);
    // --- insert the native constraints ---
    patterns.add<Pow2PdllOptPattern>(&*context);
    // Here, we wrap the applyPatternsAndFoldGreedily in a lambda function and
    // pass it to the MLIR Action.
    Operation *opp = getOperation();
    ArrayRef<IRUnit> irUnits{opp};
    context->executeAction<EchoAction>(
        [&]() {
          // Here is the pass body.
          if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
            signalPassFailure();
        },
        /*irUnits=*/irUnits, /*iteration=*/10);
    // Above, we pass the irUnits and iteration to the EchoAction.
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
  return std::make_unique<SubstitutePow2PdllGenPass>();
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

  const char *str1 = "NAME";
  const char *str2 = "#N DIRECT CALLS";

  llvm::dbgs() << "================================================="
               << "\n";
  llvm::dbgs() << "MLIR-PASS-TUTOR: static analysis results\n";
  llvm::dbgs() << "=================================================\n";
  llvm::dbgs() << llvm::format("%-20s %-10s\n", str1, str2);
  llvm::dbgs() << "-------------------------------------------------"
               << "\n";
  for (auto &CallCount : myAnalysis.getOpCount()) {
    llvm::dbgs() << llvm::format("%-20s %-10lu\n",
                                 CallCount.first().str().c_str(),
                                 CallCount.getValue());
  }

  llvm::dbgs() << "-------------------------------------------------"
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

struct DominanceCounterInstrumentation : public PassInstrumentation {
  /// The cumulative count of how many times dominance has been calculated.
  unsigned &count;

  DominanceCounterInstrumentation(unsigned &count) : count(count) {}
  void runBeforePass(Pass *pass, Operation *op) override {
    llvm::dbgs() << "Before pass: " << pass->getName() << "\n";
    op->dump();
  }
  void runAfterPass(Pass *pass, Operation *op) override {
    llvm::dbgs() << "After pass: " << pass->getName() << "\n";
    op->dump();
  }
};

std::unique_ptr<mlir::PassInstrumentation>
mhlo::createDominanceCounterInstrumentation(unsigned &count) {
  return std::make_unique<DominanceCounterInstrumentation>(count);
}
