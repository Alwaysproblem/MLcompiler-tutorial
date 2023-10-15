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

#include "passes/Inline.h"

#include <memory>
#include <numeric>

using namespace mlir;

namespace {

void RegionOfFunc(mlir::PatternRewriter &rewriter, mlir::Value callee,
                  mlir::Value input) {
  // mlir::Operation *RegionOfFunc(mlir::PatternRewriter &rewriter,
  //                               mlir::Value callee, mlir::Value input) {
  auto calleeOp = callee.getDefiningOp<func::CallOp>();
  if (!calleeOp) {
    return;
  }
  llvm::StringRef callee_name = calleeOp.getCallee();
  mlir::ModuleOp module_op = llvm::dyn_cast<mlir::ModuleOp>(
      callee.getParentRegion()->getParentOp()->getParentOp());
  auto calleeFuncOp = module_op.lookupSymbol<func::FuncOp>(callee_name);
  mlir::Block *callee_block = &calleeFuncOp.getBlocks().front();
  mlir::Region &calleeFuncOpRegion = calleeFuncOp.getBody();
  callee_block->without_terminator();
  mlir::Operation *callee_ptr = calleeOp;
  mlir::Block *inlineBlock = callee.getParentBlock();
  mlir::Block *postinlineBlock =
      inlineBlock->splitBlock(callee_ptr->getIterator());

  inlineBlock->getParent()->getBlocks().splice(
      postinlineBlock->getIterator(), calleeFuncOpRegion.getBlocks(),
      calleeFuncOpRegion.begin(), calleeFuncOpRegion.end());
  auto newBlocks = llvm::make_range(std::next(inlineBlock->getIterator()),
                                    postinlineBlock->getIterator());

  Block *firstNewBlock = &*newBlocks.begin();

  // auto *firstBlockTerminator = firstNewBlock->getTerminator();
  // rewriter.eraseOp(firstBlockTerminator);
  // firstNewBlock->getOperations().splice(firstNewBlock->end(),
  //                                         postinlineBlock->getOperations());

  firstNewBlock->print(llvm::outs());
  rewriter.eraseBlock(postinlineBlock);
  rewriter.eraseOp(calleeFuncOp);
  rewriter.mergeBlocks(firstNewBlock, inlineBlock, input);
  llvm::outs() << "callee: "
               << "\n";
  module_op.print(llvm::outs());
  // return rewriter.create<mhlo::AbsOp>(callee.getLoc(), input);
}

} // namespace

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "Inline.inc"
#include "InlinePdll.inc"
} // namespace

namespace {
struct InlinePass
    : public PassWrapper<InlinePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InlinePass)

  /// Initialize the canonicalizer by building the set of patterns used during
  /// execution. Here, The `initialize` will call only once when the pass is
  /// created. but the `runOnOperation` will call many times in different
  /// threads. if we set up a new `RewritePatternSet` in the `runOnOperation`
  /// function, it could riase `Segmentation Fault` error. However, if there is
  /// only an Op (in this case, is func::FuncOp), this problem will not happen.
  /// For example, the mlir like:
  /// module {
  ///   func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) ->
  ///   tensor<2x2xf32> {
  ///     ...
  ///     func.return %2 : tensor<2x2xf32>
  ///   }
  ///   func.func @hello(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) ->
  ///   tensor<2x2xf32> {
  ///     ...
  ///     func.return %2 : tensor<2x2xf32>
  ///   }
  /// }
  /// And the rewrite action is implemented in the `runOnOperation`. This will
  /// cause the Error because the pass will call `mlir::getorLoadDialect` in
  /// multi-thread. So, the solution is implemented the `initialize` function to
  /// create a new `RewritePatternSet` since the `initialize` function will only
  /// call once.
  /// NB: the `op->erase` action will cause the `Segmentation Fault` error.
  LogicalResult initialize(MLIRContext *context) override {
    RewritePatternSet owningPatterns(context);
    // registerNativeConstraints(owningPatterns);
    owningPatterns.add<InlinePattern>(context);

    patterns = FrozenRewritePatternSet(std::move(owningPatterns));
    return success();
  }

  void runOnOperation() final;
  FrozenRewritePatternSet patterns;
};
} // namespace

void InlinePass::runOnOperation() {
  auto op = getOperation();
  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
    signalPassFailure();
}

namespace {
struct InlinePdllPass
    : public PassWrapper<InlinePdllPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InlinePdllPass)

  LogicalResult initialize(MLIRContext *context) override {
    RewritePatternSet owningPatterns(context);
    // registerNativeConstraints(owningPatterns);
    owningPatterns.add<InlinePdllOptPattern>(context);

    patterns = FrozenRewritePatternSet(std::move(owningPatterns));
    return success();
  }

  void runOnOperation() final;
  FrozenRewritePatternSet patterns;
};
} // namespace

void InlinePdllPass::runOnOperation() {
  auto op = getOperation();
  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> mhlo::createInlinePass() {
  // There are 2 methods to achieve the same goal:
  // 1. use the tddr rules to rewrite the IR
  // return std::make_unique<InlinePass>();
  // 2. use the pdll to rewrite the IR
  return std::make_unique<InlinePdllPass>();
}
