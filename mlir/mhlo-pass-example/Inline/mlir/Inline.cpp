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

// The first way to do Inline action.
// This need `erase callee;` action in pdll files.
void RegionOfFunc(mlir::PatternRewriter &rewriter, mlir::Value callee,
                  mlir::Value input) {
  auto calleeOp = callee.getDefiningOp<func::CallOp>();
  llvm::StringRef callee_name = calleeOp.getCallee();
  mlir::ModuleOp module_op = llvm::dyn_cast<mlir::ModuleOp>(
      callee.getParentRegion()->getParentOp()->getParentOp());
  auto calleeFuncOp = module_op.lookupSymbol<func::FuncOp>(callee_name);
  mlir::Region &calleeFuncOpRegion = calleeFuncOp.getBody();
  mlir::Region &main_region = *callee.getParentRegion();

  rewriter.inlineRegionBefore(calleeFuncOpRegion, main_region,
                              main_region.end());

  mlir::Block *inlineBlock = callee.getParentBlock();
  mlir::Block *interm_b = &*std::next(inlineBlock->getIterator());
  mlir::Operation *term = interm_b->getTerminator();
  mlir::Value term_oper = term->getOperands().front();
  rewriter.eraseOp(interm_b->getTerminator());
  rewriter.mergeBlockBefore(interm_b, calleeOp, input);
  inlineBlock->getTerminator()->getOpOperands().front().set(term_oper);
  rewriter.eraseOp(calleeFuncOp);
}

// The second way to do Inline action
// This need to delete `erase callee;` action in pdll files.
// But this is not real inline action, it only merge the callee region to the
// caller region. and discard the block after the callee region.
// void RegionOfFunc(mlir::PatternRewriter &rewriter, mlir::Value callee,
//                   mlir::Value input) {
//   auto calleeOp = callee.getDefiningOp<func::CallOp>();
//   llvm::StringRef callee_name = calleeOp.getCallee();
//   mlir::ModuleOp module_op = llvm::dyn_cast<mlir::ModuleOp>(
//       callee.getParentRegion()->getParentOp()->getParentOp());
//   auto calleeFuncOp = module_op.lookupSymbol<func::FuncOp>(callee_name);
//   mlir::Block *callee_block = &calleeFuncOp.getBlocks().front();
//   mlir::Region &calleeFuncOpRegion = calleeFuncOp.getBody();

//   mlir::Operation *callee_ptr = calleeOp;
//   mlir::Block *inlineBlock = callee.getParentBlock();
//   mlir::Block *postinlineBlock =
//       inlineBlock->splitBlock(callee_ptr->getIterator());

//   inlineBlock->getParent()->getBlocks().splice(
//       postinlineBlock->getIterator(), calleeFuncOpRegion.getBlocks(),
//       calleeFuncOpRegion.begin(), calleeFuncOpRegion.end());
//   auto newBlocks = llvm::make_range(std::next(inlineBlock->getIterator()),
//                                     postinlineBlock->getIterator());

//   Block *firstNewBlock = &*newBlocks.begin();

//   rewriter.eraseBlock(postinlineBlock);
//   rewriter.eraseOp(calleeFuncOp);
//   rewriter.mergeBlocks(firstNewBlock, inlineBlock, input);
// }

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
  // Here we only apply pass on the `main` function.
  if (op.getName() != "main")
    return;
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

// --------------- unused code ----------------

// auto *firstBlockTerminator = firstNewBlock->getTerminator();
// rewriter.eraseOp(firstBlockTerminator);
// firstNewBlock->getOperations().splice(firstNewBlock->end(),
//                                         postinlineBlock->getOperations());

// void RegionOfFunc(mlir::PatternRewriter &rewriter, mlir::Value callee,
//                   mlir::Value input) {
//   // mlir::Operation *RegionOfFunc(mlir::PatternRewriter &rewriter,
//   //                               mlir::Value callee, mlir::Value input) {
//   auto calleeOp = callee.getDefiningOp<func::CallOp>();
//   if (!calleeOp) {
//     return;
//   }
//   llvm::StringRef callee_name = calleeOp.getCallee();
//   mlir::ModuleOp module_op = llvm::dyn_cast<mlir::ModuleOp>(
//       callee.getParentRegion()->getParentOp()->getParentOp());
//   auto calleeFuncOp = module_op.lookupSymbol<func::FuncOp>(callee_name);
//   mlir::Block *callee_block = &calleeFuncOp.getBlocks().front();
//   mlir::Region &calleeFuncOpRegion = calleeFuncOp.getBody();
//   callee_block->without_terminator();
//   mlir::Operation *callee_ptr = calleeOp;
//   mlir::Block *inlineBlock = callee.getParentBlock();
//   mlir::Block *postinlineBlock =
//       inlineBlock->splitBlock(callee_ptr->getIterator());

//   inlineBlock->getParent()->getBlocks().splice(
//       postinlineBlock->getIterator(), calleeFuncOpRegion.getBlocks(),
//       calleeFuncOpRegion.begin(), calleeFuncOpRegion.end());
//   auto newBlocks = llvm::make_range(std::next(inlineBlock->getIterator()),
//                                     postinlineBlock->getIterator());

//   Block *firstNewBlock = &*newBlocks.begin();

//   auto *firstBlockTerminator = firstNewBlock->getTerminator();
//   rewriter.eraseOp(firstBlockTerminator);
//   firstNewBlock->getOperations().splice(firstNewBlock->end(),
//                                           postinlineBlock->getOperations());

//   rewriter.mergeBlocks(firstNewBlock, inlineBlock, input);
//   inlineBlock->print(llvm::outs());
//   // rewriter.eraseBlock(postinlineBlock);
//   // rewriter.eraseOp(calleeFuncOp);
//   llvm::outs() << "callee: "
//                << "\n";
//   module_op.print(llvm::outs());
//   // return rewriter.create<mhlo::AbsOp>(callee.getLoc(), input);
// }

// mlir::Block *imd_blk = rewriter.createBlock(postinlineBlock);
// // rewriter.mergeBlocks(callee_block, imd_blk, input);

// imd_blk->print(llvm::outs());

// inlineBlock->getParent()->getBlocks().splice(
//     postinlineBlock->getIterator(), calleeFuncOpRegion.getBlocks(),
//     calleeFuncOpRegion.begin(), calleeFuncOpRegion.end());
// auto newBlocks = llvm::make_range(std::next(inlineBlock->getIterator()),
//                                   postinlineBlock->getIterator());

// Block *firstNewBlock = &*newBlocks.begin();
// auto *firstBlockTerminator = firstNewBlock->getTerminator();
// rewriter.eraseOp(firstBlockTerminator);

// rewriter.mergeBlocks(firstNewBlock, inlineBlock, input);
// rewriter.mergeBlocks(postinlineBlock, inlineBlock, input);

// inlineBlock->print(llvm::outs());
// inlineBlock->walk([&](func::CallOp op) {
//   op.erase();
// });

// rewriter.mergeBlockBefore();
// calleeFuncOpRegion.begin()->print(llvm::outs());

// mlir::Operation *callee_ptr = calleeOp;
// mlir::Block *inlineBlock = callee.getParentBlock();
// mlir::Block *postinlineBlock =
//     rewriter.splitBlock(inlineBlock, callee_ptr->getIterator());
// postinlineBlock->print(llvm::outs());

// rewriter.inlineRegionBefore(calleeFuncOpRegion, postinlineBlock);
// mlir::Block *interm_b = &*std::next(inlineBlock->getIterator());
// rewriter.eraseOp(interm_b->getTerminator());

// rewriter.mergeBlocks(inlineBlock, postinlineBlock);
// rewriter.mergeBlocks(interm_b, postinlineBlock);
// // rewriter.mergeBlocks(interm_b, inlineBlock, input);
// // rewriter.mergeBlocks(postinlineBlock, inlineBlock);

// postinlineBlock->print(llvm::outs());
// rewriter.mergeBlockBefore(inlineBlock, calleeOp);
// rewriter.mergeBlockBefore(interm_b, calleeOp, input);

// postinlineBlock->print(llvm::outs());

// mlir::Block *callee_block = &calleeFuncOp.getBlocks().front();
