#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::affine;

static bool isPerfectTwoLevelNest(AffineForOp outer, AffineForOp &inner) {
  Block &body = outer.getRegion().front();

  Operation *firstNonTerminator = nullptr;
  for (Operation &op : body.without_terminator()) {
    if (firstNonTerminator)
      return false; // 外层 body 里不止一个非 terminator op
    firstNonTerminator = &op;
  }

  if (!firstNonTerminator)
    return false;

  inner = dyn_cast<AffineForOp>(firstNonTerminator);
  return inner != nullptr;
}

static bool shouldInterchangeByLastIndexHeuristic(AffineForOp outer,
                                                  AffineForOp inner) {
  Value outerIV = outer.getInductionVar();
  Value innerIV = inner.getInductionVar();

  bool outerUsedAsLastIndex = false;
  bool innerUsedAsLastIndex = false;

  inner.walk([&](Operation *op) {
    if (auto load = dyn_cast<AffineLoadOp>(op)) {
      auto indices = load.getIndices();
      if (!indices.empty()) {
        if (indices.back() == outerIV)
          outerUsedAsLastIndex = true;
        if (indices.back() == innerIV)
          innerUsedAsLastIndex = true;
      }
    }
    if (auto store = dyn_cast<AffineStoreOp>(op)) {
      auto indices = store.getIndices();
      if (!indices.empty()) {
        if (indices.back() == outerIV)
          outerUsedAsLastIndex = true;
        if (indices.back() == innerIV)
          innerUsedAsLastIndex = true;
      }
    }
  });

  // 如果外层 iv 作为最右索引更常见，而内层不是，则值得尝试交换
  return outerUsedAsLastIndex && !innerUsedAsLastIndex;
}

namespace {
struct SimpleLoopInterchangePass
    : public PassWrapper<SimpleLoopInterchangePass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SimpleLoopInterchangePass)

  StringRef getArgument() const final { return "lab-simple-loop-interchange"; }
  StringRef getDescription() const final {
    return "A simple affine loop interchange pass for perfect 2-level nests";
  }

  void runOnOperation() override;
};

void SimpleLoopInterchangePass::runOnOperation() {
  func::FuncOp func = getOperation();

  SmallVector<AffineForOp> candidates;
  func.walk([&](AffineForOp forOp) { candidates.push_back(forOp); });

  for (AffineForOp outer : candidates) {
    AffineForOp inner;
    if (!isPerfectTwoLevelNest(outer, inner))
      continue;

    if (!shouldInterchangeByLastIndexHeuristic(outer, inner))
      continue;

    SmallVector<AffineForOp> loops = {outer, inner};
    SmallVector<unsigned> perm = {1, 0}; // 交换两层

    if (!isValidLoopInterchangePermutation(loops, perm))
      continue;

    interchangeLoops(outer, inner);
  }
}
} // namespace

namespace mlir {
std::unique_ptr<Pass> createSimpleLoopInterchangePass() {
  return std::make_unique<SimpleLoopInterchangePass>();
}
} // namespace mlir
