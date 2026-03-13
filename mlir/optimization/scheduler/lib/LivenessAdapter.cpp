#include "lab/LivenessAdapter.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::lab;

static int64_t getElementTypeBytes(Type elemTy) {
  if (auto ft = dyn_cast<FloatType>(elemTy))
    return ft.getWidth() / 8;
  if (auto it = dyn_cast<IntegerType>(elemTy))
    return it.getWidth() / 8;
  return -1;
}

int64_t MlirLivenessAdapter::getValueSizeBytes(Value v) {
  auto shapedTy = dyn_cast<ShapedType>(v.getType());
  if (!shapedTy || !shapedTy.hasStaticShape())
    return -1;

  int64_t elemBytes = getElementTypeBytes(shapedTy.getElementType());
  if (elemBytes < 0)
    return -1;

  return shapedTy.getNumElements() * elemBytes;
}

MlirLivenessAdapter::MlirLivenessAdapter(Operation *scope)
    : topScope(scope), liveness(scope) {
  buildOperationOrder(scope);
  buildValueLifetimeInfo(scope);
  buildPeakLiveBytes(scope);
}

std::optional<ValueLifetimeInfo> MlirLivenessAdapter::lookup(Value v) const {
  auto it = valueInfo.find(v);
  if (it == valueInfo.end())
    return std::nullopt;
  return it->second;
}

int64_t MlirLivenessAdapter::getPeakLiveBytes(Operation *scope) const {
  (void)scope;
  return cachedPeakLiveBytes;
}

void MlirLivenessAdapter::buildOperationOrder(Operation *scope) {
  int64_t index = 0;
  scope->walk([&](Operation *op) { opOrder[op] = index++; });
}

void MlirLivenessAdapter::buildValueLifetimeInfo(Operation *scope) {
  llvm::SmallVector<Operation *> ops;
  scope->walk([&](Operation *op) {
    Block *block = op->getBlock();
    if (!block)
      return;
    if (!liveness.getLiveness(block))
      return;
    ops.push_back(op);
  });

  auto updateValue = [&](Value v) {
    if (valueInfo.contains(v))
      return;

    ValueLifetimeInfo info;
    info.sizeBytes = getValueSizeBytes(v);

    // start
    if (auto result = dyn_cast<OpResult>(v)) {
      Operation *defOp = result.getOwner();
      auto it = opOrder.find(defOp);
      if (it != opOrder.end())
        info.start = it->second;
    } else if (auto barg = dyn_cast<BlockArgument>(v)) {
      Block *block = barg.getOwner();
      if (!block->empty()) {
        Operation &front = block->front();
        auto it = opOrder.find(&front);
        if (it != opOrder.end())
          info.start = it->second;
      } else {
        info.start = 0;
      }
    }

    // end: 找最后一个“在该 op 之后仍然 live”的 op
    int64_t lastLive = info.start;
    for (Operation *op : ops) {
      if (!liveness.isDeadAfter(v, op)) {
        auto it = opOrder.find(op);
        if (it != opOrder.end())
          lastLive = std::max(lastLive, it->second);
      }
    }
    info.end = lastLive;

    valueInfo[v] = info;
  };

  // 收集所有 block arguments
  scope->walk([&](Block *block) {
    for (BlockArgument arg : block->getArguments())
      updateValue(arg);
  });

  // 收集所有 op results
  scope->walk([&](Operation *op) {
    for (Value result : op->getResults())
      updateValue(result);
  });
}

void MlirLivenessAdapter::buildPeakLiveBytes(Operation *scope) {
  (void)scope;
  int64_t maxOrder = -1;
  for (const auto &it : opOrder)
    maxOrder = std::max(maxOrder, it.second);

  int64_t peak = 0;
  for (int64_t i = 0; i <= maxOrder; ++i) {
    int64_t liveBytes = 0;
    for (const auto &kv : valueInfo) {
      const ValueLifetimeInfo &info = kv.second;
      if (info.sizeBytes <= 0)
        continue;
      if (info.start <= i && i <= info.end)
        liveBytes += info.sizeBytes;
    }
    peak = std::max(peak, liveBytes);
  }

  cachedPeakLiveBytes = peak;
}
