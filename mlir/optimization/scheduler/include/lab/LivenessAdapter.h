#pragma once

#include "mlir/Analysis/Liveness.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include <cstdint>
#include <optional>

namespace mlir {
namespace lab {

struct ValueLifetimeInfo {
  int64_t start = -1;
  int64_t end = -1;
  int64_t sizeBytes = 0;
};

class LivenessAdapter {
public:
  virtual ~LivenessAdapter() = default;
  virtual std::optional<ValueLifetimeInfo> lookup(Value v) const = 0;
  virtual int64_t getPeakLiveBytes(Operation *scope) const = 0;
};

class MlirLivenessAdapter final : public LivenessAdapter {
public:
  explicit MlirLivenessAdapter(Operation *scope);

  std::optional<ValueLifetimeInfo> lookup(Value v) const override;
  int64_t getPeakLiveBytes(Operation *scope) const override;

private:
  Operation *topScope;
  mlir::Liveness liveness;

  llvm::DenseMap<Operation *, int64_t> opOrder;
  llvm::DenseMap<Value, ValueLifetimeInfo> valueInfo;
  int64_t cachedPeakLiveBytes = 0;

  void buildOperationOrder(Operation *scope);
  void buildValueLifetimeInfo(Operation *scope);
  void buildPeakLiveBytes(Operation *scope);

  static int64_t getValueSizeBytes(Value v);
};

} // namespace lab
} // namespace mlir