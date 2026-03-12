#pragma once
#include "mlir/IR/Value.h"

struct Slot {
  int id;
  int64_t sizeBytes;
  int64_t availableAfter;
};

struct MemorySlotAssignment {
  int64_t slotId = -1;
  int64_t offset = 0;
};

struct MemoryPlanResult {
  llvm::DenseMap<mlir::Value, MemorySlotAssignment> assignments;
  int64_t totalPoolBytes = 0;
};
