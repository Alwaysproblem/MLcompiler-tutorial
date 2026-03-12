#pragma once
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

struct ScheduleCandidate {
  llvm::SmallVector<int64_t> tileSizes;
  bool promote = false;
  bool fuse = false;
  bool doubleBuffer = false;
  int pipelineStages = 1;

  int64_t estimatedTrafficBytes = 0;
  int64_t estimatedPeakMemoryBytes = 0;
  double estimatedCost = 0.0;
};