#pragma once

#include "mlir/Analysis/Liveness.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>

#include "mlir/Analysis/Liveness.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include <cstdint>
#include <optional>

namespace mlir {
namespace lab {

enum class FusionReasonKind {
  // positive
  DirectUse,
  SingleUse,
  SupportedProducer,
  SupportedConsumer,
  ElementwiseConsumer,
  StaticShapeCompatible,
  NoSideEffect,
  IntermediateEliminable,
  PeakMemoryAcceptable,
  TrafficReductionExpected,

  // negative
  NullProducer,
  MultiUseProducer,
  UnsupportedProducer,
  UnsupportedConsumer,
  NonElementwiseConsumer,
  DynamicShapeUnsupported,
  ShapeMismatch,
  SideEffectingProducer,
  SideEffectingConsumer,
  IntermediateNotEliminable,
  PeakMemoryTooHigh
};

struct FusionFeasibilityResult {
  Operation *producer = nullptr;
  Operation *consumer = nullptr;
  OpOperand *fusedOperand = nullptr;

  bool isFusable = false;
  bool isProfitable = false;

  int64_t eliminatedIntermediateBytes = 0;
  int64_t estimatedTrafficSavedBytes = 0;
  int64_t estimatedPeakBeforeBytes = 0;
  int64_t estimatedPeakAfterBytes = 0;
  int64_t extraLivenessGrowthBytes = 0;

  double score = 0.0;

  llvm::SmallVector<FusionReasonKind> reasons;
};

const char *toString(FusionReasonKind kind);

} // namespace lab
} // namespace mlir