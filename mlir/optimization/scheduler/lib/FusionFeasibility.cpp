#include "lab/FusionFeasibility.h"
#include "lab/LivenessAdapter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::lab;

const char *mlir::lab::toString(FusionReasonKind kind) {
  switch (kind) {
  case FusionReasonKind::DirectUse:
    return "direct-use";
  case FusionReasonKind::SingleUse:
    return "single-use";
  case FusionReasonKind::SupportedProducer:
    return "supported-producer";
  case FusionReasonKind::SupportedConsumer:
    return "supported-consumer";
  case FusionReasonKind::ElementwiseConsumer:
    return "elementwise-consumer";
  case FusionReasonKind::StaticShapeCompatible:
    return "static-shape-compatible";
  case FusionReasonKind::NoSideEffect:
    return "no-side-effect";
  case FusionReasonKind::IntermediateEliminable:
    return "intermediate-eliminable";
  case FusionReasonKind::PeakMemoryAcceptable:
    return "peak-memory-acceptable";
  case FusionReasonKind::TrafficReductionExpected:
    return "traffic-reduction-expected";
  case FusionReasonKind::NullProducer:
    return "null-producer";
  case FusionReasonKind::MultiUseProducer:
    return "multi-use-producer";
  case FusionReasonKind::UnsupportedProducer:
    return "unsupported-producer";
  case FusionReasonKind::UnsupportedConsumer:
    return "unsupported-consumer";
  case FusionReasonKind::NonElementwiseConsumer:
    return "non-elementwise-consumer";
  case FusionReasonKind::DynamicShapeUnsupported:
    return "dynamic-shape-unsupported";
  case FusionReasonKind::ShapeMismatch:
    return "shape-mismatch";
  case FusionReasonKind::SideEffectingProducer:
    return "side-effecting-producer";
  case FusionReasonKind::SideEffectingConsumer:
    return "side-effecting-consumer";
  case FusionReasonKind::IntermediateNotEliminable:
    return "intermediate-not-eliminable";
  case FusionReasonKind::PeakMemoryTooHigh:
    return "peak-memory-too-high";
  }
  return "unknown";
}

static int64_t getElementTypeBytes(Type elemTy) {
  if (auto ft = dyn_cast<FloatType>(elemTy))
    return ft.getWidth() / 8;
  if (auto it = dyn_cast<IntegerType>(elemTy))
    return it.getWidth() / 8;
  return -1;
}

static int64_t getShapedTypeBytes(Type ty) {
  auto shaped = dyn_cast<ShapedType>(ty);
  if (!shaped || !shaped.hasStaticShape())
    return -1;
  int64_t elemBytes = getElementTypeBytes(shaped.getElementType());
  if (elemBytes < 0)
    return -1;
  return shaped.getNumElements() * elemBytes;
}

static bool sameStaticShapeAndElemType(Type a, Type b) {
  auto ta = dyn_cast<ShapedType>(a);
  auto tb = dyn_cast<ShapedType>(b);
  if (!ta || !tb || !ta.hasStaticShape() || !tb.hasStaticShape())
    return false;
  return ta.getShape() == tb.getShape() &&
         ta.getElementType() == tb.getElementType();
}

static bool isConsumerIterationSpaceCompatible(Value producerVal,
                                               Operation *consumer,
                                               OpOperand &fusedOperand) {
  auto consumerLinalg = dyn_cast<linalg::LinalgOp>(consumer);
  auto producerType = dyn_cast<ShapedType>(producerVal.getType());
  if (!consumerLinalg || !producerType || !producerType.hasStaticShape())
    return false;

  AffineMap operandMap = consumerLinalg.getMatchingIndexingMap(&fusedOperand);
  if (!operandMap.isPermutation())
    return false;

  SmallVector<int64_t> loopRanges = consumerLinalg.getStaticLoopRanges();
  if (llvm::any_of(loopRanges, ShapedType::isDynamic))
    return false;

  if (operandMap.getNumResults() != producerType.getRank() ||
      static_cast<int64_t>(loopRanges.size()) != producerType.getRank())
    return false;

  for (const auto &[dimIndex, expr] :
       llvm::enumerate(operandMap.getResults())) {
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr)
      return false;

    unsigned loopDim = dimExpr.getPosition();
    if (loopDim >= loopRanges.size())
      return false;
    if (producerType.getDimSize(dimIndex) != loopRanges[loopDim])
      return false;
  }

  return true;
}

static bool isSupportedProducer(Operation *op) {
  return isa<linalg::GenericOp, linalg::MatmulOp,
             linalg::ConvolutionOpInterface>(op);
}

static bool isSupportedConsumer(Operation *op) {
  return isa<linalg::GenericOp>(op);
}

static bool isElementwiseConsumer(Operation *op) {
  auto generic = dyn_cast<linalg::GenericOp>(op);
  if (!generic)
    return false;
  return generic.getNumLoops() == generic.getNumParallelLoops();
}

static bool hasNoSideEffects(Operation *op) {
  // 教学版：如果实现了内存副作用接口，就要求没有 effect；
  // 否则先保守地当作无副作用或按项目需要更严格处理。
  if (auto mem = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<MemoryEffects::EffectInstance, 4> effects;
    mem.getEffects(effects);
    return effects.empty();
  }
  return true;
}

static int64_t
estimateExtraLivenessGrowthBytes(Operation *producer, Operation *consumer,
                                 const LivenessAdapter *liveness) {
  if (!liveness)
    return 0;

  int64_t penalty = 0;

  for (Value v : producer->getOperands()) {
    auto info = liveness->lookup(v);
    if (!info)
      continue;

    // 教学版近似：
    // 如果 producer 输入在原来 producer 附近就死掉，但 consumer 更晚，
    // 假设 fusion 可能把它延长到 consumer。
    // 这里没有精确 op 序号接口时，你可以先简单保守累加所有 producer inputs。
    penalty += info->sizeBytes;
  }

  return penalty;
}

static FusionFeasibilityResult
analyzeFusionCandidate(Operation *producer, Operation *consumer,
                       OpOperand &fusedOperand,
                       const LivenessAdapter *liveness) {
  FusionFeasibilityResult r;
  r.producer = producer;
  r.consumer = consumer;
  r.fusedOperand = &fusedOperand;

  if (!producer) {
    r.reasons.push_back(FusionReasonKind::NullProducer);
    return r;
  }

  Value producerVal = fusedOperand.get();
  if (producerVal.getDefiningOp() != producer) {
    r.reasons.push_back(FusionReasonKind::NullProducer);
    return r;
  }
  r.reasons.push_back(FusionReasonKind::DirectUse);

  if (!isSupportedProducer(producer)) {
    r.reasons.push_back(FusionReasonKind::UnsupportedProducer);
    return r;
  }
  r.reasons.push_back(FusionReasonKind::SupportedProducer);

  if (!isSupportedConsumer(consumer)) {
    r.reasons.push_back(FusionReasonKind::UnsupportedConsumer);
    return r;
  }
  r.reasons.push_back(FusionReasonKind::SupportedConsumer);

  if (!isElementwiseConsumer(consumer)) {
    r.reasons.push_back(FusionReasonKind::NonElementwiseConsumer);
    return r;
  }
  r.reasons.push_back(FusionReasonKind::ElementwiseConsumer);

  if (!producerVal.hasOneUse()) {
    r.reasons.push_back(FusionReasonKind::MultiUseProducer);
    return r;
  }
  r.reasons.push_back(FusionReasonKind::SingleUse);

  if (!hasNoSideEffects(producer)) {
    r.reasons.push_back(FusionReasonKind::SideEffectingProducer);
    return r;
  }
  if (!hasNoSideEffects(consumer)) {
    r.reasons.push_back(FusionReasonKind::SideEffectingConsumer);
    return r;
  }
  r.reasons.push_back(FusionReasonKind::NoSideEffect);

  int64_t bytes = getShapedTypeBytes(producerVal.getType());
  if (bytes < 0) {
    r.reasons.push_back(FusionReasonKind::DynamicShapeUnsupported);
    return r;
  }

  if (!isConsumerIterationSpaceCompatible(producerVal, consumer,
                                          fusedOperand)) {
    r.reasons.push_back(FusionReasonKind::ShapeMismatch);
    return r;
  }
  r.reasons.push_back(FusionReasonKind::StaticShapeCompatible);

  r.eliminatedIntermediateBytes = bytes;
  r.estimatedTrafficSavedBytes = 2 * bytes;
  r.reasons.push_back(FusionReasonKind::IntermediateEliminable);
  r.reasons.push_back(FusionReasonKind::TrafficReductionExpected);

  r.isFusable = true;

  // Liveness / peak 估计
  r.estimatedPeakBeforeBytes =
      liveness ? liveness->getPeakLiveBytes(consumer->getParentOp()) : 0;
  r.extraLivenessGrowthBytes =
      estimateExtraLivenessGrowthBytes(producer, consumer, liveness);
  r.estimatedPeakAfterBytes = std::max<int64_t>(
      0, r.estimatedPeakBeforeBytes - r.eliminatedIntermediateBytes +
             r.extraLivenessGrowthBytes);

  if (!liveness || r.estimatedPeakAfterBytes <=
                       static_cast<int64_t>(1.2 * r.estimatedPeakBeforeBytes)) {
    r.reasons.push_back(FusionReasonKind::PeakMemoryAcceptable);
  } else {
    r.reasons.push_back(FusionReasonKind::PeakMemoryTooHigh);
  }

  r.score =
      1.0 * static_cast<double>(r.eliminatedIntermediateBytes) +
      0.5 * static_cast<double>(r.estimatedTrafficSavedBytes) -
      2.0 * static_cast<double>(r.extraLivenessGrowthBytes) -
      4.0 * static_cast<double>(std::max<int64_t>(
                0, r.estimatedPeakAfterBytes - r.estimatedPeakBeforeBytes));

  r.isProfitable =
      r.isFusable &&
      llvm::is_contained(r.reasons, FusionReasonKind::PeakMemoryAcceptable) &&
      r.score > 0.0;

  return r;
}

namespace {

struct LabFusionFeasibilityPass
    : public PassWrapper<LabFusionFeasibilityPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LabFusionFeasibilityPass)

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    MlirLivenessAdapter liveness(func);
    const LivenessAdapter *livenessPtr = &liveness;

    int64_t totalCandidates = 0;
    int64_t fusableCount = 0;
    int64_t profitableCount = 0;

    func.walk([&](Operation *consumer) {
      for (OpOperand &operand : consumer->getOpOperands()) {
        Operation *producer = operand.get().getDefiningOp();
        if (!producer)
          continue;

        ++totalCandidates;
        FusionFeasibilityResult result =
            analyzeFusionCandidate(producer, consumer, operand, livenessPtr);

        if (result.isFusable)
          ++fusableCount;
        if (result.isProfitable)
          ++profitableCount;

        consumer->emitRemark()
            << "[lab-fusion-feasibility] "
            << "producer=" << producer->getName().getStringRef()
            << " consumer=" << consumer->getName().getStringRef()
            << " fusable=" << (result.isFusable ? "true" : "false")
            << " profitable=" << (result.isProfitable ? "true" : "false")
            << " elim_bytes=" << result.eliminatedIntermediateBytes
            << " traffic_saved=" << result.estimatedTrafficSavedBytes
            << " peak_before=" << result.estimatedPeakBeforeBytes
            << " peak_after=" << result.estimatedPeakAfterBytes
            << " extra_live=" << result.extraLivenessGrowthBytes
            << " score=" << result.score;

        for (FusionReasonKind reason : result.reasons) {
          consumer->emitRemark() << "  reason: " << toString(reason);
        }
      }
    });

    func.emitRemark() << "[lab-fusion-feasibility-summary] candidates="
                      << totalCandidates << " fusable=" << fusableCount
                      << " profitable=" << profitableCount;
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createLabFusionFeasibilityPass() {
  return std::make_unique<LabFusionFeasibilityPass>();
}
} // namespace mlir
