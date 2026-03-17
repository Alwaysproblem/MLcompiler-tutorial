#pragma once

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

class Pass;

std::unique_ptr<Pass> createLabOpStatsPass();
std::unique_ptr<Pass> createLabBufferStatsPass();
std::unique_ptr<Pass> createLabFusionFeasibilityPass();
std::unique_ptr<Pass> createLabMatmulTilePass();
std::unique_ptr<Pass> createLabPipelinePlanPass();
std::unique_ptr<Pass> createLabLivenessPass();
std::unique_ptr<Pass> createLabMemrefLifetimePass();
std::unique_ptr<Pass> createLabFusionFeasibilityPass();
std::unique_ptr<Pass> createAsyncLocalSchedulePass();
std::unique_ptr<Pass> createSimpleLoopInterchangePass();
std::unique_ptr<Pass> createResidencyAnalysisPass();

} // namespace mlir
