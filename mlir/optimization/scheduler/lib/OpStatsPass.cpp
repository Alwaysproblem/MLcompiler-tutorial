#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
struct LabOpStatsPass
    : public PassWrapper<LabOpStatsPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LabOpStatsPass)

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    func.walk([&](Operation *op) {
      if (auto matmul = dyn_cast<linalg::MatmulOp>(op)) {
        analyzeMatmul(matmul);
      } else if (auto generic = dyn_cast<linalg::GenericOp>(op)) {
        analyzeGeneric(generic);
      }
    });
  }

  static int64_t getElementBytes(Type t) {
    if (auto ft = dyn_cast<FloatType>(t))
      return ft.getWidth() / 8;
    if (auto it = dyn_cast<IntegerType>(t))
      return it.getWidth() / 8;
    return 0;
  }

  void analyzeMatmul(linalg::MatmulOp op) {
    auto aType =
        dyn_cast<ShapedType>(op.getDpsInputOperand(0)->get().getType());
    auto bType =
        dyn_cast<ShapedType>(op.getDpsInputOperand(1)->get().getType());
    auto cType = dyn_cast<ShapedType>(op.getDpsInitOperand(0)->get().getType());

    if (!aType || !bType || !cType || !aType.hasStaticShape() ||
        !bType.hasStaticShape() || !cType.hasStaticShape()) {
      op.emitRemark() << "[lab-op-stats] dynamic shape matmul, skip";
      return;
    }

    int64_t M = aType.getShape()[0];
    int64_t K = aType.getShape()[1];
    int64_t N = bType.getShape()[1];

    int64_t elemBytes = getElementBytes(aType.getElementType());
    if (elemBytes == 0) {
      op.emitRemark() << "[lab-op-stats] unsupported element type";
      return;
    }

    int64_t flops = 2 * M * N * K;
    int64_t aBytes = aType.getNumElements() * elemBytes;
    int64_t bBytes = bType.getNumElements() * elemBytes;
    int64_t cBytes = cType.getNumElements() * elemBytes;
    int64_t totalBytes = aBytes + bBytes + cBytes;

    double intensity = totalBytes > 0 ? static_cast<double>(flops) /
                                            static_cast<double>(totalBytes)
                                      : 0.0;

    op.emitRemark() << "[lab-op-stats] matmul "
                    << "M=" << M << " N=" << N << " K=" << K
                    << " flops=" << flops << " bytes=" << totalBytes
                    << " intensity=" << intensity;
  }

  void analyzeGeneric(linalg::GenericOp op) {
    unsigned numLoops = op.getNumLoops();
    unsigned numParallel = op.getNumParallelLoops();
    unsigned numReduction = numLoops - numParallel;

    op.emitRemark() << "[lab-op-stats] generic "
                    << "loops=" << numLoops << " parallel=" << numParallel
                    << " reduction=" << numReduction;
  }
};
} // namespace

namespace mlir {
std::unique_ptr<Pass> createLabOpStatsPass() {
  return std::make_unique<LabOpStatsPass>();
}
} // namespace mlir