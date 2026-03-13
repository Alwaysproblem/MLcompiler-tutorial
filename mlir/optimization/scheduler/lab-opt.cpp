#include "lab/LabPasses.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::arith::ArithDialect, mlir::tensor::TensorDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                  mlir::affine::AffineDialect, mlir::cf::ControlFlowDialect>();

  mlir::registerAllPasses();
  mlir::PassPipelineRegistration<>("lab-op-stats", "Lab Op Stats Pass",
                                   [](mlir::OpPassManager &pm) {
                                     pm.addPass(mlir::createLabOpStatsPass());
                                   });
  mlir::PassPipelineRegistration<>(
      "lab-buffer-stats", "Lab Buffer Stats Pass", [](mlir::OpPassManager &pm) {
        pm.addPass(mlir::createLabBufferStatsPass());
      });
  mlir::PassPipelineRegistration<>("lab-liveness", "Lab Liveness Pass",
                                   [](mlir::OpPassManager &pm) {
                                     pm.addPass(mlir::createLabLivenessPass());
                                   });
  mlir::PassPipelineRegistration<>(
      "lab-memref-lifetime", "Lab Memref Lifetime Pass",
      [](mlir::OpPassManager &pm) {
        pm.addPass(mlir::createLabMemrefLifetimePass());
      });
  mlir::PassPipelineRegistration<>(
      "lab-fusion-feasibility", "Lab Fusion Feasibility Pass",
      [](mlir::OpPassManager &pm) {
        pm.addPass(mlir::createLabFusionFeasibilityPass());
      });

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Lab optimizer\n", registry));
}
