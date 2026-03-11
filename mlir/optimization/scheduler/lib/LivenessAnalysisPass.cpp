#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

struct LabLivenessPass
    : public PassWrapper<LabLivenessPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LabLivenessPass)

  StringRef getArgument() const final { return "lab-liveness"; }
  StringRef getDescription() const final {
    return "Example pass that prints MLIR liveness information";
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // 1) 构建 liveness 分析
    Liveness liveness(funcOp);

    AsmState asmState(funcOp);

    llvm::outs() << "=== Liveness for function @" << funcOp.getName()
                 << " ===\n";

    // 2) 遍历每个 block，打印 live-in / live-out
    for (Block &block : funcOp.getBlocks()) {
      llvm::outs() << "\nBlock ";
      if (block.getParentOp() == funcOp)
        llvm::outs() << "(top-level)";
      llvm::outs() << " {\n";

      llvm::outs() << "  live-in : ";
      printValueSet(liveness.getLiveIn(&block), asmState);
      llvm::outs() << "\n";

      llvm::outs() << "  live-out: ";
      printValueSet(liveness.getLiveOut(&block), asmState);
      llvm::outs() << "\n";

      // 3) 遍历 block 内 op，检查 operand 在该 op 后是否 dead
      for (Operation &op : block.getOperations()) {
        llvm::outs() << "  op: " << op.getName() << "\n";

        for (Value operand : op.getOperands()) {
          llvm::outs() << "    operand=";

          operand.printAsOperand(llvm::outs(), asmState);

          bool deadAfter = liveness.isDeadAfter(operand, &op);
          llvm::outs() << " dead_after_op=" << (deadAfter ? "true" : "false")
                       << "\n";
        }
      }

      llvm::outs() << "}\n";
    }

    // 4) 也可以直接整体打印
    llvm::outs() << "\n--- Full liveness dump ---\n";
    liveness.print(llvm::outs());
    llvm::outs() << "\n";
  }

  static void printValueSet(const Liveness::ValueSetT &values,
                            AsmState &asmState) {
    llvm::outs() << "{";
    bool first = true;
    for (Value v : values) {
      if (!first)
        llvm::outs() << ", ";
      v.printAsOperand(llvm::outs(), asmState);
      first = false;
    }
    llvm::outs() << "}";
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createLabLivenessPass() {
  return std::make_unique<LabLivenessPass>();
}
} // namespace mlir
