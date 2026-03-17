#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace mlir::dataflow;

enum class ResidencyKind {
  Uninitialized, // 由框架隐式表示，通常不单独存
  Unknown,       // 什么都不知道
  DDR,           // 主存/外存
  FastMem,       // L1/L2/shared/local SRAM
  Conflict       // 来自不同来源且无法统一
};

struct ResidencyValue {
  enum Kind { Unknown, DDR, FastMem, Conflict } kind = Unknown;

  static ResidencyValue getPessimisticValueState(MLIRContext *) {
    return ResidencyValue{Unknown};
  }

  static ResidencyValue getPessimisticValueState(Value value) {
    if (auto mt = dyn_cast<MemRefType>(value.getType())) {
      if (!mt.getMemorySpace())
        return ResidencyValue{DDR};

      if (auto i = dyn_cast<IntegerAttr>(mt.getMemorySpace())) {
        if (i.getInt() == 0)
          return ResidencyValue{DDR};
        if (i.getInt() == 1)
          return ResidencyValue{FastMem};
      }
    }
    return ResidencyValue{Unknown};
  }

  static ResidencyValue join(const ResidencyValue &lhs,
                             const ResidencyValue &rhs) {
    if (lhs.kind == rhs.kind)
      return lhs;
    if (lhs.kind == Unknown)
      return rhs;
    if (rhs.kind == Unknown)
      return lhs;
    return ResidencyValue{Conflict};
  }

  bool operator==(const ResidencyValue &rhs) const { return kind == rhs.kind; }

  void print(raw_ostream &os) const {
    switch (kind) {
    case Unknown:
      os << "unknown";
      break;
    case DDR:
      os << "ddr";
      break;
    case FastMem:
      os << "fastmem";
      break;
    case Conflict:
      os << "conflict";
      break;
    }
  }
};

using ResidencyLattice = Lattice<ResidencyValue>;

class ResidencyAnalysis
    : public SparseForwardDataFlowAnalysis<ResidencyLattice> {
public:
  using SparseForwardDataFlowAnalysis<
      ResidencyLattice>::SparseForwardDataFlowAnalysis;

  LogicalResult visitOperation(Operation *op,
                               ArrayRef<const ResidencyLattice *> operands,
                               ArrayRef<ResidencyLattice *> results) override {

    // Rule 1: memref.alloc
    if (auto alloc = dyn_cast<memref::AllocOp>(op)) {
      ResidencyValue v =
          ResidencyValue::getPessimisticValueState(alloc.getResult());
      auto *lattice = getLatticeElement(alloc.getResult());
      propagateIfChanged(lattice, lattice->join(v));
      return success();
    }

    // Rule 2: memref.subview / cast / reinterpret_cast
    if (isa<memref::SubViewOp, memref::CastOp, memref::ReinterpretCastOp>(op)) {
      if (op->getNumOperands() >= 1 && op->getNumResults() >= 1) {
        auto *srcLat = operands.front();
        if (srcLat)
          join(results.front(), *srcLat);
      }
      return success();
    }

    // Rule 3: memref.copy
    if (auto copy = dyn_cast<memref::CopyOp>(op)) {
      Value target = copy.getTarget();
      auto *lattice = getLatticeElement(target);
      if (const ResidencyLattice *srcLat = operands.front())
        propagateIfChanged(lattice, lattice->join(srcLat->getValue()));

      ResidencyValue dst = ResidencyValue::getPessimisticValueState(target);
      propagateIfChanged(lattice, lattice->join(dst));
      return success();
    }

    // Rule 4: linalg generic / matmul
    if (isa<linalg::LinalgOp>(op)) {
      auto linalgOp = cast<linalg::LinalgOp>(op);
      for (Value v : linalgOp.getDpsInits()) {
        ResidencyValue rv = ResidencyValue::getPessimisticValueState(v);
        auto *lattice = getLatticeElement(v);
        propagateIfChanged(lattice, lattice->join(rv));
      }
      return success();
    }

    if (auto msc = dyn_cast<memref::MemorySpaceCastOp>(op)) {
      if (!operands.empty() && operands.front() && !results.empty()) {
        join(results.front(), *operands.front());
      }
      return success();
    }

    // 默认策略：如果 op 只是透传一个值，可传播第一个操作数状态
    if (op->getNumOperands() == 1 && op->getNumResults() == 1) {
      if (!operands.empty() && operands.front())
        join(results.front(), *operands.front());
      return success();
    }

    setAllToEntryStates(results);
    return success();
  }

protected:
  void setToEntryState(ResidencyLattice *lattice) override {
    propagateIfChanged(lattice,
                       lattice->join(ResidencyValue::getPessimisticValueState(
                           lattice->getAnchor())));
  }
};

struct ResidencyAnalysisPass
    : public PassWrapper<ResidencyAnalysisPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ResidencyAnalysisPass)

  static void printMemrefMemoryInfo(Value value, raw_ostream &os,
                                    DataFlowSolver &solver) {
    os << "    value=";
    value.print(os);
    os << " type=";
    value.getType().print(os);

    auto memrefType = dyn_cast<MemRefType>(value.getType());
    if (!memrefType) {
      os << " memory_space=<not-memref>\n";
      return;
    }

    os << " memory_space=";
    Attribute memorySpace = memrefType.getMemorySpace();
    if (!memorySpace) {
      os << "default";
    } else if (auto intAttr = dyn_cast<IntegerAttr>(memorySpace)) {
      os << intAttr.getInt();
    } else {
      memorySpace.print(os);
    }
    os << " residency=";
    solver.lookupState<ResidencyLattice>(value)->getValue().print(os);

    if (Operation *defOp = value.getDefiningOp())
      os << " defined_by=" << defOp->getName();
    os << "\n";
  }

  static void printLinalgOperandMemoryInfo(linalg::LinalgOp linalgOp,
                                           DataFlowSolver &solver) {
    llvm::errs() << "linalg op: " << linalgOp->getName() << "\n";
    llvm::errs() << "  ins:\n";
    for (Value value : linalgOp.getDpsInputs())
      printMemrefMemoryInfo(value, llvm::errs(), solver);

    llvm::errs() << "  outs:\n";
    for (Value value : linalgOp.getDpsInits())
      printMemrefMemoryInfo(value, llvm::errs(), solver);
  }

  static StringRef getResidencyTag(ResidencyValue::Kind kind) {
    switch (kind) {
    case ResidencyValue::Unknown:
      return "unknown";
    case ResidencyValue::DDR:
      return "ddr";
    case ResidencyValue::FastMem:
      return "fastmem";
    case ResidencyValue::Conflict:
      return "conflict";
    }

    llvm_unreachable("unexpected residency kind");
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    DataFlowSolver solver;
    solver.load<DeadCodeAnalysis>();
    solver.load<ResidencyAnalysis>();
    if (failed(solver.initializeAndRun(func))) {
      signalPassFailure();
      return;
    }

    func.walk([&](Operation *op) {
      if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op))
        printLinalgOperandMemoryInfo(linalgOp, solver);

      for (Value result : op->getResults()) {
        if (!isa<MemRefType>(result.getType()))
          continue;

        auto *lat = solver.lookupState<ResidencyLattice>(result);
        if (!lat)
          continue;

        op->emitRemark() << "result residency = "
                         << getResidencyTag(lat->getValue().kind);
      }
    });

    func.emitRemark() << "=== function args ===";
    for (BlockArgument arg : func.getArguments()) {
      if (!isa<MemRefType>(arg.getType()))
        continue;

      if (auto *lat = solver.lookupState<ResidencyLattice>(arg)) {
        llvm::errs() << "arg: ";
        lat->getValue().print(llvm::errs());
        llvm::errs() << "\n";
      }
    }

    func.walk([&](Operation *op) {
      for (Value result : op->getResults()) {
        if (!isa<MemRefType>(result.getType()))
          continue;

        if (auto *lat = solver.lookupState<ResidencyLattice>(result)) {
          llvm::errs() << "op result @" << op->getName() << " : ";
          lat->getValue().print(llvm::errs());
          llvm::errs() << "\n";
        }
      }
    });
  }
};

namespace mlir {
std::unique_ptr<Pass> createResidencyAnalysisPass() {
  return std::make_unique<ResidencyAnalysisPass>();
}
} // namespace mlir
