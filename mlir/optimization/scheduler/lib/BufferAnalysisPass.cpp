#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <sys/types.h>

using namespace mlir;

namespace {

class LabBufferStatsAnalysis {
public:
  struct BufferRecord {
    Value value;
    int64_t sizeBytes;
    int defIndex;
    int lastUseIndex;
  };

  struct States {
    uint64_t num_allocs = 0;
    uint64_t num_allocas = 0;
    uint64_t num_deallocs = 0;
    uint64_t num_reuses = 0;
    uint64_t num_dynamic = 0;
    uint64_t peak_memory = 0;
  };

  explicit LabBufferStatsAnalysis(Operation *op) {
    auto func = cast<func::FuncOp>(op);

    func.walk([&](Operation *op) {
      if (isa<memref::AllocOp>(op)) {
        states.num_allocs++;
        auto memrefType = llvm::cast<MemRefType>(op->getResult(0).getType());
        if (memrefType.hasStaticShape() == false) {
          states.num_dynamic++;
        }
        uint64_t size = 1;
        for (auto dim : memrefType.getShape()) {
          size *= dim;
        }
        size *= getElementBytes(memrefType.getElementType());
      } else if (isa<memref::AllocaOp>(op)) {
        states.num_allocas++;
        auto memrefType = llvm::cast<MemRefType>(op->getResult(0).getType());
        if (memrefType.hasStaticShape() == false) {
          states.num_dynamic++;
        }
        uint64_t size = 1;
        for (auto dim : memrefType.getShape()) {
          size *= dim;
        }
        size *= getElementBytes(memrefType.getElementType());
      } else if (isa<memref::DeallocOp>(op)) {
        states.num_deallocs++;
      } else if (isa<linalg::CopyOp, memref::CopyOp>(op)) {
        // This is a very naive way to detect buffer reuse, but it serves as a
        // starting point.
        auto srcType = llvm::cast<MemRefType>(op->getOperand(0).getType());
        auto dstType = llvm::cast<MemRefType>(op->getOperand(1).getType());
        if (srcType && dstType && srcType == dstType) {
          states.num_reuses++;
        }
      }
    });

    DenseMap<Operation *, int> opIndex;
    SmallVector<Operation *> orderedOps;

    // Step 1: 给 op 编号
    int nextIndex = 0;
    func.walk([&](Operation *op) {
      opIndex[op] = nextIndex++;
      orderedOps.push_back(op);
    });

    for (uint32_t i = 0; i < orderedOps.size(); ++i) {
      assert(opIndex[orderedOps[i]] == (int)i && "op index mismatch");
      LDBG() << "op #" << i << ": " << orderedOps[i]->getName() << "\n";
    }

    // Step 2: 找 buffer owner，并计算 def/lastUse/size
    func.walk([&](Operation *op) {
      Value result;
      mlir::Liveness liveness(func);

      if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
        result = allocOp.getResult();
      } else if (auto allocaOp = dyn_cast<memref::AllocaOp>(op)) {
        result = allocaOp.getResult();
      } else {
        return;
      }

      auto memrefTy = dyn_cast<MemRefType>(result.getType());
      if (!memrefTy)
        return;

      auto sizeBytes = getStaticMemRefSizeInBytes(memrefTy);
      if (!sizeBytes)
        return; // 教学版：先跳过动态 shape

      int def = opIndex[op];

      // First version: 直接找最后一个使用点
      // int lastUse = def;

      // for (OpOperand &use : result.getUses()) {
      //   Operation *user = use.getOwner();
      //   auto it = opIndex.find(user);
      //   if (it != opIndex.end())
      //     lastUse = std::max(lastUse, it->second);
      // }

      auto indexOp = findLastSemanticUser(func, liveness, result);
      if (!indexOp)
        return; // 没有语义使用点？先跳过
      int lastUse = opIndex[indexOp];


      buffers.push_back(BufferRecord{
          result,
          *sizeBytes,
          def,
          lastUse,
      });
    });

    // Step 3: 事件扫描求 peak
    DenseMap<int, int64_t> delta;
    for (const auto &buf : buffers) {
      delta[buf.defIndex] += buf.sizeBytes;
      delta[buf.lastUseIndex + 1] -= buf.sizeBytes; // 闭区间 [def, lastUse]
    }

    int64_t curLive = 0;
    int64_t peakLive = 0;
    for (int i = 0; i <= (int)orderedOps.size(); ++i) {
      auto it = delta.find(i);
      if (it != delta.end())
        curLive += it->second;
      peakLive = std::max(peakLive, curLive);
    }
    states.peak_memory = peakLive;
  }

  static int64_t getElementBytes(Type t) {
    if (auto ft = dyn_cast<FloatType>(t))
      return ft.getWidth() / 8;
    if (auto it = dyn_cast<IntegerType>(t))
      return it.getWidth() / 8;
    return 0;
  }

  static std::optional<int64_t> getStaticMemRefSizeInBytes(MemRefType ty) {
    if (!ty.hasStaticShape())
      return std::nullopt;

    auto elemBytes = getElementBytes(ty.getElementType());
    if (elemBytes == 0)
      return std::nullopt;

    int64_t numElems = 1;
    for (int64_t d : ty.getShape()) {
      if (d < 0)
        return std::nullopt;
      numElems *= d;
    }
    return numElems * elemBytes;
  }

  static Operation *findLastSemanticUser(func::FuncOp funcOp,
                                        mlir::Liveness &liveness,
                                        Value value) {
    Operation *lastUser = nullptr;

    funcOp.walk([&](Operation *op) {
      bool usesValue = false;
      for (Value operand : op->getOperands()) {
        if (operand == value) {
          usesValue = true;
          break;
        }
      }
      if (!usesValue)
        return;

      // 如果这个 op 使用了 value，并且 op 之后 value 已死，
      // 就把它视为“最后语义使用点”的候选。
      if (liveness.isDeadAfter(value, op))
        lastUser = op;
    });

    return lastUser;
  }

  static bool isBufferOwner(Value v) {
    Operation *defOp = v.getDefiningOp();
    return isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(defOp);
  }

  const States &getStates() const { return states; }
  const SmallVector<BufferRecord> &getBuffers() const { return buffers; }

private:
  States states;
  SmallVector<BufferRecord> buffers;
};

struct LabBufferStats
    : public PassWrapper<LabBufferStats, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LabBufferStats)

  StringRef getArgument() const override { return "lab-buffer-stats"; }

  void runOnOperation() override {
    auto func = getOperation();
    auto &analysis = getAnalysis<LabBufferStatsAnalysis>();
    const auto &states = analysis.getStates();

    func.emitRemark() << "[lab-buffer-stats] Allocs=" << states.num_allocs
                      << " Allocas=" << states.num_allocas
                      << " Deallocs=" << states.num_deallocs
                      << " Reuses=" << states.num_reuses
                      << " Dynamic=" << states.num_dynamic
                      << " PeakMemory=" << states.peak_memory;
    const auto &buffers = analysis.getBuffers();

    llvm::outs() << "[lab-buffer-stats]\n";
    for (const auto &buf : buffers) {
      std::string valueStr;
      llvm::raw_string_ostream rso(valueStr);
      AsmState asmState(func);
      buf.value.printAsOperand(rso, asmState);
      llvm::outs() << "  value=" << rso.str() << " size=" << buf.sizeBytes << "B"
                   << " def=#" << buf.defIndex
                   << " last_use=#" << buf.lastUseIndex
                   << " lifetime=[" << buf.defIndex
                   << "," << buf.lastUseIndex << "]\n";
    }
    llvm::outs() << "  peak_live_memory=" << analysis.getStates().peak_memory << "B\n";

  }
};
} // namespace

namespace mlir {
std::unique_ptr<Pass> createLabBufferStatsPass() {
  return std::make_unique<LabBufferStats>();
}
} // namespace mlir
