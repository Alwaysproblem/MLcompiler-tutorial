#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

struct BufferLifetimeInfo {
  Value value;
  int64_t sizeBytes = 0;

  int allocIndex = -1;      // storage start
  int defIndex = -1;        // semantic start (对 alloc/alloca 来说通常等于 allocIndex)
  int lastUseIndex = -1;    // semantic end
  int releaseIndex = -1;    // storage end

  bool hasExplicitRelease = false;
  bool isAlloca = false;
};

static std::optional<int64_t> getElementTypeBytes(Type elemTy) {
  if (auto intTy = dyn_cast<IntegerType>(elemTy))
    return (intTy.getWidth() + 7) / 8;
  if (elemTy.isF16() || elemTy.isBF16())
    return 2;
  if (elemTy.isF32())
    return 4;
  if (elemTy.isF64())
    return 8;
  return std::nullopt;
}

static std::optional<int64_t> getStaticMemRefSizeInBytes(MemRefType ty) {
  if (!ty.hasStaticShape())
    return std::nullopt;

  auto elemBytes = getElementTypeBytes(ty.getElementType());
  if (!elemBytes)
    return std::nullopt;

  int64_t numElems = 1;
  for (int64_t d : ty.getShape()) {
    if (d < 0)
      return std::nullopt;
    numElems *= d;
  }
  return numElems * (*elemBytes);
}

/// 找最后一次“语义使用”
/// 条件：
/// 1. op 使用了 value
/// 2. 该 op 后 value 已死
static Operation *findLastSemanticUser(func::FuncOp funcOp,
                                       Liveness &liveness,
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

    if (liveness.isDeadAfter(value, op))
      lastUser = op;
  });

  return lastUser;
}

/// 找显式 release（这里先只支持 memref.dealloc）
static Operation *findExplicitRelease(func::FuncOp funcOp, Value value) {
  Operation *releaseOp = nullptr;

  funcOp.walk([&](memref::DeallocOp deallocOp) {
    if (deallocOp.getMemref() == value)
      releaseOp = deallocOp.getOperation();
  });

  return releaseOp;
}

static void addClosedInterval(DenseMap<int, int64_t> &delta,
                              int begin,
                              int end,
                              int64_t bytes) {
  if (begin < 0 || end < 0 || end < begin)
    return;
  delta[begin] += bytes;
  delta[end + 1] -= bytes;
}

static int64_t computePeakFromDelta(const DenseMap<int, int64_t> &delta,
                                    int numProgramPoints) {
  int64_t cur = 0;
  int64_t peak = 0;
  for (int i = 0; i <= numProgramPoints; ++i) {
    auto it = delta.find(i);
    if (it != delta.end())
      cur += it->second;
    peak = std::max(peak, cur);
  }
  return peak;
}

struct LabMemrefLifetimePass
    : public PassWrapper<LabMemrefLifetimePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LabMemrefLifetimePass)

  StringRef getArgument() const final { return "lab-memref-lifetime"; }
  StringRef getDescription() const final {
    return "Print semantic/storage lifetime and estimate semantic/storage peak memory";
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // 1) Liveness analysis
    Liveness liveness(funcOp);

    // 2) 线性编号：用于打印和近似峰值计算
    DenseMap<Operation *, int> opIndex;
    SmallVector<Operation *> orderedOps;
    int nextIndex = 0;
    funcOp.walk([&](Operation *op) {
      opIndex[op] = nextIndex++;
      orderedOps.push_back(op);
    });

    SmallVector<BufferLifetimeInfo> buffers;

    // 3) 收集 memref.alloc
    funcOp.walk([&](memref::AllocOp allocOp) {
      Value v = allocOp.getResult();
      auto ty = dyn_cast<MemRefType>(v.getType());
      if (!ty)
        return;

      auto sizeBytes = getStaticMemRefSizeInBytes(ty);
      if (!sizeBytes)
        return; // 教学版：跳过动态 shape

      BufferLifetimeInfo info;
      info.value = v;
      info.sizeBytes = *sizeBytes;
      info.allocIndex = opIndex[allocOp.getOperation()];
      info.defIndex = info.allocIndex;
      info.isAlloca = false;

      if (Operation *lastUser = findLastSemanticUser(funcOp, liveness, v))
        info.lastUseIndex = opIndex[lastUser];
      else
        info.lastUseIndex = info.defIndex; // 没人用，则退化到定义点

      if (Operation *release = findExplicitRelease(funcOp, v)) {
        info.releaseIndex = opIndex[release];
        info.hasExplicitRelease = true;
      } else {
        // 没显式 dealloc，则先退化为 last use
        info.releaseIndex = info.lastUseIndex;
      }

      buffers.push_back(info);
    });

    // 4) 收集 memref.alloca
    funcOp.walk([&](memref::AllocaOp allocaOp) {
      Value v = allocaOp.getResult();
      auto ty = dyn_cast<MemRefType>(v.getType());
      if (!ty)
        return;

      auto sizeBytes = getStaticMemRefSizeInBytes(ty);
      if (!sizeBytes)
        return;

      BufferLifetimeInfo info;
      info.value = v;
      info.sizeBytes = *sizeBytes;
      info.allocIndex = opIndex[allocaOp.getOperation()];
      info.defIndex = info.allocIndex;
      info.isAlloca = true;

      if (Operation *lastUser = findLastSemanticUser(funcOp, liveness, v))
        info.lastUseIndex = opIndex[lastUser];
      else
        info.lastUseIndex = info.defIndex;

      // 教学版策略：
      // alloca 没有显式 free，保守地认为活到所在 block 的末尾
      Block *block = allocaOp->getBlock();
      Operation *lastOpInBlock = &block->back();
      info.releaseIndex = opIndex[lastOpInBlock];
      info.hasExplicitRelease = false;

      buffers.push_back(info);
    });

    // 5) 分别构建 semantic / storage 两套事件
    DenseMap<int, int64_t> semanticDelta;
    DenseMap<int, int64_t> storageDelta;

    for (const auto &buf : buffers) {
      addClosedInterval(semanticDelta,
                        buf.defIndex,
                        buf.lastUseIndex,
                        buf.sizeBytes);

      addClosedInterval(storageDelta,
                        buf.allocIndex,
                        buf.releaseIndex,
                        buf.sizeBytes);
    }

    int64_t peakSemantic =
        computePeakFromDelta(semanticDelta, orderedOps.size());
    int64_t peakStorage =
        computePeakFromDelta(storageDelta, orderedOps.size());

    // 6) 打印
    llvm::outs() << "[lab-memref-lifetime]\n";
    for (const auto &buf : buffers) {
      llvm::outs() << "  value=";
      AsmState asmState(funcOp);
      buf.value.printAsOperand(llvm::outs(), /*printType=*/asmState);
      llvm::outs() << " size=" << buf.sizeBytes << "B"
                   << " alloc=#" << buf.allocIndex
                   << " def=#" << buf.defIndex
                   << " last_use=#" << buf.lastUseIndex
                   << " release=#" << buf.releaseIndex;

      if (buf.isAlloca)
        llvm::outs() << " kind=alloca";
      else
        llvm::outs() << " kind=alloc";

      if (buf.hasExplicitRelease)
        llvm::outs() << " explicit_release=true";
      else
        llvm::outs() << " explicit_release=false";

      llvm::outs() << "\n";

      llvm::outs() << "    semantic_lifetime=["
                   << buf.defIndex << "," << buf.lastUseIndex << "]\n";
      llvm::outs() << "    storage_lifetime=["
                   << buf.allocIndex << "," << buf.releaseIndex << "]\n";
    }

    llvm::outs() << "  peak_semantic_live_memory=" << peakSemantic << "B\n";
    llvm::outs() << "  peak_storage_live_memory=" << peakStorage << "B\n";
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createLabMemrefLifetimePass() {
  return std::make_unique<LabMemrefLifetimePass>();
}
} // namespace
