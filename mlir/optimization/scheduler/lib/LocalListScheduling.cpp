#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

enum class NodeKind {
  AsyncExecute,
  AsyncAwait,
  PureCompute,
  BarrierLike,
  Other
};

struct SchedNode {
  Operation *op = nullptr;
  NodeKind kind = NodeKind::Other;
  SmallVector<int> preds;
  SmallVector<int> succs;
  int indegree = 0;
  int originalOrder = -1;
};

static bool isAsyncType(Type ty) {
  return isa<async::TokenType>(ty) || isa<async::ValueType>(ty);
}

static NodeKind classifyOp(Operation *op) {
  if (isa<async::ExecuteOp>(op))
    return NodeKind::AsyncExecute;
  if (isa<async::AwaitOp>(op))
    return NodeKind::AsyncAwait;

  // terminator / region branch 直接看作 barrier
  if (op->hasTrait<OpTrait::IsTerminator>())
    return NodeKind::BarrierLike;

  // 无 side effect 的普通算子，视作纯计算
  if (isMemoryEffectFree(op))
    return NodeKind::PureCompute;

  return NodeKind::Other;
}

static bool isBarrier(Operation *op) {
  if (op->hasTrait<OpTrait::IsTerminator>())
    return true;

  // async.execute / async.await 本身不是 barrier
  if (isa<async::ExecuteOp, async::AwaitOp>(op))
    return false;

  // 纯 op 允许参与窗口调度
  if (isMemoryEffectFree(op))
    return false;

  // 其余统统保守视为 barrier
  return true;
}

static DenseMap<Operation *, int> buildOpIndex(ArrayRef<Operation *> ops) {
  DenseMap<Operation *, int> map;
  for (auto [i, op] : llvm::enumerate(ops))
    map[op] = i;
  return map;
}

static void addEdge(SmallVectorImpl<SchedNode> &nodes, int u, int v) {
  if (u == v)
    return;

  // 避免重复边
  if (llvm::is_contained(nodes[u].succs, v))
    return;

  nodes[u].succs.push_back(v);
  nodes[v].preds.push_back(u);
}

static void buildSSADependencies(ArrayRef<Operation *> ops,
                                 SmallVectorImpl<SchedNode> &nodes) {
  auto opToIdx = buildOpIndex(ops);

  for (auto [i, op] : llvm::enumerate(ops)) {
    for (Value operand : op->getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (!def)
        continue;

      auto it = opToIdx.find(def);
      if (it == opToIdx.end())
        continue;

      addEdge(nodes, it->second, i);
    }
  }
}

static bool needsConservativeOrder(Operation *a, Operation *b) {
  bool pureA = isMemoryEffectFree(a);
  bool pureB = isMemoryEffectFree(b);

  // 两个都纯，则不需要额外约束
  if (pureA && pureB)
    return false;

  // async.execute / async.await 与纯 op 混排时，第一版我们也允许
  // 只要它们的 SSA 依赖满足即可。
  if ((isa<async::ExecuteOp, async::AwaitOp>(a) || pureA) &&
      (isa<async::ExecuteOp, async::AwaitOp>(b) || pureB)) {
    return false;
  }

  // 其余情况保守约束
  return true;
}

static void buildConservativeOrderEdges(ArrayRef<Operation *> ops,
                                        SmallVectorImpl<SchedNode> &nodes) {
  for (int i = 0, e = static_cast<int>(ops.size()); i < e; ++i) {
    for (int j = i + 1; j < e; ++j) {
      if (needsConservativeOrder(ops[i], ops[j]))
        addEdge(nodes, i, j);
    }
  }
}

static int priorityOf(NodeKind kind) {
  switch (kind) {
  case NodeKind::AsyncExecute:
    return 300;
  case NodeKind::PureCompute:
    return 200;
  case NodeKind::AsyncAwait:
    return 100;
  case NodeKind::Other:
    return 50;
  case NodeKind::BarrierLike:
    return 0;
  }
  return 0;
}

static SmallVector<int> scheduleWindow(ArrayRef<SchedNode> inputNodes) {
  SmallVector<SchedNode> nodes(inputNodes.begin(), inputNodes.end());

  for (auto &n : nodes)
    n.indegree = static_cast<int>(n.preds.size());

  SmallVector<int> ready;
  for (int i = 0, e = static_cast<int>(nodes.size()); i < e; ++i) {
    if (nodes[i].indegree == 0)
      ready.push_back(i);
  }

  SmallVector<int> order;
  order.reserve(nodes.size());

  while (!ready.empty()) {
    int bestPos = 0;
    for (int k = 1, e = static_cast<int>(ready.size()); k < e; ++k) {
      int lhs = ready[k];
      int rhs = ready[bestPos];

      int pl = priorityOf(nodes[lhs].kind);
      int pr = priorityOf(nodes[rhs].kind);

      if (pl > pr) {
        bestPos = k;
        continue;
      }
      if (pl == pr && nodes[lhs].originalOrder < nodes[rhs].originalOrder) {
        bestPos = k;
      }
    }

    int u = ready[bestPos];
    ready.erase(ready.begin() + bestPos);
    order.push_back(u);

    for (int v : nodes[u].succs) {
      nodes[v].indegree--;
      if (nodes[v].indegree == 0)
        ready.push_back(v);
    }
  }

  if (order.size() != nodes.size())
    return {}; // 有环，放弃该窗口

  return order;
}

static SmallVector<SmallVector<Operation *>> collectWindows(Block &block) {
  SmallVector<SmallVector<Operation *>> windows;
  SmallVector<Operation *> current;

  for (Operation &op : block) {
    if (isBarrier(&op)) {
      if (!current.empty()) {
        windows.push_back(std::move(current));
        current.clear();
      }
      continue;
    }
    current.push_back(&op);
  }

  if (!current.empty())
    windows.push_back(std::move(current));

  return windows;
}

static bool reorderWindow(ArrayRef<Operation *> ops) {
  if (ops.size() < 2)
    return false;

  SmallVector<SchedNode> nodes;
  nodes.reserve(ops.size());

  for (auto [i, op] : llvm::enumerate(ops)) {
    nodes.push_back(SchedNode{
        .op = op,
        .kind = classifyOp(op),
        .preds = {},
        .succs = {},
        .indegree = 0,
        .originalOrder = static_cast<int>(i),
    });
  }

  buildSSADependencies(ops, nodes);
  buildConservativeOrderEdges(ops, nodes);

  SmallVector<int> newOrder = scheduleWindow(nodes);
  if (newOrder.empty())
    return false;

  bool changed = false;
  for (int i = 0, e = static_cast<int>(newOrder.size()); i < e; ++i) {
    if (newOrder[i] != i) {
      changed = true;
      break;
    }
  }
  if (!changed)
    return false;

  // 锚点：窗口结束位置（最后一个 op 的 next）
  Operation *afterWindow = ops.back()->getNextNode();

  for (int idx : newOrder) {
    Operation *op = nodes[idx].op;
    if (afterWindow)
      op->moveBefore(afterWindow);
    else
      op->moveBefore(op->getBlock(), Block::iterator());
  }

  return true;
}

struct AsyncLocalSchedulePass
    : public PassWrapper<AsyncLocalSchedulePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AsyncLocalSchedulePass)

  StringRef getArgument() const final { return "lab-async-local-schedule"; }
  StringRef getDescription() const final {
    return "Locally reorder async.execute/async.await inside a block";
  }

  void runOnOperation() override;
};

void AsyncLocalSchedulePass::runOnOperation() {
  func::FuncOp func = getOperation();

  bool changed = false;
  for (Block &block : func.getBody()) {
    auto windows = collectWindows(block);
    for (auto &window : windows) {
      changed |= reorderWindow(window);
    }
  }

  (void)changed;
}

} // namespace

namespace mlir {
std::unique_ptr<Pass> createAsyncLocalSchedulePass() {
  return std::make_unique<AsyncLocalSchedulePass>();
}
} // namespace mlir
