#include "cuda_shim/SupportOps.hpp"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "toy/Dialect.h"
#include "toy/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/DebugLog.h"

#include <memory>
#include <string>

#define DEBUG_TYPE "toy-gpu-outline"

namespace {

static bool isGpuOperation(mlir::Operation *op,
                           const llvm::SmallSet<llvm::StringRef, 4> &gpuOps) {
  llvm::StringRef opName = op->getName().getStringRef().split('.').second;
  return gpuOps.contains(opName);
}

static llvm::SmallVector<int64_t, 3> parseGrid(llvm::StringRef gridStr) {
  llvm::SmallVector<int64_t, 3> dims;
  llvm::SmallVector<llvm::StringRef, 4> pieces;
  gridStr.split(pieces, ',');
  for (llvm::StringRef piece : pieces) {
    int64_t value = 0;
    if (!piece.empty() && llvm::to_integer(piece.trim(), value))
      dims.push_back(value);
  }
  if (dims.size() != 3)
    dims = {1, 1, 1};
  return dims;
}

struct GpuOutlinePass
    : public mlir::PassWrapper<GpuOutlinePass,
                               mlir::OperationPass<mlir::toy::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GpuOutlinePass)

  std::string grid{"1,1,1"};

  llvm::StringRef getArgument() const override { return "toy-gpu-outline"; }

  void initializeOptions(std::string grid) { this->grid = grid; }

  void runOnOperation() override {
    auto func = getOperation();
    if (func.getName() != "main")
      return;

    llvm::SmallSet<llvm::StringRef, 4> gpuOperations = SUPPORT_OPS;

    // // Collect GPU-eligible ops in block order for deterministic cloning.
    // llvm::SmallDenseSet<mlir::Operation *, 8> gpuOpSet;
    // llvm::SmallVector<mlir::Operation *> gpuOps;

    // for (mlir::Operation &op : func.front()) {
    //   if (isGpuOperation(&op, gpuOperations)) {
    //     gpuOpSet.insert(&op);
    //     gpuOps.push_back(&op);
    //   }
    // }

    // if (gpuOps.empty())
    //   return;

    llvm::SmallVector<int64_t, 3> gridDims = parseGrid(grid);

    llvm::SmallVector<llvm::SmallVector<mlir::Operation *>> gpuSubgraphs;

    // Find a gpu subgraph like
    // [[gpuOps, ...], [gpuOps, ...], ...]
    // original sequence:
    // [..., non-gpu-op, [gpu-op, gpu-op], non-gpu-op, [gpu-op, ...]]
    func.walk([&](mlir::Operation *op) {
      if (isGpuOperation(op, gpuOperations)) {
        if (gpuSubgraphs.empty()) {
          gpuSubgraphs.push_back({op});
        } else {
          gpuSubgraphs.back().push_back(op);
        }
      } else {
        if (gpuSubgraphs.empty()) {
          gpuSubgraphs.push_back({});
        } else if (!gpuSubgraphs.back().empty()) {
          gpuSubgraphs.push_back({});
        }
      }
    });

    if (gpuSubgraphs.empty())
      return;

    bool allEmpty = llvm::all_of(
        gpuSubgraphs, [](const llvm::SmallVector<mlir::Operation *> &sg) {
          return sg.empty();
        });

    if (allEmpty)
      return;

    if (gpuSubgraphs.back().empty()) {
      gpuSubgraphs.pop_back();
    }

    for (const auto &gpuSubgraph : gpuSubgraphs) {
      LDBG() << "----GPU subgraph----";
      for (const auto &op : gpuSubgraph) {
        LDBG() << *op;
      }
      LDBG() << "--------------------";
    }

    llvm::SmallVector<std::string> outlinedFuncNames;
    llvm::SmallVector<mlir::Operation *> insertPoints;

    // the logic to outline each gpu subgraph
    // 1. find operands or input for the subgraph (exclude the input inside
    // subgraph).
    // 2. find results or output for the subgraph (exclude the output inside
    // subgraph).
    // 3. create a new function with operands as input and results as output.
    // 4. insert a LaunchGpuOp to call the outlined function at the insert point

    for (const auto &[index, gpuSubgraph] : llvm::enumerate(gpuSubgraphs)) {
      if (!gpuSubgraph.empty()) {
        LDBG() << "----GPU subgraph----";
        for (const auto &op : gpuSubgraph) {
          LDBG() << *op;
        }

        // Identify its operands.
        llvm::SmallVector<mlir::Value, 8> Operands;
        llvm::SmallPtrSet<mlir::Value, 8> OperandSet;
        for (mlir::Operation *op : gpuSubgraph) {
          for (mlir::Value operand : op->getOperands()) {
            auto *def = operand.getDefiningOp();
            if (!def || !isGpuOperation(def, gpuOperations)) {
              if (OperandSet.insert(operand).second)
                Operands.push_back(operand);
            }
          }
        }

        LDBG() << "Operands:";
        for (mlir::Value &operand : Operands) {
          LDBG() << "  " << operand;
        }

        llvm::SmallVector<mlir::Value, 2> Results;
        llvm::SmallPtrSet<mlir::Value, 2> ResultSet;

        for (mlir::Operation *op : gpuSubgraph) {
          for (mlir::Value result : op->getResults()) {
            bool escapes =
                llvm::any_of(result.getUsers(), [&](mlir::Operation *user) {
                  return !isGpuOperation(user, gpuOperations);
                });
            if (escapes && ResultSet.insert(result).second)
              Results.push_back(result);
          }
        }

        LDBG() << "Results:";
        for (mlir::Value &result : Results) {
          LDBG() << "  " << result;
        }

        if (Results.size() != 1) {
          llvm::errs()
              << "Currently only support single result GPU kernel "
              << "Since the toy return op only supports single return value "
              << "Found " << Results.size() << " results";
          return signalPassFailure();
        }

        // buid the kernel for each subgraph
        llvm::SmallVector<mlir::Type, 8> argTypes;
        argTypes.reserve(Operands.size());
        for (mlir::Value v : Operands)
          argTypes.push_back(v.getType());

        llvm::SmallVector<mlir::Type> resultTypes;
        resultTypes.reserve(Results.size());
        for (mlir::Value v : Results)
          resultTypes.push_back(v.getType());

        mlir::ModuleOp module = func->getParentOfType<mlir::ModuleOp>();
        mlir::SymbolTable symbolTable(module);
        std::string outline_func_name =
            "outlined_gpu_kernel_" + std::to_string(index);

        unsigned suffix = 0;
        while (symbolTable.lookup(outline_func_name))
          outline_func_name =
              outline_func_name + "_" + std::to_string(++suffix);

        insertPoints.push_back(gpuSubgraph.front());

        {
          mlir::OpBuilder moduleBuilder(module.getContext());
          mlir::OpBuilder::InsertionGuard guard(moduleBuilder);
          moduleBuilder.setInsertionPointToEnd(module.getBody());
          auto funcType = moduleBuilder.getFunctionType(argTypes, resultTypes);
          auto gpuFunc = mlir::toy::GPUFuncOp::create(
              moduleBuilder, func.getLoc(), outline_func_name, funcType);

          mlir::Block &kernelEntry = gpuFunc.getBody().front();
          mlir::OpBuilder kernelBuilder =
              mlir::OpBuilder::atBlockEnd(&kernelEntry);

          mlir::IRMapping mapping;
          for (auto [blockArg, captured] :
               llvm::zip(kernelEntry.getArguments(), Operands))
            mapping.map(captured, blockArg);

          for (mlir::Operation *op : gpuSubgraph) {
            kernelBuilder.clone(*op, mapping);
          }
          llvm::SmallVector<mlir::Value> mappedResults;
          mappedResults.reserve(Results.size());
          for (mlir::Value res : Results)
            mappedResults.push_back(mapping.lookup(res));
          mlir::toy::ReturnOp::create(kernelBuilder, func.getLoc(),
                                      mappedResults);

          LDBG() << "Created GPU kernel: " << gpuFunc;
        }

        outlinedFuncNames.push_back(outline_func_name);

        {
          mlir::OpBuilder hostBuilder(func.getContext());
          mlir::OpBuilder::InsertionGuard guard(hostBuilder);
          // Insert the host launch in place of the first outlined op.
          hostBuilder.setInsertionPoint(gpuSubgraph.back()->getNextNode());

          auto calleeAttr = mlir::SymbolRefAttr::get(
              func.getContext(), llvm::StringRef(outline_func_name));

          auto gridAttr = hostBuilder.getDenseI64ArrayAttr(gridDims);

          auto launch = mlir::toy::LaunchGpuOp::create(
              hostBuilder, func.getLoc(), resultTypes, Operands,
              {{"callee", calleeAttr}, {"grid", gridAttr}});

          for (auto [idx, res] : llvm::enumerate(Results))
            res.replaceAllUsesWith(launch.getResult(idx));

          for (mlir::Operation *op : llvm::reverse(gpuSubgraph))
            op->erase();
          LDBG() << "Inserted LaunchGpuOp: " << launch;
        }
        LDBG() << "--------------------";
      }
    }
  };
};
}; // namespace

namespace mlir::toy {

std::unique_ptr<mlir::Pass> createGpuOutlinePass(std::string grid) {
  auto pass = std::make_unique<GpuOutlinePass>();
  pass->initializeOptions(grid); // You can change the grid dimensions here
  return pass;
};

}; // namespace mlir::toy
