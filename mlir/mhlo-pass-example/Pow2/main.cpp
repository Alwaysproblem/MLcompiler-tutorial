#include <llvm/Support/CommandLine.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <mhlo/IR/hlo_ops.h>
#include <mlir/Dialect/Func/Extensions/AllExtensions.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/PDL/IR/PDL.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "passes/Pow2.h"

namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));

int loadMhlo(llvm::SourceMgr &sourceMgr, mlir::MLIRContext &context,
             mlir::OwningOpRef<mlir::ModuleOp> &module,
             std::string &inputFilename) {
  // Otherwise, the input is '.mlir'.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return -1;
  }

  // Parse the input mlir.
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return 3;
  }
  return 0;
}

int main(int argc, char *argv[]) {

  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  cl::ParseCommandLineOptions(argc, argv, "pass example compiler\n");

  // mlir::MLIRContext context;
  // context.loadDialect<mlir::mhlo::MhloDialect, mlir::func::FuncDialect,
  //                     mlir::pdl::PDLDialect>();

  // Here we can initialize the context with registered dialects
  mlir::DialectRegistry registry;
  mlir::func::registerAllExtensions(registry);
  // registry.insert<mlir::mhlo::MhloDialect, mlir::func::FuncDialect,
  //                 mlir::pdl::PDLDialect>();
  mlir::MLIRContext context(registry);

  context.getOrLoadDialect<mlir::mhlo::MhloDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::pdl::PDLDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module;
  llvm::SourceMgr sourceMgr;
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);

  if (int err = loadMhlo(sourceMgr, context, module, inputFilename)) {
    llvm::errs() << "Error loading MHLO module from file " << inputFilename
                 << ": " << err << "\n";
    return err;
  };

  llvm::dbgs() << "Input mhlo mlir:" << '\n';
  module->dump();

  unsigned domInfoCount = 0;

  if (enableOpt) {
    mlir::PassManager pm(&context);
    // Apply any generic pass manager command line options and run the pipeline.
    if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))
      return 4;

    // Enable the statistics (only works on debug mode)
    // pm.enableStatistics();

    // Add a run of the canonicalizer to optimize the mlir module.
    // pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::mhlo::createSubstitutePow2Pass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::mhlo::createStaticOpCounter());
    pm.addInstrumentation(
        mlir::mhlo::createDominanceCounterInstrumentation(domInfoCount));

    if (mlir::failed(pm.run(*module)))
      return 4;
  }

  llvm::dbgs() << "After Conversion:" << '\n';
  module->dump();
  llvm::dbgs()
      << "------------------------------------------------------------\n";
  module->print(llvm::outs());
  llvm::dbgs()
      << "------------------------------------------------------------\n";
  llvm::dbgs() << "DominanceInfo count: " << domInfoCount << "\n";

  return 0;
}
