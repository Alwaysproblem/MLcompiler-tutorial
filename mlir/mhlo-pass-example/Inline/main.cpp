#include <llvm/Support/CommandLine.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <mhlo/IR/hlo_ops.h>
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

#include "passes/Inline.h"

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

  mlir::MLIRContext context;
  context.loadDialect<mlir::mhlo::MhloDialect, mlir::func::FuncDialect,
                      mlir::pdl::PDLDialect>();

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

  if (enableOpt) {
    mlir::PassManager pm(&context);
    // Apply any generic pass manager command line options and run the pipeline.
    applyPassManagerCLOptions(pm);

    // Add a run of the canonicalizer to optimize the mlir module.
    // pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
    mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
    // optPM.addNestedPass<mlir::func::FuncOp>(mlir::mhlo::createInlinePass());
    optPM.addPass(mlir::mhlo::createInlinePass());

    if (mlir::failed(pm.run(*module)))
      return 4;
  }

  llvm::dbgs() << "After Conversion:"
               << "\n\n";
  module->dump();
  module->print(llvm::outs());

  return 0;
}
