// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
// SDK version: 4.0.0-EA.1+1470

#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <mhlo/IR/hlo_ops.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <iostream>
#include <memory>
#include <numeric>
#include <popir/Compile.hpp>
#include <popir/Context.hpp>
#include <popir/dialect/phlo/PhloBuildUtils.hpp>
#include <popir/dialect/popit/PopitDialect.hpp>
#include <popir/poplar_lowering/PoplarLoweringContext.hpp>
#include <popit/Device.hpp>
#include <popit/popit.hpp>
#include <sstream>
#include <string>
#include <vector>

int loadMHLO(llvm::SourceMgr &sourceMgr, mlir::MLIRContext &context,
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

  if (argc < 2) {
    llvm::errs() << "usage: " << argv[0] << " <mlir file name>" << '\n';
    return 1;
  }

  std::string inputFilename = argv[1];

  mlir::MLIRContext context;
  context.loadDialect<mlir::mhlo::MhloDialect, mlir::func::FuncDialect,
                      mlir::phlo::PhloDialect, mlir::popit::PopitDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module;
  llvm::SourceMgr sourceMgr;
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);

  loadMHLO(sourceMgr, context, module, inputFilename);

  llvm::outs() << "Input mhlo mlir:" << '\n';
  module->dump();

  return 0;
}
