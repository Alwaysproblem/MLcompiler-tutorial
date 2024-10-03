//===-- pass-tutor-opt.cpp - pass tutorial entry point --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mhlo/IR/hlo_ops.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "passes/Pow2.h"
#include <cstdlib>

int main(int argc, char **argv) {
  // Register all "core" dialects
  mlir::DialectRegistry registry;
  registry.insert<mlir::mhlo::MhloDialect>();
  mlir::registerAllDialects(registry);
  mlir::mhlo::registerPasses();

  // Register a handful of cleanup passes that we can run to make the output IR
  // look nicer.
  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();
  mlir::registerSymbolDCEPass();

  // Delegate to the MLIR utility for parsing and pass management.
  return mlir::MlirOptMain(argc, argv, "pass-tutor-opt", registry).succeeded()
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
