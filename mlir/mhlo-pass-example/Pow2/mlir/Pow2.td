//===- ToyCombine.td - Pattern Match Optimizations for Toy -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines language-specific pattern match optimizations for Toy using
// Declarative Rewrite Rules (DRR) specified using TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef POW_2
#define POW_2

include "mlir/IR/PatternBase.td"
include "mhlo/IR/hlo_ops.td"

/// Note: The DRR definition used for defining patterns is shown below:
///
/// class Pattern<
///    dag sourcePattern, list<dag> resultPatterns,
///    list<dag> additionalConstraints = [],
///    dag benefitsAdded = (addBenefit 0)
/// >;

def ReshapeReshapeOptPattern : Pat<(MHLO_ReshapeOp(MHLO_ReshapeOp $arg)),
                                   (MHLO_ReshapeOp $arg)>;

#endif // POW_2
