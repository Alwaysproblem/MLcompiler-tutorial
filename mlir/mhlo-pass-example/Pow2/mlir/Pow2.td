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


def TypesAreIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;

def Eqn2 : Constraint<CPred<"::ValueEql2($0)">>;

def Pow2OptPattern : Pat<(MHLO_PowOp $arg, (MHLO_ConstantOp:$cst $cstVal)),
                         (MHLO_MulOp $arg, $arg),
                         [(TypesAreIdentical $arg, $cst),
                          (Eqn2 $cst)]>;

#endif // POW_2
