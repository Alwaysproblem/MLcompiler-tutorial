module {
  func.func @foo(%arg0: i32, %arg1: i32, %arg2: i1) -> i32 {
    %0 = arith.addi %arg0, %arg1 : i32
    cf.cond_br %arg2, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %1 = arith.muli %0, %arg0 : i32
    cf.br ^bb3(%1 : i32)
  ^bb2:  // pred: ^bb0
    %2 = arith.subi %0, %arg1 : i32
    cf.br ^bb3(%2 : i32)
  ^bb3(%3: i32):  // 2 preds: ^bb1, ^bb2
    %4 = arith.addi %3, %0 : i32
    return %4 : i32
  }
}