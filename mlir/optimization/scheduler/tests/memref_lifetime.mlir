module {
  func.func @demo(%cond: i1) {
    %c0 = arith.constant 0.0 : f32

    %0 = memref.alloc() : memref<1x3x160x240xf32>
    linalg.fill ins(%c0 : f32) outs(%0 : memref<1x3x160x240xf32>)

    cf.cond_br %cond, ^bb1, ^bb2

  ^bb1:
    %1 = memref.alloc() : memref<1x3x160x240xf32>
    memref.copy %0, %1 : memref<1x3x160x240xf32> to memref<1x3x160x240xf32>
    memref.dealloc %0 : memref<1x3x160x240xf32>
    cf.br ^bb3(%1 : memref<1x3x160x240xf32>)

  ^bb2:
    %2 = memref.alloca() : memref<1x3x160x240xf32>
    memref.copy %0, %2 : memref<1x3x160x240xf32> to memref<1x3x160x240xf32>
    cf.br ^bb3(%2 : memref<1x3x160x240xf32>)

  ^bb3(%x: memref<1x3x160x240xf32>):
    %3 = memref.alloc() : memref<1x3x160x240xf32>
    memref.copy %x, %3 : memref<1x3x160x240xf32> to memref<1x3x160x240xf32>
    memref.dealloc %3 : memref<1x3x160x240xf32>
    return
  }
}
