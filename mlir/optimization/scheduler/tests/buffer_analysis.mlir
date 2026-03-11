module {
  func.func @main() {
    %c0 = arith.constant 0.0 : f32
    %0 = memref.alloc() : memref<1x3x160x240xf32>   // 1*3*160*240*4 = 460800
    linalg.fill ins(%c0 : f32) outs(%0 : memref<1x3x160x240xf32>)

    %1 = memref.alloc() : memref<1x3x160x240xf32>
    memref.copy %0, %1 : memref<1x3x160x240xf32> to memref<1x3x160x240xf32>

    memref.dealloc %0 : memref<1x3x160x240xf32>
    memref.dealloc %1 : memref<1x3x160x240xf32>

    %acc = memref.alloca() : memref<2x200000xf32>
    memref.dealloc %acc : memref<2x200000xf32>

    return
  }
}
