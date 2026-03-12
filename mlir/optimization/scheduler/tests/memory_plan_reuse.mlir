// RUN: ../build/scheduler/lab-scheduler %s --pass-pipeline='builtin.module(func.func(lab-buffer-stats))' --mlir-disable-threading

module {
  func.func @memory_plan_reuse_demo() {
    %c0 = arith.constant 0.0 : f32

    // Buffer A lives first.
    %a = memref.alloc() : memref<64x64xf32>
    linalg.fill ins(%c0 : f32) outs(%a : memref<64x64xf32>)

    // Buffer B overlaps with A during the copy, so it cannot reuse A's slot.
    %b = memref.alloc() : memref<64x64xf32>
    memref.copy %a, %b : memref<64x64xf32> to memref<64x64xf32>
    memref.dealloc %a : memref<64x64xf32>

    // Buffer C starts after A is dead and is smaller, so it is a good reuse candidate.
    %c = memref.alloc() : memref<32x32xf32>
    linalg.fill ins(%c0 : f32) outs(%c : memref<32x32xf32>)
    memref.dealloc %c : memref<32x32xf32>

    // Buffer D overlaps with B during the copy, so it needs another live slot.
    %d = memref.alloc() : memref<64x64xf32>
    memref.copy %b, %d : memref<64x64xf32> to memref<64x64xf32>

    memref.dealloc %b : memref<64x64xf32>
    memref.dealloc %d : memref<64x64xf32>
    return
  }
}