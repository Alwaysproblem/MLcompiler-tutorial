func.func @interchange_me(%A: memref<64x64xf32>, %B: memref<64x64xf32>) {
  affine.for %i = 0 to 64 {
    affine.for %j = 0 to 64 {
      %v = affine.load %A[%j, %i] : memref<64x64xf32>
      %c = arith.constant 1.0 : f32
      %r = arith.addf %v, %c : f32
      affine.store %r, %B[%j, %i] : memref<64x64xf32>
    }
  }
  return
}

// we assume the tensor is stored in row-major order, so the original loop order is i-j.
// expected to be transformed to:
// func.func @interchange_me(%A: memref<64x64xf32>, %B: memref<64x64xf32>) {
//   affine.for %j = 0 to 64 {
//     affine.for %i = 0 to 64 {
//       %v = affine.load %A[%j, %i] : memref<64x64xf32>
//       %c = arith.constant 1.0 : f32
//       %r = arith.addf %v, %c : f32
//       affine.store %r, %B[%j, %i] : memref<64x64xf32>
//     }
//   }
//   return
// }
