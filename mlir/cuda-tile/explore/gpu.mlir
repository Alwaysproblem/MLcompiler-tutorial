// module attributes {gpu.container_module} {
//   // ---- Device side (GPU) ----
//   gpu.module @kernels {
//     gpu.func @kernel(%n : index, %A : memref<?xf32, 1>, %B : memref<?xf32, 1>)
//         attributes { gpu.kernel } {
//       %tid = gpu.thread_id x
//       %pred = arith.cmpi slt, %tid, %n : index
//       scf.if %pred {
//         %a = memref.load %A[%tid] : memref<?xf32, 1>
//         memref.store %a, %B[%tid] : memref<?xf32, 1>
//       }
//       gpu.return
//     }
//   }

//   // ---- Host side (CPU) ----
//   func.func @main(%n : index, %hA : memref<?xf32>, %hB : memref<?xf32>) {
//     %dA = gpu.alloc(%n) : memref<?xf32, 1>
//     %dB = gpu.alloc(%n) : memref<?xf32, 1>
//     gpu.memcpy %dA, %hA : memref<?xf32, 1>, memref<?xf32>
//     // launch kernel（blocks/threads 这里先写死成 1D）
//     %c1 = arith.constant 1 : index
//     gpu.launch_func @kernels::@kernel
//       blocks in (%c1, %c1, %c1) threads in (%n, %c1, %c1)
//       args(%n : index, %dA : memref<?xf32, 1>, %dB : memref<?xf32, 1>)

//     gpu.memcpy %hB, %dB : memref<?xf32>, memref<?xf32, 1>
//     gpu.dealloc %dA : memref<?xf32, 1>
//     gpu.dealloc %dB : memref<?xf32, 1>
//     return
//   }
// }

module attributes {gpu.container_module} {

  gpu.module @kernels {
    gpu.func @kernel(%n : index, %A : memref<6xf32, 1>, %B : memref<6xf32, 1>)
        attributes { gpu.kernel } {
      %tid = gpu.thread_id x
      %pred = arith.cmpi slt, %tid, %n : index
      scf.if %pred {
        %a = memref.load %A[%tid] : memref<6xf32, 1>
        %b = arith.addf %a, %a : f32
        memref.store %b, %B[%tid] : memref<6xf32, 1>
      }
      gpu.return
    }
  }

  func.func @main() {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 6.000000e+00 : f32
    %cst_0 = arith.constant 5.000000e+00 : f32
    %cst_1 = arith.constant 4.000000e+00 : f32
    %cst_2 = arith.constant 3.000000e+00 : f32
    %cst_3 = arith.constant 2.000000e+00 : f32
    %cst_4 = arith.constant 1.000000e+00 : f32

    %0 = memref.alloc() : memref<6xf32>
    %1 = memref.alloc() : memref<6xf32>

    affine.store %cst_4, %1[0] : memref<6xf32>
    affine.store %cst_3, %1[1] : memref<6xf32>
    affine.store %cst_2, %1[2] : memref<6xf32>
    affine.store %cst_1, %1[3] : memref<6xf32>
    affine.store %cst_0, %1[4] : memref<6xf32>
    affine.store %cst,   %1[5] : memref<6xf32>

    %n = arith.constant 2 : index

    %dA = gpu.alloc() : memref<6xf32, 1>
    %dB = gpu.alloc() : memref<6xf32, 1>
    gpu.memcpy %dA, %1 : memref<6xf32, 1>, memref<6xf32>

    // launch kernel（blocks/threads 这里先写死成 1D）
    gpu.launch_func @kernels::@kernel
      blocks in (%c1, %c1, %c1) threads in (%n, %c1, %c1)
      args(%n : index, %dA : memref<6xf32, 1>, %dB : memref<6xf32, 1>)

    gpu.memcpy %0, %dB : memref<6xf32>, memref<6xf32, 1>
    gpu.dealloc %dA : memref<6xf32, 1>
    gpu.dealloc %dB : memref<6xf32, 1>
    memref.dealloc %1 : memref<6xf32>
    memref.dealloc %0 : memref<6xf32>
    return
  }
}
// func.func @main() {
//     %c2 = arith.constant 2 : index
//     %c1 = arith.constant 1 : index
//     gpu.launch
//         blocks(%0, %1, %2) in (%3 = %c1, %4 = %c1, %5 = %c1)
//         threads(%6, %7, %8) in (%9 = %c2, %10 = %c1, %11 = %c1) {
//         gpu.printf "Hello from %d\n", %6 : index
//         gpu.terminator
//     }
//     return
// }