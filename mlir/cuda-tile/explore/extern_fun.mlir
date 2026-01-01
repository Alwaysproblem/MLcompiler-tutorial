module {
  // libc
  func.func private @malloc(i64) -> memref<*xi8>
  func.func private @free(memref<*xi8>)

  // 轻量包装：仅用整数/布尔/opaque memref，避免 llvm.ptr 类型
  func.func private @shimMemAlloc(i64) -> i64
  func.func private @shimMemFree(i64)
  func.func private @shimMemcpyHtoD(i64, memref<6xf32>)
  func.func private @shimMemcpyDtoH(memref<6xf32>, i64)
  func.func private @shimCtxSynchronize()

  func.func @main() {
    %size_bytes = arith.constant 24 : i64                 // 6 * sizeof(f32)
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %f1 = arith.constant 1.0 : f32
    %f2 = arith.constant 2.0 : f32
    %f3 = arith.constant 3.0 : f32
    %f4 = arith.constant 4.0 : f32
    %f5 = arith.constant 5.0 : f32
    %f6 = arith.constant 6.0 : f32

    // host buffer as memref
    %h = memref.alloc() : memref<6xf32>
    memref.store %f1, %h[%c0] : memref<6xf32>
    memref.store %f2, %h[%c1] : memref<6xf32>
    memref.store %f3, %h[%c2] : memref<6xf32>
    memref.store %f4, %h[%c3] : memref<6xf32>
    memref.store %f5, %h[%c4] : memref<6xf32>
    memref.store %f6, %h[%c5] : memref<6xf32>

    // device alloc handle (as i64 pointer-sized integer)
    %d = func.call @shimMemAlloc(%size_bytes) : (i64) -> i64

    func.call @shimMemcpyHtoD(%d, %h) : (i64, memref<6xf32>) -> ()
    func.call @shimCtxSynchronize() : () -> ()
    func.call @shimMemcpyDtoH(%h, %d) : (memref<6xf32>, i64) -> ()
    func.call @shimCtxSynchronize() : () -> ()

    func.call @shimMemFree(%d) : (i64) -> ()
    memref.dealloc %h : memref<6xf32>
    func.return
  }
}

// module {
//   // libc
//   llvm.func @malloc(i64) -> !llvm.ptr
//   llvm.func @free(!llvm.ptr)

//   // cuda_shim C 接口（来自 cuda_shim.cpp）
//   llvm.func @mgpuMemAlloc(i64, !llvm.ptr, i1) -> !llvm.ptr
//   llvm.func @mgpuMemFree(!llvm.ptr, !llvm.ptr)
//   llvm.func @mgpuMemcpyHtoD(!llvm.ptr, !llvm.ptr, i64)
//   llvm.func @mgpuMemcpyDtoH(!llvm.ptr, !llvm.ptr, i64)
//   llvm.func @mgpuCtxSynchronize()

//   llvm.func @main() {
//     %size = llvm.mlir.constant(24 : i64) : i64          // 6 * sizeof(f32)
//     %zero_ptr = llvm.mlir.zero : !llvm.ptr              // 空 stream
//     %false = llvm.mlir.constant(false) : i1

//     // host buffer
//     %h = llvm.call @malloc(%size) : (i64) -> !llvm.ptr

//     // 写入 1..6 到 host
//     %c0 = llvm.mlir.constant(0 : index) : i64
//     %c1 = llvm.mlir.constant(1 : index) : i64
//     %c2 = llvm.mlir.constant(2 : index) : i64
//     %c3 = llvm.mlir.constant(3 : index) : i64
//     %c4 = llvm.mlir.constant(4 : index) : i64
//     %c5 = llvm.mlir.constant(5 : index) : i64
//     %f1 = llvm.mlir.constant(1.0 : f32) : f32
//     %f2 = llvm.mlir.constant(2.0 : f32) : f32
//     %f3 = llvm.mlir.constant(3.0 : f32) : f32
//     %f4 = llvm.mlir.constant(4.0 : f32) : f32
//     %f5 = llvm.mlir.constant(5.0 : f32) : f32
//     %f6 = llvm.mlir.constant(6.0 : f32) : f32

//     %p0 = llvm.getelementptr %h[%c0] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//     llvm.store %f1, %p0 : f32, !llvm.ptr
//     %p1 = llvm.getelementptr %h[%c1] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//     llvm.store %f2, %p1 : f32, !llvm.ptr
//     %p2 = llvm.getelementptr %h[%c2] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//     llvm.store %f3, %p2 : f32, !llvm.ptr
//     %p3 = llvm.getelementptr %h[%c3] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//     llvm.store %f4, %p3 : f32, !llvm.ptr
//     %p4 = llvm.getelementptr %h[%c4] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//     llvm.store %f5, %p4 : f32, !llvm.ptr
//     %p5 = llvm.getelementptr %h[%c5] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//     llvm.store %f6, %p5 : f32, !llvm.ptr

//     // device alloc (isHostShared = false)
//     %d = llvm.call @mgpuMemAlloc(%size, %zero_ptr, %false)
//         : (i64, !llvm.ptr, i1) -> !llvm.ptr

//     // HtoD then DtoH (round-trip)
//     llvm.call @mgpuMemcpyHtoD(%d, %h, %size) : (!llvm.ptr, !llvm.ptr, i64) -> ()
//     llvm.call @mgpuCtxSynchronize() : () -> ()
//     llvm.call @mgpuMemcpyDtoH(%h, %d, %size) : (!llvm.ptr, !llvm.ptr, i64) -> ()
//     llvm.call @mgpuCtxSynchronize() : () -> ()

//     // free
//     llvm.call @mgpuMemFree(%d, %zero_ptr) : (!llvm.ptr, !llvm.ptr) -> ()
//     llvm.call @free(%h) : (!llvm.ptr) -> ()
//     llvm.return
//   }
// }