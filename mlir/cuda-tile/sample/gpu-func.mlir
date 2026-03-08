module {
  // --- CUDA shim externs (ABI: all pointers/handles are i64) ---
  func.func private @cuda_shim_load_module_from_image(i64, i64) -> i64
  func.func private @cuda_shim_load_module_from_file(i64, i64) -> i64
  func.func private @cuda_shim_unload_module(i64) -> ()
  func.func private @cuda_shim_stream_create() -> i64
  func.func private @cuda_shim_stream_destroy(i64) -> ()
  func.func private @cuda_shim_stream_synchronize(i64) -> ()
  func.func private @cuda_shim_malloc(i64, i64, i1) -> i64
  func.func private @cuda_shim_free(i64, i64) -> ()
  func.func private @cuda_shim_memcpy_h2d(i64, i64, i64) -> ()
  func.func private @cuda_shim_memcpy_d2h(i64, i64, i64) -> ()
  func.func private @cuda_shim_launch_packed(
      i64, i64,
      i32, i32, i32,
      i32, i32, i32,
      i32,
      i64,
      i64, i64,
      i32) -> ()
  func.func private @cuda_debug_dump_float(i64, i32) -> ()

  // // --- GPU blob embedded (placeholder bytes for "cuda_tile.cubin") ---
  // memref.global "private" constant @cuda_blob : memref<16xi8> = dense<
  //   [99, 117, 100, 97, 95, 116, 105, 108, 101, 46, 99, 117, 98, 105, 110, 0]
  // > : memref<16xi8>

  // // --- Kernel name as a C string (NUL-terminated) ---
  // // 注意：如果 driver 侧用 name 查找函数，这个字符串必须以 0 结尾。
  // memref.global "private" constant @kname : memref<22xi8> = dense<[
  //   111,117,116,108,105,110,101,100,95,103,112,117,95,107,101,114,110,101,108,95,48,0
  // ]> : memref<22xi8>

  memref.global "private" constant @cuda_blob : memref<16xi8> =
  dense<"0x637564615f74696c652e637562696e00">

  memref.global "private" constant @kname : memref<22xi8> =
    dense<"0x6f75746c696e65645f6770755f6b65726e656c5f3000">

  func.func @main() {
    // ---------- Host buffers (after bufferization) ----------
    %hA = memref.alloc() : memref<2x4xf32>
    %hB = memref.alloc() : memref<2x4xf32>
    %hOut = memref.alloc() : memref<2x4xf32>

    // Fill constants (为了示例直接用 store 展开；真实 pipeline 通常会从 memref.global copy)
    // A
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index

    %cf1 = arith.constant 1.0 : f32
    %cf2 = arith.constant 2.0 : f32
    %cf3 = arith.constant 3.0 : f32
    %cf9 = arith.constant 9.0 : f32

    %cf4 = arith.constant 4.0 : f32
    %cf5 = arith.constant 5.0 : f32
    %cf6 = arith.constant 6.0 : f32
    %cf10 = arith.constant 10.0 : f32

    %cf11 = arith.constant 11.0 : f32
    %cf12 = arith.constant 12.0 : f32
    %cf13 = arith.constant 13.0 : f32
    %cf14 = arith.constant 14.0 : f32
    %cf15 = arith.constant 15.0 : f32
    %cf16 = arith.constant 16.0 : f32
    %cf17 = arith.constant 17.0 : f32
    %cf18 = arith.constant 18.0 : f32

    // row0
    memref.store %cf1, %hA[%c0, %c0] : memref<2x4xf32>
    memref.store %cf2, %hA[%c0, %c1] : memref<2x4xf32>
    memref.store %cf3, %hA[%c0, %c2] : memref<2x4xf32>
    memref.store %cf9, %hA[%c0, %c3] : memref<2x4xf32>

    // row1
    memref.store %cf4, %hA[%c1, %c0] : memref<2x4xf32>
    memref.store %cf5, %hA[%c1, %c1] : memref<2x4xf32>
    memref.store %cf6, %hA[%c1, %c2] : memref<2x4xf32>
    memref.store %cf10, %hA[%c1, %c3] : memref<2x4xf32>

    // B = %1 in your original (这里假设 %1 是第二个输入；你原 op 里是 (%0, %2, %1)，请按你 kernel 的真实语义对齐)
    memref.store %cf11, %hB[%c0, %c0] : memref<2x4xf32>
    memref.store %cf12, %hB[%c0, %c1] : memref<2x4xf32>
    memref.store %cf13, %hB[%c0, %c2] : memref<2x4xf32>
    memref.store %cf14, %hB[%c0, %c3] : memref<2x4xf32>
    memref.store %cf15, %hB[%c1, %c0] : memref<2x4xf32>
    memref.store %cf16, %hB[%c1, %c1] : memref<2x4xf32>
    memref.store %cf17, %hB[%c1, %c2] : memref<2x4xf32>
    memref.store %cf18, %hB[%c1, %c3] : memref<2x4xf32>

    // ---------- Load module ----------
    %blob = memref.get_global @cuda_blob : memref<16xi8>
    %blob_ptr_idx = memref.extract_aligned_pointer_as_index %blob : memref<16xi8>  -> index
    %blob_ptr_i64 = arith.index_cast %blob_ptr_idx : index to i64
    %blobSize = arith.constant 16 : i64
    %mod = func.call @cuda_shim_load_module_from_file(%blob_ptr_i64, %blobSize) : (i64, i64) -> i64

    // kernel name pointer
    %kn = memref.get_global @kname : memref<22xi8>
    %kname_ptr_idx = memref.extract_aligned_pointer_as_index %kn : memref<22xi8>  -> index
    %kname_ptr_i64 = arith.index_cast %kname_ptr_idx : index to i64

    // ---------- Stream + device alloc ----------
    %stream = func.call @cuda_shim_stream_create() : () -> i64
    %isHostShared = arith.constant 0 : i1

    %nElems = arith.constant 8 : i32
    %nbytes = arith.constant 32 : i64  // 2*4*f32 = 8 * 4 = 32 bytes

    %dA = func.call @cuda_shim_malloc(%nbytes, %stream, %isHostShared) : (i64, i64, i1) -> i64
    %dB = func.call @cuda_shim_malloc(%nbytes, %stream, %isHostShared) : (i64, i64, i1) -> i64
    %dOut = func.call @cuda_shim_malloc(%nbytes, %stream, %isHostShared) : (i64, i64, i1) -> i64

    // host ptrs (as i64)
    %hAptr = memref.extract_aligned_pointer_as_index %hA : memref<2x4xf32>  -> index
    %hBptr = memref.extract_aligned_pointer_as_index %hB : memref<2x4xf32>  -> index
    %hOutptr = memref.extract_aligned_pointer_as_index %hOut : memref<2x4xf32>  -> index

    // host memrefs -> i64
    %hA_ptr_i64 = arith.index_cast %hAptr : index to i64
    %hB_ptr_i64 = arith.index_cast %hBptr : index to i64
    %hOut_ptr_i64 = arith.index_cast %hOutptr : index to i64

    func.call @cuda_shim_memcpy_h2d(%dA, %hA_ptr_i64, %nbytes) : (i64, i64, i64) -> ()
    func.call @cuda_shim_memcpy_h2d(%dB, %hB_ptr_i64, %nbytes) : (i64, i64, i64) -> ()

    // ---------- Build argSlots / argSizes (方案 A) ----------
    // 这里 num_args=4： (A, B, Out, N)
    // 注意：参数顺序必须和 @outlined_gpu_kernel_0 的 PTX param_0.. 一致
    %numArgs = arith.constant 4 : index
    %argSlots = memref.alloc() : memref<4xi64>
    %argSizes = memref.alloc() : memref<4xi64>
    %c8 = arith.constant 8 : i64
    %ci4 = arith.constant 4 : i64

    // num_args = 4
    // i=0 a0
    memref.store %c8, %argSizes[%c0] : memref<4xi64>
    memref.store %dA, %argSlots[%c0] : memref<4xi64>

    // i=1 a1
    memref.store %c8, %argSizes[%c1] : memref<4xi64>
    memref.store %dB, %argSlots[%c1] : memref<4xi64>

    // i=2 a2   (你需要一个 dC，对应第三个输入)
    memref.store %c8, %argSizes[%c2] : memref<4xi64>
    memref.store %dB, %argSlots[%c2] : memref<4xi64>

    // i=3 out
    memref.store %c8, %argSizes[%c3] : memref<4xi64>
    memref.store %dOut, %argSlots[%c3] : memref<4xi64>

    // pointers to argSlots/argSizes (as i64)
    %argSlotsptr = memref.extract_aligned_pointer_as_index %argSlots : memref<4xi64>  -> index
    %argSlots_ptr_i64 = arith.index_cast %argSlotsptr : index to i64
    %argSizesptr = memref.extract_aligned_pointer_as_index %argSizes : memref<4xi64>  -> index
    %argSizes_ptr_i64 = arith.index_cast %argSizesptr : index to i64

    // ---------- Launch ----------
    %gridX = arith.constant 1 : i32
    %gridY = arith.constant 1 : i32
    %gridZ = arith.constant 1 : i32
    %blockX = arith.constant 8 : i32
    %blockY = arith.constant 1 : i32
    %blockZ = arith.constant 1 : i32
    %shmem = arith.constant 0 : i32
    %numArgsI32 = arith.constant 4 : i32

    func.call @cuda_shim_launch_packed(
      %mod, %kname_ptr_i64,
      %gridX, %gridY, %gridZ,
      %blockX, %blockY, %blockZ,
      %shmem, %stream,
      %argSlots_ptr_i64, %argSizes_ptr_i64, %numArgsI32
    ) : (i64, i64, i32, i32, i32, i32, i32, i32, i32, i64, i64, i64, i32) -> ()

    func.call @cuda_shim_stream_synchronize(%stream) : (i64) -> ()
    func.call @cuda_shim_memcpy_d2h(%hOut_ptr_i64, %dOut, %nbytes) : (i64, i64, i64) -> ()

    %ci8 = arith.constant 8 : i32
    func.call @cuda_debug_dump_float(%hOut_ptr_i64, %ci8) : (i64, i32) -> ()

    // ---------- Cleanup ----------
    func.call @cuda_shim_free(%dOut, %stream) : (i64, i64) -> ()
    func.call @cuda_shim_free(%dA, %stream) : (i64, i64) -> ()
    func.call @cuda_shim_free(%dB, %stream) : (i64, i64) -> ()
    func.call @cuda_shim_stream_destroy(%stream) : (i64) -> ()
    func.call @cuda_shim_unload_module(%mod) : (i64) -> ()

    return
  }
}
