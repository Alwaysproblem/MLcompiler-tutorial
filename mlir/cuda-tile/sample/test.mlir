module {
  func.func private @cuda_shim_unload_module(i64)
  func.func private @cuda_shim_stream_destroy(i64)
  func.func private @cuda_shim_free(i64, i64)
  func.func private @cuda_shim_stream_synchronize(i64)
  func.func private @cuda_shim_launch_grid_packed(i64, i64, i32, i32, i32, i64, i64, i64, i32)
  func.func private @cuda_shim_memcpy_d2h(i64, i64, i64)
  func.func private @cuda_shim_memcpy_h2d(i64, i64, i64)
  func.func private @cuda_shim_malloc(i64, i64, i1) -> i64
  func.func private @cuda_shim_stream_create() -> i64
  func.func private @cuda_shim_load_module_from_file(i64, i64) -> i64
  func.func private @cuda_debug_dump_float(i64, i32)
  memref.global "private" constant @kname : memref<22xi8> = dense<[111, 117, 116, 108, 105, 110, 101, 100, 95, 103, 112, 117, 95, 107, 101, 114, 110, 101, 108, 95, 48, 0]>
  memref.global "private" constant @cuda_blob : memref<26xi8> = dense<[47, 116, 109, 112, 47, 99, 117, 100, 97, 95, 116, 105, 108, 101, 45, 57, 52, 100, 50, 56, 48, 46, 98, 105, 110, 0]>
  func.func @main() {
    %alloc = memref.alloc() : memref<2x4xf32>
    %alloc_0 = memref.alloc() : memref<2x4xf32>
    %alloc_1 = memref.alloc() : memref<2x4xf32>
    %alloc_2 = memref.alloc() : memref<2x4xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %cst = arith.constant 1.000000e+00 : f32
    affine.store %cst, %alloc_2[%c0, %c0] : memref<2x4xf32>
    %cst_3 = arith.constant 2.000000e+00 : f32
    affine.store %cst_3, %alloc_2[%c0, %c1] : memref<2x4xf32>
    %cst_4 = arith.constant 3.000000e+00 : f32
    affine.store %cst_4, %alloc_2[%c0, %c2] : memref<2x4xf32>
    %cst_5 = arith.constant 9.000000e+00 : f32
    affine.store %cst_5, %alloc_2[%c0, %c3] : memref<2x4xf32>
    %cst_6 = arith.constant 4.000000e+00 : f32
    affine.store %cst_6, %alloc_2[%c1, %c0] : memref<2x4xf32>
    %cst_7 = arith.constant 5.000000e+00 : f32
    affine.store %cst_7, %alloc_2[%c1, %c1] : memref<2x4xf32>
    %cst_8 = arith.constant 6.000000e+00 : f32
    affine.store %cst_8, %alloc_2[%c1, %c2] : memref<2x4xf32>
    %cst_9 = arith.constant 1.000000e+01 : f32
    affine.store %cst_9, %alloc_2[%c1, %c3] : memref<2x4xf32>
    %c0_10 = arith.constant 0 : index
    %c1_11 = arith.constant 1 : index
    %c2_12 = arith.constant 2 : index
    %c3_13 = arith.constant 3 : index
    %cst_14 = arith.constant 1.100000e+01 : f32
    affine.store %cst_14, %alloc_1[%c0_10, %c0_10] : memref<2x4xf32>
    %cst_15 = arith.constant 1.200000e+01 : f32
    affine.store %cst_15, %alloc_1[%c0_10, %c1_11] : memref<2x4xf32>
    %cst_16 = arith.constant 1.300000e+01 : f32
    affine.store %cst_16, %alloc_1[%c0_10, %c2_12] : memref<2x4xf32>
    %cst_17 = arith.constant 1.400000e+01 : f32
    affine.store %cst_17, %alloc_1[%c0_10, %c3_13] : memref<2x4xf32>
    %cst_18 = arith.constant 1.500000e+01 : f32
    affine.store %cst_18, %alloc_1[%c1_11, %c0_10] : memref<2x4xf32>
    %cst_19 = arith.constant 1.600000e+01 : f32
    affine.store %cst_19, %alloc_1[%c1_11, %c1_11] : memref<2x4xf32>
    %cst_20 = arith.constant 1.700000e+01 : f32
    affine.store %cst_20, %alloc_1[%c1_11, %c2_12] : memref<2x4xf32>
    %cst_21 = arith.constant 1.800000e+01 : f32
    affine.store %cst_21, %alloc_1[%c1_11, %c3_13] : memref<2x4xf32>
    %c0_22 = arith.constant 0 : index
    %c1_23 = arith.constant 1 : index
    %c2_24 = arith.constant 2 : index
    %c3_25 = arith.constant 3 : index
    %cst_26 = arith.constant 7.000000e+00 : f32
    affine.store %cst_26, %alloc_0[%c0_22, %c0_22] : memref<2x4xf32>
    %cst_27 = arith.constant 8.000000e+00 : f32
    affine.store %cst_27, %alloc_0[%c0_22, %c1_23] : memref<2x4xf32>
    %cst_28 = arith.constant 9.000000e+00 : f32
    affine.store %cst_28, %alloc_0[%c0_22, %c2_24] : memref<2x4xf32>
    %cst_29 = arith.constant 1.300000e+01 : f32
    affine.store %cst_29, %alloc_0[%c0_22, %c3_25] : memref<2x4xf32>
    %cst_30 = arith.constant 1.000000e+01 : f32
    affine.store %cst_30, %alloc_0[%c1_23, %c0_22] : memref<2x4xf32>
    %cst_31 = arith.constant 1.100000e+01 : f32
    affine.store %cst_31, %alloc_0[%c1_23, %c1_23] : memref<2x4xf32>
    %cst_32 = arith.constant 1.200000e+01 : f32
    affine.store %cst_32, %alloc_0[%c1_23, %c2_24] : memref<2x4xf32>
    %cst_33 = arith.constant 1.400000e+01 : f32
    affine.store %cst_33, %alloc_0[%c1_23, %c3_25] : memref<2x4xf32>
    %0 = memref.get_global @cuda_blob : memref<26xi8>
    %intptr = memref.extract_aligned_pointer_as_index %0 : memref<26xi8> -> index
    %1 = arith.index_cast %intptr : index to i64
    %2 = memref.get_global @kname : memref<22xi8>
    %intptr_34 = memref.extract_aligned_pointer_as_index %2 : memref<22xi8> -> index
    %3 = arith.index_cast %intptr_34 : index to i64
    %c26_i64 = arith.constant 26 : i64
    %4 = call @cuda_shim_load_module_from_file(%1, %c26_i64) : (i64, i64) -> i64
    %5 = call @cuda_shim_stream_create() : () -> i64
    %alloc_35 = memref.alloc() : memref<4xi64>
    %alloc_36 = memref.alloc() : memref<4xi64>
    %c32_i64 = arith.constant 32 : i64
    %false = arith.constant false
    %6 = call @cuda_shim_malloc(%c32_i64, %5, %false) : (i64, i64, i1) -> i64
    %intptr_37 = memref.extract_aligned_pointer_as_index %alloc_2 : memref<2x4xf32> -> index
    %7 = arith.index_cast %intptr_37 : index to i64
    call @cuda_shim_memcpy_h2d(%6, %7, %c32_i64) : (i64, i64, i64) -> ()
    %c0_38 = arith.constant 0 : index
    memref.store %6, %alloc_35[%c0_38] : memref<4xi64>
    %c8_i64 = arith.constant 8 : i64
    memref.store %c8_i64, %alloc_36[%c0_38] : memref<4xi64>
    %c32_i64_39 = arith.constant 32 : i64
    %false_40 = arith.constant false
    %8 = call @cuda_shim_malloc(%c32_i64_39, %5, %false_40) : (i64, i64, i1) -> i64
    %intptr_41 = memref.extract_aligned_pointer_as_index %alloc_0 : memref<2x4xf32> -> index
    %9 = arith.index_cast %intptr_41 : index to i64
    call @cuda_shim_memcpy_h2d(%8, %9, %c32_i64_39) : (i64, i64, i64) -> ()
    %c1_42 = arith.constant 1 : index
    memref.store %8, %alloc_35[%c1_42] : memref<4xi64>
    %c8_i64_43 = arith.constant 8 : i64
    memref.store %c8_i64_43, %alloc_36[%c1_42] : memref<4xi64>
    %c32_i64_44 = arith.constant 32 : i64
    %false_45 = arith.constant false
    %10 = call @cuda_shim_malloc(%c32_i64_44, %5, %false_45) : (i64, i64, i1) -> i64
    %intptr_46 = memref.extract_aligned_pointer_as_index %alloc_1 : memref<2x4xf32> -> index
    %11 = arith.index_cast %intptr_46 : index to i64
    call @cuda_shim_memcpy_h2d(%10, %11, %c32_i64_44) : (i64, i64, i64) -> ()
    %c2_47 = arith.constant 2 : index
    memref.store %10, %alloc_35[%c2_47] : memref<4xi64>
    %c8_i64_48 = arith.constant 8 : i64
    memref.store %c8_i64_48, %alloc_36[%c2_47] : memref<4xi64>
    %c32_i64_49 = arith.constant 32 : i64
    %false_50 = arith.constant false
    %12 = call @cuda_shim_malloc(%c32_i64_49, %5, %false_50) : (i64, i64, i1) -> i64
    %intptr_51 = memref.extract_aligned_pointer_as_index %alloc : memref<2x4xf32> -> index
    %13 = arith.index_cast %intptr_51 : index to i64
    %c3_52 = arith.constant 3 : index
    memref.store %12, %alloc_35[%c3_52] : memref<4xi64>
    %c8_i64_53 = arith.constant 8 : i64
    memref.store %c8_i64_53, %alloc_36[%c3_52] : memref<4xi64>
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1_i32_54 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %intptr_55 = memref.extract_aligned_pointer_as_index %alloc_35 : memref<4xi64> -> index
    %14 = arith.index_cast %intptr_55 : index to i64
    %intptr_56 = memref.extract_aligned_pointer_as_index %alloc_36 : memref<4xi64> -> index
    %15 = arith.index_cast %intptr_56 : index to i64
    call @cuda_shim_launch_grid_packed(%4, %3, %c8_i32, %c1_i32, %c1_i32_54, %5, %14, %15, %c4_i32) : (i64, i64, i32, i32, i32, i64, i64, i64, i32) -> ()
    call @cuda_shim_stream_synchronize(%5) : (i64) -> ()
    call @cuda_shim_memcpy_d2h(%13, %12, %c32_i64_49) : (i64, i64, i64) -> ()
    memref.dealloc %alloc_35 : memref<4xi64>
    memref.dealloc %alloc_36 : memref<4xi64>
    call @cuda_shim_free(%12, %5) : (i64, i64) -> ()
    call @cuda_shim_free(%10, %5) : (i64, i64) -> ()
    call @cuda_shim_free(%8, %5) : (i64, i64) -> ()
    call @cuda_shim_free(%6, %5) : (i64, i64) -> ()
    call @cuda_shim_stream_destroy(%5) : (i64) -> ()
    call @cuda_shim_unload_module(%4) : (i64) -> ()

    // toy.print %alloc : memref<2x4xf32>
    %ci8 = arith.constant 8 : i32
    func.call @cuda_debug_dump_float(%13, %ci8) : (i64, i32) -> ()

    memref.dealloc %alloc_2 : memref<2x4xf32>
    memref.dealloc %alloc_1 : memref<2x4xf32>
    memref.dealloc %alloc_0 : memref<2x4xf32>
    memref.dealloc %alloc : memref<2x4xf32>
    return
  }
}
