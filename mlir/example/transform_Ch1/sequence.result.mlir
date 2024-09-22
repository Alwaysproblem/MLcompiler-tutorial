#map = affine_map<(d0) -> (d0 * 8)>
#map1 = affine_map<(d0) -> (d0 * 32)>
#map2 = affine_map<(d0) -> (d0 * 4)>
module {
  func.func @microkernel(%arg0: tensor<4x512xf32>, %arg1: tensor<512x4xf32>, %arg2: tensor<4x4xf32>, %arg3: tensor<4x4xf32>, %arg4: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = scf.forall (%arg5) in (4) shared_outs(%arg6 = %arg4) -> (tensor<4x4xf32>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg5, 0] [1, 512] [1, 1] : tensor<4x512xf32> to tensor<1x512xf32>
      %extracted_slice_0 = tensor.extract_slice %arg2[%arg5, 0] [1, 4] [1, 1] : tensor<4x4xf32> to tensor<1x4xf32>
      %1 = linalg.matmul ins(%extracted_slice, %arg1 : tensor<1x512xf32>, tensor<512x4xf32>) outs(%extracted_slice_0 : tensor<1x4xf32>) -> tensor<1x4xf32>
      %extracted_slice_1 = tensor.extract_slice %arg3[%arg5, 0] [1, 4] [1, 1] : tensor<4x4xf32> to tensor<1x4xf32>
      %extracted_slice_2 = tensor.extract_slice %arg6[%arg5, 0] [1, 4] [1, 1] : tensor<4x4xf32> to tensor<1x4xf32>
      %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %extracted_slice_1 : tensor<1x4xf32>, tensor<1x4xf32>) outs(%extracted_slice_2 : tensor<1x4xf32>) -> tensor<1x4xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %2 into %arg6[%arg5, 0] [1, 4] [1, 1] : tensor<1x4xf32> into tensor<4x4xf32>
      }
    }
    return %0 : tensor<4x4xf32>
  }
  func.func @fc_relu(%arg0: tensor<512x512xf32>, %arg1: tensor<512x512xf32>, %arg2: tensor<512x512xf32>, %arg3: tensor<512x512xf32>) -> tensor<512x512xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = scf.forall (%arg4, %arg5) in (64, 16) shared_outs(%arg6 = %arg3) -> (tensor<512x512xf32>) {
      %1 = affine.apply #map(%arg4)
      %2 = affine.apply #map1(%arg5)
      %extracted_slice = tensor.extract_slice %arg0[%1, 0] [8, 512] [1, 1] : tensor<512x512xf32> to tensor<8x512xf32>
      %extracted_slice_0 = tensor.extract_slice %arg1[0, %2] [512, 32] [1, 1] : tensor<512x512xf32> to tensor<512x32xf32>
      %extracted_slice_1 = tensor.extract_slice %arg3[%1, %2] [8, 32] [1, 1] : tensor<512x512xf32> to tensor<8x32xf32>
      %extracted_slice_2 = tensor.extract_slice %arg2[%1, %2] [8, 32] [1, 1] : tensor<512x512xf32> to tensor<8x32xf32>
      %3 = scf.forall (%arg7, %arg8) in (2, 8) shared_outs(%arg9 = %extracted_slice_1) -> (tensor<8x32xf32>) {
        %5 = affine.apply #map2(%arg7)
        %6 = affine.apply #map2(%arg8)
        %extracted_slice_4 = tensor.extract_slice %extracted_slice[%5, 0] [4, 512] [1, 1] : tensor<8x512xf32> to tensor<4x512xf32>
        %extracted_slice_5 = tensor.extract_slice %extracted_slice_0[0, %6] [512, 4] [1, 1] : tensor<512x32xf32> to tensor<512x4xf32>
        %extracted_slice_6 = tensor.extract_slice %extracted_slice_1[%5, %6] [4, 4] [1, 1] : tensor<8x32xf32> to tensor<4x4xf32>
        %extracted_slice_7 = tensor.extract_slice %extracted_slice_2[%5, %6] [4, 4] [1, 1] : tensor<8x32xf32> to tensor<4x4xf32>
        %extracted_slice_8 = tensor.extract_slice %arg9[%5, %6] [4, 4] [1, 1] : tensor<8x32xf32> to tensor<4x4xf32>
        %7 = func.call @microkernel(%extracted_slice_4, %extracted_slice_5, %extracted_slice_6, %extracted_slice_7, %extracted_slice_8) : (tensor<4x512xf32>, tensor<512x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %7 into %arg9[%5, %6] [4, 4] [1, 1] : tensor<4x4xf32> into tensor<8x32xf32>
        }
      }
      %extracted_slice_3 = tensor.extract_slice %arg6[%1, %2] [8, 32] [1, 1] : tensor<512x512xf32> to tensor<8x32xf32>
      %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins(%3, %cst : tensor<8x32xf32>, f32) outs(%extracted_slice_3 : tensor<8x32xf32>) -> tensor<8x32xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %4 into %arg6[%1, %2] [8, 32] [1, 1] : tensor<8x32xf32> into tensor<512x512xf32>
      }
    }
    return %0 : tensor<512x512xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op, %arg1: !transform.op<"linalg.matmul">, %arg2: !transform.op<"linalg.elemwise_binary">) {
      %0:2 = transform.split_handle %arg2 : (!transform.op<"linalg.elemwise_binary">) -> (!transform.any_op, !transform.any_op)
      %tiled_op, %forall_op = transform.structured.tile_using_forall %0#1 tile_sizes [8, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %0#0 into %forall_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      %fused_op_0, %new_containing_op_1 = transform.structured.fuse_into_containing_op %arg1 into %new_containing_op : (!transform.op<"linalg.matmul">, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      %tiled_op_2, %forall_op_3 = transform.structured.tile_using_forall %fused_op tile_sizes [4, 4] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      %fused_op_4, %new_containing_op_5 = transform.structured.fuse_into_containing_op %fused_op_0 into %forall_op_3 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      %tiled_op_6, %forall_op_7 = transform.structured.tile_using_forall %tiled_op_2 tile_sizes [1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      %fused_op_8, %new_containing_op_9 = transform.structured.fuse_into_containing_op %fused_op_4 into %forall_op_7 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      %function, %call = transform.loop.outline %forall_op_7 {func_name = "microkernel"} : (!transform.any_op) -> (!transform.any_op, !transform.op<"func.call">)
      transform.yield
    }
  }
}
