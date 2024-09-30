// RUN: mlir-opt %s --transform-interpreter |\
// RUN: FileCheck %s

// Original function to optimize.
func.func @reduction(%arg0: tensor<512x512xf32>, %bias: tensor<512xf32>, %out: tensor<512xf32>)
                   -> tensor<512xf32> {
  %bias_init = tensor.empty() : tensor<512x512xf32>
  %biased = linalg.broadcast ins(%bias: tensor<512xf32>)
        outs(%bias_init: tensor<512x512xf32>)
        dimensions = [1]

  %red = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0)>],
  iterator_types = ["parallel", "reduction"]}
  ins(%arg0, %biased : tensor<512x512xf32>, tensor<512x512xf32>)
  outs(%out : tensor<512xf32>) {
    ^bb0(%arg7: f32, %arg8: f32, %arg9: f32):
    %1 = arith.addf %arg7, %arg8 : f32
    %2 = arith.addf %1, %arg9 : f32
    linalg.yield %2 : f32
  } -> tensor<512xf32>
  return %red : tensor<512xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op) {
    %bias = transform.structured.match ops {["linalg.broadcast"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %red = transform.structured.match ops {["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op

    %body, %loop = transform.structured.tile_using_forall %red
                  tile_sizes [4] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)


    %body_with_bias, %loop_2 = transform.structured.fuse_into_containing_op %bias into %loop : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op)

    %red_fill, %conv4, %combining, %rz_ry_rx
    = transform.structured.tile_reduction_using_for %body by
      tile_sizes=[0, 1]
      : (!transform.any_op)
      -> (!transform.any_op, !transform.any_op, !transform.any_op,
          !transform.any_op)

    transform.structured.generalize %body_with_bias
      : (!transform.any_op) -> !transform.any_op

    %f00 = transform.structured.match ops{["func.func"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f00 {
    } : !transform.any_op

    transform.yield
  }
}
