// RUN: mlir-opt %s --transform-interpreter |\
// RUN: FileCheck %s

func.func @fc_relu(%lhs: tensor<512x512xf32>, %rhs: tensor<512x512xf32>,
                   %bias: tensor<512x512xf32>, %output: tensor<512x512xf32>)
                   -> tensor<512x512xf32> {
  // Matrix-matrix multiplication.
  %matmul = linalg.matmul ins(%lhs, %rhs: tensor<512x512xf32>, tensor<512x512xf32>)
                          outs(%output: tensor<512x512xf32>) -> tensor<512x512xf32>

  // Elementwise addition.
  %biased = linalg.elemwise_binary { fun = #linalg.binary_fn<add> }
    ins(%matmul, %bias : tensor<512x512xf32>, tensor<512x512xf32>)
    outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>

  // Elementwise max with 0 (ReLU).
  %c0f = arith.constant 0.0 : f32
  %relued = linalg.elemwise_binary { fun = #linalg.binary_fn<max_signed> }
    ins(%biased, %c0f : tensor<512x512xf32>, f32)
    outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>
  func.return %relued : tensor<512x512xf32>
}

// The module containing named sequences must have an attribute allowing them
// to enable verification.
module @transforms attributes { transform.with_named_sequence } {
  // Entry point. This takes as the only argument the root operation (typically
  // pass root) given to the transform interpreter.
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    // Collect operations that match the criteria specified in named sequence.
    // If the named sequence fails with a silenceable failure, silences it (the
    // message is forwarded to the debug stream). If the named sequence
    // succeeds, appends its results to the results of this operation.
    %elemwise = transform.collect_matching @match_elemwise in %root
      : (!transform.any_op) -> !transform.any_op
    %matmul = transform.collect_matching @match_matmul in %root
      : (!transform.any_op) -> !transform.any_op
    transform.include @print_elemwise failures(propagate)  (%elemwise)
      : (!transform.any_op) -> ()
    transform.include @print_matmul failures(propagate)  (%matmul)
      : (!transform.any_op) -> ()

    transform.yield
  }

  // This is a matcher sequence. It is given an operation to match and the
  // match is considered successful unless any nested operation produces a
  // failure. The values yielded by this operation will be forwarded to the
  // rewriter sequence on success.
  transform.named_sequence @match_elemwise(
      %entry: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %entry ["linalg.elemwise_binary"]
      : !transform.any_op
    transform.yield %entry : !transform.any_op
  }

  transform.named_sequence @match_matmul(
      %entry: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %entry ["linalg.matmul"] : !transform.any_op
    transform.yield %entry : !transform.any_op
  }

  // This is a rewriter sequence.
  transform.named_sequence @print_elemwise(
      %elemwise_binary: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at
      %elemwise_binary, "elementwise binary" : !transform.any_op
    transform.yield
  }

  transform.named_sequence @print_matmul(
      %matmul: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %matmul, "matmul" : !transform.any_op
    transform.yield
  }
}
