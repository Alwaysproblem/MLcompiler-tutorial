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

    %matmul, %el1, %el2 = transform.collect_matching @match_matmul_elemwise in %root
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    %elemwise = transform.merge_handles %el1, %el2 : !transform.any_op

    transform.include @print_elemwise failures(propagate)  (%elemwise)
      : (!transform.any_op) -> ()
    transform.include @print_matmul failures(propagate)  (%matmul)
      : (!transform.any_op) -> ()

    transform.yield
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

  // This is also a matcher sequence. It is similarly given an operation to
  // match and nested operations must succeed in order for a match to be deemed
  // successful. It starts matching from the last operation in the use-def chain
  // and goes back because each operand (use) has exactly one definition.
  transform.named_sequence @match_matmul_elemwise(
      %last: !transform.any_op {transform.readonly})
      -> (!transform.any_op, !transform.any_op, !transform.any_op) {
    // The last operation must be an elementwise binary.
    transform.match.operation_name %last ["linalg.elemwise_binary"]
      : !transform.any_op
    // Its first operand must be defined by another operation, to which we
    // will get a handle here. We are guaranteed that the first operand exists
    // because we know the operation is binary, but even in absence of such a
    // guarantee, this operation would have produced a silenceable failure when
    // `%last` does not have enough operands.
    %middle = transform.get_producer_of_operand %last[0]
      : (!transform.any_op) -> !transform.any_op
    // The defining operation must itself be an elementwise binary.
    transform.match.operation_name %middle ["linalg.elemwise_binary"]
      : !transform.any_op
    // And the first operand of that operation must be defined by yet another
    // operation.
    %matmul = transform.get_producer_of_operand %middle[0]
      : (!transform.any_op) -> !transform.any_op
    // And that operation is a matmul.
    transform.match.operation_name %matmul ["linalg.matmul"] : !transform.any_op
    // We will yield the handles to the matmul and the two elementwise
    // operations separately.
    transform.yield %matmul, %middle, %last
      : !transform.any_op, !transform.any_op, !transform.any_op
  }
}
