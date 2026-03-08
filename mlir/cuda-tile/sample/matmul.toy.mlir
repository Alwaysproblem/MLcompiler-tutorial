toy.func private @matmul_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
  %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
  %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64>
  %2 = toy.matmul(%0 : tensor<*xf64>, %1 : tensor<*xf64>) to tensor<*xf64>
  toy.return %2 : tensor<*xf64>
}

toy.func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64>
  %2 = toy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>
  %3 = toy.reshape(%2 : tensor<6xf64>) to tensor<3x2xf64>
  %4 = toy.generic_call @matmul_transpose(%1, %3) : (tensor<2x3xf64>, tensor<3x2xf64>) -> tensor<*xf64>
  toy.print %4 : tensor<*xf64>
  toy.return
}
