toy.gpu_func @my_kernel(%arg0: tensor<2x3xf32>, %arg1: tensor<3x2xf32>) -> tensor<2x2xf32> {
  %2 = toy.matmul(%arg0 : tensor<2x3xf32>, %arg1 : tensor<3x2xf32>) to tensor<2x2xf32>
  toy.return %2 : tensor<2x2xf32>
}

toy.func @main() {
  %1 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf32>
  %3 = toy.constant dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00], [5.000000e+00, 6.000000e+00]]> : tensor<3x2xf32>
  %4 = toy.launch_gpu @my_kernel(%1, %3) {grid = [16, 16, 1]}
        : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
  toy.print %4 : tensor<2x2xf32>
  toy.return
}