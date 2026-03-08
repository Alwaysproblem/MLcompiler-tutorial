module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf32>
    %1 = toy.constant dense<[[1.100000e+01, 1.200000e+01, 1.300000e+01], [1.400000e+01, 1.500000e+01, 1.600000e+01]]> : tensor<2x3xf32>
    %2 = toy.launch_gpu @outlined_gpu_kernel_0(%1, %0) {grid = array<i64: 4, 2, 1>} : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x2xf32>
    toy.print %2 : tensor<2x2xf32>
    %3 = toy.constant dense<[[7.000000e+00, 8.000000e+00, 9.000000e+00], [1.000000e+01, 1.100000e+01, 1.200000e+01]]> : tensor<2x3xf32>
    %4 = toy.launch_gpu @outlined_gpu_kernel_1(%0, %3, %1) {grid = array<i64: 4, 2, 1>} : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    toy.print %4 : tensor<2x3xf32>
    toy.return
  }
  toy.gpu_func @outlined_gpu_kernel_0(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x2xf32> {
    %0 = toy.transpose(%arg0 : tensor<2x3xf32>) to tensor<3x2xf32>
    %1 = toy.matmul(%arg1 : tensor<2x3xf32>, %0 : tensor<3x2xf32>) to tensor<2x2xf32>
    toy.return %1 : tensor<2x2xf32>
  }
  toy.gpu_func @outlined_gpu_kernel_1(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>, %arg2: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %0 = toy.mul %arg0, %arg1 : tensor<2x3xf32>
    %1 = toy.add %0, %arg2 : tensor<2x3xf32>
    toy.return %1 : tensor<2x3xf32>
  }
}
