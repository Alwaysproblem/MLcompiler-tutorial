module {
  func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = "mhlo.add"(%arg0, %arg1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %1 = mhlo.constant dense<2.0> : tensor<2x2xf32>
    %2 = "mhlo.power"(%0, %1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    func.return %2 : tensor<2x2xf32>
  }
}

// after conversion:
// module {
//   func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
//     %0 = "mhlo.add"(%arg0, %arg1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
//     %2 = "mhlo.multiply"(%0, %0) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
//     func.return %2 : tensor<2x2xf32>
//   }
// }
