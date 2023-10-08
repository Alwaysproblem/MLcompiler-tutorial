module {
  func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = "mhlo.log"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %1 = "mhlo.log"(%arg1) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %2 = "mhlo.add"(%0, %1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %3 = "mhlo.exponential"(%2) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    func.return %3 : tensor<2x2xf32>
  }
}

// after conversion:
// module {
//   func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
//     %0 = mhlo.multiply %arg0, %arg1 : tensor<2x2xf32>
//     return %0 : tensor<2x2xf32>
//   }
// }
