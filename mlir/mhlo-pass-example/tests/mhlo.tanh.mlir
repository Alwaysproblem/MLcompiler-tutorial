// RUN: tanh %s -opt | FileCheck %s

module {
  func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = "mhlo.add"(%arg0, %arg1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %1 = "mhlo.tanh"(%0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    func.return %1 : tensor<2x2xf32>
  }
}

// CHECK-LABEL: func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK-NEXT:   %0 = mhlo.add %arg0, %arg1 : tensor<2x2xf32>
// CHECK-NEXT:   %1 = mhlo.exponential %0 : tensor<2x2xf32>
// CHECK-NEXT:   %2 = mhlo.negate %0 : tensor<2x2xf32>
// CHECK-NEXT:   %3 = mhlo.exponential %2 : tensor<2x2xf32>
// CHECK-NEXT:   %4 = mhlo.subtract %1, %3 : tensor<2x2xf32>
// CHECK-NEXT:   %5 = mhlo.add %1, %3 : tensor<2x2xf32>
// CHECK-NEXT:   %6 = mhlo.divide %4, %5 : tensor<2x2xf32>
// CHECK-NEXT:   return %6 : tensor<2x2xf32>
// CHECK-NEXT: }
