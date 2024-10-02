// RUN: pass-tutor-opt %s --pow-2 | FileCheck %s

module {
  func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = "mhlo.add"(%arg0, %arg1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %1 = mhlo.constant dense<2.0> : tensor<2x2xf32>
    %2 = "mhlo.power"(%0, %1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    func.return %2 : tensor<2x2xf32>
  }
}


// CHECK-LABEL: func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK-NEXT:   %[[ADD:.*]] = mhlo.add %arg0, %arg1 : tensor<2x2xf32>
// CHECK-NEXT:   %[[MUL:.*]] = mhlo.multiply %[[ADD]], %[[ADD]] : tensor<2x2xf32>
// CHECK-NEXT:   return %[[MUL]] : tensor<2x2xf32>
// CHECK-NEXT: }
