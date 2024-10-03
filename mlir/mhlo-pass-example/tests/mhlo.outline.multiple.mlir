// RUN: outline %s -opt --mlir-disable-threading | FileCheck %s

module {
  func.func @main0(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = mhlo.add %arg0, %arg1 : tensor<2x2xf32>
    %1 = mhlo.exponential %0 : tensor<2x2xf32>
    %2 = mhlo.negate %0 : tensor<2x2xf32>
    %3 = mhlo.exponential %2 : tensor<2x2xf32>
    %4 = mhlo.subtract %1, %3 : tensor<2x2xf32>
    %5 = mhlo.add %1, %3 : tensor<2x2xf32>
    %6 = mhlo.divide %4, %5 : tensor<2x2xf32>
    return %6 : tensor<2x2xf32>
  }
  func.func @main1(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = mhlo.add %arg0, %arg1 : tensor<2x2xf32>
    %1 = mhlo.exponential %0 : tensor<2x2xf32>
    %2 = mhlo.negate %0 : tensor<2x2xf32>
    %3 = mhlo.exponential %2 : tensor<2x2xf32>
    %4 = mhlo.subtract %1, %3 : tensor<2x2xf32>
    %5 = mhlo.add %1, %3 : tensor<2x2xf32>
    %6 = mhlo.divide %4, %5 : tensor<2x2xf32>
    return %6 : tensor<2x2xf32>
  }
}



// CHECK-LABEL: func.func @tanh_function(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK-NEXT:    %0 = mhlo.exponential %arg0 : tensor<2x2xf32>
// CHECK-NEXT:    %1 = mhlo.negate %arg0 : tensor<2x2xf32>
// CHECK-NEXT:    %2 = mhlo.exponential %1 : tensor<2x2xf32>
// CHECK-NEXT:    %3 = mhlo.subtract %0, %2 : tensor<2x2xf32>
// CHECK-NEXT:    %4 = mhlo.add %0, %2 : tensor<2x2xf32>
// CHECK-NEXT:    %5 = mhlo.divide %3, %4 : tensor<2x2xf32>
// CHECK-NEXT:    return %5 : tensor<2x2xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @main0(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK-NEXT:    %0 = mhlo.add %arg0, %arg1 : tensor<2x2xf32>
// CHECK-NEXT:    %1 = call @tanh_function(%0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK-NEXT:    return %1 : tensor<2x2xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @main1(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK-NEXT:    %0 = mhlo.add %arg0, %arg1 : tensor<2x2xf32>
// CHECK-NEXT:    %1 = call @tanh_function(%0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK-NEXT:    return %1 : tensor<2x2xf32>
// CHECK-NEXT:  }
