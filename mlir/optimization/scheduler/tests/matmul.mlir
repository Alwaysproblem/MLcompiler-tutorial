module {
  func.func @matmul(%A: tensor<128x256xf32>,
                    %B: tensor<256x128xf32>,
                    %C: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %0 = linalg.matmul
      ins(%A, %B : tensor<128x256xf32>, tensor<256x128xf32>)
      outs(%C : tensor<128x128xf32>) -> tensor<128x128xf32>
    return %0 : tensor<128x128xf32>
  }
}
