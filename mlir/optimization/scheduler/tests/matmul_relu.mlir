module {
  func.func @matmul(%A: tensor<128x256xf32>,
                    %B: tensor<256x128xf32>,
                    %C: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = linalg.matmul
      ins(%A, %B : tensor<128x256xf32>, tensor<256x128xf32>)
      outs(%C : tensor<128x128xf32>) -> tensor<128x128xf32>

    %1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%0 : tensor<128x128xf32>) outs(%C : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.maxnumf %in, %cst : f32
      linalg.yield %2 : f32
    } -> tensor<128x128xf32>
    return %1 : tensor<128x128xf32>
  }
}
