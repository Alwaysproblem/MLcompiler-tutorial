module {
  func.func @conv_relu(%input: tensor<1x16x32x32xf32>,
                       %filter: tensor<32x16x3x3xf32>,
                       %init: tensor<1x32x30x30xf32>) -> tensor<1x32x30x30xf32> {
    %0 = linalg.conv_2d_nchw_fchw
      ins(%input, %filter : tensor<1x16x32x32xf32>, tensor<32x16x3x3xf32>)
      outs(%init : tensor<1x32x30x30xf32>) -> tensor<1x32x30x30xf32>

    %cst = arith.constant 0.0 : f32
    %1 = linalg.generic
      {indexing_maps = [
        affine_map<(n, c, h, w) -> (n, c, h, w)>,
        affine_map<(n, c, h, w) -> ()>,
        affine_map<(n, c, h, w) -> (n, c, h, w)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%0, %cst : tensor<1x32x30x30xf32>, f32)
      outs(%init : tensor<1x32x30x30xf32>) {
      ^bb0(%x: f32, %zero: f32, %out: f32):
        %cmp = arith.cmpf oge, %x, %zero : f32
        %sel = arith.select %cmp, %x, %zero : f32
        linalg.yield %sel : f32
      } -> tensor<1x32x30x30xf32>

    return %1 : tensor<1x32x30x30xf32>
  }
}
