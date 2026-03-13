module {
  func.func @multi_use(%A: tensor<64x64xf32>,
                       %B: tensor<64x64xf32>,
                       %C: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>) {
    %0 = linalg.matmul
      ins(%A, %B : tensor<64x64xf32>, tensor<64x64xf32>)
      outs(%C : tensor<64x64xf32>) -> tensor<64x64xf32>

    %init0 = tensor.empty() : tensor<64x64xf32>
    %init1 = tensor.empty() : tensor<64x64xf32>
    %c0 = arith.constant 0.0 : f32

    %1 = linalg.generic
      {indexing_maps = [
         affine_map<(i, j) -> (i, j)>,
         affine_map<(i, j) -> ()>,
         affine_map<(i, j) -> (i, j)>
       ],
       iterator_types = ["parallel", "parallel"]}
      ins(%0, %c0 : tensor<64x64xf32>, f32)
      outs(%init0 : tensor<64x64xf32>) {
      ^bb0(%x: f32, %zero: f32, %out: f32):
        %cmp = arith.cmpf oge, %x, %zero : f32
        %sel = arith.select %cmp, %x, %zero : f32
        linalg.yield %sel : f32
      } -> tensor<64x64xf32>

    %2 = linalg.generic
      {indexing_maps = [
         affine_map<(i, j) -> (i, j)>,
         affine_map<(i, j) -> ()>,
         affine_map<(i, j) -> (i, j)>
       ],
       iterator_types = ["parallel", "parallel"]}
      ins(%0, %c0 : tensor<64x64xf32>, f32)
      outs(%init1 : tensor<64x64xf32>) {
      ^bb0(%x: f32, %zero: f32, %out: f32):
        %cmp = arith.cmpf ogt, %x, %zero : f32
        %sel = arith.select %cmp, %x, %zero : f32
        linalg.yield %sel : f32
      } -> tensor<64x64xf32>

    return %1, %2 : tensor<64x64xf32>, tensor<64x64xf32>
  }
}