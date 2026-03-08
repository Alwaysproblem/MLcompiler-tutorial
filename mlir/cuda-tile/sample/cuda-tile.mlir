module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00, 9.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00, 1.000000e+01]]> : tensor<2x4xf32>
    %1 = toy.constant dense<[[1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01], [1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01]]> : tensor<2x4xf32>
    %2 = toy.constant dense<[[7.000000e+00, 8.000000e+00, 9.000000e+00, 1.300000e+01], [1.000000e+01, 1.100000e+01, 1.200000e+01, 1.400000e+01]]> : tensor<2x4xf32>
    %3 = toy.launch_gpu @outlined_gpu_kernel_0(%0, %2, %1) {grid = array<i64: 1, 1, 1>} : (tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    toy.print %3 : tensor<2x4xf32>
    toy.return
  }
  cuda_tile.module @cuda_tile_module {
    entry @outlined_gpu_kernel_0(%arg0: tile<ptr<f32>>, %arg1: tile<ptr<f32>>, %arg2: tile<ptr<f32>>, %arg3: tile<ptr<f32>>) {
      %tview = make_tensor_view %arg0, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xf32, strides=[4,1]>
      %pview = make_partition_view %tview : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>
      %cst_0_i32 = constant <i32: 0> : tile<i32>
      %tile, %result_token = load_view_tko weak %pview[%cst_0_i32, %cst_0_i32] : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>, tile<i32> -> tile<2x4xf32>, token
      %tview_0 = make_tensor_view %arg1, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xf32, strides=[4,1]>
      %pview_1 = make_partition_view %tview_0 : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>
      %tile_2, %result_token_3 = load_view_tko weak %pview_1[%cst_0_i32, %cst_0_i32] : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>, tile<i32> -> tile<2x4xf32>, token
      %0 = mulf %tile, %tile_2  : tile<2x4xf32>
      %tview_4 = make_tensor_view %arg2, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xf32, strides=[4,1]>
      %pview_5 = make_partition_view %tview_4 : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>
      %tile_6, %result_token_7 = load_view_tko weak %pview_5[%cst_0_i32, %cst_0_i32] : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>, tile<i32> -> tile<2x4xf32>, token
      %tile_8, %result_token_9 = load_view_tko weak %pview_1[%cst_0_i32, %cst_0_i32] : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>, tile<i32> -> tile<2x4xf32>, token
      %1 = mulf %tile_6, %tile_8  : tile<2x4xf32>
      %2 = addf %0, %1  : tile<2x4xf32>
      %tview_10 = make_tensor_view %arg3, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xf32, strides=[4,1]>
      %pview_11 = make_partition_view %tview_10 : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>
      %3 = store_view_tko weak %2, %pview_11[%cst_0_i32, %cst_0_i32] : tile<2x4xf32>, partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>, tile<i32> -> token
      return
    }
  }
}
