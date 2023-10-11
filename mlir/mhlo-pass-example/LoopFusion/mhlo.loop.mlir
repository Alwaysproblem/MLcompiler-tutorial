module {
  func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> (tensor<i64>, tensor<i64>) {
    %init_i = mhlo.constant dense<1> : tensor<i64>
    %init_sum = mhlo.constant dense<1> : tensor<i64>
    %init_sum_plus_1 = mhlo.constant dense<0> : tensor<i64>
    %one = mhlo.constant dense<1> : tensor<i64>
    %two = mhlo.constant dense<2> : tensor<i64>
    %ten = mhlo.constant dense<10> : tensor<i64>

    // sum += 2
    %result:2 = mhlo.while(%iterArg = %init_i, %iterArg_0 = %init_sum) : tensor<i64>, tensor<i64>
     cond {
      %cond = mhlo.compare  LT, %init_i, %ten : (tensor<i64>, tensor<i64>) -> tensor<i1>
      mhlo.return %cond : tensor<i1>
    } do {
      %new_sum = mhlo.add %iterArg_0, %two : tensor<i64>
      %new_i = mhlo.add %iterArg, %one : tensor<i64>
      mhlo.return %new_i, %new_sum : tensor<i64>, tensor<i64>
    }

    // sum += 1
    %result_plus_1:2 = mhlo.while(%iterArg = %init_i, %iterArg_0 = %init_sum_plus_1) : tensor<i64>, tensor<i64>
     cond {
      %cond = mhlo.compare  LT, %init_i, %ten : (tensor<i64>, tensor<i64>) -> tensor<i1>
      mhlo.return %cond : tensor<i1>
    } do {
      %new_sum = mhlo.add %iterArg_0, %one : tensor<i64>
      %new_i = mhlo.add %iterArg, %one : tensor<i64>
      mhlo.return %new_i, %new_sum : tensor<i64>, tensor<i64>
    }

    return %result#1, %result_plus_1#1 : tensor<i64>, tensor<i64>
  }
}

// after conversion:
// module {
//   func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> (tensor<i64>, tensor<i64>) {
//     %init_i = mhlo.constant dense<1> : tensor<i64>
//     %init_sum = mhlo.constant dense<1> : tensor<i64>
//     %init_sum_plus_1 = mhlo.constant dense<0> : tensor<i64>
//     %one = mhlo.constant dense<1> : tensor<i64>
//     %two = mhlo.constant dense<2> : tensor<i64>
//     %ten = mhlo.constant dense<10> : tensor<i64>

//     // sum += 2
//     %result:3 = mhlo.while(%iterArg = %init_i, %iterArg_0 = %init_sum, %iterArg_1 = %init_sum_plus_1) : tensor<i64>, tensor<i64>, tensor<i64>
//      cond {
//       %cond = mhlo.compare  LT, %init_i, %ten : (tensor<i64>, tensor<i64>) -> tensor<i1>
//       mhlo.return %cond : tensor<i1>
//     } do {
//       %new_sum = mhlo.add %iterArg_0, %two : tensor<i64>
//       %new_sum_plus_1 = mhlo.add %iterArg_1, %one : tensor<i64>
//       %new_i = mhlo.add %iterArg, %one : tensor<i64>
//       mhlo.return %new_i, %new_sum, %new_sum_plus_1 : tensor<i64>, tensor<i64>, tensor<i64>
//     }
//     return %result#1, %result#2 : tensor<i64>, tensor<i64>
//   }
// }


// ---- mhlo while Op example -----
// module {
//   func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<i64> {
//     %init_i = "mhlo.constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
//     %init_sum = "mhlo.constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
//     %one = "mhlo.constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
//     %two = "mhlo.constant"() {value = dense<2> : tensor<i64>} : () -> tensor<i64>
//     %ten = "mhlo.constant"() {value = dense<10> : tensor<i64>} : () -> tensor<i64>

//     %result_cnt, %result_sum = "mhlo.while"(%init_i, %init_sum) ({
//       ^bb0(%a0: tensor<i64>, %a1: tensor<i64>):
//         %cond = "mhlo.compare"(%init_i, %ten) {
//           comparison_direction = #mhlo<comparison_direction LT>
//         } : (tensor<i64>, tensor<i64>) -> tensor<i1>
//         mhlo.return %cond : tensor<i1>
//       }, {
//       ^bb0(%a0: tensor<i64>, %a1: tensor<i64>):
//         %new_sum = mhlo.add %a1, %two : tensor<i64>
//         %new_i = mhlo.add %init_i, %one : tensor<i64>
//         mhlo.return %new_i, %new_sum : tensor<i64>, tensor<i64>
//     }) : (tensor<i64>, tensor<i64>) -> (tensor<i64>, tensor<i64>)

//     func.return %result_sum : tensor<i64>
//   }
// }
