func.func @test(%c0 : i32, %c1 : i32) -> i32 {
  %token0, %t0 = async.execute -> !async.value<i32> {
    async.yield %c0 : i32
  }
  %v0 = async.await %t0 : !async.value<i32>

  %token1, %t1 = async.execute -> !async.value<i32> {
    async.yield %c1 : i32
  }
  %v1 = async.await %t1 : !async.value<i32>

  %sum = arith.addi %v0, %v1 : i32
  return %sum : i32
}