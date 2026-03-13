// 依赖 DAG：
//   task0 (a+b)  \
//                 -> task3 (t0+t1) \
//   task1 (c*d)  /                  \
//                                    -> task4 (t2*t3) -> await -> return
//   task2 (e-a) ————————————————————/
//
// 教学点：
// 1) 这里故意把 t0/t1/t2 的 await 写得很早（次优顺序）。
// 2) task3/task4 通过 async.value 显式传参，避免重排后出现支配关系错误。
// 3) 通过一个有副作用的 call 充当 barrier，形成两个调度窗口。
// 4) pass 只能在每个窗口内重排，不能跨越 barrier。

func.func private @barrier_probe(%x : i32)

func.func @complex_reorder(%a : i32, %b : i32, %c : i32,
                           %d : i32, %e : i32) -> i32 {
  // task0: a + b  (独立)
  %token0, %t0 = async.execute -> !async.value<i32> {
    %r = arith.addi %a, %b : i32
    async.yield %r : i32
  }
  // 故意过早 await：会压缩并发窗口
  %v0 = async.await %t0 : !async.value<i32>

  // task1: c * d  (与 task0 独立)
  %token1, %t1 = async.execute -> !async.value<i32> {
    %r = arith.muli %c, %d : i32
    async.yield %r : i32
  }
  // 故意过早 await：会压缩并发窗口
  %v1 = async.await %t1 : !async.value<i32>

  // task2: e - a  (与 task0、task1 独立)
  %token2, %t2 = async.execute -> !async.value<i32> {
    %r = arith.subi %e, %a : i32
    async.yield %r : i32
  }
  // 故意过早 await：会压缩并发窗口
  %v2 = async.await %t2 : !async.value<i32>

  // barrier: side-effecting call，调度器不会把窗口两侧的 op 互相穿越
  func.call @barrier_probe(%v2) : (i32) -> ()

  // task3: await(t0) + await(t1)  (显式依赖 task0, task1)
  %token3, %t3 = async.execute (%t0 as %x0: !async.value<i32>,
                                %t1 as %x1: !async.value<i32>) -> !async.value<i32> {
    %r = arith.addi %x0, %x1 : i32
    async.yield %r : i32
  }

  // task4: await(t2) * await(t3)  (显式依赖 task2, task3)
  %token4, %t4 = async.execute (%t2 as %x2: !async.value<i32>,
                                %t3 as %x3: !async.value<i32>) -> !async.value<i32> {
    %r = arith.muli %x2, %x3 : i32
    async.yield %r : i32
  }
  %v4 = async.await %t4 : !async.value<i32>

  // 使用早期 await 的值，保证它们是语义必需，而非死代码。
  %s01 = arith.addi %v0, %v1 : i32
  %s012 = arith.addi %s01, %v2 : i32
  %out = arith.addi %s012, %v4 : i32

  return %out : i32
}
