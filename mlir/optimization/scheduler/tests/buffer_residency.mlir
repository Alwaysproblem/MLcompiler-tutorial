func.func @ex1(%A: memref<128x128xf32>) {
  %buf_fast = memref.alloc() : memref<128x128xf32, 1>

  %generic = memref.memory_space_cast %buf_fast
    : memref<128x128xf32, 1> to memref<128x128xf32>

  %tile = memref.subview %generic[0, 0][64, 64][1, 1]
    : memref<128x128xf32>
    to memref<64x64xf32, strided<[128, 1], offset: 0>>

  return
}

// arg: ddr
// op result @memref.alloc : fastmem
// op result @memref.memory_space_cast : fastmem
// op result @memref.subview : fastmem

func.func @ex4(%A: memref<128x128xf32>,
                %B: memref<128x128xf32>,
                %C: memref<128x128xf32>) {
  %bufA = memref.alloc() : memref<128x128xf32, 1>
  %bufB = memref.alloc() : memref<128x128xf32, 1>

  %a = memref.memory_space_cast %bufA
    : memref<128x128xf32, 1> to memref<128x128xf32>
  %b = memref.subview %bufB[0, 0][128, 128][1, 1]
    : memref<128x128xf32, 1>
    to memref<128x128xf32, strided<[128, 1], offset: 0>, 1>

  linalg.matmul
    ins(%a, %b : memref<128x128xf32>,
                  memref<128x128xf32, strided<[128, 1], offset: 0>, 1>)
    outs(%C : memref<128x128xf32>)
  return
}

// arg: ddr
// arg: ddr
// arg: ddr
// op result @memref.alloc : fastmem
// op result @memref.alloc : fastmem
// op result @memref.memory_space_cast : fastmem
// op result @memref.subview : fastmem
// linalg op: linalg.matmul
//   ins:
//     value=%memspacecast = memref.memory_space_cast %alloc : memref<128x128xf32, 1> to memref<128x128xf32> type=memref<128x128xf32> memory_space=default residency=fastmem defined_by=memref.memory_space_cast
//     value=%subview = memref.subview %alloc_0[0, 0] [128, 128] [1, 1] : memref<128x128xf32, 1> to memref<128x128xf32, strided<[128, 1]>, 1> type=memref<128x128xf32, strided<[128, 1]>, 1> memory_space=1 residency=fastmem defined_by=memref.subview
//   outs:
//     value=<block argument> of type 'memref<128x128xf32>' at index: 2 type=memref<128x128xf32> memory_space=default residency=ddr
