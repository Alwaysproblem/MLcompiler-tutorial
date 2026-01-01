; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

declare ptr @mgpuMemAlloc(i64, ptr, i1)

declare void @mgpuMemFree(ptr, ptr)

declare void @mgpuMemcpyHtoD(ptr, ptr, i64)

declare void @mgpuMemcpyDtoH(ptr, ptr, i64)

declare void @mgpuCtxSynchronize()

define void @main() {
  %1 = call ptr @malloc(i64 24)
  %2 = getelementptr float, ptr %1, i64 0
  store float 1.000000e+00, ptr %2, align 4
  %3 = getelementptr float, ptr %1, i64 1
  store float 2.000000e+00, ptr %3, align 4
  %4 = getelementptr float, ptr %1, i64 2
  store float 3.000000e+00, ptr %4, align 4
  %5 = getelementptr float, ptr %1, i64 3
  store float 4.000000e+00, ptr %5, align 4
  %6 = getelementptr float, ptr %1, i64 4
  store float 5.000000e+00, ptr %6, align 4
  %7 = getelementptr float, ptr %1, i64 5
  store float 6.000000e+00, ptr %7, align 4
  %8 = call ptr @mgpuMemAlloc(i64 24, ptr null, i1 false)
  call void @mgpuMemcpyHtoD(ptr %8, ptr %1, i64 24)
  call void @mgpuCtxSynchronize()
  call void @mgpuMemcpyDtoH(ptr %1, ptr %8, i64 24)
  call void @mgpuCtxSynchronize()
  call void @mgpuMemFree(ptr %8, ptr null)
  call void @free(ptr %1)
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
