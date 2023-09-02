# Pytorch

## Build from source

```bash
# gcc-10, bazel
conda create -n tf2-build python=3.10 requests numpy wheel build -c conda-forge  -y

DEBUG=1 USE_DISTRIBUTED=0 USE_MKLDNN=0 USE_CUDA=0 BUILD_TEST=0 USE_FBGEMM=0 USE_NNPACK=0 USE_QNNPACK=0 USE_XNNPACK=0 python setup.py develop # CPU debug version pytorch whl.
```

[build from source](https://github.com/openxla/xla/blob/main/docs/build_from_source.md)

## Env for debugging

```shell
cp -r vscode/pytorch/vscode/ pytorch/.vscode/
```

## Pytorch Jit trace

```txt

trace (standalone function) -> ScriptFunction
# python: torch.jit.trace
# cpp: _create_function_from_trace
{
  # cpp: toTraceableStack # this will change the input into c10::IValue
  # cpp: tracer::createGraphByTracing
  {
    # cpp: tracer::trace ->
    {
      # build graph
      # trace the decorated funcion with `trace_fn`
      # and get local and global env variable with
      # inspect python package.
    }
  }
  # cpp: cu->create_function -> GraphFuncion
  # registed the `function ptr` and `compilation unit`
}


run (ScriptFunction) -> torch.Tensor
# python: traced_foo
# python: __call__
# pytbind11-cpp: ScriptFunction.__call__
# cpp: invokeScriptFunctionFromPython
# cpp: runAndInsertCall
# cpp: GraphFunction::run(Stack& stack)
{
  # cpp: get_executor -> graphexecutor
  {
    # cpp: optimized_graph
    {
      # cpp: preoptimizeGraph
      ...
      torch/jit/api/graph ...
      ...
    }
  }
}

{
  # cpp: run
  # cpp: GraphExecutorImplBase::run
  {
    # cpp: getPlanFor(stack)
    # cpp: InterpreterState(plan.code).run(stack);
    {
      # cpp: InterpreterStateImpl::run
      {
        # cpp: InterpreterStateImpl::runImpl(stack) ## Finite State Machine for run the python stack.
        # for every instruction from the frame intialized with `plan.code`
        #   the stack will push the input or arguments
        #   and then change the status to `OP`
        #   and find the op in the `frame.function->operator_table_[inst.X]`
        #   and call that op with input stack.
        #   after that, store the output into the stack for next instruction.
      }
    }
  }
}

```

## pytorch log level

- pass name, `>>` means `Graph_UPDATE` log level

```shell
PYTORCH_JIT_LOG_LEVEL='>>dead_code_elimination' python pytorch-test/aa.py
```

## debug trick

```gdb
# debug Stack
# -exec call ((c10::IValue *)<>)->dump()
-exec call ((c10::IValue *)stack._M_impl._M_start)->dump()
-exec call graph->dump()
```

## pytorch script

```txt
# python: torch.jit.script
{
  # python: _check_directly_compile_overloaded
  # python: _try_get_jit_cached_function
  # python: get_jit_def
  {
    # find the source code from file
    # parse the source code with python ast
    # python: build_def
    {
      # python: build_param_list
      {
        # substitute the arguments with `torch._C._jit_tree_views.Param`
      }
      # substitute the python ast to _jit_tree_view
    }
    #** after `build_def`, the ast wille be like this.
    #** (def
    #**   (ident f)
    #**   (decl
    #**     (list
    #**       (param
    #**         (ident a)
    #**         (option)
    #**         (option)
    #**         (False))
    #**       (param
    #**         (ident b)
    #**         (option)
    #**         (option)
    #**         (False)))
    #**     (option))
    #**   (list
    #**     (assign
    #**       (list (variable (ident c)))
    #**       (option
    #**         (+
    #**           (variable (ident a))
    #**           (variable (ident b))))
    #**       (option))
    #**     (assign
    #**       (list (variable (ident d)))
    #**       (option
    #**         (*
    #**           (variable (ident c))
    #**           (variable (ident c))))
    #**       (option))
    #**     (assign
    #**       (list (variable (ident e)))
    #**       (option
    #**         (apply
    #**           (.
    #**             (variable (ident torch))
    #**             (ident tanh))
    #**           (list
    #**             (*
    #**               (variable (ident d))
    #**               (variable (ident c))))
    #**           (list)))
    #**       (option))
    #**     (return
    #**       (+
    #**         (variable (ident d))
    #**         (+
    #**           (variable (ident e))
    #**           (variable (ident e)))))))
    # python: torch._C._jit_script_compile
    # cpp: pybind11::_jit_script_compile, loc `script_init.cpp`
    {
      # cpp: script_compile_function -> GraphExecutor
      {
        # this part analyse the property, here we will not describe the detail
        #
        # this is for-loop to define the python ast object
        # cpp: for:
        {
          # cpp: CompilationUnit::define -> GraphFunction (traversing all the graph and operator by DFS.)
          {
            # return a GraphFunction but not convert to torch IR
            # in `ensure_define` call the `creater` to build torch IR
            # cpp: for
            {
              # cpp: CompilationUnit::define -> GraphFunction (recursive)
            }
          }
          # record the function ptr into a function table
        }

        # cpp: call `ensure_defined` for all function in the function table
        #      (each op will be a single graph or function).
        {
          # here will call the function creator
          # the `to_ir` function will be called in the creator.
          #
          # cpp: struct to_ir and build for every type and property.
          {
            # 1. push start frame and create enviornment
            # 2. set the method schema (here will recursive call the lower structure.)
            # 3. `ReplaceOldOperatorsWithUpgraders`, `ConvertToSSA`, `CanonicalizeModifiedLoops`
            #    `NormalizeOps` and `runCleanupPasses` passes.
          }
        }
      }
    }
  }
}

{
  # cpp: run
  # cpp: GraphExecutorImplBase::run
  {
    # cpp: getPlanFor(stack)
    {
      # cpp: getOptimizedPlanFor(stack, remaining_bailout_depth){
        # cpp: runProfilingInsensitiveOptimizations
        {
          Inline(*graph);
          ClearProfilingInformation(graph);
          LowerGradOf(*graph);
          ClearUndefinedness(graph);
          RemoveExpands(graph);
          CanonicalizeOps(graph);
          EliminateDeadCode(graph);
          DecomposeOps(graph); # this will process `layer_norm` and `addmm` (DFS recuresive check)
          ConstantPropagation(graph);
          EliminateDeadCode(graph);
          EliminateCommonSubexpression(graph);
          ConstantPooling(graph);
          PeepholeOptimize(graph);
          EliminateDeadCode(graph);
          LowerSimpleTuples(graph);
          CheckInplace(graph);
        }

        # after the getOptimization plan
        # ---------- original graph ---------
        #-- graph(%a.1 : Tensor,
        #--       %b.1 : Tensor):
        #--   %2 : int = prim::Constant[value=1]()
        #--   %c.1 : Tensor = aten::add(%a.1, %b.1, %2) # /root/Desktop/dockerVolumn/MLcompiler-tutorial/torch/pytorch/pytorch-test/aa.py:9:6
        #--   %d.1 : Tensor = aten::mul(%c.1, %c.1) # /root/Desktop/dockerVolumn/MLcompiler-tutorial/torch/pytorch/pytorch-test/aa.py:10:6
        #--   return (%d.1)
        # ---------- profiling graph ---------
        #-- graph(%a.1 : Tensor,
        #--       %b.1 : Tensor):
        #--   %2 : int = prim::Constant[value=1]()
        #--   %5 : Tensor = prim::profile[profiled_type=Tensor, seen_none=0](%a.1)
        #--   %6 : Tensor = prim::profile[profiled_type=Tensor, seen_none=0](%b.1)
        #--   %c.1 : Tensor = aten::add(%5, %6, %2) # /root/Desktop/dockerVolumn/MLcompiler-tutorial/torch/pytorch/pytorch-test/aa.py:9:6
        #--   %7 : Tensor = prim::profile[profiled_type=Tensor, seen_none=0](%c.1)
        #--   %8 : Tensor = prim::profile[profiled_type=Tensor, seen_none=0](%c.1)
        #--   %d.1 : Tensor = aten::mul(%7, %8) # /root/Desktop/dockerVolumn/MLcompiler-tutorial/torch/pytorch/pytorch-test/aa.py:10:6
        #--   %9 : Tensor = prim::profile[profiled_type=Tensor, seen_none=0](%d.1)
        #--    = prim::profile()
        #--   return (%9)
      }
    }
    # cpp: InterpreterState(plan.code).run(stack);
    {
      # cpp: InterpreterStateImpl::run
      {
        # cpp: InterpreterStateImpl::runImpl(stack) ## Finite State Machine for run the python stack.
        # for every instruction from the frame intialized with `plan.code`
        #   the stack will push the input or arguments
        #   and then change the status to `OP`
        #   and find the op in the `frame.function->operator_table_[inst.X]`
        #   and call that op with input stack.
        #   after that, store the output into the stack for next instruction.
      }
    }
  }
}

```

## pytorch dynamo (version > 2.0)

The dynamo compiler configuration

```python
{'debug': False, 'disable_progress': True, 'verbose_progress': False, 'cpp_wrapper': False, 'dce': False, 'static_weight_shapes': True, 'size_asserts': True, 'pick_loop_orders': True, 'inplace_buffers': True, 'allow_buffer_reuse': True, 'benchmark_harness': True, 'epilogue_fusion': True, 'epilogue_fusion_first': False, 'pattern_matcher': True, 'split_cat_fx_passes': True, 'group_fusion': False, 'batch_fusion': True, 'reordering': True, 'use_mixed_mm': False, 'force_mixed_mm': False, 'aot_inductor_output_path': '', 'max_autotune': False, 'max_autotune_pointwise': False, 'max_autotune_gemm': False, 'max_autotune_gemm_backends': 'ATEN,TRITON', 'search_autotune_cache': False, 'save_args': False, 'autotune_in_subproc': False, 'coordinate_descent_tuning': False, 'coordinate_descent_check_all_directions': False, 'coordinate_descent_search_radius': 1, 'layout_optimization': True, 'keep_output_stride': True, 'warn_mix_layout': False, 'realize_reads_threshold': 4, 'realize_bytes_threshold': 2000, 'realize_acc_reads_threshold': 8, 'fallback_random': False, 'implicit_fallbacks': True, 'aggressive_fusion': False, 'max_fusion_size': 64, 'unroll_reductions_threshold': 8, 'comment_origin': False, 'conv_1x1_as_mm': False, 'split_reductions': True, 'benchmark_kernel': False, 'constant_and_index_propagation': True, 'joint_graph_constant_folding': True, 'debug_index_asserts': False, 'is_nightly_or_source': True, 'developer_warnings': True, 'compile_threads': 32, 'global_cache_dir': None, 'kernel_name_max_ops': 10, 'shape_padding': True, 'permute_fusion': False, 'profiler_mark_wrapper_call': False, 'generate_intermediate_hooks': False, 'debug_ir_traceback': False, '_raise_error_for_testing': False, '_profile_var': '', 'profile_bandwidth': False, 'profile_bandwidth_regex': '', 'disable_cpp_codegen': False, 'freezing': False, 'freezing_discard_parameters': False, 'cpp.threads': -1, 'cpp.no_redundant_loops': True, 'cpp.dynamic_threads': False, 'cpp.simdlen': None, 'cpp.min_chunk_size': 4096, 'cpp.cxx': (None, 'g++'), 'cpp.enable_kernel_profile': False, 'cpp.weight_prepack': True, 'cpp.inject_relu_bug_TESTING_ONLY': None, 'cpp.inject_log1p_bug_TESTING_ONLY': None, 'cpp.vec_isa_ok': None, 'cpp.descriptive_names': 'original_aten', 'cpp.max_horizontal_fusion_size': 16, 'triton.cudagraphs': False, 'triton.cudagraph_trees': True, 'triton.slow_path_cudagraph_asserts': True, 'triton.cudagraph_trees_history_recording': False, 'triton.fast_path_cudagraph_asserts': False, 'triton.skip_cudagraph_warmup': False, 'triton.debug_sync_graph': False, 'triton.debug_sync_kernel': False, 'triton.dense_indexing': False, 'triton.max_tiles': 2, 'triton.autotune_pointwise': True, 'triton.autotune_cublasLt': True, 'triton.tiling_prevents_pointwise_fusion': True, 'triton.tiling_prevents_reduction_fusion': True, 'triton.assert_indirect_indexing': True, 'triton.unique_kernel_names': False, 'triton.descriptive_names': 'original_aten', 'triton.persistent_reductions': True, 'triton.divisible_by_16': True, 'triton.max_block': {'X': 2048, 'Y': 1024, 'Z': 1024}, 'triton.store_cubin': False, 'triton.spill_threshold': 16, 'triton.inject_relu_bug_TESTING_ONLY': None, 'trace.enabled': False, 'trace.debug_log': False, 'trace.info_log': False, 'trace.fx_graph': True, 'trace.fx_graph_transformed': True, 'trace.ir_pre_fusion': True, 'trace.ir_post_fusion': True, 'trace.output_code': True, 'trace.graph_diagram': False, 'trace.compile_profile': False, 'trace.upload_tar': None, '_save_config_ignore': {'trace.upload_tar'}}
```

When tracing the python code, the input will be set `FakeTensor(..., size=[4, 4], requires_grad=True)` and building the `torch.fx.GraphModule`.

The `torch.compile` registed some graph converter callback. when first runing, the graph will be call those callback to process the unexpected format and call `make_fx` to do the conversion of `torch.fx.GraphModule`. During this, if input tensor has `require_grad=True`, `make_fx` also call the `torch.autograd.grad` to build a tangents backward graph for the input tensor (the `torch.autograd.grad` will call the cpp function to compute). after buiding the `torch.fx.GraphModule`, it also generates the python definition code.

after `aot_dispatch_autograd_graph` called, the `fx_g` will be like (the forward graph is `aten.mm(m1, m2)`):

```python
class joint_helper(torch.nn.Module):
    def forward(self, primals, tangents):
        primals_1: f32[4, 4], primals_2: f32[4, 4], tangents_1: f32[4, 4], = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
        # No stacktrace found for following nodes
        mm: f32[4, 4] = torch.ops.aten.mm.default(primals_1, primals_2)
        permute: f32[4, 4] = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        mm_1: f32[4, 4] = torch.ops.aten.mm.default(permute, tangents_1);  permute = None
        permute_1: f32[4, 4] = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
        mm_2: f32[4, 4] = torch.ops.aten.mm.default(tangents_1, permute_1);  tangents_1 = permute_1 = None
        return pytree.tree_unflatten([mm, mm_2, mm_1], self._out_spec)
```

**Note that: the mm (aten.mm) and sfdp (attention compute unit) will be registed or compile in the `lazy_init` function, this is the same as `jit.script` and `jit.trace`.**



compile_fx_inner
  fx_codegen_and_compile
    GraphLowering().run()
      # substitute the input argument by TensorBox
      # substitute op by detail inplementation decomp_fn
      _register_lowering
    compiled_fn = graph.compile_to_fn()
      graph.compile_to_module().call

python source code:

```
# torch_graph
x = torch.FloatTensor([1, 2, 3])
y = torch.FloatTensor([4, 5, 6])
x + y
```

`torch.fx.GraphModule`:

```
class GraphModule(torch.nn.Module):
    def forward(self, L_a_ : torch.Tensor, L_b_ : torch.Tensor):
        l_a_ = L_a_
        l_b_ = L_b_

        # File: /root/Desktop/dockerVolumn/MLcompiler-tutorial/torch/pytorch/pytorch-test/dynami.py:9, code: return a + b
        add = l_a_ + l_b_;  l_a_ = l_b_ = None
        return (add,)
```

lowering to:

```
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: f32[3], arg1_1: f32[3]):
        # File: /root/Desktop/dockerVolumn/MLcompiler-tutorial/torch/pytorch/pytorch-test/dynami.py:9, code: return a + b
        add: f32[3] = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        return (add,)
```

IR:

```
graph():
    %arg0_1 : [num_users=1] = placeholder[target=arg0_1]
    %arg1_1 : [num_users=1] = placeholder[target=arg1_1]
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, %arg1_1), kwargs = {})
    return (add,)
```

after `Interpreter.run_node()`:

```
arg0_1 = TensorBox(StorageBox(
  InputBuffer(name='arg0_1', layout=FixedLayout('cpu', torch.float32, size=[3], stride=[1]))
))

arg1_1 = TensorBox(StorageBox(
  InputBuffer(name='arg1_1', layout=FixedLayout('cpu', torch.float32, size=[3], stride=[1]))
))

add = TensorBox(StorageBox(
  ComputedBuffer(name='buf0', layout=FixedLayout('cpu', torch.float32, size=[3], stride=[1]), data=Pointwise(
    'cpu',
    torch.float32,
    def inner_fn(index):
        i0 = index
        tmp0 = ops.load(arg0_1, i0)
        tmp1 = ops.load(arg1_1, i0)
        tmp2 = tmp0 + tmp1
        return tmp2
    ,
    ranges=[3],
    origin_node=add,
    origins={add}
  ))
))
```

### codegen

It is really code generation inside of the llvm-like codegen for Cpp codegen of pytorch, the model template can be found in the source code in `WrapperCodeGen` class.

for `WrapperCodeGen` `write_header` funcion:

```python
    def write_header(self):
        self.header.splice(
            f"""
                from ctypes import c_void_p, c_long
                import torch
                import math
                import random
                import os
                import tempfile
                from math import inf, nan
                from torch._inductor.hooks import run_intermediate_hooks
                from torch._inductor.utils import maybe_profile

                from torch import empty_strided, as_strided, device
                from {codecache.__name__} import AsyncCompile
                from torch._inductor.select_algorithm import extern_kernels

                aten = torch.ops.aten
                assert_size_stride = torch._C._dynamo.guards.assert_size_stride
                async_compile = AsyncCompile()

            """
        )
```

For example code, the code generateive result will be like:

```python
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile

from torch import empty_strided, as_strided, device
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompile()


cpp_fused_add_0 = async_compile.cpp('''
#include "/tmp/torchinductor_root/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        #pragma omp simd simdlen(4)
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(3L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = in_ptr0[static_cast<long>(i0)];
            auto tmp1 = in_ptr1[static_cast<long>(i0)];
            auto tmp2 = tmp0 + tmp1;
            out_ptr0[static_cast<long>(i0)] = tmp2;
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (3, ), (1, ))
    assert_size_stride(arg1_1, (3, ), (1, ))
    buf0 = empty_strided((3, ), (1, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_0(c_void_p(arg0_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(buf0.data_ptr()))
    del arg0_1
    del arg1_1
    return (buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((3, ), (1, ), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((3, ), (1, ), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
```

the scheduler will call the `generate` method to generate the cpu code.

Here is the cpp header file in code example above:

```cpp
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <omp.h>

#include <ATen/NumericUtils.h>
#include <ATen/core/PhiloxRNGEngine.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/Math.h>

#include <c10/util/BFloat16.h>
#include <c10/util/BFloat16-math.h>
#include <c10/util/Half.h>

#if defined(CPU_CAPABILITY_AVX512) || defined(CPU_CAPABILITY_AVX2)
#define INDUCTOR_USE_VECTOR_TYPES() 1
#else
#define INDUCTOR_USE_VECTOR_TYPES() 0
#endif

#if INDUCTOR_USE_VECTOR_TYPES()
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#endif

typedef at::Half half;
typedef at::BFloat16 bfloat16;

template <typename T>
struct Welford {
  T mean = T(0);
  T m2 = T(0);
  T weight = T(0);
};


template <typename T>
struct IsVecType: std::false_type {};

#if INDUCTOR_USE_VECTOR_TYPES()
template <typename T>
struct IsVecType<at::vec::Vectorized<T>>: std::true_type {};
#endif

template <typename T>
Welford<T> welford_combine(const Welford<T> &a, const Welford<T> &b) {
  if constexpr (!IsVecType<T>::value) {
    if (a.weight == 0) {
      return b;
    }
    if (b.weight == 0) {
      return a;
    }
  }
  auto delta = b.mean - a.mean;
  auto new_weight = a.weight + b.weight;
  auto wb_over_w = b.weight / new_weight;
  if constexpr (IsVecType<T>::value) {
    // Guard against division by zero
    wb_over_w = T::blendv(wb_over_w, T(0), new_weight == T(0));
  }
  auto result = Welford<T>{
    a.mean + delta * wb_over_w,
    a.m2 + b.m2 + delta * delta * a.weight * wb_over_w,
    new_weight
  };
  return result;
}

template <typename T>
Welford<T> welford_combine(const Welford<T> &acc, T data) {
  // Add a single data point
  auto delta = data - acc.mean;
  auto new_weight = acc.weight + T(1);
  auto new_mean = acc.mean + delta / new_weight;
  auto new_delta = data - new_mean;
  auto result = Welford<T>{
    new_mean,
    acc.m2 + delta * new_delta,
    new_weight
  };
  return result;
}


#if INDUCTOR_USE_VECTOR_TYPES()
template <typename scalar_t>
inline at::vec::Vectorized<scalar_t> vec_shuffle_down(at::vec::Vectorized<scalar_t> x, size_t n) {
  using Vec = at::vec::Vectorized<scalar_t>;
  alignas(alignof(Vec)) scalar_t array[Vec::size()];
  x.store(array);
  for (size_t i = 0; i + n < Vec::size(); i += 2 * n) {
    array[i] = array[i + n];
  }
  return Vec::loadu(array);
}

#ifdef CPU_CAPABILITY_AVX2
inline at::vec::Vectorized<float> vec_shuffle_down(at::vec::Vectorized<float> x, size_t n) {
  using vec_t = at::vec::Vectorized<float>;
#define SHUFFLE_MASK(z, y, x, w) ((z << 6) | (y << 4) | (x << 2) | w)
  switch (n) {
  case 1:
    return vec_t(_mm256_permute_ps(x, SHUFFLE_MASK(1, 1, 3, 3)));
  case 2:
    return vec_t(_mm256_permute_ps(x, SHUFFLE_MASK(2, 2, 2, 2)));
  case 4:
    return vec_t(_mm256_permute2f128_ps(x, x, SHUFFLE_MASK(1, 1, 1, 1)));
  }
  TORCH_CHECK(false, "Unhandled vec_shuffle_down value ", n);
}
#endif

template <typename scalar_t>
Welford<scalar_t> welford_vec_reduce_all(Welford<at::vec::Vectorized<scalar_t>> acc) {
  using Vec = at::vec::Vectorized<scalar_t>;
  for (size_t n = 1; n < Vec::size(); n *= 2) {
    auto shuffled = Welford<Vec>{
      vec_shuffle_down(acc.mean, n),
      vec_shuffle_down(acc.m2, n),
      vec_shuffle_down(acc.weight, n)
    };
    acc = welford_combine(acc, shuffled);
  }

  Welford<scalar_t> result;
  alignas(alignof(Vec)) scalar_t array[Vec::size()];
  acc.mean.store(array);
  result.mean = array[0];

  acc.m2.store(array);
  result.m2 = array[0];

  acc.weight.store(array);
  result.weight = array[0];

  return result;
}
#endif


template <typename T> inline T mod(T a, T b) { return a % b; }
template <> inline float mod(float a, float b) { return std::fmod(a, b); }
template <> inline double mod(double a, double b) { return std::fmod(a, b); }

template <typename scalar_t>
inline scalar_t max_propagate_nan(scalar_t a, scalar_t b) {
  if (at::_isnan(a)) {
    return a;
  }
  return a > b ? a : b;
}

template <typename scalar_t>
inline scalar_t min_propagate_nan(scalar_t a, scalar_t b) {
  if (at::_isnan(a)) {
    return a;
  }
  return a < b ? a : b;
}

constexpr float uint32_to_uniform_float(uint32_t value) {
  // maximum value such that `MAX_INT * scale < 1.0` (with float rounding)
  constexpr float scale = 4.6566127342e-10;
  return static_cast<float>(value & 0x7FFFFFFF) * scale;
}

float normalized_rand_cpu(uint32_t seed, uint32_t offset) {
  return uint32_to_uniform_float(at::Philox4_32(seed, 0, offset)());
}

float randn_cpu(uint32_t seed, uint32_t offset) {
  at::Philox4_32 engine(seed, 0, offset);
  return engine.randn(10);
}

uint64_t randint64_cpu(uint32_t seed, uint32_t offset, int64_t low, int64_t high) {
  auto gen = at::Philox4_32(seed, 0, offset);
  uint64_t r0 = gen();
  uint64_t r1 = gen();
  uint64_t result = r0 | (r1 << 32);
  return (result % static_cast<uint64_t>(high - low)) + low;
}

template <typename T> struct AsIntegerType { typedef T type; };
template <> struct AsIntegerType<float> { typedef uint32_t type; };
template <> struct AsIntegerType<double> { typedef uint64_t type; };
template <> struct AsIntegerType<bfloat16> { typedef uint16_t type; };

template <typename T>
typename std::enable_if<!std::is_reduced_floating_point<T>::value, T>::type
inline fetch_value(volatile T *addr) {
  return *addr;
}

template <typename T>
typename std::enable_if<std::is_reduced_floating_point<T>::value, T>::type
inline fetch_value(volatile T *addr) {
  return T(addr->x, T::from_bits());
}

template <typename T>
typename std::enable_if<!std::is_integral<T>::value>::type
atomic_add(volatile T *addr, T offset) {
  typedef typename AsIntegerType<T>::type alt_type;

  static_assert(sizeof(std::atomic<alt_type>) == sizeof(T),
                "std::atomic issue");

  alt_type expected;

  alt_type desired;

  std::atomic<alt_type> *atomic_addr = (std::atomic<alt_type> *)addr;
  do {
    T val = fetch_value(addr);
    reinterpret_cast<T *>(&expected)[0] = val;
    reinterpret_cast<T *>(&desired)[0] = val + offset;
  } while (!atomic_addr->compare_exchange_weak(expected, desired,
                                               std::memory_order_relaxed));
}

// Since C++20 float is supported by fetch_add, but the performance may not
// better than compare_exchange_weak, which can be checked by microbenchmark
// inductor_cpu_atomic.py
template <typename T>
typename std::enable_if<std::is_integral<T>::value>::type
atomic_add(volatile T *addr, T offset) {
  static_assert(sizeof(std::atomic<T>) == sizeof(T),
                "std::atomic issue");
  std::atomic<T> *atomic_addr = (std::atomic<T> *)addr;
  atomic_addr->fetch_add(offset, std::memory_order_relaxed);
}

// This function is used to convert bool or uint8 to float mask for
// vectorization. The caller needs to make sure the src represents TRUE/FALSE
// correctly.
template <typename T>
inline float flag_to_float_scalar(T src) {
  float ret;
  *(uint32_t*)(&ret) = src ? 0xFFFFFFFF : 0;
  return ret;
}

#if defined(CPU_CAPABILITY_AVX512) || defined(CPU_CAPABILITY_AVX2)

inline at::vec::Vectorized<float> masked_load(const float* src, at::vec::Vectorized<float> mask) {
  at::vec::Vectorized<float> zero_vec(0);
# if defined(CPU_CAPABILITY_AVX512)
    auto all_ones = _mm512_set1_epi32(0xFFFFFFFF);
    auto mmask = _mm512_cmp_epi32_mask(_mm512_castps_si512(mask), all_ones, _MM_CMPINT_EQ);
    return _mm512_mask_loadu_ps(zero_vec, mmask, src);
# else // AVX2
    auto all_ones = _mm256_set1_epi32(0xFFFFFFFF);
    auto mmask = _mm256_cmpeq_epi32(_mm256_castps_si256(mask), all_ones);
    return _mm256_maskload_ps(src, mmask);
# endif
}

template <typename T>
typename std::enable_if<std::is_same<T, bfloat16>::value || std::is_same<T, half>::value, at::vec::Vectorized<T>>::type
inline masked_load(const T* src, at::vec::Vectorized<float> mask) {
# if defined(CPU_CAPABILITY_AVX512)
  auto all_ones = _mm512_set1_epi32(0xFFFFFFFF);
  auto mmask = _mm512_cmp_epi32_mask(_mm512_castps_si512(mask), all_ones, _MM_CMPINT_EQ);
  auto zero = _mm256_set1_epi16(0);
  auto temp = _mm256_mask_loadu_epi16(zero, mmask, src);
  return _mm512_inserti32x8(_mm512_castsi256_si512(temp), zero, 1);
# else // AVX2
  auto all_ones = _mm256_set1_epi32(0xFFFFFFFF);
  auto mmask_vec = _mm256_cmpeq_epi32(_mm256_castps_si256(mask), all_ones);
  __at_align__ uint32_t mmask[8];
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(mmask), mmask_vec);
  __at_align__ uint16_t result[16];
  for (auto i = 0; i < 8; i++) {
    result[i] = mmask[i] == 0xFFFFFFFF ? src[i].x: uint16_t(0);
  }
  return at::vec::Vectorized<T>::loadu(result);
# endif
}

inline at::vec::Vectorized<uint8_t> masked_load(const uint8_t* src, at::vec::Vectorized<float> mask) {
# if defined(CPU_CAPABILITY_AVX512)
    auto all_ones = _mm512_set1_epi32(0xFFFFFFFF);
    auto mmask = _mm512_cmp_epi32_mask(_mm512_castps_si512(mask), all_ones, _MM_CMPINT_EQ);
    auto zero = _mm_set1_epi8(0);
    auto temp = _mm_mask_loadu_epi8(zero, mmask, src);
    return _mm512_inserti64x2(_mm512_set1_epi32(0), temp, 0);
# else // AVX2
    auto all_ones = _mm256_set1_epi32(0xFFFFFFFF);
    auto mmask_vec = _mm256_cmpeq_epi32(_mm256_castps_si256(mask), all_ones);
    __at_align__ uint32_t mmask[8];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(mmask), mmask_vec);
    __at_align__ uint8_t result[32];
    for (auto i = 0; i < 8; i++) {
      result[i] = mmask[i] == 0xFFFFFFFF ? src[i]: uint8_t(0);
    }
    return at::vec::Vectorized<uint8_t>::loadu(result);
# endif
}

template <typename T>
inline at::vec::Vectorized<float> flag_to_float_vec(const T* src) {
  __at_align__ float dst_tmp[at::vec::Vectorized<float>::size()];
  #pragma unroll
  for (int64_t i = 0; i < at::vec::Vectorized<float>::size(); i++) {
    dst_tmp[i] = flag_to_float_scalar(src[i]);
  }
  return at::vec::Vectorized<float>::loadu(dst_tmp);
}

template <typename scalar_t>
inline at::vec::Vectorized<float> cvt_lowp_fp_to_fp32(
    at::vec::Vectorized<scalar_t> src) {
  at::vec::Vectorized<float> res_vec1(0);
  at::vec::Vectorized<float> res_vec2(0);
  std::tie(res_vec1, res_vec2) = at::vec::convert_to_float<scalar_t>(src);
  return res_vec1;
}

template <typename scalar_t>
inline at::vec::Vectorized<scalar_t> cvt_fp32_to_lowp_fp(
    at::vec::Vectorized<float> src) {
  return at::vec::convert_from_float<scalar_t>(src, src);
}

inline at::vec::Vectorized<float> mask_convert_to_float(at::vec::Vectorized<float> src) {
  auto zeros = at::vec::Vectorized<float>(0);
  auto ones = at::vec::Vectorized<float>(1);
  return at::vec::Vectorized<float>::blendv(zeros, ones, src);
}

template <typename SRC>
inline at::vec::Vectorized<float> vec_convert_to_mask(at::vec::Vectorized<SRC> src) {
  assert(
      at::vec::Vectorized<float>::size() == at::vec::Vectorized<SRC>::size());
  at::vec::Vectorized<float> res_vec(0);
  __at_align__ float dst_tmp[at::vec::Vectorized<float>::size()];
  __at_align__ SRC src_tmp[at::vec::Vectorized<SRC>::size()];
  src.store(src_tmp);

#pragma unroll
  for (int i = 0; i < at::vec::Vectorized<float>::size(); i++) {
    *(uint32_t*)(dst_tmp + i) = src_tmp[i] ? 0xFFFFFFFF : 0;
  }

  return res_vec.loadu(dst_tmp);
}

template <typename SRC>
inline at::vec::Vectorized<float> to_float_mask(at::vec::Vectorized<SRC> src) {
  return vec_convert_to_mask(src);
}

template <>
inline at::vec::Vectorized<float> to_float_mask(at::vec::Vectorized<int> src) {
#if defined(CPU_CAPABILITY_AVX2)
  return at::vec::Vectorized<float>(_mm256_castsi256_ps(src));
#else
  return at::vec::Vectorized<float>(_mm512_castsi512_ps(src));
#endif
}

template <>
inline at::vec::Vectorized<float> to_float_mask(at::vec::Vectorized<float> src) {
  return src;
}
#endif
```

after `graph.compile_to_fn()`, the `call` in the python code example above will be return and assigned into `compiled_fn`.

I assume that it is also works for `Triton language`.

after compilation, it will add some CUDA operaion like rng seed and offset into input argument, so this is not for CPU backend.

**Note: For release 2.1, the compile is still prototype, so this is only right for current pytorch version.**

#### cpp

for cpp jit extension, here is the code from the pytorch release 2.1 source code example:

```python
>>> from torch.utils.cpp_extension import load_inline
>>> source = """
at::Tensor sin_add(at::Tensor x, at::Tensor y) {
  return x.sin() + y.sin();
}
"""
>>> module = load_inline(name='inline_extension',
...                      cpp_sources=[source],
...                      functions=['sin_add'])
```

The cpp source code can be directly load in python.

### modify the python bytecode

After compilation, the `compiled_fn` will convert into the python instruction and insert into the originial python bytecode of source code.

```python
# add the compiled_fn into global var.
Instruction(opcode=116, opname='LOAD_GLOBAL', arg=False, argval='__compiled_fn_0', offset=None, starts_line=None, is_jump_target=False, positions=None, target=None, exn_tab_entry=None)
# load the 2 input arguments.
Instruction(opcode=124, opname='LOAD_FAST', arg=None, argval='a', offset=None, starts_line=None, is_jump_target=False, positions=None, target=None, exn_tab_entry=None)
Instruction(opcode=124, opname='LOAD_FAST', arg=None, argval='b', offset=None, starts_line=None, is_jump_target=False, positions=None, target=None, exn_tab_entry=None)
# call the torch function
Instruction(opcode=131, opname='CALL_FUNCTION', arg=2, argval=<class 'torch._dynamo.bytecode_transformation._NotProvided'>, offset=None, starts_line=None, is_jump_target=False, positions=None, target=None, exn_tab_entry=None)
# unpack the result.
Instruction(opcode=92, opname='UNPACK_SEQUENCE', arg=1, argval=<class 'torch._dynamo.bytecode_transformation._NotProvided'>, offset=None, starts_line=None, is_jump_target=False, positions=None, target=None, exn_tab_entry=None)
# return op
Instruction(opcode=83, opname='RETURN_VALUE', arg=None, argval=<class 'torch._dynamo.bytecode_transformation._NotProvided'>, offset=None, starts_line=None, is_jump_target=False, positions=None, target=None, exn_tab_entry=None)
```

Those python bytecode instruction will be injected into original bytecoe or do the substitution.

by `transform_code_object` , the code instruction will be:

```
8           0 LOAD_GLOBAL              0 (__compiled_fn_0)
              2 LOAD_FAST                0 (a)
              4 LOAD_FAST                1 (b)
              6 CALL_FUNCTION            2
              8 UNPACK_SEQUENCE          1
             10 RETURN_VALUE
```

Reference:

- https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/OVERVIEW.md#jit-logging
