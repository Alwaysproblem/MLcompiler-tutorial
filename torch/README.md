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

Reference:

- https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/OVERVIEW.md#jit-logging
