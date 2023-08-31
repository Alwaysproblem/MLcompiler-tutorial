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
```

Reference:

- https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/OVERVIEW.md#jit-logging
