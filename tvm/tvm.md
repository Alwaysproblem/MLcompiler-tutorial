## TVM export the runtime module

```text
python: lib initialized
python: -> self.module = tvm.graph_executor_factory.create(libmod, libmod_name, *args);
cpp:      -> call `make_object<GraphExecutorFactory>@src/runtime/graph_executor/graph_executor_factory.cc:214` make a `GraphExecutorFactory`
          -> call `exec->Import(args[1]);`; import from the graph_json_str

python: lib.export_library
python:  -> self.module.export_library
python:    -> export_library@python/tvm/runtime/module.py:535
python:       -> save@python/tvm/runtime/module.py:316
cpp:            -> LLVMModuleNode::SaveToFile@src/target/llvm/llvm_module.cc:242 [This is for lib0.o]
cpp:            -> PackImportsToLLVM@src/target/codegen.cc:362 [This is for devc.o]
cpp:              -> call `CodeGenBlob@src/target/llvm/codegen_blob.cc:64` [will initialize the `LLVMJITEngine`]




## TVM runtime source workflow

```text
python: load_module
  -> call `_ffi_api.ModuleLoadFromFile@python/tvm/runtime/module.py:708`
    -> call `Module::LoadFromFile@src/runtime/module.cc:79`
      -> call `runtime.module.loadfile_so@src/runtime/dso_library.cc:152`
        -> call `CreateDSOLibraryObject`
           This function will call the `dlopen` and `dlsym` to analysis the callee function.
        -> call `CreateModuleFromLibrary@src/runtime/library_module.cc:204`
          -> call `ProcessModuleBlob@src/runtime/library_module.cc:137`
            -> There are 3 tkey can be load
            -> `tkey == "_lib"` call `make_object<LibraryModuleNode>`
            -> `tkey == "_import_tree"` read `import_tree_row_ptr` and `import_tree_child_indices`
            -> `tkey == <others>` call `LoadModuleFromBinary@src/runtime/library_module.cc:107`
              -> call `runtime.module.loadbinary_<tkey>` for rcar target, it is `runtime.module.loadbinary_rcar_v4x`
                -> call `LoadFromBinary@src/runtime/contrib/rcar_v4x/rcar_v4x_runtime.cc:140`
            -> push back into the Module List
        -> return the root module which is the first module in the list.
```

During the `CreateModuleFromLibrary` function, the `net_onnx_rdl.tar` will be untar and with `g++  -shared -fPIC  -o <output_name>.so devc.o lib0.o` to create the shared library.


```log
python: tvm.graph_executor.create
-> call the `GraphExecutorCreate@src/runtime/graph_executor/graph_executor.cc:788`
  -> call the `exec->Init`
    -> call the `this->Load(&reader)` # This is the `Load` function.
      -> in this load function, it will parse the metadata.json like this (in the graph_json chapter):
      -> Here it will extract the `arg_nodes`, `heads`, `nodes`, `node_row_ptr`, `attrs` and `shape` information.
      -> and init the vector of Node in the `GraphExecutor` object.
      -> and all the nodes in the json will be extracted to the `TVMOpParam`.
    -> call `GraphExecutor::SetupStorage()`
      -> calculate the tensor space and create a storage list and allocate the memory. 
    -> call `GraphExecutor::SetupOpExecs()`
      -> create the input and output tensor list and create the `CreateTVMOp` list.
      -> `CreateTVMOp@src/runtime/graph_executor/graph_executor.cc:603`
        -> scan the input and output and create the TVMArgs
        -> `module_.GetFunction(param.func_name, true);`
          # Here the param.func_name is the function name in the json.
          # So, This will call the `RCarV4XRuntime::GetFunction` since the name is for rcar target.
          # -> `ModuleNode::GetFunction@src/runtime/module.cc:64`
        -> return the function ptr and args ptr.

python:
  -> call `GraphExecutor.run`
    -> call the `GraphExecutor::GetFunction@src/runtime/graph_executor/graph_executor.cc:653`
      -> call `GraphExecutor::Run()@src/runtime/graph_executor/graph_executor.cc@69`
        -> you can find source code that the run the iteration of the `CreateTVMOp` list.
```

Note: call the GetFuntion(`default`) function, This will convert the tensor into device tensor.

## graph_json

```json
{
  "nodes": [
    {
      "op": "null", 
      "name": "input", 
      "inputs": []
    }, 
    {
      "op": "tvm_op", 
      "name": "tvmgen_foo_rcardkl_main_0", 
      "attrs": {
        "num_outputs": "1", 
        "num_inputs": "1", 
        "flatten_data": "0", 
        "func_name": "tvmgen_foo_rcardkl_main_0"
      }, 
      "inputs": [
        [
          0, 
          0, 
          0
        ]
      ]
    }, 
    {
      "op": "tvm_op", 
      "name": "tvmgen_foo_tvmgen_foo_rcar_imp_main_0", 
      "attrs": {
        "flatten_data": "0", 
        "func_name": "tvmgen_foo_tvmgen_foo_rcar_imp_main_0", 
        "out_layout": "", 
        "data_layout": "NCHW", 
        "hash": "3fb7d5cff8f3e218", 
        "kernel_layout": "OIHW", 
        "num_outputs": "1", 
        "num_inputs": "1", 
        "global_symbol": "tvmgen_foo_tvmgen_foo_rcar_imp_main_0", 
        "Compiler": "rcar_imp"
      }, 
      "inputs": [
        [
          1, 
          0, 
          0
        ]
      ]
    }, 
    {
      "op": "tvm_op", 
      "name": "tvmgen_foo_rcardkl_main_3", 
      "attrs": {
        "num_outputs": "1", 
        "num_inputs": "1", 
        "flatten_data": "0", 
        "func_name": "tvmgen_foo_rcardkl_main_3"
      }, 
      "inputs": [
        [
          2, 
          0, 
          0
        ]
      ]
    }
  ], 
  "arg_nodes": [0], 
  "heads": [
    [
      3, 
      0, 
      0
    ]
  ], 
  "attrs": {
    "dltype": [
      "list_str", 
      [
        "float32", 
        "int8", 
        "int8", 
        "float32"
      ]
    ], 
    "device_index": [
      "list_int", 
      [1, 1, 1, 1]
    ], 
    "storage_id": [
      "list_int", 
      [0, 1, 2, 3]
    ], 
    "shape": [
      "list_shape", 
      [
        [1, 3, 224, 224], 
        [1, 3, 224, 224], 
        [1, 512, 1, 1], 
        [1, 1000]
      ]
    ]
  }, 
  "node_row_ptr": [0, 1, 2, 3, 4]
}
```