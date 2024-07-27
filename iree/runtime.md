# iree runtime source code investigation

## the factory registration

```text
iree_hal_register_all_available_drivers(iree_hal_driver_registry_t * registry) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/hal/drivers/init.c:69)
iree_runtime_instance_options_use_all_available_drivers(iree_runtime_instance_options_t * options) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/instance.c:33)
iree_runtime_demo_main() (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:81)
main(int argc, char ** argv) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:28)
```

## System Library Loader

```text
main(int argc, char ** argv) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:28)
iree_runtime_demo_main() (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:94)
iree_runtime_demo_run_session(iree_runtime_instance_t * instance) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:125)
iree_runtime_instance_try_create_default_device(iree_runtime_instance_t * instance, iree_string_view_t driver_name, iree_hal_device_t ** out_device) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/instance.c:158)
iree_hal_driver_registry_try_create(iree_hal_driver_registry_t * registry, iree_string_view_t driver_name, iree_allocator_t host_allocator, iree_hal_driver_t ** out_driver) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/hal/driver_registry.c:314)
iree_hal_local_task_driver_factory_try_create(void * self, iree_string_view_t driver_name, iree_allocator_t host_allocator, iree_hal_driver_t ** out_driver) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/hal/drivers/local_task/registration/driver_module.c:71)
iree_hal_create_all_available_executable_loaders(iree_hal_executable_plugin_manager_t * plugin_manager, iree_host_size_t capacity, iree_host_size_t * out_count, iree_hal_executable_loader_t ** loaders, iree_allocator_t host_allocator) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/hal/local/loaders/registration/init.c:55)
iree_hal_system_library_loader_create(iree_hal_executable_plugin_manager_t * plugin_manager, iree_allocator_t host_allocator, iree_hal_executable_loader_t ** out_executable_loader) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/hal/local/loaders/system_library_loader.c:378)
```

In here, the `iree_hal_system_library_loader_create` function will create the executable loader for the system library.
copy the `iree_hal_system_library_loader_vtable` to the `iree_hal_executable_loader_t` structure.

```cpp
static const iree_hal_executable_loader_vtable_t
    iree_hal_system_library_loader_vtable = {
        .destroy = iree_hal_system_library_loader_destroy,
        .query_support = iree_hal_system_library_loader_query_support,
        .try_load = iree_hal_system_library_loader_try_load,
};
```

The `iree_hal_system_library_loader_try_load` function will load the system library and create the executable.

```text
iree_hal_system_library_loader_try_load@runtime/src/iree/hal/local/loaders/system_library_loader.c:435
iree_hal_system_executable_create@runtime/src/iree/hal/local/loaders/system_library_loader.c:213
iree_hal_local_executable_initialize@runtime/src/iree/hal/local/local_executable.c:11
```

Here, assign the `iree_hal_system_executable_vtable` to `iree_hal_system_executable_t` structure.

```cpp
static const iree_hal_local_executable_vtable_t
    iree_hal_system_executable_vtable = {
        .base =
            {
                .destroy = iree_hal_system_executable_destroy,
            },
        .issue_call = iree_hal_system_executable_issue_call,
};
```

### Load the dynamic library workflow

#### Load the dynamic library

```text
iree_hal_system_library_loader_try_load@runtime/src/iree/hal/local/loaders/system_library_loader.c:435
iree_hal_system_executable_create@runtime/src/iree/hal/local/loaders/system_library_loader.c:213
iree_hal_system_executable_load@runtime/src/iree/hal/local/loaders/system_library_loader.c:101
iree_dynamic_library_load_from_memory@runtime/src/iree/base/internal/dynamic_library_posix.c:232
iree_dynamic_library_load_from_file@runtime/src/iree/base/internal/dynamic_library_posix.c:138
iree_dynamic_library_load_from_files@runtime/src/iree/base/internal/dynamic_library_posix.c:145
```

Here, create an temperary `.so` file that saves the binary data and read the handle from the dynamic library with `dlopen` function.


#### Load the function from the dynamic library

```text
iree_hal_system_library_loader_try_load@runtime/src/iree/hal/local/loaders/system_library_loader.c:435
iree_hal_system_executable_create@runtime/src/iree/hal/local/loaders/system_library_loader.c:213
iree_hal_system_executable_query_library@runtime/src/iree/hal/local/loaders/system_library_loader.c:140
iree_dynamic_library_lookup_symbol@runtime/src/iree/base/internal/dynamic_library_posix.c:309
```

Here, use the `dlsym` function to load the function from the dynamic library.

## ELF Loader

```log
main(int argc, char ** argv) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:28)
iree_runtime_demo_main() (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:94)
iree_runtime_demo_run_session(iree_runtime_instance_t * instance) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:125)
iree_runtime_instance_try_create_default_device(iree_runtime_instance_t * instance, iree_string_view_t driver_name, iree_hal_device_t ** out_device) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/instance.c:158)
iree_hal_driver_registry_try_create(iree_hal_driver_registry_t * registry, iree_string_view_t driver_name, iree_allocator_t host_allocator, iree_hal_driver_t ** out_driver) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/hal/driver_registry.c:314)
iree_hal_local_task_driver_factory_try_create(void * self, iree_string_view_t driver_name, iree_allocator_t host_allocator, iree_hal_driver_t ** out_driver) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/hal/drivers/local_task/registration/driver_module.c:71)
iree_hal_create_all_available_executable_loaders(iree_hal_executable_plugin_manager_t * plugin_manager, iree_host_size_t capacity, iree_host_size_t * out_count, iree_hal_executable_loader_t ** loaders, iree_allocator_t host_allocator) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/hal/local/loaders/registration/init.c:62)
iree_hal_embedded_elf_loader_create(iree_hal_executable_plugin_manager_t * plugin_manager, iree_allocator_t host_allocator, iree_hal_executable_loader_t ** out_executable_loader) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/hal/local/loaders/embedded_elf_loader.c:258)
```

In here, the `iree_hal_embedded_elf_loader_create` function will create the executable loader for the system library.
copy the `iree_hal_embedded_elf_loader_vtable` to the `iree_hal_executable_loader_t` structure.

```cpp
static const iree_hal_executable_loader_vtable_t
    iree_hal_embedded_elf_loader_vtable = {
        .destroy = iree_hal_embedded_elf_loader_destroy,
        .query_support = iree_hal_embedded_elf_loader_query_support,
        .try_load = iree_hal_embedded_elf_loader_try_load,
};
```

The `iree_hal_embedded_elf_loader_try_load` function will load the system library and create the executable.


```text
iree_hal_embedded_elf_loader_try_load@runtime/src/iree/hal/local/loaders/embedded_elf_loader.c:294
iree_hal_elf_executable_create@runtime/src/iree/hal/local/loaders/embedded_elf_loader.c:303
iree_hal_local_executable_initialize@runtime/src/iree/hal/local/loaders/embedded_elf_loader.c:113
```

Here, assign the `iree_hal_elf_executable_vtable` to `iree_hal_elf_executable_t` structure.

```cpp
static const iree_hal_local_executable_vtable_t iree_hal_elf_executable_vtable =
    {
        .base =
            {
                .destroy = iree_hal_elf_executable_destroy,
            },
        .issue_call = iree_hal_elf_executable_issue_call,
};
```

### Initialize ELF workflow

#### Load the elf from memory

```text
iree_hal_embedded_elf_loader_try_load@runtime/src/iree/hal/local/loaders/embedded_elf_loader.c:294
iree_hal_elf_executable_create@runtime/src/iree/hal/local/loaders/embedded_elf_loader.c:87
iree_elf_module_initialize_from_memory@runtime/src/iree/hal/local/elf/elf_module.c:573 <- Only FatELF supported.
```

#### Parse the ELF file

```text
iree_hal_embedded_elf_loader_try_load@runtime/src/iree/hal/local/loaders/embedded_elf_loader.c:294
iree_hal_elf_executable_create@runtime/src/iree/hal/local/loaders/embedded_elf_loader.c:87
iree_elf_module_initialize_from_memory@runtime/src/iree/hal/local/elf/elf_module.c:573 <- Only FatELF supported.
iree_elf_module_parse_headers@runtime/src/iree/hal/local/elf/elf_module.c:162
```

Here, parse the ELF file and get the ELF header, program header, and section header.

#### Allocate and load the ELF into memory

```text
iree_hal_embedded_elf_loader_try_load@runtime/src/iree/hal/local/loaders/embedded_elf_loader.c:294
iree_hal_elf_executable_create@runtime/src/iree/hal/local/loaders/embedded_elf_loader.c:87
iree_elf_module_initialize_from_memory@runtime/src/iree/hal/local/elf/elf_module.c:573 <- Only FatELF supported.
iree_elf_module_load_segments@runtime/src/iree/hal/local/elf/elf_module.c:234
```

1. 计算 ELF 文件中所有 PT_LOAD 段的虚拟地址范围。
2. 在主机的虚拟内存空间中预留一块内存。这块内存初始时未实际分配，只有在需要时才会分配物理内存。module->vaddr_bias 是一个偏移量，用于调整 ELF 文件的虚拟地址使其匹配预留的内存地址。
3. 遍历所有的 Program Headers，检查它们是否为 PT_LOAD 类型。如果是 PT_LOAD 段，则执行以下步骤:

    - 提交内存范围：调用 iree_memory_view_commit_ranges 函数，提交需要加载段的内存范围，并设置为可读写。(这里用 mmap 函数实现虚拟进程内存映射分配)
    - 拷贝数据：如果段在文件中有数据（p_filesz > 0），则从原始数据 (raw_data) 中拷贝对应的数据到预留的内存区域。(这里用 memcpy 函数实现数据拷贝)

#### Parse required dynamic symbol tables

```text
iree_hal_embedded_elf_loader_try_load@runtime/src/iree/hal/local/loaders/embedded_elf_loader.c:294
iree_hal_elf_executable_create@runtime/src/iree/hal/local/loaders/embedded_elf_loader.c:87
iree_elf_module_initialize_from_memory@runtime/src/iree/hal/local/elf/elf_module.c:573 <- Only FatELF supported.
iree_elf_module_parse_dynamic_tables@runtime/src/iree/hal/local/elf/elf_module.c:370
```

1. 查找动态段：
   - 遍历程序头表（Program Header Table），找到类型为 PT_DYNAMIC 的段。
   - 获取动态段的地址和大小，并保存到 load_state 中。
2. 解析动态段条目：
   - 遍历动态段中的每个条目，根据 d_tag 的值进行处理。
   - 处理常见的动态段条目，包括 .dynstr、.dynsym、初始化函数和初始化函数数组等。
3. 设置模块的动态符号和字符串表：
   - 根据解析的结果，设置模块结构体中的 .dynstr 和 .dynsym 表及其大小和条目数。
4. 验证必要的段：
   - 确保 .dynstr 和 .dynsym 表及其大小和条目数都已经正确设置。
   - 如果缺少必要的段，返回错误状态。
5. 返回成功状态：
   - 所有必要的信息都已解析和设置，返回成功状态。

#### Apply relocations to the loaded pages.

```text
iree_hal_embedded_elf_loader_try_load@runtime/src/iree/hal/local/loaders/embedded_elf_loader.c:294
iree_hal_elf_executable_create@runtime/src/iree/hal/local/loaders/embedded_elf_loader.c:87
iree_elf_module_initialize_from_memory@runtime/src/iree/hal/local/elf/elf_module.c:573 <- Only FatELF supported.
iree_elf_module_apply_relocations@runtime/src/iree/hal/local/elf/elf_module.c:493
iree_elf_arch_apply_relocations@runtime/src/iree/hal/local/elf/arch/x86_64.c:110
```

1. 遍历重定位表：
   - 循环遍历 rela_table 中的每个重定位条目。
   - 从 rela->r_info 中提取重定位类型 type，跳过类型为 IREE_ELF_R_X86_64_NONE 的条目。
2. 处理符号地址：
   - 提取符号索引 sym_ordinal。
   - 如果符号索引不为 0，则查找符号表以获取符号地址 sym_addr。
   - 检查符号索引是否有效，如果无效则返回错误状态。
3. 计算指令指针：
   - 计算需要重定位的地址 instr_ptr，这是基地址 state->vaddr_bias 加上重定位偏移量 rela->r_offset。
4. 应用重定位：
   - 根据重定位类型 type，应用相应的重定位：
   - IREE_ELF_R_X86_64_RELATIVE：地址 = 基地址 + 偏移量。
   - IREE_ELF_R_X86_64_JUMP_SLOT、IREE_ELF_R_X86_64_GLOB_DAT、IREE_ELF_R_X86_64_COPY：符号地址写入指令指针位置。
   - IREE_ELF_R_X86_64_64：符号地址加上附加值写入指令指针位置。
   - IREE_ELF_R_X86_64_32：32 位地址重定位，符号地址加上附加值写入指令指针位置。
   - IREE_ELF_R_X86_64_32S：32 位有符号地址重定位，符号地址加上附加值写入指令指针位置。
   - IREE_ELF_R_X86_64_PC32：计算相对地址并写入指令指针位置。
   - 其他未实现的重定位类型返回错误状态。
5. 返回成功状态：
   - 所有重定位条目处理完毕后，返回成功状态。

#### Apply final protections to the loaded pages

```text
iree_hal_embedded_elf_loader_try_load@runtime/src/iree/hal/local/loaders/embedded_elf_loader.c:294
iree_hal_elf_executable_create@runtime/src/iree/hal/local/loaders/embedded_elf_loader.c:87
iree_elf_module_initialize_from_memory@runtime/src/iree/hal/local/elf/elf_module.c:573 <- Only FatELF supported.
iree_elf_module_apply_relocations@runtime/src/iree/hal/local/elf/elf_module.c:493
iree_elf_module_protect_segments@runtime/src/iree/hal/local/elf/elf_module.c:294
```

1. 遍历程序头表：
   - 遍历程序头表（Program Header Table），处理类型为 PT_LOAD 和 PT_GNU_RELRO 的段。
2. 处理 PT_LOAD 段：
   - 解释段的访问权限位（R、W、X），确定内存访问权限。
   - 检查并禁止可执行且可写的段。
   - 应用新的访问保护。
   - 如果段是可执行的，则刷新指令缓存。
3. 处理 PT_GNU_RELRO 段：
   - 将 PT_GNU_RELRO 段设置为只读，以加强安全性。

Here use the `mprotect` function to set the memory protection.

#### Run initializers prior to returning to the caller

```text
iree_hal_embedded_elf_loader_try_load@runtime/src/iree/hal/local/loaders/embedded_elf_loader.c:294
iree_hal_elf_executable_create@runtime/src/iree/hal/local/loaders/embedded_elf_loader.c:87
iree_elf_module_initialize_from_memory@runtime/src/iree/hal/local/elf/elf_module.c:573 <- Only FatELF supported.
iree_elf_module_apply_relocations@runtime/src/iree/hal/local/elf/elf_module.c:493
iree_elf_module_run_initializers@runtime/src/iree/hal/local/elf/elf_module.c:512
```

Here, directly run the function address after relocation and forcing type conversion.

```cpp
void iree_elf_call_v_v(const void* symbol_ptr) {
  typedef void (*ptr_t)(void);
  ((ptr_t)symbol_ptr)();
}
```

### ELF executable query library

```text
iree_hal_embedded_elf_loader_try_load@runtime/src/iree/hal/local/loaders/embedded_elf_loader.c:294
iree_hal_elf_executable_create@runtime/src/iree/hal/local/loaders/embedded_elf_loader.c:87
iree_hal_elf_executable_query_library@runtime/src/iree/hal/local/loaders/embedded_elf_loader.c:44
```

1. 获取导出符号：
   - 查找并获取用于查询库元数据的导出符号（query_fn）。
2. 查询库版本：
   - 使用 query_fn 查询与运行时兼容的库版本。
   - 如果不支持当前运行时版本，则返回错误。
3. 检查库的安全性：
   - 检查库是否需要特定的安全检测（sanitizer）。
   - 如果库需要的安全检测在当前环境中不支持，则返回错误。
4. 设置可执行文件属性：
   - 设置可执行文件的标识符和调度属性。

Q: I can not find the implementation of the `iree_hal_executable_library_query` function.



## device create

```text
iree_hal_local_task_driver_factory_try_create(void * self, iree_string_view_t driver_name, iree_allocator_t host_allocator, iree_hal_driver_t ** out_driver) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/hal/drivers/local_task/registration/driver_module.c:68)
iree_hal_driver_registry_try_create(iree_hal_driver_registry_t * registry, iree_string_view_t driver_name, iree_allocator_t host_allocator, iree_hal_driver_t ** out_driver) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/hal/driver_registry.c:314)
iree_runtime_instance_try_create_default_device(iree_runtime_instance_t * instance, iree_string_view_t driver_name, iree_hal_device_t ** out_device) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/instance.c:158)
iree_runtime_demo_run_session(iree_runtime_instance_t * instance) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:125)
iree_runtime_demo_main() (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:94)
main(int argc, char ** argv) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:28)
```

## Load module


run the init function in the vm module.

```text
main(int argc, char ** argv) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:28)
iree_runtime_demo_main() (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:94)
iree_runtime_demo_run_session(iree_runtime_instance_t * instance) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:135)
iree_runtime_session_create_with_device(iree_runtime_instance_t * instance, const iree_runtime_session_options_t * options, iree_hal_device_t * device, iree_allocator_t host_allocator, iree_runtime_session_t ** out_session) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/session.c:102)
iree_vm_context_register_modules(iree_vm_context_t * context, iree_host_size_t module_count, iree_vm_module_t ** modules) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/context.c:525) <- register the module and run the `@__init` function in the vm module.
```


iree_runtime_session_append_bytecode_module_from_memory@runtime/src/iree/runtime/demo/hello_world_terse.c:57
  -> iree_vm_bytecode_module_create@runtime/src/iree/runtime/session.c:214
     1. check the flatbuffer is valid
     2. initalized the model function pointer:
        iree_vm_bytecode_module_destroy
        iree_vm_bytecode_module_name
        iree_vm_bytecode_module_signature
        iree_vm_bytecode_module_get_module_attr
        iree_vm_bytecode_module_enumerate_dependencies: Note register the callback function
        iree_vm_bytecode_module_lookup_function
        iree_vm_bytecode_module_get_function
        iree_vm_bytecode_module_get_function_attr
        iree_vm_bytecode_module_resolve_source_location
        iree_vm_bytecode_module_alloc_state
        iree_vm_bytecode_module_free_state
        iree_vm_bytecode_module_resolve_import
        iree_vm_bytecode_module_notify
        iree_vm_bytecode_module_begin_call
        iree_vm_bytecode_module_resume_call


iree_runtime_call_invoke
  -> iree_runtime_session_call
    -> iree_vm_invoke
      -> iree_vm_begin_invoke@runtime/src/iree/vm/invocation.c:302
        -> 
      ->

----

```cpp
static const iree_hal_executable_loader_vtable_t
    iree_hal_vmvx_module_loader_vtable = {
        .destroy = iree_hal_vmvx_module_loader_destroy,
        .query_support = iree_hal_vmvx_module_loader_query_support,
        .try_load = iree_hal_vmvx_module_loader_try_load,
};
```

## Run


## Debug call stack information


```text
main(int argc, char ** argv) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:28)
iree_runtime_demo_main() (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:94)
iree_runtime_demo_run_session(iree_runtime_instance_t * instance) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:149)
iree_runtime_demo_perform_mul(iree_runtime_session_t * session) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:254)
iree_runtime_call_invoke(iree_runtime_call_t * call, iree_runtime_call_flags_t flags) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/call.c:97)
iree_runtime_session_call(iree_runtime_session_t * session, const iree_vm_function_t * function, iree_vm_list_t * input_list, iree_vm_list_t * output_list) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/session.c:301)
iree_vm_invoke(iree_vm_context_t * context, iree_vm_function_t function, iree_vm_invocation_flags_t flags, const iree_vm_invocation_policy_t * policy, const iree_vm_list_t * inputs, iree_vm_list_t * outputs, iree_allocator_t host_allocator) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/invocation.c:302)
iree_vm_begin_invoke(iree_vm_invoke_state_t * state, iree_vm_context_t * context, iree_vm_function_t function, iree_vm_invocation_flags_t flags, const iree_vm_invocation_policy_t * policy, const iree_vm_list_t * inputs, iree_allocator_t host_allocator) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/invocation.c:504)
iree_vm_bytecode_module_begin_call(void * self, iree_vm_stack_t * stack, iree_vm_function_call_t call) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/bytecode/module.c:788)
iree_vm_bytecode_dispatch_begin(iree_vm_stack_t * stack, iree_vm_bytecode_module_t * module, const iree_vm_function_call_t call, iree_string_view_t cconv_arguments, iree_string_view_t cconv_results) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/bytecode/dispatch.c:636)
iree_vm_bytecode_dispatch(iree_vm_stack_t * restrict stack, iree_vm_bytecode_module_t * restrict module, iree_vm_stack_frame_t * restrict current_frame, iree_vm_registers_t regs, iree_byte_span_t call_results) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/bytecode/dispatch.c:1680)
iree_vm_bytecode_call_import(iree_vm_stack_t * stack, const iree_vm_bytecode_module_state_t * module_state, uint32_t import_ordinal, const iree_vm_registers_t caller_registers, const iree_vm_register_list_t * restrict src_reg_list, const iree_vm_register_list_t * restrict dst_reg_list, iree_vm_stack_frame_t * restrict * out_caller_frame, iree_vm_registers_t * out_caller_registers) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/bytecode/dispatch.c:568)
iree_vm_bytecode_issue_import_call(iree_vm_stack_t * stack, const iree_vm_function_call_t call, iree_string_view_t cconv_results, const iree_vm_register_list_t * restrict dst_reg_list, iree_vm_stack_frame_t * restrict * out_caller_frame, iree_vm_registers_t * out_caller_registers) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/bytecode/dispatch.c:452)
iree_vm_native_module_begin_call(void * self, iree_vm_stack_t * stack, iree_vm_function_call_t call) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/native_module.c:402)
iree_vm_native_module_issue_call(iree_vm_native_module_t * module, iree_vm_stack_t * stack, iree_vm_stack_frame_t * callee_frame, iree_vm_native_function_flags_t flags, iree_byte_span_t args_storage, iree_byte_span_t rets_storage) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/native_module.c:346)
iree_vm_shim_i_r(iree_vm_stack_t * restrict stack, iree_vm_native_function_flags_t flags, iree_byte_span_t args_storage, iree_byte_span_t rets_storage, iree_vm_native_function_target2_t target_fn, void * restrict module, void * restrict module_state) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/shims.c:10)
iree_hal_module_devices_get(iree_vm_stack_t * restrict stack, void * restrict module, iree_hal_module_state_t * restrict state, iree_vm_abi_i_t * restrict args, iree_vm_abi_r_t * restrict rets) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/modules/hal/module.c:1178)
```

```text
main(int argc, char ** argv) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:28)
iree_runtime_demo_main() (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:94)
iree_runtime_demo_run_session(iree_runtime_instance_t * instance) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:144)
iree_runtime_demo_load_module(iree_runtime_session_t * session) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:34)
iree_runtime_session_append_bytecode_module_from_file(iree_runtime_session_t * session, const char * file_path) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/session.c:249)
iree_runtime_session_append_bytecode_module_from_memory(iree_runtime_session_t * session, iree_const_byte_span_t flatbuffer_data, iree_allocator_t flatbuffer_allocator) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/session.c:220)
iree_runtime_session_append_module(iree_runtime_session_t * session, iree_vm_module_t * module) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/session.c:197)
iree_vm_context_register_modules(iree_vm_context_t * context, iree_host_size_t module_count, iree_vm_module_t ** modules) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/context.c:525)
iree_vm_context_run_function(iree_vm_context_t * context, iree_vm_stack_t * stack, iree_vm_module_t * module, iree_string_view_t function_name) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/context.c:91)
iree_vm_bytecode_module_begin_call(void * self, iree_vm_stack_t * stack, iree_vm_function_call_t call) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/bytecode/module.c:788)
iree_vm_bytecode_dispatch_begin(iree_vm_stack_t * stack, iree_vm_bytecode_module_t * module, const iree_vm_function_call_t call, iree_string_view_t cconv_arguments, iree_string_view_t cconv_results) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/bytecode/dispatch.c:636)
iree_vm_bytecode_dispatch(iree_vm_stack_t * restrict stack, iree_vm_bytecode_module_t * restrict module, iree_vm_stack_frame_t * restrict current_frame, iree_vm_registers_t regs, iree_byte_span_t call_results) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/bytecode/dispatch.c:1680)
iree_vm_bytecode_call_import(iree_vm_stack_t * stack, const iree_vm_bytecode_module_state_t * module_state, uint32_t import_ordinal, const iree_vm_registers_t caller_registers, const iree_vm_register_list_t * restrict src_reg_list, const iree_vm_register_list_t * restrict dst_reg_list, iree_vm_stack_frame_t * restrict * out_caller_frame, iree_vm_registers_t * out_caller_registers) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/bytecode/dispatch.c:568)
iree_vm_bytecode_issue_import_call(iree_vm_stack_t * stack, const iree_vm_function_call_t call, iree_string_view_t cconv_results, const iree_vm_register_list_t * restrict dst_reg_list, iree_vm_stack_frame_t * restrict * out_caller_frame, iree_vm_registers_t * out_caller_registers) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/bytecode/dispatch.c:452)
iree_vm_native_module_begin_call(void * self, iree_vm_stack_t * stack, iree_vm_function_call_t call) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/native_module.c:402)
iree_vm_native_module_issue_call(iree_vm_native_module_t * module, iree_vm_stack_t * stack, iree_vm_stack_frame_t * callee_frame, iree_vm_native_function_flags_t flags, iree_byte_span_t args_storage, iree_byte_span_t rets_storage) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/native_module.c:339)
```
