# iree runtime source code investigation

## the factory registration

```text
iree_hal_register_all_available_drivers(iree_hal_driver_registry_t * registry) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/hal/drivers/init.c:69)
iree_runtime_instance_options_use_all_available_drivers(iree_runtime_instance_options_t * options) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/instance.c:33)
iree_runtime_demo_main() (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:81)
main(int argc, char ** argv) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:28)
```

## Type registration

In the `iree_runtime_instance_create` function will register the type:

- `vm.buffer`
- `vm.list`
- `hal.allocator`
- `hal.channel`
- `hal.command_buffer`
- `hal.descriptor_set_layout`
- `hal.device`
- `hal.event`
- `hal.fence`
- `hal.file`
- `hal.pipeline_layout`
- `hal.semaphore`

## Device Creation

```text
main(int argc, char ** argv) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:28)
iree_runtime_demo_main() (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:94)
iree_runtime_demo_run_session(iree_runtime_instance_t * instance) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:125)
iree_runtime_instance_try_create_default_device(iree_runtime_instance_t * instance, iree_string_view_t driver_name, iree_hal_device_t ** out_device) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/instance.c:158)
iree_hal_driver_registry_try_create(iree_hal_driver_registry_t * registry, iree_string_view_t driver_name, iree_allocator_t host_allocator, iree_hal_driver_t ** out_driver) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/hal/driver_registry.c:314)
iree_hal_local_task_driver_factory_try_create(void * self, iree_string_view_t driver_name, iree_allocator_t host_allocator, iree_hal_driver_t ** out_driver) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/hal/drivers/local_task/registration/driver_module.c:64)
iree_hal_task_driver_create@runtime/src/iree/hal/drivers/local_task/task_driver.c:37
```

Here will register the `iree_hal_driver_vtable_t` into `iree_hal_resource_t` in `driver->resource`

```cpp
static const iree_hal_driver_vtable_t iree_hal_task_driver_vtable = {
    .destroy = iree_hal_task_driver_destroy,
    .query_available_devices = iree_hal_task_driver_query_available_devices,
    .dump_device_info = iree_hal_task_driver_dump_device_info,
    .create_device_by_id = iree_hal_task_driver_create_device_by_id,
    .create_device_by_path = iree_hal_task_driver_create_device_by_path,
};
```


### create default device

call the function `create_device_by_id` after registering the driver in the function `iree_hal_local_task_driver_factory_try_create`

```text
iree_hal_task_driver_create_device_by_id@runtime/src/iree/hal/drivers/local_task/task_driver.c:147
iree_hal_task_device_create@runtime/src/iree/hal/drivers/local_task/task_device.c:85
```

Here will register the `iree_hal_device_vtable_t` into `iree_hal_resource_t` in `device->resource`

```cpp
static const iree_hal_device_vtable_t iree_hal_task_device_vtable = {
    .destroy = iree_hal_task_device_destroy,
    .id = iree_hal_task_device_id,
    .host_allocator = iree_hal_task_device_host_allocator,
    .device_allocator = iree_hal_task_device_allocator,
    .replace_device_allocator = iree_hal_task_replace_device_allocator,
    .replace_channel_provider = iree_hal_task_replace_channel_provider,
    .trim = iree_hal_task_device_trim,
    .query_i64 = iree_hal_task_device_query_i64,
    .create_channel = iree_hal_task_device_create_channel,
    .create_command_buffer = iree_hal_task_device_create_command_buffer,
    .create_descriptor_set_layout =
        iree_hal_task_device_create_descriptor_set_layout,
    .create_event = iree_hal_task_device_create_event,
    .create_executable_cache = iree_hal_task_device_create_executable_cache,
    .import_file = iree_hal_task_device_import_file,
    .create_pipeline_layout = iree_hal_task_device_create_pipeline_layout,
    .create_semaphore = iree_hal_task_device_create_semaphore,
    .query_semaphore_compatibility =
        iree_hal_task_device_query_semaphore_compatibility,
    .queue_alloca = iree_hal_task_device_queue_alloca,
    .queue_dealloca = iree_hal_task_device_queue_dealloca,
    .queue_read = iree_hal_task_device_queue_read,
    .queue_write = iree_hal_task_device_queue_write,
    .queue_execute = iree_hal_task_device_queue_execute,
    .queue_flush = iree_hal_task_device_queue_flush,
    .wait_semaphores = iree_hal_task_device_wait_semaphores,
    .profiling_begin = iree_hal_task_device_profiling_begin,
    .profiling_flush = iree_hal_task_device_profiling_flush,
    .profiling_end = iree_hal_task_device_profiling_end,
};
```

After this, it will also initialize the buffer, queue, memory block pool, and loaders.

## create cuda device

```log
main (/root/Desktop/dockerVolumn/iree/tools/iree-run-module-main.c:43)
iree_tooling_run_module_from_flags (/root/Desktop/dockerVolumn/iree/runtime/src/iree/tooling/run_module.c:387)
iree_tooling_run_module_with_data (/root/Desktop/dockerVolumn/iree/runtime/src/iree/tooling/run_module.c:404)
iree_tooling_create_run_context (/root/Desktop/dockerVolumn/iree/runtime/src/iree/tooling/run_module.c:151)
iree_tooling_create_context_from_flags (/root/Desktop/dockerVolumn/iree/runtime/src/iree/tooling/context_util.c:610)
iree_tooling_resolve_modules (/root/Desktop/dockerVolumn/iree/runtime/src/iree/tooling/context_util.c:485)
iree_vm_module_enumerate_dependencies (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/module.c:278)
iree_vm_bytecode_module_enumerate_dependencies (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/bytecode/module.c:232)
iree_tooling_resolve_module_dependency_callback (/root/Desktop/dockerVolumn/iree/runtime/src/iree/tooling/context_util.c:425)
iree_tooling_load_hal_async_module (/root/Desktop/dockerVolumn/iree/runtime/src/iree/tooling/context_util.c:204)
iree_hal_create_devices_from_flags (/root/Desktop/dockerVolumn/iree/runtime/src/iree/tooling/device_util.c:392)
iree_hal_create_device (/root/Desktop/dockerVolumn/iree/runtime/src/iree/hal/driver_registry.c:350)
iree_hal_driver_create_device_by_uri (/root/Desktop/dockerVolumn/iree/runtime/src/iree/hal/driver.c:159)
iree_hal_driver_create_device_by_path (/root/Desktop/dockerVolumn/iree/runtime/src/iree/hal/driver.c:120)
iree_hal_cuda_driver_create_device_by_path (/root/Desktop/dockerVolumn/iree/runtime/src/iree/hal/drivers/cuda/cuda_driver.c:596)
iree_hal_cuda_driver_create_device_by_id (/root/Desktop/dockerVolumn/iree/runtime/src/iree/hal/drivers/cuda/cuda_driver.c:472)
iree_hal_cuda_device_create (/root/Desktop/dockerVolumn/iree/runtime/src/iree/hal/drivers/cuda/cuda_device.c:403)
iree_hal_cuda_device_create_internal (/root/Desktop/dockerVolumn/iree/runtime/src/iree/hal/drivers/cuda/cuda_device.c:334)
```

Here the logic in `iree_hal_cuda_driver_create_device_by_id`:

1. cuda init
2. select default device
3. create the iree cuda device struct


Here the logic in `iree_hal_cuda_device_create`:

1. check the cuda device is valid
2. create the cuda context
3. create the cuda stream
4. call `iree_hal_cuda_device_create_internal` to create the device struct
5. create iree host event pool allocator
6. create the iree cuda event pool allocator
7. create the iree cuda timepoint pool allocator


Here the logic in `iree_hal_cuda_device_create_internal`:

1. Initalize the resource with `iree_hal_resource_initialize` and register the `iree_hal_cuda_device_vtable` in `device->resource`

```cpp
static const iree_hal_device_vtable_t iree_hal_cuda_device_vtable = {
    .destroy = iree_hal_cuda_device_destroy,
    .id = iree_hal_cuda_device_id,
    .host_allocator = iree_hal_cuda_device_host_allocator,
    .device_allocator = iree_hal_cuda_device_allocator,
    .replace_device_allocator = iree_hal_cuda_replace_device_allocator,
    .replace_channel_provider = iree_hal_cuda_replace_channel_provider,
    .trim = iree_hal_cuda_device_trim,
    .query_i64 = iree_hal_cuda_device_query_i64,
    .create_channel = iree_hal_cuda_device_create_channel,
    .create_command_buffer = iree_hal_cuda_device_create_command_buffer,
    .create_descriptor_set_layout =
        iree_hal_cuda_device_create_descriptor_set_layout,
    .create_event = iree_hal_cuda_device_create_event,
    .create_executable_cache = iree_hal_cuda_device_create_executable_cache,
    .import_file = iree_hal_cuda_device_import_file,
    .create_pipeline_layout = iree_hal_cuda_device_create_pipeline_layout,
    .create_semaphore = iree_hal_cuda_device_create_semaphore,
    .query_semaphore_compatibility =
        iree_hal_cuda_device_query_semaphore_compatibility,
    .queue_alloca = iree_hal_cuda_device_queue_alloca,
    .queue_dealloca = iree_hal_cuda_device_queue_dealloca,
    .queue_read = iree_hal_cuda_device_queue_read,
    .queue_write = iree_hal_cuda_device_queue_write,
    .queue_execute = iree_hal_cuda_device_queue_execute,
    .queue_flush = iree_hal_cuda_device_queue_flush,
    .wait_semaphores = iree_hal_cuda_device_wait_semaphores,
    .profiling_begin = iree_hal_cuda_device_profiling_begin,
    .profiling_flush = iree_hal_cuda_device_profiling_flush,
    .profiling_end = iree_hal_cuda_device_profiling_end,
};
```

2. create the arena pool
3. move the driver info, context, symbol etc to the device struct
4. register the vtable `iree_hal_cuda_deferred_work_queue_device_interface_vtable` for device_interface

```cpp
static const iree_hal_deferred_work_queue_device_interface_vtable_t
    iree_hal_cuda_deferred_work_queue_device_interface_vtable = {
        .destroy = iree_hal_cuda_deferred_work_queue_device_interface_destroy,
        .bind_to_thread =
            iree_hal_cuda_deferred_work_queue_device_interface_bind_to_thread,
        .wait_native_event =
            iree_hal_cuda_deferred_work_queue_device_interface_wait_native_event,
        .create_native_event =
            iree_hal_cuda_deferred_work_queue_device_interface_create_native_event,
        .record_native_event =
            iree_hal_cuda_deferred_work_queue_device_interface_record_native_event,
        .synchronize_native_event =
            iree_hal_cuda_deferred_work_queue_device_interface_synchronize_native_event,
        .destroy_native_event =
            iree_hal_cuda_deferred_work_queue_device_interface_destroy_native_event,
        .semaphore_acquire_timepoint_device_signal_native_event =
            iree_hal_cuda_deferred_work_queue_device_interface_semaphore_acquire_timepoint_device_signal_native_event,
        .acquire_host_wait_event =
            iree_hal_cuda_deferred_work_queue_device_interface_acquire_host_wait_event,
        .device_wait_on_host_event =
            iree_hal_cuda_deferred_work_queue_device_interface_device_wait_on_host_event,
        .release_wait_event =
            iree_hal_cuda_deferred_work_queue_device_interface_release_wait_event,
        .native_event_from_wait_event =
            iree_hal_cuda_deferred_work_queue_device_interface_native_event_from_wait_event,
        .create_stream_command_buffer =
            iree_hal_cuda_deferred_work_queue_device_interface_create_stream_command_buffer,
        .submit_command_buffer =
            iree_hal_cuda_deferred_work_queue_device_interface_submit_command_buffer,
};
```
5. create an queue with `iree_hal_deferred_work_queue_create`, in this function will create an queue and threads.
    1. create a queue with work queue and complete queue
    2. start the threads with `iree_hal_deferred_work_queue_worker_execute` and `iree_hal_deferred_work_queue_completion_execute`

6. if tracing is enabled, create the tracing event
7. create the memory pool allocator

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

## VMVX Loader

### Initalize the vm module

```text
main(int argc, char ** argv) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:28)
iree_runtime_demo_main() (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:94)
iree_runtime_demo_run_session(iree_runtime_instance_t * instance) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:125)
iree_runtime_instance_try_create_default_device(iree_runtime_instance_t * instance, iree_string_view_t driver_name, iree_hal_device_t ** out_device) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/instance.c:158)
iree_hal_driver_registry_try_create(iree_hal_driver_registry_t * registry, iree_string_view_t driver_name, iree_allocator_t host_allocator, iree_hal_driver_t ** out_driver) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/hal/driver_registry.c:314)
iree_hal_local_task_driver_factory_try_create(void * self, iree_string_view_t driver_name, iree_allocator_t host_allocator, iree_hal_driver_t ** out_driver) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/hal/drivers/local_task/registration/driver_module.c:71)
iree_hal_create_all_available_executable_loaders(iree_hal_executable_plugin_manager_t * plugin_manager, iree_host_size_t capacity, iree_host_size_t * out_count, iree_hal_executable_loader_t ** loaders, iree_allocator_t host_allocator) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/hal/local/loaders/registration/init.c:69)
iree_hal_vmvx_module_loader_create_isolated(iree_host_size_t user_module_count, iree_vm_module_t ** user_modules, iree_allocator_t host_allocator, iree_hal_executable_loader_t ** out_executable_loader) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/hal/local/loaders/vmvx_module_loader.c:620)
iree_hal_vmvx_module_loader_create(iree_vm_instance_t * instance, iree_host_size_t user_module_count, iree_vm_module_t ** user_modules, iree_allocator_t host_allocator, iree_hal_executable_loader_t ** out_executable_loader) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/hal/local/loaders/vmvx_module_loader.c:572)
iree_vmvx_module_create(iree_vm_instance_t * instance, iree_allocator_t host_allocator, iree_vm_module_t ** out_module) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/modules/vmvx/module.c:808)
iree_vm_native_module_initialize(const iree_vm_module_t * module_interface, const iree_vm_native_module_descriptor_t * module_descriptor, iree_vm_instance_t * instance, iree_allocator_t allocator, iree_vm_module_t * base_module) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/native_module.c:488)
```

In here, the `iree_hal_vmvx_module_loader_create` function will create the executable loader for the vmvx module.
copy the `iree_hal_vmvx_module_loader_vtable` to the `iree_hal_executable_loader_t` structure.

```cpp
static const iree_hal_executable_loader_vtable_t
    iree_hal_vmvx_module_loader_vtable = {
        .destroy = iree_hal_vmvx_module_loader_destroy,
        .query_support = iree_hal_vmvx_module_loader_query_support,
        .try_load = iree_hal_vmvx_module_loader_try_load,
};
```

Here, initalize the vm module and assign the interface to the module.

```cpp
  // Base interface that routes through our thunks.
  iree_vm_module_initialize(&module->base_interface, module);
  module->base_interface.destroy = iree_vm_native_module_destroy;
  module->base_interface.name = iree_vm_native_module_name;
  module->base_interface.signature = iree_vm_native_module_signature;
  module->base_interface.get_module_attr =
      iree_vm_native_module_get_module_attr;
  module->base_interface.enumerate_dependencies =
      iree_vm_native_module_enumerate_dependencies;
  module->base_interface.lookup_function =
      iree_vm_native_module_lookup_function;
  module->base_interface.get_function = iree_vm_native_module_get_function;
  module->base_interface.get_function_attr =
      iree_vm_native_module_get_function_attr;
  module->base_interface.alloc_state = iree_vm_native_module_alloc_state;
  module->base_interface.free_state = iree_vm_native_module_free_state;
  module->base_interface.resolve_import = iree_vm_native_module_resolve_import;
  module->base_interface.notify = iree_vm_native_module_notify;
  module->base_interface.begin_call = iree_vm_native_module_begin_call;
  module->base_interface.resume_call = iree_vm_native_module_resume_call;
```

### hal executable loader initialization

```text
iree_hal_vmvx_module_loader_try_load@runtime/src/iree/hal/local/loaders/vmvx_module_loader.c:655
iree_vm_bytecode_module_create@runtime/src/iree/vm/bytecode/module.c:799
```

Here do the following things:

1. check the flatbuffer is valid
2. initalized the model function pointer:

```cpp
  iree_vm_module_initialize(&module->interface, module);
  module->interface.destroy = iree_vm_bytecode_module_destroy;
  module->interface.name = iree_vm_bytecode_module_name;
  module->interface.signature = iree_vm_bytecode_module_signature;
  module->interface.get_module_attr = iree_vm_bytecode_module_get_module_attr;
  module->interface.enumerate_dependencies =
      iree_vm_bytecode_module_enumerate_dependencies;
  module->interface.lookup_function = iree_vm_bytecode_module_lookup_function;
  module->interface.get_function = iree_vm_bytecode_module_get_function;
  module->interface.get_function_attr =
      iree_vm_bytecode_module_get_function_attr;
#if IREE_VM_BACKTRACE_ENABLE
  module->interface.resolve_source_location =
      iree_vm_bytecode_module_resolve_source_location;
#endif  // IREE_VM_BACKTRACE_ENABLE
  module->interface.alloc_state = iree_vm_bytecode_module_alloc_state;
  module->interface.free_state = iree_vm_bytecode_module_free_state;
  module->interface.resolve_import = iree_vm_bytecode_module_resolve_import;
  module->interface.notify = iree_vm_bytecode_module_notify;
  module->interface.begin_call = iree_vm_bytecode_module_begin_call;
  module->interface.resume_call = iree_vm_bytecode_module_resume_call;

```

### create hal executable

```text
iree_hal_vmvx_module_loader_try_load@runtime/src/iree/hal/local/loaders/vmvx_module_loader.c:655
iree_hal_vmvx_executable_create@runtime/src/iree/hal/local/loaders/vmvx_module_loader.c:215
```

Here assign the `iree_hal_local_executable_vtable_t` to `iree_hal_vmvx_executable_t` structure.

```cpp
static const iree_hal_local_executable_vtable_t
    iree_hal_vmvx_executable_vtable = {
        .base =
            {
                .destroy = iree_hal_vmvx_executable_destroy,
            },
        .issue_call = iree_hal_vmvx_executable_issue_call,
};
```

and then Find all the entry points and move them to `entry_fn_ordinals` field in the executable.
after that, Query the optional local workgroup size from each entry point.
Finally, Initialize a context per worker requested.


## Create a session

```text
main(int argc, char ** argv) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:28)
iree_runtime_demo_main() (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:94)
iree_runtime_demo_run_session(iree_runtime_instance_t * instance) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:135)
iree_runtime_session_create_with_device(iree_runtime_instance_t * instance, const iree_runtime_session_options_t * options, iree_hal_device_t * device, iree_allocator_t host_allocator, iree_runtime_session_t ** out_session) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/session.c:94)
```

Here:

1. Create a session with empty vm context.
2. Add the HAL initialized module
3. VM context register HAL modules

  1. iree hal resource initialize

  ```text
  main(int argc, char ** argv) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:28)
  iree_runtime_demo_main() (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:94)
  iree_runtime_demo_run_session(iree_runtime_instance_t * instance) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:135)
  iree_runtime_session_create_with_device(iree_runtime_instance_t * instance, const iree_runtime_session_options_t * options, iree_hal_device_t * device, iree_allocator_t host_allocator, iree_runtime_session_t ** out_session) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/session.c:102)
  iree_vm_context_register_modules(iree_vm_context_t * context, iree_host_size_t module_count, iree_vm_module_t ** modules) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/context.c:501)
  iree_vm_native_module_alloc_state(void * self, iree_allocator_t allocator, iree_vm_module_state_t ** out_module_state) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/native_module.c:295) <- Here is called by `module->alloc_state`
  iree_hal_module_alloc_state(void * self, iree_allocator_t host_allocator, iree_vm_module_state_t ** out_module_state) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/modules/hal/module.c:103)
  iree_hal_executable_cache_create(iree_hal_device_t * device, iree_string_view_t identifier, iree_loop_t loop, iree_hal_executable_cache_t ** out_executable_cache) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/hal/executable_cache.c:37)
  iree_hal_task_device_create_executable_cache(iree_hal_device_t * base_device, iree_string_view_t identifier, iree_loop_t loop, iree_hal_executable_cache_t ** out_executable_cache) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/hal/drivers/local_task/task_device.c:350)
  iree_hal_local_executable_cache_create(iree_string_view_t identifier, iree_host_size_t worker_capacity, iree_host_size_t loader_count, iree_hal_executable_loader_t ** loaders, iree_allocator_t host_allocator, iree_hal_executable_cache_t ** out_executable_cache) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/hal/local/local_executable_cache.c:49)
  ```

  Here the `iree_hal_resource_initialize` will register the vtable `iree_hal_executable_cache_vtable_t` into the `executable_cache->resource`

  ```cpp
  static const iree_hal_executable_cache_vtable_t
      iree_hal_local_executable_cache_vtable = {
          .destroy = iree_hal_local_executable_cache_destroy,
          .can_prepare_format =
              iree_hal_local_executable_cache_can_prepare_format,
          .prepare_executable =
              iree_hal_local_executable_cache_prepare_executable,
  };
  ```

  For `iree_hal_local_executable_cache_can_prepare_format` function, it just prepare the executable format like "embedded-elf-<targe>" strings.
  For `iree_hal_executable_loader_try_load` function, it will try to load the executable from the loader.

  1. vm context resolve module imports

  ```text
  main(int argc, char ** argv) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:28)
  iree_runtime_demo_main() (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:94)
  iree_runtime_demo_run_session(iree_runtime_instance_t * instance) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:135)
  iree_runtime_session_create_with_device(iree_runtime_instance_t * instance, const iree_runtime_session_options_t * options, iree_hal_device_t * device, iree_allocator_t host_allocator, iree_runtime_session_t ** out_session) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/session.c:102)
  iree_vm_context_register_modules(iree_vm_context_t * context, iree_host_size_t module_count, iree_vm_module_t ** modules) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/context.c:501)
  iree_vm_context_resolve_module_imports@runtime/src/iree/vm/context.c:160
  ```

     1. Check module presence/versions before individual imports.
     2. Find the symbol that we can call using the `lookup_function` function.

4. Run the `@__init` Funtion in the vm module.

```text
main(int argc, char ** argv) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:28)
iree_runtime_demo_main() (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:94)
iree_runtime_demo_run_session(iree_runtime_instance_t * instance) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:135)
iree_runtime_session_create_with_device(iree_runtime_instance_t * instance, const iree_runtime_session_options_t * options, iree_hal_device_t * device, iree_allocator_t host_allocator, iree_runtime_session_t ** out_session) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/session.c:102)
iree_vm_context_register_modules(iree_vm_context_t * context, iree_host_size_t module_count, iree_vm_module_t ** modules) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/context.c:525)
iree_vm_context_run_function(iree_vm_context_t * context, iree_vm_stack_t * stack, iree_vm_module_t * module, iree_string_view_t function_name) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/context.c:73) <-- Here run the `@__init` function in the vm module.
```

But, for HAL modules, it will not run the `@__init` function since there is no such function in the HAL module.

5. iree_vm_context_resolve_module_state: find and assign the module state to the `hal_module_state`.

## Load module

iree_runtime_session_append_bytecode_module_from_memory@runtime/src/iree/runtime/demo/hello_world_terse.c:57
  -> iree_vm_bytecode_module_create@runtime/src/iree/vm/bytecode/module.c:799
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

      3. append the module to the session

```log
main(int argc, char ** argv) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:28)
iree_runtime_demo_main() (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:94)
iree_runtime_demo_run_session(iree_runtime_instance_t * instance) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:144)
iree_runtime_demo_load_module(iree_runtime_session_t * session) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:34)
iree_runtime_session_append_bytecode_module_from_file(iree_runtime_session_t * session, const char * file_path) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/session.c:249)
iree_runtime_session_append_bytecode_module_from_memory(iree_runtime_session_t * session, iree_const_byte_span_t flatbuffer_data, iree_allocator_t flatbuffer_allocator) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/session.c:220)
iree_runtime_session_append_module(iree_runtime_session_t * session, iree_vm_module_t * module) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/session.c:197)
iree_vm_context_register_modules(iree_vm_context_t * context, iree_host_size_t module_count, iree_vm_module_t ** modules) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/context.c:437)
```

This is the same the process in the session creation (`iree_vm_context_register_modules`).

1. The `iree_vm_context_resolve_module_imports` function will check the dependency:

```cpp
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_vm_module_enumerate_dependencies(
              module, iree_vm_context_check_module_dependency, context));
```

call frame:
```log
main(int argc, char ** argv) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:28)
iree_runtime_demo_main() (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:94)
iree_runtime_demo_run_session(iree_runtime_instance_t * instance) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:144)
iree_runtime_demo_load_module(iree_runtime_session_t * session) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:34)
iree_runtime_session_append_bytecode_module_from_file(iree_runtime_session_t * session, const char * file_path) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/session.c:249)
iree_runtime_session_append_bytecode_module_from_memory(iree_runtime_session_t * session, iree_const_byte_span_t flatbuffer_data, iree_allocator_t flatbuffer_allocator) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/session.c:220)
iree_vm_context_resolve_module_imports(iree_vm_context_t * context, iree_vm_module_t * module, iree_vm_module_state_t * module_state) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/context.c:168)
iree_runtime_session_append_module(iree_runtime_session_t * session, iree_vm_module_t * module) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/session.c:197)
iree_vm_context_register_modules(iree_vm_context_t * context, iree_host_size_t module_count, iree_vm_module_t ** modules) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/context.c:510)
iree_vm_module_enumerate_dependencies(iree_vm_module_t * module, iree_vm_module_dependency_callback_t callback, void * user_data) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/module.c:278)
iree_vm_bytecode_module_enumerate_dependencies(void * self, iree_vm_module_dependency_callback_t callback, void * user_data) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/bytecode/module.c:215)
iree_vm_context_check_module_dependency(void * user_data_ptr, const iree_vm_module_dependency_t * dependency) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/context.c:126)
```

```log
# Here is from the dumped vmfb file:
Module Dependencies:
  hal, version >= 2, required
```

This is from the vmfb file that dumped from logs.

2. find the import function with the `iree_vm_native_module_lookup_function` function.

```log
main(int argc, char ** argv) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:28)
iree_runtime_demo_main() (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:94)
iree_runtime_demo_run_session(iree_runtime_instance_t * instance) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:144)
iree_runtime_demo_load_module(iree_runtime_session_t * session) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:34)
iree_runtime_session_append_bytecode_module_from_file(iree_runtime_session_t * session, const char * file_path) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/session.c:249)
iree_runtime_session_append_bytecode_module_from_memory(iree_runtime_session_t * session, iree_const_byte_span_t flatbuffer_data, iree_allocator_t flatbuffer_allocator) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/session.c:220)
iree_runtime_session_append_module(iree_runtime_session_t * session, iree_vm_module_t * module) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/session.c:197)
iree_vm_context_resolve_module_imports(iree_vm_context_t * context, iree_vm_module_t * module, iree_vm_module_state_t * module_state) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/context.c:190)
iree_vm_context_register_modules(iree_vm_context_t * context, iree_host_size_t module_count, iree_vm_module_t ** modules) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/context.c:510)
iree_vm_context_resolve_function_impl(const iree_vm_context_t * context, iree_string_view_t full_name, const iree_vm_function_signature_t * expected_signature, iree_vm_function_t * out_function) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/context.c:590)
iree_vm_native_module_lookup_function(void * self, iree_vm_function_linkage_t linkage, iree_string_view_t name, const iree_vm_function_signature_t * expected_signature, iree_vm_function_t * out_function) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/native_module.c:248)
iree_vm_native_module_get_function(void * self, iree_vm_function_linkage_t linkage, iree_host_size_t ordinal, iree_vm_function_t * out_function, iree_string_view_t * out_name, iree_vm_function_signature_t * out_signature) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/native_module.c:208)
iree_vm_native_module_get_export_function(iree_vm_native_module_t * module, iree_host_size_t ordinal, iree_vm_function_t * out_function, iree_string_view_t * out_name, iree_vm_function_signature_t * out_signature) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/native_module.c:187)
```

Here will find the imported function (list below) from the `module->descriptor->exports[ordinal]`

```log
# Here is from the dumped vmfb file:
Imported Functions:
  [  0] hal.buffer.assert(!vm.ref<?>, !vm.ref<?>, !vm.ref<?>, i64, i32, i32) -> ()
  [  1] hal.buffer_view.create(!vm.ref<?>, i64, i64, i32, i32, tuple<i64>...) -> (!vm.ref<?>)
  [  2] hal.buffer_view.assert(!vm.ref<?>, !vm.ref<?>, i32, i32, tuple<i64>...) -> ()
  [  3] hal.buffer_view.buffer(!vm.ref<?>) -> (!vm.ref<?>)
  [  4] hal.command_buffer.create(!vm.ref<?>, i32, i32, i32) -> (!vm.ref<?>)
  [  5] hal.command_buffer.finalize(!vm.ref<?>) -> ()
  [  6] hal.command_buffer.execution_barrier(!vm.ref<?>, i32, i32, i32) -> ()
  [  7] hal.command_buffer.push_descriptor_set(!vm.ref<?>, !vm.ref<?>, i32, tuple<i32, i32, !vm.ref<?>, i64, i64>...) -> ()
  [  8] hal.command_buffer.dispatch(!vm.ref<?>, !vm.ref<?>, i32, i32, i32, i32) -> ()
  [  9] hal.descriptor_set_layout.create(!vm.ref<?>, i32, tuple<i32, i32, i32>...) -> (!vm.ref<?>)
  [ 10] hal.device.allocator(!vm.ref<?>) -> (!vm.ref<?>)
  [ 11] hal.device.query.i64(!vm.ref<?>, !vm.ref<?>, !vm.ref<?>) -> (i32, i64)
  [ 12] hal.device.queue.alloca(!vm.ref<?>, i64, !vm.ref<?>, !vm.ref<?>, i32, i32, i32, i64) -> (!vm.ref<?>)
  [ 13] hal.device.queue.execute(!vm.ref<?>, i64, !vm.ref<?>, !vm.ref<?>, tuple<!vm.ref<?>>...) -> ()
  [ 14] hal.devices.get(i32) -> (!vm.ref<?>)
  [ 15] hal.executable.create(!vm.ref<?>, !vm.ref<?>, !vm.ref<?>, !vm.ref<?>, tuple<!vm.ref<?>>...) -> (!vm.ref<?>)
  [ 16] hal.fence.create(!vm.ref<?>, i32) -> (!vm.ref<?>)
  [ 17] hal.fence.await(i32, tuple<!vm.ref<?>>...) -> (i32)
  [ 18] hal.pipeline_layout.create(!vm.ref<?>, i32, tuple<!vm.ref<?>>...) -> (!vm.ref<?>)
```

However, the bytecode module has the `@__init` function. so this time will run the `@__init` function in the vm module.

## Call the function

1. prepare the call stack, frame, registers (simulated), and call results.
2. call the `iree_vm_bytecode_dispatch`

```log
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
iree_vm_bytecode_dispatch(iree_vm_stack_t * restrict stack, iree_vm_bytecode_module_t * restrict module, iree_vm_stack_frame_t * restrict current_frame, iree_vm_registers_t regs, iree_byte_span_t call_results) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/bytecode/dispatch.c:673)
```

In the `iree_vm_bytecode_dispatch` function, the function will be called by `bytecode_data[pc]`. The `bytecode_data` the enum number of the function, which can be found `runtime/src/iree/vm/bytecode/utils/generated/op_table.h`

`bytedata` pc will be aligned with `uint16_t` 4 bytes.

For example:

before run:

```
bytedata[0] = `block`
bytedata[1] = `vm.const.ref.zero`

bytedata[4] = `0xd`: `vm.const.i32`
bytedata[5] = `2`
bytedata[6] = `0`
bytedata[7] = `0`
bytedata[8] = `0`
bytedata[9] = `0`
bytedata[10] = `0`
bytedata[11] = `0xd`: vm.const.i32
---
regs_i32[0] = 0
regs_i32[1] = 0

```

After run the code unfold with macro:

```cpp
// pc = 5
DISPATCH_OP(CORE, ConstI32, {
  int32_t value = iree_unaligned_load_le((uint32_t*)&bytecode_data[pc + (0)]);
  pc += 4;
  int32_t* result = &regs_i32[iree_unaligned_load_le((uint16_t*)&bytecode_data[pc + (0)])];
  pc += 2;;
  *result = value;
});
```

This see that value will be the `bytedata[5] = 2`
and `pc = 9` and then the result will be written
to the `regs_i32[bytedata[9]]` which is the first register.

For Call operator:

```cpp
// pc = 49
bytecode_data[49] = `0x58`: `vm.call`
bytecode_data[50] = `14`: `function_ordinal` // This is the ordinal number of the "Imported Functions" in the vmfb dumped file.
// the `%ref = vm.call @hal.devices.get(%zero_0) {live = ["%c-1", "%c1", "%c14", "%c2", "%c7", "%null", "%ref", "%zero", "%zero_0"], nosideeffects, result_registers = ["r1"]} : (i32) -> !vm.ref<!hal.device>` in the vm.mlir file
*(const iree_vm_register_list_t*)&bytecode_data[54] == {size = 1, registers = 0x555555750f78} : `operands`
*(const iree_vm_register_list_t*)&bytecode_data[58] == {size = 1, registers = 0x555555750f7c} : `result`
```

before issue the import call, the cconv argument should be assigned with values:
the `src_reg_list->registers` will record the caller registers offset.

```cpp
caller_registers.i32[src_reg_list->registers[0]] == 0 // %zero_0
```

and call the `hal.devices.get`. Here is the call stack of the running function `hal.devices.get` will be:

```log
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

Here is the call stack through the `hal.devices.get` function.

```log
iree_hal_module_devices_get(iree_vm_stack_t * restrict stack, void * restrict module, iree_hal_module_state_t * restrict state, iree_vm_abi_i_t * restrict args, iree_vm_abi_r_t * restrict rets) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/modules/hal/module.c:1178)
iree_vm_shim_i_r(iree_vm_stack_t * restrict stack, iree_vm_native_function_flags_t flags, iree_byte_span_t args_storage, iree_byte_span_t rets_storage, iree_vm_native_function_target2_t target_fn, void * restrict module, void * restrict module_state) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/shims.c:10)
iree_vm_native_module_issue_call(iree_vm_native_module_t * module, iree_vm_stack_t * stack, iree_vm_stack_frame_t * callee_frame, iree_vm_native_function_flags_t flags, iree_byte_span_t args_storage, iree_byte_span_t rets_storage) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/native_module.c:346)
iree_vm_native_module_begin_call(void * self, iree_vm_stack_t * stack, iree_vm_function_call_t call) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/native_module.c:402)
iree_vm_bytecode_issue_import_call(iree_vm_stack_t * stack, const iree_vm_function_call_t call, iree_string_view_t cconv_results, const iree_vm_register_list_t * restrict dst_reg_list, iree_vm_stack_frame_t * restrict * out_caller_frame, iree_vm_registers_t * out_caller_registers) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/bytecode/dispatch.c:452)
iree_vm_bytecode_call_import(iree_vm_stack_t * stack, const iree_vm_bytecode_module_state_t * module_state, uint32_t import_ordinal, const iree_vm_registers_t caller_registers, const iree_vm_register_list_t * restrict src_reg_list, const iree_vm_register_list_t * restrict dst_reg_list, iree_vm_stack_frame_t * restrict * out_caller_frame, iree_vm_registers_t * out_caller_registers) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/bytecode/dispatch.c:568)
iree_vm_bytecode_dispatch(iree_vm_stack_t * restrict stack, iree_vm_bytecode_module_t * restrict module, iree_vm_stack_frame_t * restrict current_frame, iree_vm_registers_t regs, iree_byte_span_t call_results) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/bytecode/dispatch.c:1680)
iree_vm_bytecode_dispatch_begin(iree_vm_stack_t * stack, iree_vm_bytecode_module_t * module, const iree_vm_function_call_t call, iree_string_view_t cconv_arguments, iree_string_view_t cconv_results) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/bytecode/dispatch.c:636)
iree_vm_bytecode_module_begin_call(void * self, iree_vm_stack_t * stack, iree_vm_function_call_t call) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/bytecode/module.c:788)
iree_vm_context_run_function(iree_vm_context_t * context, iree_vm_stack_t * stack, iree_vm_module_t * module, iree_string_view_t function_name) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/context.c:91)
iree_vm_context_register_modules(iree_vm_context_t * context, iree_host_size_t module_count, iree_vm_module_t ** modules) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/context.c:525)
iree_runtime_session_append_module(iree_runtime_session_t * session, iree_vm_module_t * module) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/session.c:197)
iree_runtime_session_append_bytecode_module_from_memory(iree_runtime_session_t * session, iree_const_byte_span_t flatbuffer_data, iree_allocator_t flatbuffer_allocator) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/session.c:220)
iree_runtime_session_append_bytecode_module_from_file(iree_runtime_session_t * session, const char * file_path) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/session.c:249)
iree_runtime_demo_load_module(iree_runtime_session_t * session) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:34)
iree_runtime_demo_run_session(iree_runtime_instance_t * instance) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:144)
iree_runtime_demo_main() (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:94)
main(int argc, char ** argv) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:28)
```

After `call.function.module->begin_call` in the `iree_vm_native_module_issue_call`. The `stack` and storage will be update. and copy the result to the caller registers.

## HAL create cuda device executable

```log
main (/root/Desktop/dockerVolumn/iree/tools/iree-run-module-main.c:43)
...
iree_vm_context_create_with_modules (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/context.c:340)
iree_vm_context_register_modules (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/context.c:525)
iree_vm_context_run_function (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/context.c:91)
iree_vm_bytecode_module_begin_call (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/bytecode/module.c:788)
iree_vm_bytecode_dispatch_begin (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/bytecode/dispatch.c:636)
iree_vm_bytecode_dispatch (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/bytecode/dispatch.c:1716)
iree_vm_bytecode_call_import_variadic (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/bytecode/dispatch.c:609)
iree_vm_bytecode_issue_import_call (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/bytecode/dispatch.c:452)
iree_vm_native_module_begin_call (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/native_module.c:402)
iree_vm_native_module_issue_call (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/native_module.c:346)
iree_vm_shim_rrrrCrD_r (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/shims.c:50)
iree_hal_module_executable_create (/root/Desktop/dockerVolumn/iree/runtime/src/iree/modules/hal/module.c:1490)
```

Here is the logic of the `iree_hal_module_executable_create` function:

1. validate the ro data and extract the executable data format.
2. lookup the executable cache with `iree_hal_module_state_lookup_executable_cache`
3. do somethings with pipeline_layout works.
4. prepare the executable data with `iree_hal_executable_cache_prepare_executable`
   This will call _VTABlE_DISPATCH macro inside the `iree_hal_executable_cache_prepare_executable` function.

    ```cpp
    _VTABLE_DISPATCH(executable_cache, prepare_executable)(
        executable_cache, executable_params, out_executable);
    // This marco above will be expanded to:
    ((const iree_hal_executable_cache_vtable_t*)((const iree_hal_resource_t*)(executable_cache))
        ->vtable)
        ->prepare_executable
    ```

    the `iree_hal_cuda_executable_cache_vtable` will be used because the resource is cuda driver.
    and then, `iree_hal_cuda_native_executable_create` will be called inside the `iree_hal_cuda_nop_executable_cache_prepare_executable` function.
    The call stack is like below:

    ```log
    ...
    iree_hal_module_executable_create (/root/Desktop/dockerVolumn/iree/runtime/src/iree/modules/hal/module.c:1489)
    iree_hal_executable_cache_prepare_executable (/root/Desktop/dockerVolumn/iree/runtime/src/iree/hal/executable_cache.c:64)
    iree_hal_cuda_nop_executable_cache_prepare_executable (/root/Desktop/dockerVolumn/iree/runtime/src/iree/hal/drivers/cuda/nop_executable_cache.c:91)
    iree_hal_cuda_native_executable_create (/root/Desktop/dockerVolumn/iree/runtime/src/iree/hal/drivers/cuda/native_executable.c:123)
    ```

    In the function `iree_hal_cuda_native_executable_create` :
    1. will valid the executable data format and extract the executable data.
    2. load ptx data with `cuModuleLoadDataEx` nvidia symbol.

    ```cpp
    iree_status_t status = IREE_CURESULT_TO_STATUS(
    symbols, cuModuleLoadDataEx(&module, ptx_image, 0, NULL, NULL),
    "cuModuleLoadDataEx");
    ```
    3. get the kernel function with `cuModuleGetFunction` nvidia symbol during the `executable` setting up.
    4. set the max shard memory and function attributes with `cuFuncSetAttribute` nvidia symbol during the `executable` setting up.


## For Cuda device-like diver creation

```log
main (/root/Desktop/dockerVolumn/iree/tools/iree-run-module-main.c:43)
iree_tooling_run_module_from_flags (/root/Desktop/dockerVolumn/iree/runtime/src/iree/tooling/run_module.c:387)
iree_tooling_run_module_with_data (/root/Desktop/dockerVolumn/iree/runtime/src/iree/tooling/run_module.c:404)
iree_tooling_create_run_context (/root/Desktop/dockerVolumn/iree/runtime/src/iree/tooling/run_module.c:151)
iree_tooling_create_context_from_flags (/root/Desktop/dockerVolumn/iree/runtime/src/iree/tooling/context_util.c:610)
iree_tooling_resolve_modules (/root/Desktop/dockerVolumn/iree/runtime/src/iree/tooling/context_util.c:485)
iree_vm_module_enumerate_dependencies (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/module.c:278)
iree_vm_bytecode_module_enumerate_dependencies (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/bytecode/module.c:232)
iree_tooling_resolve_module_dependency_callback (/root/Desktop/dockerVolumn/iree/runtime/src/iree/tooling/context_util.c:425)
iree_tooling_load_hal_async_module (/root/Desktop/dockerVolumn/iree/runtime/src/iree/tooling/context_util.c:204)
iree_hal_create_devices_from_flags (/root/Desktop/dockerVolumn/iree/runtime/src/iree/tooling/device_util.c:392)
iree_hal_create_device (/root/Desktop/dockerVolumn/iree/runtime/src/iree/hal/driver_registry.c:342)
iree_hal_driver_registry_try_create (/root/Desktop/dockerVolumn/iree/runtime/src/iree/hal/driver_registry.c:314)
iree_hal_cuda_driver_factory_try_create (/root/Desktop/dockerVolumn/iree/runtime/src/iree/hal/drivers/cuda/registration/driver_module.c:118)
iree_hal_cuda_driver_create (/root/Desktop/dockerVolumn/iree/runtime/src/iree/hal/drivers/cuda/cuda_driver.c:111)
iree_hal_cuda_driver_create_internal (/root/Desktop/dockerVolumn/iree/runtime/src/iree/hal/drivers/cuda/cuda_driver.c:79)
iree_hal_cuda_dynamic_symbols_initialize (/root/Desktop/dockerVolumn/iree/runtime/src/iree/hal/drivers/cuda/cuda_dynamic_symbols.c:61)
```

Here the cuda symbol will be loaded with `iree_hal_cuda_dynamic_symbols_initialize`

1. load cuda symbols with `iree_dynamic_library_load_from_files` (with dlopen `libcuda.so`)
2. resolve the symbols with `iree_dynamic_library_lookup_symbol` (with dlsym)
    1. find the `cuGetProcAddress` function. This function will be used to load the cuda driver functions.
    2. lookup the cuda functions with `cuGetProcAddress` function.
3. load nccl symbols with `iree_dynamic_library_load_from_files` (with dlopen `libnccl.so`)

<!-- ## Debug call stack information


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



------------------------

```log
iree_vm_native_module_get_export_function(iree_vm_native_module_t * module, iree_host_size_t ordinal, iree_vm_function_t * out_function, iree_string_view_t * out_name, iree_vm_function_signature_t * out_signature) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/native_module.c:176)
iree_vm_native_module_get_function(void * self, iree_vm_function_linkage_t linkage, iree_host_size_t ordinal, iree_vm_function_t * out_function, iree_string_view_t * out_name, iree_vm_function_signature_t * out_signature) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/native_module.c:208)
iree_vm_native_module_lookup_function(void * self, iree_vm_function_linkage_t linkage, iree_string_view_t name, const iree_vm_function_signature_t * expected_signature, iree_vm_function_t * out_function) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/native_module.c:272)
iree_vm_context_resolve_function_impl(const iree_vm_context_t * context, iree_string_view_t full_name, const iree_vm_function_signature_t * expected_signature, iree_vm_function_t * out_function) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/context.c:590)
iree_vm_context_resolve_module_imports(iree_vm_context_t * context, iree_vm_module_t * module, iree_vm_module_state_t * module_state) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/context.c:190)
iree_vm_context_register_modules(iree_vm_context_t * context, iree_host_size_t module_count, iree_vm_module_t ** modules) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/vm/context.c:510)
iree_runtime_session_append_module(iree_runtime_session_t * session, iree_vm_module_t * module) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/session.c:197)
iree_runtime_session_append_bytecode_module_from_memory(iree_runtime_session_t * session, iree_const_byte_span_t flatbuffer_data, iree_allocator_t flatbuffer_allocator) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/session.c:220)
iree_runtime_session_append_bytecode_module_from_file(iree_runtime_session_t * session, const char * file_path) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/session.c:249)
iree_runtime_demo_load_module(iree_runtime_session_t * session) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:34)
iree_runtime_demo_run_session(iree_runtime_instance_t * instance) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:144)
iree_runtime_demo_main() (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:94)
main(int argc, char ** argv) (/root/Desktop/dockerVolumn/iree/runtime/src/iree/runtime/demo/hello_world_explained.c:28)
```
 -->
