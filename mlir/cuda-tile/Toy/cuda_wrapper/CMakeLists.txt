add_library(cuda_shim STATIC cuda_shim.cpp)
target_include_directories(cuda_shim PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(cuda_shim PRIVATE CUDA::cuda_driver CUDA::cudart_static)
