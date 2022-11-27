cmake_minimum_required(VERSION 3.10)

# note: fix ztd terminfo not found
project(mlir-example LANGUAGES C CXX)

find_package(LLVM CONFIG PATHS
             ${CMAKE_CURRENT_SOURCE_DIR}/third_party/lib/cmake/llvm)
find_package(MLIR CONFIG PATHS
             ${CMAKE_CURRENT_SOURCE_DIR}/third_party/lib/cmake/mlir)

include_directories(${LLVM_INCLUDE_DIR})
include_directories(${MLIR_INCLUDE_DIR})

include(LLVMDistributionSupport)
include(TableGen)
include(AddMLIR)

# note: fix the llvm::cl undefined reference problem
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fno-rtti")

add_subdirectory(Ch1)
add_subdirectory(Ch2)
add_subdirectory(Ch3)
add_subdirectory(Ch4)
add_subdirectory(Ch5)
add_subdirectory(Ch6)
add_subdirectory(Ch7)
