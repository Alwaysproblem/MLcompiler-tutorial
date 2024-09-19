cmake_minimum_required(VERSION 3.10)

# note: fix ztd terminfo not found
project(mlir-example LANGUAGES C CXX)

# ############## For conda users.################################
find_package(LLVM CONFIG REQUIRED)
find_package(MLIR CONFIG REQUIRED)
# set(MLIR_TABLEGEN_EXE /root/anaconda3/envs/mlir/bin/mlir-tblgen)
# ##############################################################################

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Found MLIR ${MLIR_PACKAGE_VERSION}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Found MLIRTableGen: ${MLIR_TABLEGEN_EXE}")
message(STATUS "LLVM_INCLUDE_DIR include dir: ${LLVM_INCLUDE_DIR}")
message(STATUS "MLIR_INCLUDE_DIR include dir: ${MLIR_INCLUDE_DIR}")

# This is for non-conda users. find_package(LLVM CONFIG PATHS
# ${CMAKE_CURRENT_SOURCE_DIR}/third_party/lib/cmake/llvm) find_package(MLIR
# CONFIG PATHS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/lib/cmake/mlir)
# set(MLIR_TABLEGEN_EXE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/bin/mlir-tblgen)

include_directories(${LLVM_INCLUDE_DIR})
include_directories(${MLIR_INCLUDE_DIR})

include(LLVMDistributionSupport)
include(TableGen)
include(AddMLIR)
include(AddLLVM)
# include(HandleLLVMOptions)

# note: fix the llvm::cl undefined reference problem
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fno-rtti")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

add_subdirectory(Ch1)
add_subdirectory(Ch2)
add_subdirectory(Ch3)
add_subdirectory(Ch4)
add_subdirectory(Ch5)
add_subdirectory(Ch6)
add_subdirectory(Ch7)
add_subdirectory(Ch8)
add_subdirectory(transform_Ch2)
add_subdirectory(transform_Ch3)
add_subdirectory(transform_Ch4)
