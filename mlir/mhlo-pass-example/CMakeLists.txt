cmake_minimum_required(VERSION 3.10)

# note: fix ztd terminfo not found
project(mlir-example LANGUAGES C CXX)

# if(LLVM_ENABLE_LLD) set(CMAKE_C_FLAGS "-fuse-ld=lld") set(CMAKE_CXX_FLAGS
# "-fuse-ld=lld") endif()

# ############## For conda users.################################
find_package(MHLO CONFIG REQUIRED)
find_package(LLVM CONFIG REQUIRED)
find_package(MLIR CONFIG REQUIRED)
# find_package(popir CONFIG REQUIRED) find_package(popit CONFIG REQUIRED)
# find_package(poplar CONFIG REQUIRED) find_package(popef CONFIG REQUIRED)

set(MLIR_TABLEGEN_EXE ${MLIR_TABLEGEN_EXEUTABLE})
# ##############################################################################

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Found MLIR ${MLIR_PACKAGE_VERSION}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Found MLIRTableGen: ${MLIR_TABLEGEN_EXE}")
message(STATUS "LLVM_INCLUDE_DIR include dir: ${LLVM_INCLUDE_DIR}")
message(STATUS "MLIR_INCLUDE_DIR include dir: ${MLIR_INCLUDE_DIR}")
message(STATUS "Using MHLOConfig.cmake in: ${MHLO_DIR}")
message(STATUS "MHLO_INCLUDE_DIRS include dir: ${MHLO_INCLUDE_DIRS}")

if(NOT POPLAR_SDK_PATH)
  set(POPLAR_SDK_PATH ${PACKAGE_PREFIX_DIR})
endif()

message(STATUS "POPLAR_SDK_PATH include dir: ${POPLAR_SDK_PATH}")

include_directories(${MHLO_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIR})
include_directories(${MLIR_INCLUDE_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/mhlo)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/mlir-hlo/mhlo)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/mlir-hlo/stablehlo)

include(LLVMDistributionSupport)
include(TableGen)
include(AddMLIR)
include(AddLLVM)

# note: fix the llvm::cl undefined reference problem
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fno-rtti")

# add_subdirectory(popit-example) add_subdirectory(mhlo-popir)
add_subdirectory(mhlo_example)