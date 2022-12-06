# For a better template to copy, see examples/standalone
include_directories(include)
add_subdirectory(include)

set(LLVM_LINK_COMPONENTS Support)

set(LLVM_TARGET_DEFINITIONS mlir/ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(ToyCh3CombineIncGen)

add_executable(mlir-example-ch3 toyc.cpp parser/AST.cpp mlir/MLIRGen.cpp
                                mlir/Dialect.cpp mlir/ToyCombine.cpp)

add_dependencies(mlir-example-ch3 ToyCh3OpsIncGen ToyCh3CombineIncGen)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
target_link_libraries(
  mlir-example-ch3 PRIVATE MLIRAnalysis MLIRIR MLIRParser MLIRPass
                           MLIRSideEffectInterfaces MLIRTransforms)
