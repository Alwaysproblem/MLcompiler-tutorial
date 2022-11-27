# For a better template to copy, see examples/standalone
add_subdirectory(include)

set(LLVM_LINK_COMPONENTS Support)

add_executable(mlir-example-ch2 toyc.cpp parser/AST.cpp mlir/MLIRGen.cpp
                                mlir/Dialect.cpp)

add_dependencies(mlir-example-ch2 ToyCh2OpsIncGen)

include_directories(include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
target_link_libraries(
  mlir-example-ch2 PRIVATE MLIRAnalysis MLIRIR MLIRParser
                           MLIRSideEffectInterfaces MLIRTransforms)
