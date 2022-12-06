# For a better template to copy, see examples/standalone
set(LLVM_LINK_COMPONENTS Support)

add_executable(mlir-example-ch1 toyc.cpp parser/AST.cpp)

include_directories(include/)
target_link_libraries(mlir-example-ch1 MLIRSupport)
