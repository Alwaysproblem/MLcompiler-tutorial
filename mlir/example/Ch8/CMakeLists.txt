# For a better template to copy, see examples/standalone
include_directories(include)
add_subdirectory(include)

set(LLVM_LINK_COMPONENTS Core Support nativecodegen OrcJIT)

set(LLVM_TARGET_DEFINITIONS mlir/ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(ToyCh8CombineIncGen)

add_executable(
  mlir-example-ch8
  toyc.cpp
  parser/AST.cpp
  mlir/MLIRGen.cpp
  mlir/Dialect.cpp
  mlir/LowerToAffineLoops.cpp
  mlir/LowerToLLVM.cpp
  mlir/ShapeInferencePass.cpp
  mlir/ToyCombine.cpp)

add_dependencies(mlir-example-ch8 ToyCh8ShapeInferenceInterfaceIncGen
                 ToyCh8OpsIncGen ToyCh8CombineIncGen)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
target_link_libraries(
  mlir-example-ch8
  PRIVATE ${dialect_libs}
          ${conversion_libs}
          ${extension_libs}
          MLIRAnalysis
          MLIRBuiltinToLLVMIRTranslation
          MLIRCallInterfaces
          MLIRCastInterfaces
          MLIRExecutionEngine
          MLIRIR
          MLIRLLVMCommonConversion
          MLIRLLVMToLLVMIRTranslation
          MLIRMemRefDialect
          MLIRParser
          MLIRPass
          MLIRSideEffectInterfaces
          MLIRTargetLLVMIRExport
          MLIRTransforms)
