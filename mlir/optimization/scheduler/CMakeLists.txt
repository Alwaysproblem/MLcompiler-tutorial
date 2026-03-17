# For a better template to copy, see examples/standalone
include_directories(include)
add_subdirectory(include)

set(LLVM_LINK_COMPONENTS Core Support nativecodegen OrcJIT)

# set(LLVM_TARGET_DEFINITIONS mlir/ToyCombine.td)
# mlir_tablegen(ToyCombine.inc -gen-rewriters)
# add_public_tablegen_target(ToyCh6CombineIncGen)

add_executable(
  lab-scheduler
  lab-opt.cpp
  lib/OpStatsPass.cpp
  lib/BufferAnalysisPass.cpp
  lib/LivenessAnalysisPass.cpp
  lib/MemrefLifetime.cpp
  lib/FusionFeasibility.cpp
  lib/LivenessAdapter.cpp
  lib/LocalListScheduling.cpp
  lib/SimpleLoopInterchange.cpp
  lib/BufferResidencyAnalysis.cpp
  )

# add_dependencies(lab-scheduler ToyCh6ShapeInferenceInterfaceIncGen
#                  ToyCh6OpsIncGen ToyCh6CombineIncGen)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
target_link_libraries(
  lab-scheduler
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
          MLIRLLVMDialect
          MLIRLLVMToLLVMIRTranslation
          MLIRMemRefDialect
          MLIRParser
          MLIRPass
          MLIRSideEffectInterfaces
          MLIRSupport
          MLIRTargetLLVMIRExport
          MLIRTransforms)
