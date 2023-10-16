# For a better template to copy, see examples/standalone
include_directories(include)
add_subdirectory(include)

set(LLVM_LINK_COMPONENTS Support)

set(LLVM_TARGET_DEFINITIONS mlir/Pow2.td)
mlir_tablegen(Pow2.inc -gen-rewriters EXTRA_INCLUDES ${MLIR_INCLUDE_DIR})
add_public_tablegen_target(Pow2IncGen)

set(LLVM_TARGET_DEFINITIONS mlir/transform/pass.td)
mlir_tablegen(Pow2Pass.inc -gen-pass-decls EXTRA_INCLUDES ${MLIR_INCLUDE_DIR})
add_public_tablegen_target(Pow2PassIncGen)

add_executable(pow2 main.cpp mlir/Pow2.cpp)

add_mlir_pdll_library(Pow2PdllIncGen mlir/Pow2.pdll Pow2Pdll.inc EXTRA_INCLUDES
                      ${MLIR_INCLUDE_DIR})

add_dependencies(pow2 Pow2IncGen Pow2PdllIncGen Pow2PassIncGen)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

target_link_libraries(
  pow2
  PRIVATE ${dialect_libs}
          MhloDialect
          MLIRAnalysis
          MLIRCallInterfaces
          MLIRCastInterfaces
          MLIRIR
          MLIRParser
          MLIRPass
          MLIRSideEffectInterfaces
          MLIRSupport
          MLIRTransforms)
