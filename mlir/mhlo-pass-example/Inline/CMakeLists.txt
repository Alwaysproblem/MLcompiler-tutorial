# For a better template to copy, see examples/standalone
include_directories(include)
add_subdirectory(include)

set(LLVM_LINK_COMPONENTS Support)

set(LLVM_TARGET_DEFINITIONS mlir/Inline.td)
mlir_tablegen(Inline.inc -gen-rewriters EXTRA_INCLUDES ${MLIR_INCLUDE_DIR})
add_public_tablegen_target(InlineIncGen)

add_executable(inline main.cpp mlir/Inline.cpp)

add_mlir_pdll_library(InlinePdllIncGen mlir/Inline.pdll InlinePdll.inc
                      EXTRA_INCLUDES ${MLIR_INCLUDE_DIR})

add_dependencies(inline InlineIncGen InlinePdllIncGen)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

target_link_libraries(
  inline
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
