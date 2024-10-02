# For a better template to copy, see examples/standalone
include_directories(include)
add_subdirectory(include)

set(LLVM_LINK_COMPONENTS Support)

set(LLVM_TARGET_DEFINITIONS mlir/Outline.td)
mlir_tablegen(Outline.inc -gen-rewriters EXTRA_INCLUDES ${MLIR_INCLUDE_DIR})
add_public_tablegen_target(OutlineIncGen)

add_executable(outline main.cpp mlir/Outline.cpp)

add_mlir_pdll_library(OutlinePdllIncGen mlir/Outline.pdll OutlinePdll.inc
                      EXTRA_INCLUDES ${MLIR_INCLUDE_DIR})

add_dependencies(outline OutlineIncGen OutlinePdllIncGen)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

target_link_libraries(
  outline
  PRIVATE ${dialect_libs}
          ${conversion_libs}
          ${extension_libs}
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
