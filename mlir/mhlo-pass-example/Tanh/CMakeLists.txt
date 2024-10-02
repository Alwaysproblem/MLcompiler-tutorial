# For a better template to copy, see examples/standalone
include_directories(include)
add_subdirectory(include)

set(LLVM_LINK_COMPONENTS Support)

set(LLVM_TARGET_DEFINITIONS mlir/Tanh.td)
mlir_tablegen(Tanh.inc -gen-rewriters EXTRA_INCLUDES ${MLIR_INCLUDE_DIR})
add_public_tablegen_target(TanhIncGen)

add_executable(tanh main.cpp mlir/Tanh.cpp)

add_dependencies(tanh TanhIncGen)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

target_link_libraries(
  tanh
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
