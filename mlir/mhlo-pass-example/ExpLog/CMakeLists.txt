# For a better template to copy, see examples/standalone
include_directories(include)
add_subdirectory(include)

set(LLVM_LINK_COMPONENTS Support)

set(LLVM_TARGET_DEFINITIONS mlir/ExpLog.td)
mlir_tablegen(ExpLog.inc -gen-rewriters EXTRA_INCLUDES ${MLIR_INCLUDE_DIR})
add_public_tablegen_target(ExpLogIncGen)

add_executable(explog main.cpp mlir/ExpLog.cpp)

add_dependencies(explog ExpLogIncGen)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

target_link_libraries(
  explog
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
