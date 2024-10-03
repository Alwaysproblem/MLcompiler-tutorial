include_directories(include)
add_subdirectory(include)

set(LLVM_LINK_COMPONENTS Support)

set(LLVM_TARGET_DEFINITIONS mlir/Pow2.td)
mlir_tablegen(Pow2.inc -gen-rewriters EXTRA_INCLUDES ${MLIR_INCLUDE_DIR})
add_public_tablegen_target(PassTutorIncGen)

set(LLVM_TARGET_DEFINITIONS mlir/transform/pass.td)
mlir_tablegen(Pow2Pass.inc -gen-pass-decls EXTRA_INCLUDES ${MLIR_INCLUDE_DIR})
mlir_tablegen(Pow2Pass.md -gen-pass-doc)
add_public_tablegen_target(PassTutorPassIncGen)

add_mlir_pdll_library(PassTutorPdllIncGen mlir/Pow2.pdll Pow2Pdll.inc
                      EXTRA_INCLUDES ${MLIR_INCLUDE_DIR})

add_executable(pass-tutor-opt main.cpp mlir/PassTutor.cpp)

add_dependencies(pass-tutor-opt PassTutorIncGen PassTutorPdllIncGen
                 PassTutorPassIncGen)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

target_link_libraries(
  pass-tutor-opt PRIVATE ${dialect_libs} ${conversion_libs} ${extension_libs}
                         MLIRIR MLIRMlirOptMain MhloDialect)
