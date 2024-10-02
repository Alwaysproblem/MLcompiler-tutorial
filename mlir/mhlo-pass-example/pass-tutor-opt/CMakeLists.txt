# For a better top-level template to copy, see examples/standalone.

include_directories(${PROJECT_BINARY_DIR}/Pow2)
include_directories(${PROJECT_SOURCE_DIR}/Pow2/include)

add_executable(pass-tutor-opt pass-tutor-opt.cpp
                              ${PROJECT_SOURCE_DIR}/Pow2/mlir/Pow2.cpp)

add_dependencies(pass-tutor-opt Pow2IncGen Pow2PdllIncGen Pow2PassIncGen)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

target_link_libraries(pass-tutor-opt PRIVATE ${dialect_libs} MLIRIR
                                             MLIRMlirOptMain MhloDialect)
