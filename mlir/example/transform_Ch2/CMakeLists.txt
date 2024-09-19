# For a better top-level template to copy, see examples/standalone.

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(include)
add_subdirectory(lib)

add_executable(transform-opt-ch2 transform-opt/transform-opt.cpp)

add_dependencies(transform-opt-ch2 MyExtensionCh2IncGen)

target_link_libraries(
  transform-opt-ch2 PRIVATE MLIRIR MLIRMlirOptMain MLIRSideEffectInterfaces
                            MyExtensionCh2)
