add_executable(mhlo-example mhlo_example.cpp)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

target_link_libraries(
  mhlo-example
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
          MLIRTransforms
          MLIRPopopsExprDialect
          MLIRSharedDialect
          MLIRPoprandDialect
          MLIRPopnnDialect
          MLIRPoplayoutDialect
          MLIRPopitTransforms
          MLIRPoplarDialect
          MLIRPopitDialect
          MLIRPhloDialect
          MLIRPoplarTransforms
          MLIRPopopsDialect
          MLIRPhloToPoplarConversion
          MLIRPoplarConversionCommon
          MLIRPhloTransforms
          MLIRPoplinDialect
          poplar
          popit
          pvti
          popef
          popops
          poprand
          popnn
          poplin
          popir)
