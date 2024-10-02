configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

set(MHLO_PASS_TUTOR_DEPENDS
    FileCheck
    count
    not
    tanh
    explog
    inline
    pow2
    outline
    pass-tutor-opt)

add_lit_testsuite(
  check-mhlo-pass-tutor "Running the standalone regression tests"
  ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${MHLO_PASS_TUTOR_DEPENDS})

set_target_properties(check-mhlo-pass-tutor PROPERTIES FOLDER "Tests")

add_lit_testsuites(STANDALONE ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS
                   ${MHLO_PASS_TUTOR_DEPENDS})
