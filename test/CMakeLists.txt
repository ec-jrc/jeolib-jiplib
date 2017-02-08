###############################################################################
#
# CMake build scripts for JIPLIB
#
#
# History
# 2016/06/24 - Created by Pieter Kempeneers
# Change log
#
###############################################################################
# find_package (Boost COMPONENTS system filesystem unit_test_framework REQUIRED)
# include_directories (${TEST_SOURCE_DIR}/src
#                      ${Boost_INCLUDE_DIRS}
#                      )
# add_definitions (-DBOOST_TEST_DYN_LINK)

set(TESTAPPS
  testMemoryLeak
  testCreateImg
  test2
  )

set(JIPLIBS
  ${JIPLIB_LIB_NAME}
  ${PKTOOLS_LIBRARIES}
  ${GSL_LIBRARIES}
  )
# set(JIPLIBS
#   base
#   algorithms
#   imageClasses
#   ${GDAL_LIBRARIES}
#   ${JIPLIB_LIB_NAME}
#   ${Boost_FILESYSTEM_LIBRARY}
#   ${Boost_SYSTEM_LIBRARY}
#   ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
#   )

foreach(TESTAPP ${TESTAPPS})
  add_executable(${TESTAPP} ${CMAKE_CURRENT_SOURCE_DIR}/${TESTAPP}.cc)
  target_link_libraries(${TESTAPP} ${JIPLIBS})
endforeach()

# Build executables

# add_executable(testCreateImg ${PROJECT_TEST_DIR}/testCreateImg.cc)
# add_executable(test2 ${PROJECT_TEST_DIR}/test2.cc)
# target_link_libraries( testCreateImg
#   ${JIPLIB_LIB_NAME}
#   ${PKTOOLS_LIBRARIES}
#   ${GSL_LIBRARIES}
#   )
# target_link_libraries( test2
#   ${JIPLIB_LIB_NAME}
#   ${PKTOOLS_LIBRARIES}
#   ${GSL_LIBRARIES}
#   )


########## CTEST ############
add_test(test1 ${CMAKE_CURRENT_BINARY_DIR}/testCreateImg --output ../Testing/test1.tif --ncol 1 --nrow 1 --oformat GTiff --otype Byte --seed 100 --mean 0 --stdev 100)
add_test(test2 ${CMAKE_CURRENT_BINARY_DIR}/test2 --output ../Testing/test2.tif --ncol 100 --nrow 100 --oformat GTiff --otype Byte --seed 100 --mean 0 --stdev 100)
add_test(test3 ${CMAKE_CURRENT_BINARY_DIR}/testMemoryLeak --number 10 --ncol 1024 --nrow 1024 --nband 10 --otype GDT_Byte)
set_tests_properties (test1
  PROPERTIES PASS_REGULAR_EXPRESSION "done")
set_tests_properties (test2
    PROPERTIES PASS_REGULAR_EXPRESSION "done")
set_tests_properties (test3
    PROPERTIES PASS_REGULAR_EXPRESSION "done")

