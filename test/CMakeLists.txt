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

find_package(PythonInterp)

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
if(PYTHONINTERP_FOUND)
  add_test(test1 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest1.py --input ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_2010.tif)
  set_tests_properties (test1 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test2 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest2.py --input ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_2010.tif --output ${CMAKE_BINARY_DIR}/Testing/modis_ndvi_2010_write.tif)
  set_tests_properties (test2 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test3 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest3.py --input ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_2010.tif)
  set_tests_properties (test3 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test4 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest4.py --input ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_2010.tif)
  set_tests_properties (test4 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test5 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest5.py --input ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_2010.tif --extent ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_training.sqlite)
  set_tests_properties (test5 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test6 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest6.py)
  set_tests_properties (test6 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test7 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest7.py)
  set_tests_properties (test7 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test8 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest8.py)
  set_tests_properties (test8 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test9 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest9.py)
  set_tests_properties (test9 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test10 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest10.py)
  set_tests_properties (test10 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test11 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest11.py)
  set_tests_properties (test11 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test12 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest12.py)
  set_tests_properties (test12 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test13 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest13.py)
  set_tests_properties (test13 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test14 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest14.py)
  set_tests_properties (test14 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test15 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest15.py --input ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_2010.tif)
  set_tests_properties (test15 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test16 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest16.py --input ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_2010.tif)
  set_tests_properties (test16 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test17 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest17.py --input ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_2010.tif)
  set_tests_properties (test17 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test18 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest18.py --input ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_2010.tif)
  set_tests_properties (test18 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test19 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest19.py --input ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_2010.tif --vectormask ${CMAKE_CURRENT_SOURCE_DIR}/nuts_italy.sqlite)
  set_tests_properties (test19 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test20 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest20.py --input ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_2010.tif)
  set_tests_properties (test20 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test21 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest21.py --input ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_2010.tif --vector ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_training.sqlite --output ${CMAKE_BINARY_DIR}/Testing/modis_ndvi_training.sqlite)
  set_tests_properties (test21 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test22 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest22.py --input ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_2010.tif --vector ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_training.sqlite --model ${CMAKE_BINARY_DIR}/Testing/model.txt --output ${CMAKE_BINARY_DIR}/Testing/modis_ndvi_classify.tif)
  set_tests_properties (test22 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
endif(PYTHONINTERP_FOUND)
