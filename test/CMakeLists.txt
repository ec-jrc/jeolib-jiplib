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
  add_test(test1 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_open.py --input ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_2010.tif)
  set_tests_properties (test1 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test2 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_write.py --input ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_2010.tif --output ${CMAKE_BINARY_DIR}/Testing/modis_ndvi_2010_write.tif)
  set_tests_properties (test2 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test3 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_downsample.py --input ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_2010.tif)
  set_tests_properties (test3 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test4 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_upsample.py --input ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_2010.tif)
  set_tests_properties (test4 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test5 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_opensubset.py --input ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_2010.tif --extent ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_training.sqlite)
  set_tests_properties (test5 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test6 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_createsimple.py)
  set_tests_properties (test6 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test7 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_refdxdy.py)
  set_tests_properties (test7 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test8 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_bb.py)
  set_tests_properties (test8 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test9 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_init.py)
  set_tests_properties (test9 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test10 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_randgaussian.py)
  set_tests_properties (test10 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test11 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_randuniform.py)
  set_tests_properties (test11 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test12 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_createreference.py)
  set_tests_properties (test12 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test13 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_copyimage.py)
  set_tests_properties (test13 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test14 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_createtemplate.py)
  set_tests_properties (test14 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test15 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_setthreshold.py --input ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_2010.tif --min 10 --max 50)
  set_tests_properties (test15 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test16 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_setthreshold.py --input ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_2010.tif --min 0 --max 1)
  set_tests_properties (test16 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test17 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_getmask.py --input ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_2010.tif)
  set_tests_properties (test17 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test18 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_setmask.py --input ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_2010.tif)
  set_tests_properties (test18 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test19 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_setmask.py --input ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_2010.tif --vectormask ${CMAKE_CURRENT_SOURCE_DIR}/nuts_italy.sqlite)
  set_tests_properties (test19 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test20 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_composite.py --input ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_2010.tif)
  set_tests_properties (test20 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test21 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_extractogr.py --input ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_2010.tif --vector ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_training.sqlite --output ${CMAKE_BINARY_DIR}/Testing/modis_ndvi_training.sqlite)
  set_tests_properties (test21 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test22 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_classify.py --input ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_2010.tif --vector ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_training.sqlite --model ${CMAKE_BINARY_DIR}/Testing/model.txt --output ${CMAKE_BINARY_DIR}/Testing/modis_ndvi_classify.tif)
  set_tests_properties (test22 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test23 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_filter2d.py --input ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_2010.tif --filter median --band 0 10 --size 5 --output ${CMAKE_BINARY_DIR}/Testing/modis_ndvi_median.tif)
  set_tests_properties (test23 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test24 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_filter2d.py --input ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_2010.tif --filter nvalid --nodata 0 --band 1 --size 3)
  set_tests_properties (test24 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test25 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_filter2d.py --input ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_2010.tif --filter var --band 2)
  set_tests_properties (test25 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test26 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_filter2d.py --input ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_2010.tif --filter min --band 3)
  set_tests_properties (test26 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test27 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_filter2d.py --input ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_2010.tif --filter max --band 4)
  set_tests_properties (test27 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test28 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_filter2d.py --input ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_2010.tif --filter sum --band 5)
  set_tests_properties (test28 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test29 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_filter2d.py --input ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_2010.tif --filter mean --band 6)
  set_tests_properties (test29 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test30 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_filter2d.py --input ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_2010.tif --filter close --band 7)
  set_tests_properties (test30 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test31 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_filter2d.py --input ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_2010.tif --filter homog --band 8)
  set_tests_properties (test31 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test32 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_filter2d.py --input ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_2010.tif --filter heterog --band 8)
  set_tests_properties (test32 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test33 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_filter2d.py --input ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_2010.tif --filter smooth --band 9)
  set_tests_properties (test33 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test34 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_filter2d.py --input ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_2010.tif --filter smoothnodata -nodata 0 --band 9)
  set_tests_properties (test34 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test35 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_filter2d.py --input ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_2010.tif --filter dwt --cut 90)
  set_tests_properties (test35 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test36 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_filter1d.py --input ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_2010.tif --filter smoothnodata -nodata 0)
  set_tests_properties (test36 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test37 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_statprofile.py --input ${CMAKE_CURRENT_SOURCE_DIR}/modis_ndvi_2010.tif --function 'min' 'mean' 'max')
  set_tests_properties (test37 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
endif(PYTHONINTERP_FOUND)
