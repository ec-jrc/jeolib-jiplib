###############################################################################
# CMakeLists build script for cmake
# Author(s): Pieter.Kempeneers@ec.europa.eu
# Copyright (c) 2016-2019 European Union (Joint Research Centre)
# License EUPLv1.2
#
# This file is part of jiplib
###############################################################################
# find_package (Boost COMPONENTS system filesystem unit_test_framework REQUIRED)
# include_directories (${TEST_SOURCE_DIR}/src
#                      ${Boost_INCLUDE_DIRS}
#                      )
# add_definitions (-DBOOST_TEST_DYN_LINK)

# INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR})
set(SRC_INCLUDE_DIR "../src")
INCLUDE_DIRECTORIES(${SRC_INCLUDE_DIR})
message(STATUS "include_directores for test: ${SRC_INCLUDE_DIR}")

find_package(PythonInterp)

set(TESTAPPS
  testMemoryLeak
  testCreateImg
  test2
  jltestexternal
  )

set(JIPLIBS
  ${JIPLIB_LIB_NAME}
  ${GSL_LIBRARIES}
  ${BOOST_LIBRARIES}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_SERIALIZATION_LIBRARY}
  )
# set(JIPLIBS
#   base
#   algorithms
#   imageClasses
#   ${GDAL_LIBRARIES}
#   ${JIPLIB_LIB_NAME}
#   ${Boost_FILESYSTEM_LIBRARY}
#   ${Boost_SYSTEM_LIBRARY}
#   ${Boost_SERIALIZATION_LIBRARY}
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
#   ${GSL_LIBRARIES}
#   )
# target_link_libraries( test2
#   ${JIPLIB_LIB_NAME}
#   ${GSL_LIBRARIES}
#   )


########## CTEST ############
if(PYTHONINTERP_FOUND)
  add_test(test0 echo "Success")
  set_tests_properties (test0 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test01 echo "${PYTHON_EXECUTABLE}")
  set_tests_properties (test01 PROPERTIES PASS_REGULAR_EXPRESSION "python")
  add_test(test1 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_open.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_2010.tif)
  set_tests_properties (test1 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test2 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_write.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_2010.tif --output ${CMAKE_BINARY_DIR}/Testing/modis_ndvi_2010_write.tif)
  set_tests_properties (test2 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test3 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_downsample.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_2010.tif)
  set_tests_properties (test3 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test4 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_upsample.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_2010.tif)
  set_tests_properties (test4 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test5 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_opensubset.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_2010.tif --extent ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_training.sqlite)
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
  add_test(test141 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_crop.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_2010.tif --extent ${CMAKE_CURRENT_SOURCE_DIR}/data/nuts_italy.sqlite --cut_to_cutline True --nodata 0 --output ${CMAKE_BINARY_DIR}/Testing/crop.tif)
  set_tests_properties (test141 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test142 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_crop.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_2010.tif --extent ${CMAKE_CURRENT_SOURCE_DIR}/data/nuts_italy.sqlite --cut_in_cutline True --nodata 255 --output ${CMAKE_BINARY_DIR}/Testing/crop.tif)
  set_tests_properties (test142 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test143 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_crop.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_2010.tif --extent ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_training.sqlite --output ${CMAKE_BINARY_DIR}/Testing/crop.tif)
  set_tests_properties (test143 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test15 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_setthreshold.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_2010.tif --min 10 --max 50)
  set_tests_properties (test15 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test16 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_setthreshold.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_2010.tif --min 0 --max 1)
  set_tests_properties (test16 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test17 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_getmask.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_2010.tif)
  set_tests_properties (test17 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test18 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_setmask.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_2010.tif)
  set_tests_properties (test18 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test19 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_setmask.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_2010.tif --vectormask ${CMAKE_CURRENT_SOURCE_DIR}/data/nuts_italy.sqlite)
  set_tests_properties (test19 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test20 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_composite.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_2010.tif)
  set_tests_properties (test20 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test211 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_extractogr.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_2010.tif --output ${CMAKE_BINARY_DIR}/Testing/modis_ndvi_grid.sqlite)
  set_tests_properties (test211 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test212 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_extractogr.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_2010.tif --vector ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_training.sqlite --output ${CMAKE_BINARY_DIR}/Testing/modis_ndvi_training.sqlite)
  set_tests_properties (test212 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test213 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_extractogr.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_2010.tif --vector ${CMAKE_CURRENT_SOURCE_DIR}/data/nuts_italy.sqlite --output ${CMAKE_BINARY_DIR}/Testing/modis_ndvi_nuts.sqlite)
  set_tests_properties (test213 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test214 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_extractogr.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_2010.tif --vector ${CMAKE_CURRENT_SOURCE_DIR}/data/nuts_italy.sqlite --output ${CMAKE_BINARY_DIR}/Testing/modis_ndvi_nuts_noread.sqlite --noread True)
  set_tests_properties (test214 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test215 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_extractSample.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/corine.tif --output ${CMAKE_BINARY_DIR}/Testing/corine_sample.sqlite --random 100)
  set_tests_properties (test215 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test22 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_classify.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_2010.tif --vector ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_training.sqlite --model ${CMAKE_BINARY_DIR}/Testing/model.txt --output ${CMAKE_BINARY_DIR}/Testing/modis_ndvi_classify.tif --classifier "svm")
  set_tests_properties (test22 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test221 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_classify_s2.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/s2.tif --reference ${CMAKE_CURRENT_SOURCE_DIR}/data/clc_32632.tif --model ${CMAKE_BINARY_DIR}/Testing/model_sml.txt --output ${CMAKE_BINARY_DIR}/Testing/sml_class.tif --classifier "sml")
  set_tests_properties (test221 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test222 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_classify_s2.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/s2.tif --reference ${CMAKE_CURRENT_SOURCE_DIR}/data/clc_32632.tif --model ${CMAKE_BINARY_DIR}/Testing/model_svm.txt --sampleSize 100 --output ${CMAKE_BINARY_DIR}/Testing/svm_class.tif --classifier "svm" --training ${CMAKE_BINARY_DIR}/Testing/training_svm.shp)
  set_tests_properties (test222 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test223 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_classify_s2.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/s2.tif --reference ${CMAKE_CURRENT_SOURCE_DIR}/data/clc_32632.tif --model ${CMAKE_BINARY_DIR}/Testing/model_ann.txt --sampleSize 100 --output ${CMAKE_BINARY_DIR}/Testing/ann_class.tif --classifier "ann")
  set_tests_properties (test223 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test224 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_classifyOgr.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_2010.tif --vector ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_training.sqlite --model ${CMAKE_BINARY_DIR}/Testing/model.txt --output ${CMAKE_BINARY_DIR}/Testing/svm_class.sqlite --classifier "svm")
  set_tests_properties (test224 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test225 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_classifyOgr.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_2010.tif --vector ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_training.sqlite --model ${CMAKE_BINARY_DIR}/Testing/model.txt --output ${CMAKE_BINARY_DIR}/Testing/ann_class.sqlite --classifier "ann")
  set_tests_properties (test225 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test23 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_filter2d.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_2010.tif --filter median --band 0 10 --size 5 --output ${CMAKE_BINARY_DIR}/Testing/modis_ndvi_median.tif)
  set_tests_properties (test23 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test24 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_filter2d.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_2010.tif --filter nvalid --nodata 0 --band 1 --size 3)
  set_tests_properties (test24 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test25 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_filter2d.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_2010.tif --filter var --band 2)
  set_tests_properties (test25 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test26 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_filter2d.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_2010.tif --filter min --band 3)
  set_tests_properties (test26 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test27 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_filter2d.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_2010.tif --filter max --band 4)
  set_tests_properties (test27 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test28 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_filter2d.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_2010.tif --filter sum --band 5)
  set_tests_properties (test28 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test29 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_filter2d.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_2010.tif --filter mean --band 6)
  set_tests_properties (test29 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test30 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_filter2d.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_2010.tif --filter close --band 7)
  set_tests_properties (test30 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test31 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_filter2d.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_2010.tif --filter homog --band 8)
  set_tests_properties (test31 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test32 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_filter2d.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_2010.tif --filter heterog --band 8)
  set_tests_properties (test32 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test33 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_filter2d.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_2010.tif --filter smooth --band 9)
  set_tests_properties (test33 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test34 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_filter2d.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_2010.tif --filter smoothnodata -nodata 0 --band 9)
  set_tests_properties (test34 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test35 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_filter2d.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_2010.tif --filter dwt --cut 90)
  set_tests_properties (test35 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test351 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_filter2d.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_2010.tif --filter tap)
  set_tests_properties (test351 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test361 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_filter1d.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_2010.tif --filter smoothnodata -nodata 0 --interp linear)
  set_tests_properties (test361 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test362 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_filter1d.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_2010.tif --filter dwt_cut --threshold 20)
  set_tests_properties (test362 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test363 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_filter1d.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_2010.tif --filter median --dz 5)
  set_tests_properties (test363 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test364 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_filter1d.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_2010.tif --filter min -nodata 0)
  set_tests_properties (test364 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test365 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_filter1d.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_2010.tif --tapz 1 1 1 --output ${CMAKE_BINARY_DIR}/Testing/filter_taps.tif)
  set_tests_properties (test365 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
  add_test(test37 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytest_statprofile.py --input ${CMAKE_CURRENT_SOURCE_DIR}/data/modis_ndvi_2010.tif --function min max median percentile --perc 10 50 90)
  set_tests_properties (test37 PROPERTIES PASS_REGULAR_EXPRESSION "Success")
endif(PYTHONINTERP_FOUND)
