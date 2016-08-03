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
find_package (Boost COMPONENTS system filesystem unit_test_framework REQUIRED)
include_directories (${TEST_SOURCE_DIR}/src
                     ${Boost_INCLUDE_DIRS}
                     )
add_definitions (-DBOOST_TEST_DYN_LINK)

set(TESTAPPS
  test1
  checkS2nodata
  )

set(JIPLIBS 
  base
  algorithms
  imageClasses
  ${GDAL_LIBRARIES}
  ${JIPLIB_LIB_NAME}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
  )

foreach(TESTAPP ${TESTAPPS})
  add_executable(${TESTAPP} ${TESTAPP}.cc)
  target_link_libraries(${TESTAPP} ${JIPLIBS})
endforeach()

