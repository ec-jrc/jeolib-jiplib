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
INCLUDE_DIRECTORIES(${PKTOOLS_INCLUDE_DIR})

set(JIPLIB_IMG_CC jim.cc)
set(JIPLIB_IMG_H jim.h)
# Define the jiplib library.
add_library( ${JIPLIB_LIB_NAME} ${JIPLIB_IMG_H} ${JIPLIB_IMG_CC})
target_link_libraries( 
  ${JIPLIB_LIB_NAME} 
  ${MIA_LIB_NAME}
  gomp
  ${GDAL_LIBRARIES}
  base
  algorithms
  imageClasses
  )
