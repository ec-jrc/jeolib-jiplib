###############################################################################
# set file locations
set(BASE_SRC_DIR base)
set(ALGOR_SRC_DIR algorithms)
set(FILECLASS_SRC_DIR fileclasses)
set(IMGCLASS_SRC_DIR imageclasses)
set(LASCLASS_SRC_DIR lasclasses)
set(APP_SRC_DIR apps)
set(PYTHON_SRC_DIR swig)

set(BASE_H
  ${BASE_SRC_DIR}/IndexValue.h
  ${BASE_SRC_DIR}/typeconversion.h
  ${BASE_SRC_DIR}/Optionpk.h
  ${BASE_SRC_DIR}/PosValue.h
  ${BASE_SRC_DIR}/Vector2d.h
  )

set(BASE_CC
  ${BASE_SRC_DIR}/typeconversion.cc
  ${BASE_SRC_DIR}/Vector2d.cc
  )

set(ALGOR_H
  ${ALGOR_SRC_DIR}/ConfusionMatrix.h
  ${ALGOR_SRC_DIR}/CostFactory.h
  ${ALGOR_SRC_DIR}/CostFactorySVM.h
  ${ALGOR_SRC_DIR}/Egcs.h
  ${ALGOR_SRC_DIR}/FeatureSelector.h
  ${ALGOR_SRC_DIR}/Filter.h
  ${ALGOR_SRC_DIR}/Filter2d.h
  ${ALGOR_SRC_DIR}/ImgRegression.h
  ${ALGOR_SRC_DIR}/StatFactory.h
  ${ALGOR_SRC_DIR}/myfann_cpp.h
  ${ALGOR_SRC_DIR}/svm.h
  )

set(ALGOR_CC
  ${ALGOR_SRC_DIR}/ConfusionMatrix.cc
  ${ALGOR_SRC_DIR}/CostFactorySVM.cc
  ${ALGOR_SRC_DIR}/CostFactorySVM.h
  ${ALGOR_SRC_DIR}/Egcs.cc
  ${ALGOR_SRC_DIR}/Filter.cc
  ${ALGOR_SRC_DIR}/Filter2d.cc
  ${ALGOR_SRC_DIR}/ImgRegression.cc
  ${ALGOR_SRC_DIR}/svm.cpp
  )

set(FILECLASS_H
  ${FILECLASS_SRC_DIR}/FileReaderAscii.h
  )

set(FILECLASS_CC
  ${FILECLASS_SRC_DIR}/FileReaderAscii.cc
  )

set(IMGCLASS_H
  ${IMGCLASS_SRC_DIR}/ImgRaster.h
  ${IMGCLASS_SRC_DIR}/ImgList.h
  ${IMGCLASS_SRC_DIR}/VectorOgr.h
  ${IMGCLASS_SRC_DIR}/ImgReaderOgr.h
  ${IMGCLASS_SRC_DIR}/ImgWriterOgr.h
  ${IMGCLASS_SRC_DIR}/pksml_lib.h
  ${IMGCLASS_SRC_DIR}/pkclassify_lib.h
  )

set(IMGCLASS_CC
  ${IMGCLASS_SRC_DIR}/ImgRaster.cc
  ${IMGCLASS_SRC_DIR}/ImgList.cc
  ${IMGCLASS_SRC_DIR}/VectorOgr.cc
  ${IMGCLASS_SRC_DIR}/ImgReaderOgr.cc
  ${IMGCLASS_SRC_DIR}/ImgWriterOgr.cc
  ${IMGCLASS_SRC_DIR}/pkcomposite_lib.cc
  ${IMGCLASS_SRC_DIR}/pkcrop_lib.cc
  ${IMGCLASS_SRC_DIR}/pkextractogr_lib.cc
  ${IMGCLASS_SRC_DIR}/pkextractimg_lib.cc
  ${IMGCLASS_SRC_DIR}/pkstatprofile_lib.cc
  ${IMGCLASS_SRC_DIR}/pkfilter1d_lib.cc
  ${IMGCLASS_SRC_DIR}/pkfilter2d_lib.cc
  ${IMGCLASS_SRC_DIR}/pkclassify_lib.cc
  ${IMGCLASS_SRC_DIR}/pksvm_lib.cc
  ${IMGCLASS_SRC_DIR}/pksvmogr_lib.cc
  ${IMGCLASS_SRC_DIR}/pkann_lib.cc
  ${IMGCLASS_SRC_DIR}/pkannogr_lib.cc
  ${IMGCLASS_SRC_DIR}/pkdiff_lib.cc
  ${IMGCLASS_SRC_DIR}/pkvalidate_lib.cc
  ${IMGCLASS_SRC_DIR}/pkstretch_lib.cc
  ${IMGCLASS_SRC_DIR}/pkdumpimg_lib.cc
  ${IMGCLASS_SRC_DIR}/pkreclass_lib.cc
  ${IMGCLASS_SRC_DIR}/pksetmask_lib.cc
  ${IMGCLASS_SRC_DIR}/pkgetmask_lib.cc
  ${IMGCLASS_SRC_DIR}/pkstat_lib.cc
  )

set(LASCLASS_H
  ${LASCLASS_SRC_DIR}/FileReaderLas.h
  )

set(LASCLASS_CC
  ${LASCLASS_SRC_DIR}/FileReaderLas.cc
  )

set(APP_H
  ${APP_SRC_DIR}/AppFactory.h
  )

set(PYTHON_CC
  ${PYTHON_SRC_DIR}/ImgRaster_wrap.cc
  )

# set(PKTOOLS_I
#   ${PYTHON_SRC_DIR}/pktools.i
#   )
###############################################################################

###############################################################################
# Group source files for IDE source explorers
source_group("CMake Files" FILES CMakeLists.txt)
source_group("src_base" FILES ${BASE_H} ${BASE_CC})
source_group("src_algor" FILES ${ALGOR_H} ${ALGOR_CC})
source_group("src_fileclass" FILES ${FILECLASS_H} ${FILECLASS_CC})
source_group("src_imgclass" FILES ${IMGCLASS_H} ${IMGCLASS_CC})
source_group("src_lasclass" FILES ${LASCLASS_H} ${LASCLASS_CC})
###############################################################################

###############################################################################
# Build and link library

if(BUILD_WITH_LIBLAS)
  add_library( ${JIPLIB_LIB_NAME} ${LASCLASS_H} ${LASCLASS_CC} ${IMGCLASS_H} ${IMGCLASS_CC} ${ALGOR_H} ${ALGOR_CC} ${FILECLASS_CC} ${FILECLASS_H} ${BASE_H} ${BASE_CC})
else(BUILD_WITH_LIBLAS)
  add_library( ${JIPLIB_LIB_NAME} ${IMGCLASS_H} ${IMGCLASS_CC} ${ALGOR_H} ${ALGOR_CC} ${FILECLASS_CC} ${FILECLASS_H} ${BASE_H} ${BASE_CC})
endif(BUILD_WITH_LIBLAS)

target_link_libraries(${JIPLIB_LIB_NAME}
  ${BOOST_LIBRARIES}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_SERIALIZATION_LIBRARY}
  ${BOOST_ARCHIVE_LIBRARY}
  ${JSONCPP_LIBRARIES}
  ${GDAL_LIBRARIES}
  ${GSL_LIBRARIES}
  ${FANN_LIBRARIES}
  )

if(BUILD_WITH_FANN)
target_link_libraries(${JIPLIB_LIB_NAME}
  ${FANN_LIBRARIES}
  )
endif(BUILD_WITH_FANN)

if(BUILD_WITH_LIBLAS)
  target_link_libraries(${JIPLIB_LIB_NAME}
    ${LIBLAS_LIBRARIES}
    )
endif(BUILD_WITH_LIBLAS)

if(BUILD_WITH_PYTHON)
  target_link_libraries(${JIPLIB_LIB_NAME}
    ${PYTHON_LIBRARIES}
    )
endif(BUILD_WITH_PYTHON)

if(BUILD_WITH_MIALIB)
  target_link_libraries(
    ${JIPLIB_LIB_NAME}
    ${MIA_LIB_NAME}
    gomp
    )
endif(BUILD_WITH_MIALIB)

###############################################################################
SET(JIPLIB_LIBRARIES ${JIPLIB_LIB_NAME})
###############################################################################
# Set target properties
SET_TARGET_PROPERTIES(${JIPLIB_LIB_NAME}
  PROPERTIES
  SOVERSION ${JIPLIB_LIB_SOVERSION}
  VERSION ${JIPLIB_LIB_VERSION}
)
###############################################################################

########## SWIG #############
IF(BUILD_WITH_PYTHON)
  FIND_PACKAGE(SWIG REQUIRED)
  FIND_PACKAGE(PythonLibs)
  # SET(PYTHON_SRC_DIR ${PROJECT_SOURCE_DIR}/swig)
  FILE(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/swig)
  INCLUDE(${SWIG_USE_FILE})
  INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR})
  INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})
  INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
  SET(CMAKE_SWIG_FLAGS "")
  SET_PROPERTY(SOURCE ${PYTHON_SRC_DIR}/jiplib.i PROPERTY CPLUSPLUS ON)
  SWIG_ADD_MODULE(jiplib python ${PYTHON_SRC_DIR}/jiplib.i ${IMGCLASS_H} ${IMGCLASS_CC})
  # SET(SWIG_MODULE_jiplib_EXTRA_DEPS ${JIPLIB_LIBRARIES})
  SET_SOURCE_FILES_PROPERTIES(${PYTHON_SRC_DIR}/jiplib.i PROPERTIES SWIG_FLAGS "-Wall")
  SWIG_LINK_LIBRARIES(jiplib ${PYTHON_LIBRARIES} ${JIPLIB_LIBRARIES})
ENDIF(BUILD_WITH_PYTHON)

########## INSTALL ##########
# INSTALL (TARGETS ${JIPLIB_BASE_LIB_NAME} ${JIPLIB_IMAGECLASSES_LIB_NAME} ${JIPLIB_ALGORITHMS_LIB_NAME} ${JIPLIB_FILECLASSES_LIB_NAME} DESTINATION ${INSTALL_LIBRARY_DIR})

INSTALL (TARGETS ${JIPLIB_LIB_NAME}
  ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR}
  LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR})
IF(BUILD_WITH_PYTHON)
  EXECUTE_PROCESS(COMMAND python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())" OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
  SET(PYTHON_INSTALL_DIR ${PYTHON_SITE_PACKAGES} CACHE PATH "Install directory for Python library")
  INSTALL(TARGETS ${SWIG_MODULE_jiplib_REAL_NAME} LIBRARY DESTINATION ${PYTHON_INSTALL_DIR})
  INSTALL(FILES  ${CMAKE_CURRENT_BINARY_DIR}/jiplib.py DESTINATION ${PYTHON_INSTALL_DIR})
  INSTALL (FILES ${PYTHON_SRC_DIR}/jiplib.i DESTINATION ${PROJECT_INCLUDE_DIR}/swig)
ENDIF(BUILD_WITH_PYTHON)
###############################################################################
###############################################################################
# Installation
# install (TARGETS ${JIPLIB_LIB_NAME} DESTINATION lib)

INSTALL (FILES ${BASE_H} DESTINATION ${PROJECT_INCLUDE_DIR}/base)
INSTALL (FILES ${IMGCLASS_H} DESTINATION ${PROJECT_INCLUDE_DIR}/imageclasses)
INSTALL (FILES ${ALGOR_H} DESTINATION ${PROJECT_INCLUDE_DIR}/algorithms)
INSTALL (FILES ${FILECLASS_H} DESTINATION ${PROJECT_INCLUDE_DIR}/fileclasses)
INSTALL (FILES ${APP_H} DESTINATION ${PROJECT_INCLUDE_DIR}/apps)

if (BUILD_WITH_LIBLAS)
  INSTALL (FILES ${LASCLASS_H} DESTINATION ${PROJECT_INCLUDE_DIR}/lasclasses)
endif(BUILD_WITH_LIBLAS)

###############################################################################
