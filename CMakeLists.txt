##############################################################################
# CMakeLists build script for cmake
# Author(s): Pieter.Kempeneers@ec.europa.eu
# Copyright (c) 2016-2018 European Union (Joint Research Centre)
# License EUPLv1.2

# This file is part of jiplib
###############################################################################
cmake_minimum_required(VERSION 3.7)

# The project's name and version
project(jiplib)

enable_testing()
include(CTest)

set (JIPLIB_VERSION_MAJOR 1)
set (JIPLIB_VERSION_MINOR 0)
set (JIPLIB_VERSION_PATCH 0)
set (JIPLIB_VERSION "${JIPLIB_VERSION_MAJOR}.${JIPLIB_VERSION_MINOR}.${JIPLIB_VERSION_PATCH}")
set (JIPLIB_SOVERSION "${JIPLIB_VERSION_MAJOR}")
set (JIPLIB_PACKAGE_BUGREPORT "pieter.kempeneers@ec.europa.eu")


set(CLASSIF 0 CACHE STRING "set a variable that can be used in Jim.cc")
set(ODOITHIN 0 CACHE STRING "set a variable that can be used in Jim.cc")
set(MCISRG 1 CACHE STRING "set a variable that can be used in Jim.cc")

set(BUILD_WITH_LIBLAS FALSE CACHE BOOL "Choose if jiplib is to be built with liblas")
set(BUILD_WITH_FANN TRUE CACHE BOOL "Choose if jiplib is to be built with fann")
set(BUILD_WITH_PYTHON TRUE CACHE BOOL "Choose if jiplib is to be built with PYTHON support")
set(PROCESS_IN_PARALLEL TRUE CACHE BOOL "Choose if jiplib should be run in parallel")
set(BUILD_WITH_MIALIB TRUE CACHE BOOL "Choose if jiplib should be compiled with mialib")

set (PROJECT_INCLUDE_DIR include/jiplib)
set (PROJECT_TEST_DIR test)

set(CMAKE_COLOR_MAKEFILE ON)

if(BUILD_WITH_MIALIB)
  set(MIA_BOOL 1)
else(BUILD_WITH_MIALIB)
  set(MIA_BOOL 0)
endif(BUILD_WITH_MIALIB)



option (BUILD_SHARED_LIBS "Build with shared library" ON)

# Platform and compiler specific settings

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-long-long")
  message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has C++11 support.")
else(COMPILER_SUPPORTS_CXX11)
  message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

########## BUILDING INSTALLLER ##########
# build a CPack driven installer package
include (InstallRequiredSystemLibraries)
set (CPACK_RESOURCE_FILE_LICENSE
     "${CMAKE_CURRENT_SOURCE_DIR}/License.txt")
set (CPACK_PACKAGE_VERSION_MAJOR "${JIPLIB_VERSION_MAJOR}")
set (CPACK_PACKAGE_VERSION_MINOR "${JIPLIB_VERSION_MINOR}")
set (CPACK_PACKAGE_VERSION_PATCH "${JIPLIB_VERSION_PATCH}")
include (CPack)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/modules")

# Make sure Boost libraries are found
set(Boost_DEBUG 0)
find_package(Boost COMPONENTS filesystem system serialization)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
  # add_definitions("-DBOOST_ALL")
  add_definitions("-DBOOST_ALL_NO_LIB")
endif()

# Make sure Python libraries are found
# set(Python_ADDITIONAL_VERSIONS 3.5)
#set(PYTHON_VERSION "2.7.13" CACHE STRING "Specify which Python version to use")
set(PYTHON_VERSION "3.6.8" CACHE STRING "Specify which Python version to use")
#from https://github.com/InsightSoftwareConsortium/ITK/blob/master/Wrapping/Generators/Python/CMakeLists.txt#L1-18
# if(PYTHON_VERSION_STRING VERSION_EQUAL PYTHONLIBS_VERSION_STRING)
# if(PYTHON_VERSION_STRING VERSION_LESS 2.6)
# if(PYTHON_VERSION_STRING VERSION_GREATER 3.0)
# FIND_PACKAGE(PythonLibs ${PYTHON_VERSION} EXACT)
FIND_PACKAGE( PythonInterp 3.6 REQUIRED )
FIND_PACKAGE( PythonLibs 3.6 REQUIRED )
if(PYTHONLIBS_FOUND)
  message(STATUS "Python version found: ${PYTHONLIBS_VERSION_STRING}")
  message(STATUS "Found Python library: ${PYTHON_LIBRARY}")
  message(STATUS "Found Python include dir: ${PYTHON_INCLUDE_DIR}")
else(PYTHONLIBS_FOUND)
  message(STATUS "Warning: PYTHON exact version ${PYTHON_VERSION} not found, trying default")
  FIND_PACKAGE(PythonLibs REQUIRED)
endif(PYTHONLIBS_FOUND)
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIR})

find_package(GDAL 2.00 REQUIRED)
if(GDAL_FOUND)
  include_directories(${GDAL_INCLUDE_DIRS})
endif()

find_package(GSL REQUIRED)
if(GSL_FOUND)
  message(STATUS "Found GSL: ${GSL_LIBRARIES}")
  include_directories(${GSL_INCLUDE_DIRS})
  add_definitions(-DGSL_DLL)
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)
if(JSONCPP_FOUND)
  message(STATUS "JSONCPP package found: OK")
else(JSONCPP_FOUND)
  message(STATUS "Warning: JSONCPP package not found!")
endif(JSONCPP_FOUND)

# link_libraries(${JSONCPP_LIBRARIES})
include_directories(${JSONCPP_INCLUDE_DIRS})

if(PROCESS_IN_PARALLEL)
  find_package(OpenMP)
  if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(PARALLEL_BOOL 1)
    message(STATUS "parallel processing: ${PARALLEL_BOOL}")
  endif(OPENMP_FOUND)
else(PROCESS_IN_PARALLEL)
  set(PARALLEL_BOOL 0)
  message(STATUS "parallel processing: ${PARALLEL_BOOL}")
endif(PROCESS_IN_PARALLEL)

###############################################################################
# Setup configure file

# configure a header file to pass some of the CMake settings to the source code
configure_file (
  "cmake-config.h.in"
  "config_jiplib.h"
  )

configure_file ( "cmake-config.in" "${CMAKE_CURRENT_BINARY_DIR}/jiplib-config" @ONLY )
add_definitions(-DHAVE_CONFIG_H)

configure_file ( "cmake.pc.in" "jiplib.pc"  @ONLY)



# Name of C++ library
set(JIPLIB_LIB_NAME jiplib)

set(BUILD_WITH_MIALIB TRUE CACHE BOOL "JIPlib needs to be built with mialiab")

# Include directories
###############################################################################
# include_directories("${CMAKE_SOURCE_DIR}")
include_directories("${PROJECT_BINARY_DIR}")# needed for config_jiplib.h
include_directories("${PROJECT_BINARY_DIR}/src")#  this is where sources will go
# INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR})
# include_directories("${CMAKE_CURRENT_BINARY_DIR}")# needed for config_jiplib.h
# include_directories("${CMAKE_CURRENT_BINARY_DIR}/src/")#
# include_directories("${CMAKE_CURRENT_BINARY_DIR}/doc")#needed for jiplib_doc.i
###############################################################################
# Instruct CMake to inspect the following subfolders (with CMakeLists.txt in each subfolder)
# add_subdirectory ("${PROJECT_DOC_DIR}")
#todo: add if condition with TEST?
add_subdirectory ("${PROJECT_TEST_DIR}")

if (BUILD_WITH_FANN)
  FIND_PACKAGE(FANN 2.2 REQUIRED)
  if(FANN_FOUND)
    include_directories(${FANN_INCLUDE_DIRS})
    add_definitions(-DFANN_DLL)
  else(FANN_FOUND)
    message(STATUS "Error: could not find FANN packagei version >= 2.2")
  endif()
endif(BUILD_WITH_FANN)

###############################################################################

#we will use a copy of the imageclasses in the binary build directory, where we can change the content (for MIALIB)

message(STATUS "cmake_current_binary_dir: ${CMAKE_CURRENT_BINARY_DIR}")
file(COPY ${PROJECT_SOURCE_DIR}/src DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

set(BASE_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR}/src/base)
set(ALGOR_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR}/src/algorithms)
set(FILECLASS_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR}/src/fileclasses)
set(IMGCLASS_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses)
set(LASCLASS_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR}/src/lasclasses)
set(APP_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR}/src/apps)
# set(PYTHON_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR}/src/swig)
# set(PYTHON_SRC_DIR src/swig)

set(BASE_H
  ${BASE_SRC_DIR}/IndexValue.h
  ${BASE_SRC_DIR}/typeconversion.h
  ${BASE_SRC_DIR}/Optionjl.h
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
  ${ALGOR_SRC_DIR}/svm.h
  )

if(BUILD_WITH_FANN)
  set(ALGOR_H ${ALGOR_H}
    ${ALGOR_SRC_DIR}/myfann_cpp.h
    )
endif(BUILD_WITH_FANN)

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

set(LASCLASS_H
  ${LASCLASS_SRC_DIR}/FileReaderLas.h
  )

set(LASCLASS_CC
  ${LASCLASS_SRC_DIR}/FileReaderLas.cc
  )

set(APP_H
  ${APP_SRC_DIR}/AppFactory.h
  )

set(IMGCLASS_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses)

set(IMGCLASS_H
  ${IMGCLASS_SRC_DIR}/Jim.h
  ${IMGCLASS_SRC_DIR}/JimList.h
  ${IMGCLASS_SRC_DIR}/VectorOgr.h
  ${IMGCLASS_SRC_DIR}/jlsml_lib.h
  ${IMGCLASS_SRC_DIR}/jlclassify_lib.h
  )

set(IMGCLASS_CC
  ${IMGCLASS_SRC_DIR}/Jim.cc
  ${IMGCLASS_SRC_DIR}/JimList.cc
  ${IMGCLASS_SRC_DIR}/VectorOgr.cc
  ${IMGCLASS_SRC_DIR}/jlcomposite_lib.cc
  ${IMGCLASS_SRC_DIR}/jlcrop_lib.cc
  ${IMGCLASS_SRC_DIR}/jlextractogr_lib.cc
  ${IMGCLASS_SRC_DIR}/jlextractimg_lib.cc
  ${IMGCLASS_SRC_DIR}/jlstatprofile_lib.cc
  ${IMGCLASS_SRC_DIR}/jlfilter1d_lib.cc
  ${IMGCLASS_SRC_DIR}/jlfilter2d_lib.cc
  ${IMGCLASS_SRC_DIR}/jlclassify_lib.cc
  ${IMGCLASS_SRC_DIR}/jlsvm_lib.cc
  ${IMGCLASS_SRC_DIR}/jlsvmogr_lib.cc
  ${IMGCLASS_SRC_DIR}/jldiff_lib.cc
  ${IMGCLASS_SRC_DIR}/jlvalidate_lib.cc
  ${IMGCLASS_SRC_DIR}/jlstretch_lib.cc
  ${IMGCLASS_SRC_DIR}/jldumpimg_lib.cc
  ${IMGCLASS_SRC_DIR}/jlreclass_lib.cc
  ${IMGCLASS_SRC_DIR}/jlsetmask_lib.cc
  ${IMGCLASS_SRC_DIR}/jlgetmask_lib.cc
  ${IMGCLASS_SRC_DIR}/jlstat_lib.cc
  ${IMGCLASS_SRC_DIR}/jlpolygonize_lib.cc
  ${IMGCLASS_SRC_DIR}/jlwarp_lib.cc
  ${IMGCLASS_SRC_DIR}/jloperators_lib.cc
  )

if(BUILD_WITH_FANN)
  set(IMGCLASS_CC ${IMGCLASS_CC}
  ${IMGCLASS_SRC_DIR}/jlann_lib.cc
  ${IMGCLASS_SRC_DIR}/jlannogr_lib.cc
  )
endif(BUILD_WITH_FANN)

if(BUILD_WITH_MIALIB)
  set(MIA_BOOL 1)
  message(STATUS "build with mialib: ${MIA_BOOL}")
  set(MIA_LIB_NAME mialib_generic)
  set(MIA_INCLUDE_DIR /usr/local/include/mialib CACHE PATH "Include PATH for MIA")
  set(MIA_LIB_PATH /usr/local/lib CACHE PATH "Library PATH for MIA")

  INCLUDE_DIRECTORIES(${MIA_INCLUDE_DIR})
  LINK_DIRECTORIES(${MIA_LIB_PATH})
else(BUILD_WITH_MIALIB)
  set(MIA_BOOL 0)
  message(STATUS "build with mialib: ${MIA_BOOL}")
endif(BUILD_WITH_MIALIB)

###############################################################################
find_package(PythonInterp)
if(PYTHONINTERP_FOUND)
  # file(WRITE ${JIPLIB_DOC} "//This is an automatically generated doc file, do not edit")
  #todo: treat mialib functions that return: int, long int, double, G_TYPE
  # G_TYPE: getpixval
  # G_TYPE* min_max
  # long int thresholdRegion_Size
  # long int thresholdRegion_Contrast
  # double julian_date
  if(BUILD_WITH_MIALIB)
    MESSAGE(STATUS "PythonInterp was found")

    file(GLOB MIA_HEADER_FILES ${MIA_INCLUDE_DIR}/mialib_*.h)
    MESSAGE(STATUS "MIA_INCLUDE_DIR: ${MIA_INCLUDE_DIR}")
    MESSAGE(STATUS "MIA_HEADER_FILES: ${MIA_HEADER_FILES}")
    IF("${MIA_HEADER_FILES}" STREQUAL "")
      MESSAGE(FATAL_ERROR "Error: no header files found for mialib")
    ENDIF("${MIA_HEADER_FILES}" STREQUAL "")
    file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/mialib_imagetype)
    file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/mialib_imagelisttype)
    file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/mialib_errortype_nd)
    file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/mialib_errortype_nm)
    file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/mialib_errortype)
    #Mialib functions have been designed as destructive. We will therefore create a copy to avoid the destructive behaviour. The destructive method will still be available with the prefix d_
    #The following functions F_ND from mialib do not modify the input image and should therefore not create a copy. They will be treated in the same way as the destructive functions (without the prefix d_)
    SET(F_ND "dumpxyz|szcompat|szgeocompat|iminfo|tiffinfo|writeGnuPlot3D|vectorizeImage|IsPartitionEqual|IsPartitionFiner|dendro|getfirstmaxpos|volume|imequalp|getmax|getminmax")
    #The following functions F_NM from mialib are not compatible for multi-band processing
    SET(F_NM "addframebox|subframebox")
    #These functions must only be included if MCISRG is set
    SET(F_MCISRG "labelImage|segmentImage")
    #Create a text file with the list of headers to MIA functions, e.g., mialib_imagetype for functions returning IMAGE *
    FOREACH(infileName ${MIA_HEADER_FILES})
      MESSAGE(STATUS "Process file: ${infileName}")
      file(STRINGS ${infileName} FUN_IMAGETYPE REGEX "^extern IMAGE \\*[^*]")
      FOREACH(fun ${FUN_IMAGETYPE})
        IF("${fun}" MATCHES "${F_MCISRG}")
          IF(MCISRG)
            file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/mialib_imagetype "${fun}\n")
          ENDIF(MCISRG)
        ELSE("${fun}" MATCHES "${F_MCISRG}")
          file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/mialib_imagetype "${fun}\n")
        ENDIF("${fun}" MATCHES "${F_MCISRG}")
      ENDFOREACH(fun)
      file(STRINGS ${infileName} FUN_IMAGELISTTYPE REGEX "^extern IMAGE \\*\\*[^*]")
      FOREACH(fun ${FUN_IMAGELISTTYPE})
        file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/mialib_imagelisttype "${fun}\n")
      ENDFOREACH(fun)
      file(STRINGS ${infileName} FUN_ERRORTYPE REGEX "^extern ERROR")
      FOREACH(fun ${FUN_ERRORTYPE})
        IF("${fun}" MATCHES "${F_ND}")
          file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/mialib_errortype_nd "${fun}\n")
        ELSE("${fun}" MATCHES "${F_ND}")
          IF("${fun}" MATCHES "${F_NM}")
            file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/mialib_errortype_nm "${fun}\n")
          ELSE("${fun}" MATCHES "${F_NM}")
            file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/mialib_errortype "${fun}\n")
          ENDIF("${fun}" MATCHES "${F_NM}")
        ENDIF("${fun}" MATCHES "${F_ND}")
      ENDFOREACH(fun)
    ENDFOREACH(infileName)

    #Create headers with new names in build directory
    EXECUTE_PROCESS(COMMAND ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/rename.sh ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses OUTPUT_QUIET)
    EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses INPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/fun2method_imagetype.py OUTPUT_QUIET)
    EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses INPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/fun2method_imagetype_multi.py OUTPUT_QUIET)
    EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses INPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/fun2method_imagetype_jimlist.py OUTPUT_QUIET)
    EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses INPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/fun2method_imagelisttype.py OUTPUT_QUIET)
    EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses INPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/fun2method_errortype.py OUTPUT_QUIET)
    EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses INPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/fun2method_errortype_d.py OUTPUT_QUIET)
    EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses INPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/fun2method_errortype_nd.py OUTPUT_QUIET)
    EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses INPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/fun2method_errortype_nm.py OUTPUT_QUIET)
    EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses INPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/fun2method_errortype_d_nm.py OUTPUT_QUIET)

    # remove all lines in between start and end in Jim.cc
    execute_process(COMMAND bash "-c" "sed -i '/\\/\\/start insert from fun2method_imagetype/,/\\/\\/end insert from fun2method_imagetype/{/\\/\\/start insert from fun2method_imagetype/!{/\\/\\/end insert from fun2method_imagetype/!d}}' Jim.h" WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses)
    execute_process(COMMAND bash "-c" "sed -i -ne '/\\/\\/start insert from fun2method_imagetype/ {p; r fun2method_imagetype.h' -e ':a; n; /\\/\\/end insert from fun2method_imagetype/ {p; b}; ba}; p' Jim.h" WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses)
    execute_process(COMMAND bash "-c" "sed -i '/\\/\\/start insert from fun2method_imagetype_multi/,/\\/\\/end insert from fun2method_imagetype_multi/{/\\/\\/start insert from fun2method_imagetype_multi/!{/\\/\\/end insert from fun2method_imagetype_multi/!d}}' Jim.h" WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses)
    execute_process(COMMAND bash "-c" "sed -i -ne '/\\/\\/start insert from fun2method_imagetype_multi/ {p; r fun2method_imagetype_multi.h' -e ':a; n; /\\/\\/end insert from fun2method_imagetype_multi/ {p; b}; ba}; p' Jim.h" WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses)
    execute_process(COMMAND bash "-c" "sed -i '/\\/\\/start insert from fun2method_imagetype_jimlist/,/\\/\\/end insert from fun2method_imagetype_jimlist/{/\\/\\/start insert from fun2method_imagetype_jimlist/!{/\\/\\/end insert from fun2method_imagetype_jimlist/!d}}' JimList.h" WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses)
    execute_process(COMMAND bash "-c" "sed -i -ne '/\\/\\/start insert from fun2method_imagetype_jimlist/ {p; r fun2method_imagetype_jimlist.h' -e ':a; n; /\\/\\/end insert from fun2method_imagetype_jimlist/ {p; b}; ba}; p' JimList.h" WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses)
    execute_process(COMMAND bash "-c" "sed -i '/\\/\\/start insert from fun2method_imagelisttype/,/\\/\\/end insert from fun2method_imagelisttype/{/\\/\\/start insert from fun2method_imagelisttype/!{/\\/\\/end insert from fun2method_imagelisttype/!d}}' JimList.h" WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses)
    execute_process(COMMAND bash "-c" "sed -i -ne '/\\/\\/start insert from fun2method_imagelisttype/ {p; r fun2method_imagelisttype.h' -e ':a; n; /\\/\\/end insert from fun2method_imagelisttype/ {p; b}; ba}; p' Jim.h" WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses)
    execute_process(COMMAND bash "-c" "sed -i '/\\/\\/start insert from fun2method_errortype/,/\\/\\/end insert from fun2method_errortype/{/\\/\\/start insert from fun2method_errortype/!{/\\/\\/end insert from fun2method_errortype/!d}}' Jim.h" WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses)
    execute_process(COMMAND bash "-c" "sed -i -ne '/\\/\\/start insert from fun2method_errortype/ {p; r fun2method_errortype.h' -e ':a; n; /\\/\\/end insert from fun2method_errortype/ {p; b}; ba}; p' Jim.h" WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses)
    execute_process(COMMAND bash "-c" "sed -i '/\\/\\/start insert from fun2method_errortype_d/,/\\/\\/end insert from fun2method_errortype_d/{/\\/\\/start insert from fun2method_errortype_d/!{/\\/\\/end insert from fun2method_errortype_d/!d}}' Jim.h" WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses)
    execute_process(COMMAND bash "-c" "sed -i -ne '/\\/\\/start insert from fun2method_errortype_d/ {p; r fun2method_errortype_d.h' -e ':a; n; /\\/\\/end insert from fun2method_errortype_d/ {p; b}; ba}; p' Jim.h" WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses)
    execute_process(COMMAND bash "-c" "sed -i '/\\/\\/start insert from fun2method_errortype_nd/,/\\/\\/end insert from fun2method_errortype_nd/{/\\/\\/start insert from fun2method_errortype_nd/!{/\\/\\/end insert from fun2method_errortype_nd/!d}}' Jim.h" WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses)
    execute_process(COMMAND bash "-c" "sed -i -ne '/\\/\\/start insert from fun2method_errortype_nd/ {p; r fun2method_errortype_nd.h' -e ':a; n; /\\/\\/end insert from fun2method_errortype_nd/ {p; b}; ba}; p' Jim.h" WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses)
    execute_process(COMMAND bash "-c" "sed -i '/\\/\\/start insert from fun2method_errortype_nm/,/\\/\\/end insert from fun2method_errortype_nm/{/\\/\\/start insert from fun2method_errortype_nm/!{/\\/\\/end insert from fun2method_errortype_nm/!d}}' Jim.h" WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses)
    execute_process(COMMAND bash "-c" "sed -i -ne '/\\/\\/start insert from fun2method_errortype_nm/ {p; r fun2method_errortype_nm.h' -e ':a; n; /\\/\\/end insert from fun2method_errortype_nm/ {p; b}; ba}; p' Jim.h" WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses)
    execute_process(COMMAND bash "-c" "sed -i '/\\/\\/start insert from fun2method_errortype_d_nm/,/\\/\\/end insert from fun2method_errortype_d_nm/{/\\/\\/start insert from fun2method_errortype_d_nm/!{/\\/\\/end insert from fun2method_errortype_d_nm/!d}}' Jim.h" WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses)
    execute_process(COMMAND bash "-c" "sed -i -ne '/\\/\\/start insert from fun2method_errortype_d_nm/ {p; r fun2method_errortype_d_nm.h' -e ':a; n; /\\/\\/end insert from fun2method_errortype_d_nm/ {p; b}; ba}; p' Jim.h" WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses)
    #Append the source files to Jim.cc
    file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/Jim.cc "#include \"${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/fun2method_imagetype.cc\"\n")
    file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/Jim.cc "#include \"${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/fun2method_imagetype_multi.cc\"\n")
    file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/Jim.cc "#include \"${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/fun2method_imagelisttype.cc\"\n")
    file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/Jim.cc "#include \"${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/fun2method_errortype.cc\"\n")
    file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/Jim.cc "#include \"${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/fun2method_errortype_d.cc\"\n")
    file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/Jim.cc "#include \"${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/fun2method_errortype_nd.cc\"\n")
    file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/Jim.cc "#include \"${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/fun2method_errortype_nm.cc\"\n")
    file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/Jim.cc "#include \"${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/fun2method_errortype_d_nm.cc\"\n")
    file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/JimList.cc "#include \"${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/fun2method_imagetype_jimlist.cc\"\n")
  endif(BUILD_WITH_MIALIB)
else(PYTHONINTERP_FOUND)
    MESSAGE(STATUS "Error: PythonInterp not found")
endif(PYTHONINTERP_FOUND)

###############################################################################
# Define the jiplib library.
if (BUILD_WITH_LIBLAS)
  add_library( ${JIPLIB_LIB_NAME} ${LASCLASS_H} ${LASCLASS_CC} ${IMGCLASS_H} ${IMGCLASS_CC} ${ALGOR_H} ${ALGOR_CC} ${FILECLASS_CC} ${FILECLASS_H} ${BASE_H} ${BASE_CC})
else(BUILD_WITH_LIBLAS)
  add_library( ${JIPLIB_LIB_NAME} ${IMGCLASS_H} ${IMGCLASS_CC} ${ALGOR_H} ${ALGOR_CC} ${FILECLASS_CC} ${FILECLASS_H} ${BASE_H} ${BASE_CC})
endif(BUILD_WITH_LIBLAS)

###############################################################################

SET_TARGET_PROPERTIES(${JIPLIB_LIB_NAME}
  PROPERTIES
  SOVERSION ${JIPLIB_SOVERSION}
  VERSION ${JIPLIB_VERSION}
)

target_link_libraries(
  ${JIPLIB_LIB_NAME}
  ${BOOST_LIBRARIES}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_SERIALIZATION_LIBRARY}
  ${BOOST_ARCHIVE_LIBRARY}
  )

if(BUILD_WITH_MIALIB)
  target_link_libraries(
    ${JIPLIB_LIB_NAME}
    ${MIA_LIB_NAME}
    )
endif(BUILD_WITH_MIALIB)

target_link_libraries(
  ${JIPLIB_LIB_NAME}
  ${PYTHON_LIBRARIES}
  gomp
  ${JSONCPP_LIBRARIES}
  ${GDAL_LIBRARIES}
  ${GSL_LIBRARIES}
  )

if(BUILD_WITH_FANN)
  target_link_libraries(
    ${JIPLIB_LIB_NAME}
    ${FANN_LIBRARIES}
    )
endif(BUILD_WITH_FANN)

########## SWIG #############
set(PYTHON_SRC_DIR src/swig)
# set(PYTHON_CC
#   ${PYTHON_SRC_DIR}/jiplib_wrap.cc
#   )

set(JIPLIB_I ${PYTHON_SRC_DIR}/jiplib.i)
FIND_PACKAGE(SWIG REQUIRED)
INCLUDE(${SWIG_USE_FILE})
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}/src)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

#test
INCLUDE_DIRECTORIES(/usr/local/include)

SET(CMAKE_SWIG_FLAGS "-keyword")
SET(CMAKE_SWIG_FLAGS "-DMIALIB=${MIA_BOOL}")
SET_PROPERTY(SOURCE ${JIPLIB_I} PROPERTY CPLUSPLUS ON)
#deprecated, use SWIG_ADD_LIBRARY instead for newer versions of cmake
# SWIG_ADD_MODULE(jiplib python ${JIPLIB_I})
 SWIG_ADD_LIBRARY(
   jiplib
   TYPE USE_BUILD_SHARED_LIBS
   LANGUAGE python
   SOURCES ${JIPLIB_I}
   )
SET_SOURCE_FILES_PROPERTIES(${JIPLIB_I} PROPERTIES SWIG_FLAGS "-W2")
SWIG_LINK_LIBRARIES(jiplib ${PYTHON_LIBRARIES} ${JIPLIB_LIB_NAME} ${BOOST_LIBRARIES})

# SET_PROPERTY(SOURCE ${PYTHON_SRC_DIR}/jiplib.i PROPERTY CPLUSPLUS ON)
# SWIG_ADD_MODULE(jiplib python ${PYTHON_SRC_DIR}/jiplib.i)
# SET_SOURCE_FILES_PROPERTIES(${PYTHON_SRC_DIR}/jiplib.i PROPERTIES SWIG_FLAGS "-W2")
# SWIG_LINK_LIBRARIES(jiplib ${PYTHON_LIBRARIES} ${JIPLIB_LIB_NAME} ${BOOST_LIBRARIES})

########## INSTALL ##########
INSTALL (TARGETS ${JIPLIB_LIB_NAME} LIBRARY DESTINATION lib)
INSTALL (FILES ${IMGCLASS_H} DESTINATION include/jiplib/imageclasses)
INSTALL (FILES ${APP_H} DESTINATION include/jiplib/apps)
INSTALL (FILES ${BASE_H} DESTINATION include/jiplib/base)
INSTALL (FILES ${ALGOR_H} DESTINATION include/jiplib/algorithms)
INSTALL (FILES ${FILECLASS_H} DESTINATION include/jiplib/fileclasses)
INSTALL (FILES ${LASCLASS_H} DESTINATION include/jiplib/lasclasses)
EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())" OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
SET(PYTHON_INSTALL_DIR ${PYTHON_SITE_PACKAGES} CACHE PATH "Install directory for Python library")
INSTALL(TARGETS ${SWIG_MODULE_jiplib_REAL_NAME} LIBRARY DESTINATION ${PYTHON_INSTALL_DIR})
INSTALL(FILES ${CMAKE_BINARY_DIR}/jiplib.py DESTINATION ${PYTHON_INSTALL_DIR})
install (FILES "${CMAKE_CURRENT_BINARY_DIR}/config_jiplib.h" DESTINATION ${PROJECT_INCLUDE_DIR})
###############################################################################
