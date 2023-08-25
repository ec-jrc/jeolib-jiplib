###############################################################################
# CMakeLists build script for cmake
# Author(s): Pieter.Kempeneers@ec.europa.eu
# Copyright (C) 2016-2023 European Union (Joint Research Centre)
#
# This file is part of jiplib.

# jiplib is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# jiplib is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with jiplib.  If not, see <https://www.gnu.org/licenses/>.
###############################################################################
cmake_minimum_required(VERSION 3.9)

# The project's name and version
project(jiplib)

# Name of C++ library
SET(JIPLIB_LIB_NAME jiplib)

enable_testing()
include(CTest)

SET(JIPLIB_VERSION_MAJOR 1)
SET(JIPLIB_VERSION_MINOR 1)
SET(JIPLIB_VERSION_PATCH 2)
SET(JIPLIB_VERSION "${JIPLIB_VERSION_MAJOR}.${JIPLIB_VERSION_MINOR}.${JIPLIB_VERSION_PATCH}")
SET(JIPLIB_SOVERSION "${JIPLIB_VERSION_MAJOR}")
SET(JIPLIB_PACKAGE_BUGREPORT "pieter.kempeneers@ec.europa.eu")

SET(CLASSIF 0 CACHE STRING "set a variable that can be used in Jim.cc")
SET(ODOITHIN 0 CACHE STRING "set a variable that can be used in Jim.cc")
SET(MCISRG 1 CACHE STRING "set a variable that can be used in Jim.cc")

SET(BUILD_WITH_LIBLAS FALSE CACHE BOOL "Choose if jiplib is to be built with liblas")
SET(BUILD_WITH_FANN TRUE CACHE BOOL "Choose if jiplib is to be built with fann")
SET(BUILD_WITH_PYTHON TRUE CACHE BOOL "Choose if jiplib is to be built with PYTHON support")
SET(PROCESS_IN_PARALLEL TRUE CACHE BOOL "Choose if jiplib should be run in parallel")
SET(BUILD_WITH_MIALLIB TRUE CACHE BOOL "Choose if jiplib should be compiled with miallib")
#from https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling
#By default if you don't change any RPATH related settings, CMake will
# link the executables and shared libraries with full RPATH to all used
# libraries in the build tree. When installing, it will clear the RPATH of
# these targets so they are installed with an empty RPATH
SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
SET(CMAKE_INSTALL_RPATH "$\{ORIGIN\}:${CMAKE_CURRENT_BINARY_DIR}")
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

SET(PROJECT_INCLUDE_DIR include/jiplib)
SET(PROJECT_TEST_DIR test)

SET(CMAKE_COLOR_MAKEFILE ON)

if(BUILD_WITH_MIALLIB)
  SET(MIAL_BOOL 1)
else(BUILD_WITH_MIALLIB)
  SET(MIAL_BOOL 0)
endif(BUILD_WITH_MIALLIB)

option (BUILD_SHARED_LIBS "Build with shared library" ON)

# Platform and compiler specific settings

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-long-long")
  message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has C++11 support.")
else(COMPILER_SUPPORTS_CXX11)
  message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

########## BUILDING INSTALLLER ##########
# build a CPack driven installer package
include (InstallRequiredSystemLibraries)
SET(CPACK_RESOURCE_FILE_LICENSE
     "${CMAKE_CURRENT_SOURCE_DIR}/COPYING")
SET(CPACK_PACKAGE_VERSION_MAJOR "${JIPLIB_VERSION_MAJOR}")
SET(CPACK_PACKAGE_VERSION_MINOR "${JIPLIB_VERSION_MINOR}")
SET(CPACK_PACKAGE_VERSION_PATCH "${JIPLIB_VERSION_PATCH}")

# Set files to ignore
list(APPEND CPACK_SOURCE_IGNORE_FILES "_CPack_Packages")
list(APPEND CPACK_SOURCE_IGNORE_FILES ".gz")
list(APPEND CPACK_SOURCE_IGNORE_FILES ".bz2")
list(APPEND CPACK_SOURCE_IGNORE_FILES ".zip")
list(APPEND CPACK_SOURCE_IGNORE_FILES ".svn")
list(APPEND CPACK_SOURCE_IGNORE_FILES ".git")
list(APPEND CPACK_SOURCE_IGNORE_FILES "HOWTORELEASE.txt")
list(APPEND CPACK_SOURCE_IGNORE_FILES "CMakeCache.txt")
list(APPEND CPACK_SOURCE_IGNORE_FILES "CPackConfig.cmake")
list(APPEND CPACK_SOURCE_IGNORE_FILES "schemas")
list(APPEND CPACK_SOURCE_IGNORE_FILES "/build/;~$;${CPACK_SOURCE_IGNORE_FILES}")

include (CPack)

add_custom_target(dist COMMAND ${CMAKE_MAKE_PROGRAM} package_source)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/modules")

# Make sure Boost libraries are found
SET(Boost_DEBUG 0)
find_package(Boost COMPONENTS filesystem system serialization REQUIRED)

if(BUILD_WITH_PYTHON)
  SET(CREATE_WHEEL TRUE CACHE BOOL "Choose if jiplib PYTHON wheel is to be created")
endif(BUILD_WITH_PYTHON)

SET(PYTHON3 ON CACHE BOOL "set ON for PYTHON3")
if(PYTHON3)
  FIND_PACKAGE( PythonInterp 3.5 REQUIRED )
  FIND_PACKAGE( PythonLibs 3.5 REQUIRED )
else(PYTHON3)
  FIND_PACKAGE( PythonInterp 2.7 REQUIRED )
  FIND_PACKAGE( PythonLibs 2.7 REQUIRED )
endif(PYTHON3)

if(PYTHONLIBS_FOUND)
  message(STATUS "Python version found: ${PYTHONLIBS_VERSION_STRING}")
  message(STATUS "Found Python library: ${PYTHON_LIBRARY}")
  message(STATUS "Found Python include dir: ${PYTHON_INCLUDE_DIR}")
else(PYTHONLIBS_FOUND)
  message(STATUS "Warning: PYTHON exact version ${PYTHON_VERSION} not found, trying default")
  FIND_PACKAGE(PythonLibs REQUIRED)
endif(PYTHONLIBS_FOUND)

function(gdal_find_major _major)
    file(READ ${GDAL_INCLUDE_DIR}/gdal_version.h versionfile)
    string(REGEX MATCH "GDAL_VERSION_MAJOR[\t ]+([0-9]+)" _ ${versionfile})
    SET(MAJOR ${CMAKE_MATCH_1})
    string(REGEX MATCH "GDAL_VERSION_MINOR[\t ]+([0-9]+)" _ ${versionfile})
    SET(MINOR ${CMAKE_MATCH_1})
    string(REGEX MATCH "GDAL_VERSION_REV[\t ]+([0-9]+)" _ ${versionfile})
    SET(REV ${CMAKE_MATCH_1})
    SET(${_major} ${MAJOR} PARENT_SCOPE)
endfunction(gdal_find_major)

function(gdal_find_minor _minor)
    file(READ ${GDAL_INCLUDE_DIR}/gdal_version.h versionfile)
    string(REGEX MATCH "GDAL_VERSION_MAJOR[\t ]+([0-9]+)" _ ${versionfile})
    SET(MAJOR ${CMAKE_MATCH_1})
    string(REGEX MATCH "GDAL_VERSION_MINOR[\t ]+([0-9]+)" _ ${versionfile})
    SET(MINOR ${CMAKE_MATCH_1})
    string(REGEX MATCH "GDAL_VERSION_REV[\t ]+([0-9]+)" _ ${versionfile})
    SET(REV ${CMAKE_MATCH_1})
    SET(${_minor} ${MINOR} PARENT_SCOPE)
endfunction(gdal_find_minor)

function(gdal_find_rev _rev)
    file(READ ${GDAL_INCLUDE_DIR}/gdal_version.h versionfile)
    string(REGEX MATCH "GDAL_VERSION_MAJOR[\t ]+([0-9]+)" _ ${versionfile})
    SET(MAJOR ${CMAKE_MATCH_1})
    string(REGEX MATCH "GDAL_VERSION_MINOR[\t ]+([0-9]+)" _ ${versionfile})
    SET(MINOR ${CMAKE_MATCH_1})
    string(REGEX MATCH "GDAL_VERSION_REV[\t ]+([0-9]+)" _ ${versionfile})
    SET(REV ${CMAKE_MATCH_1})
    SET(${_rev} ${REV} PARENT_SCOPE)
endfunction(gdal_find_rev)

function(gdal_find_version _version)
    file(READ ${GDAL_INCLUDE_DIR}/gdal_version.h versionfile)
    string(REGEX MATCH "GDAL_VERSION_MAJOR[\t ]+([0-9]+)" _ ${versionfile})
    SET(MAJOR ${CMAKE_MATCH_1})
    string(REGEX MATCH "GDAL_VERSION_MINOR[\t ]+([0-9]+)" _ ${versionfile})
    SET(MINOR ${CMAKE_MATCH_1})
    string(REGEX MATCH "GDAL_VERSION_REV[\t ]+([0-9]+)" _ ${versionfile})
    SET(REV ${CMAKE_MATCH_1})
    SET(${_version} ${MAJOR}.${MINOR}.${REV} PARENT_SCOPE)
endfunction(gdal_find_version)

SET(GDAL3 ON CACHE BOOL "set ON for GDAL3")
if(GDAL3)
  find_package(GDAL 3.00 REQUIRED)
else(GDAL3)
  find_package(GDAL 2.00 REQUIRED)
endif(GDAL3)

if(GDAL_FOUND)
  gdal_find_version(GDAL_VERSION)
  gdal_find_major(GDAL_VERSION_MAJOR)
  gdal_find_minor(GDAL_VERSION_MINOR)

  SET(CMAKE_SWIG_FLAGS "-DGDAL_VERSION_MAJOR=${GDAL_VERSION_MAJOR}")
  SET(CMAKE_SWIG_FLAGS "-DGDAL_VERSION_MINOR=${GDAL_VERSION_MINOR}")
  # gdal_find_version(GDAL_VERSION)
  # SET(GDAL_VERSION_MAJOR "2" CACHE STRING " ")
  message(STATUS "GDAL version found: ${GDAL_VERSION}")
  message(STATUS "GDAL version major: ${GDAL_VERSION_MAJOR}")
  message(STATUS "GDAL version minor: ${GDAL_VERSION_MINOR}")
endif()

find_package(GSL REQUIRED)
if(GSL_FOUND)
  message(STATUS "Found GSL: ${GSL_LIBRARIES}")
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)
if(JSONCPP_FOUND)
  message(STATUS "JSONCPP package found: OK")
else(JSONCPP_FOUND)
  message(STATUS "Warning: JSONCPP package not found!")
endif(JSONCPP_FOUND)

# link_libraries(${JSONCPP_LIBRARIES})

if(PROCESS_IN_PARALLEL)
  find_package(OpenMP)
  if (OPENMP_FOUND)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    SET(PARALLEL_BOOL 1)
    message(STATUS "parallel processing: ${PARALLEL_BOOL}")
  endif(OPENMP_FOUND)
else(PROCESS_IN_PARALLEL)
  SET(PARALLEL_BOOL 0)
  message(STATUS "parallel processing: ${PARALLEL_BOOL}")
endif(PROCESS_IN_PARALLEL)

###############################################################################
# Setup configure file

include(GNUInstallDirs)
SET(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR}/jiplib
  CACHE PATH "Location of header files" )
SET(SYSCONFIG_INSTALL_DIR ${CMAKE_INSTALL_SYSCONFDIR}/jiplib
  CACHE PATH "Location of configuration files" )

# configure a header file to pass some of the CMake settings to the source code
configure_file (
  "cmake-config.h.in"
  "config_jiplib.h"
  )

# # configure a header file to pass some of the CMake settings to the source code
# configure_file (
#   "cmake-config.in"
#   "${CMAKE_CURRENT_BINARY_DIR}/jiplib-config"
#   @ONLY
#   )

# configure_file (
#   "cmake.pc.in"
#   "jiplib.pc"
#   @ONLY
#   )

#Generating Package Files
include(CMakePackageConfigHelpers)

configure_package_config_file(jiplib-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/jiplib-config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/jiplib
  PATH_VARS INCLUDE_INSTALL_DIR SYSCONFIG_INSTALL_DIR
  )
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/jiplib-config-version.cmake
  VERSION ${JIPLIB_VERSION}
  COMPATIBILITY AnyNewerVersion
  )
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/jiplib-config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/jiplib-config-version.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/jiplib )

configure_file ( "cmake.pc.in" "jiplib.pc"  @ONLY)


SET(BUILD_WITH_MIALLIB TRUE CACHE BOOL "JIPlib needs to be built with mialiab")

# Instruct CMake to inspect the following subfolders (with CMakeLists.txt in each subfolder)
add_subdirectory ("${PROJECT_TEST_DIR}")

if (BUILD_WITH_FANN)
  FIND_PACKAGE(FANN 2.2 REQUIRED)
endif(BUILD_WITH_FANN)

###############################################################################

#we will use a copy of the imageclasses in the binary build directory, where we can change the content (for MIALLIB)

message(STATUS "cmake_current_binary_dir: ${CMAKE_CURRENT_BINARY_DIR}")
file(COPY ${PROJECT_SOURCE_DIR}/src DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

SET(BASE_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR}/src/base)
SET(ALGOR_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR}/src/algorithms)
SET(FILECLASS_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR}/src/fileclasses)
SET(IMGCLASS_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses)
SET(LASCLASS_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR}/src/lasclasses)
SET(APP_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR}/src/apps)
# SET(PYTHON_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR}/src/swig)
# SET(PYTHON_SRC_DIR src/swig)

SET(BASE_H
  ${BASE_SRC_DIR}/IndexValue.h
  ${BASE_SRC_DIR}/typeconversion.h
  ${BASE_SRC_DIR}/Optionjl.h
  ${BASE_SRC_DIR}/PosValue.h
  ${BASE_SRC_DIR}/Vector2d.h
  )

SET(BASE_CC
  ${BASE_SRC_DIR}/typeconversion.cc
  ${BASE_SRC_DIR}/Vector2d.cc
  )

SET(ALGOR_H
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
  SET(ALGOR_H ${ALGOR_H}
    ${ALGOR_SRC_DIR}/myfann_cpp.h
    )
endif(BUILD_WITH_FANN)

SET(ALGOR_CC
  ${ALGOR_SRC_DIR}/ConfusionMatrix.cc
  ${ALGOR_SRC_DIR}/CostFactorySVM.cc
  ${ALGOR_SRC_DIR}/CostFactorySVM.h
  ${ALGOR_SRC_DIR}/Egcs.cc
  ${ALGOR_SRC_DIR}/Filter.cc
  ${ALGOR_SRC_DIR}/Filter2d.cc
  ${ALGOR_SRC_DIR}/ImgRegression.cc
  ${ALGOR_SRC_DIR}/svm.cpp
  )

SET(FILECLASS_H
  ${FILECLASS_SRC_DIR}/FileReaderAscii.h
  )

SET(FILECLASS_CC
  ${FILECLASS_SRC_DIR}/FileReaderAscii.cc
  )

SET(LASCLASS_H
  ${LASCLASS_SRC_DIR}/FileReaderLas.h
  )

SET(LASCLASS_CC
  ${LASCLASS_SRC_DIR}/FileReaderLas.cc
  )

SET(APP_H
  ${APP_SRC_DIR}/AppFactory.h
  )

SET(IMGCLASS_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses)

SET(IMGCLASS_H
  ${IMGCLASS_SRC_DIR}/Jim.h
  ${IMGCLASS_SRC_DIR}/Mia.h
  ${IMGCLASS_SRC_DIR}/JimList.h
  ${IMGCLASS_SRC_DIR}/VectorOgr.h
  ${IMGCLASS_SRC_DIR}/jlcrop_lib.h
  ${IMGCLASS_SRC_DIR}/jlfilter1d_lib.h
  ${IMGCLASS_SRC_DIR}/jlfilter2d_lib.h
  ${IMGCLASS_SRC_DIR}/jlclassify_lib.h
  ${IMGCLASS_SRC_DIR}/jlsml_lib.h
  ${IMGCLASS_SRC_DIR}/jlsetmask_lib.h
  ${IMGCLASS_SRC_DIR}/jloperators_lib.h
  ${IMGCLASS_SRC_DIR}/jlextractimg_lib.h
  ${IMGCLASS_SRC_DIR}/jldem_lib.h
  )

SET(IMGCLASS_CC
  ${IMGCLASS_SRC_DIR}/Jim.cc
  ${IMGCLASS_SRC_DIR}/Mia.cc
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
  ${IMGCLASS_SRC_DIR}/jlsml_lib.cc
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
  ${IMGCLASS_SRC_DIR}/jldem_lib.cc
  )

if(BUILD_WITH_FANN)
  SET(IMGCLASS_CC ${IMGCLASS_CC}
  ${IMGCLASS_SRC_DIR}/jlann_lib.cc
  ${IMGCLASS_SRC_DIR}/jlannogr_lib.cc
  )
endif(BUILD_WITH_FANN)

if(BUILD_WITH_MIALLIB)
  find_package(MIALLIB REQUIRED)
  if(MIALLIB_FOUND)
    ADD_LIBRARY(miallib::miallib SHARED IMPORTED)
    SET_TARGET_PROPERTIES(
      miallib::miallib
      PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${MIALLIB_INCLUDE_DIRS}"
      IMPORTED_LOCATION ${MIALLIB_LIBRARY}
      )
    message(STATUS "MIALLIB was found")
    message(STATUS "MIALLIB was found with version ${MIALLIB_VERSION}")
    message(STATUS "Found MIALLIB libraries: ${MIALLIB_LIBRARIES}")
    message(STATUS "Found MIALLIB include dir: ${MIALLIB_INCLUDE_DIRS}")
  endif()

  SET(MIAL_BOOL 1)
  message(STATUS "build with miallib: ${MIAL_BOOL}")

else(BUILD_WITH_MIALLIB)
  SET(MIAL_BOOL 0)
  message(STATUS "build with miallib: ${MIAL_BOOL}")
endif(BUILD_WITH_MIALLIB)

###############################################################################

find_package(PythonInterp)
if(PYTHONINTERP_FOUND)
  # file(WRITE ${JIPLIB_DOC} "//This is an automatically generated doc file, do not edit")
  #todo: treat miallib functions that return: int, long int, double, G_TYPE
  # G_TYPE: getpixval
  # G_TYPE* min_max
  # long int thresholdRegion_Size
  # long int thresholdRegion_Contrast
  # double julian_date
  if(BUILD_WITH_MIALLIB)
    MESSAGE(STATUS "PythonInterp was found")

    file(GLOB MIALLIB_HEADER_FILES ${MIALLIB_INCLUDE_DIR}/miallib_*.h)
    MESSAGE(STATUS "MIALLIB_HEADER_FILES: ${MIALLIB_HEADER_FILES}")
    IF("${MIALLIB_HEADER_FILES}" STREQUAL "")
      MESSAGE(FATAL_ERROR "Error: no header files found for miallib")
    ENDIF("${MIALLIB_HEADER_FILES}" STREQUAL "")
    file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/miallib_imagetype)
    file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/miallib_imagelisttype)
    file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/miallib_errortype_nd)
    file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/miallib_errortype_nm)
    file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/miallib_errortype)
    #Miallib functions have been designed as destructive. We will therefore create a copy to avoid the destructive behaviour. The destructive method will still be available with the prefix d_
    #The following functions F_ND from miallib do not modify the input image and should therefore not create a copy. They will be treated in the same way as the destructive functions (without the prefix d_)
    SET(F_ND "dumpxyz|szcompat|szgeocompat|iminfo|tiffinfo|writeGnuPlot3D|vectorizeImage|IsPartitionEqual|IsPartitionFiner|dendro|getfirstmaxpos|volume|imequalp|getmax|getminmax")
    #The following functions F_NM from miallib are not compatible for multi-band processing
    SET(F_NM "addframebox|subframebox")
    #These functions must only be included if MCISRG is set
    SET(F_MCISRG "labelImage|segmentImage")
    #Create a text file with the list of headers to MIALLIB functions, e.g., miallib_imagetype for functions returning IMAGE *
    FOREACH(infileName ${MIALLIB_HEADER_FILES})
      MESSAGE(STATUS "Process file: ${infileName}")
      file(STRINGS ${infileName} FUN_IMAGETYPE REGEX "^extern IMAGE \\*[^*]")
      FOREACH(fun ${FUN_IMAGETYPE})
        IF("${fun}" MATCHES "${F_MCISRG}")
          IF(MCISRG)
            file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/miallib_imagetype "${fun}\n")
          ENDIF(MCISRG)
        ELSE("${fun}" MATCHES "${F_MCISRG}")
          file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/miallib_imagetype "${fun}\n")
        ENDIF("${fun}" MATCHES "${F_MCISRG}")
      ENDFOREACH(fun)
      file(STRINGS ${infileName} FUN_IMAGELISTTYPE REGEX "^extern IMAGE \\*\\*[^*]")
      FOREACH(fun ${FUN_IMAGELISTTYPE})
        file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/miallib_imagelisttype "${fun}\n")
      ENDFOREACH(fun)
      file(STRINGS ${infileName} FUN_ERRORTYPE REGEX "^extern ERROR")
      FOREACH(fun ${FUN_ERRORTYPE})
        IF("${fun}" MATCHES "${F_ND}")
          file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/miallib_errortype_nd "${fun}\n")
        ELSE("${fun}" MATCHES "${F_ND}")
          IF("${fun}" MATCHES "${F_NM}")
            file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/miallib_errortype_nm "${fun}\n")
          ELSE("${fun}" MATCHES "${F_NM}")
            file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/miallib_errortype "${fun}\n")
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
  endif(BUILD_WITH_MIALLIB)
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

# add_definitions(-DMIALLIB_DLL)
# add_definitions("-DBOOST_ALL_NO_LIB")
# add_definitions(-DHAVE_CONFIG_H)
# add_definitions(-DGSL_DLL)
# add_definitions(-DFANN_DLL)
TARGET_COMPILE_DEFINITIONS(${JIPLIB_LIB_NAME} PUBLIC MIALLIB_DLL)
if (BUILD_WITH_FANN)
  TARGET_COMPILE_DEFINITIONS(${JIPLIB_LIB_NAME} PUBLIC FANN_DLL)
endif(BUILD_WITH_FANN)
TARGET_COMPILE_DEFINITIONS(${JIPLIB_LIB_NAME} PUBLIC GSL_DLL)
TARGET_COMPILE_DEFINITIONS(${JIPLIB_LIB_NAME} PUBLIC HAVE_CONFIG_H)
TARGET_INCLUDE_DIRECTORIES(${JIPLIB_LIB_NAME} PUBLIC ${Boost_INCLUDE_DIRS})
TARGET_COMPILE_DEFINITIONS(${JIPLIB_LIB_NAME} PUBLIC BOOST_ALL_NO_LIB)

TARGET_INCLUDE_DIRECTORIES(${JIPLIB_LIB_NAME} PUBLIC ${GDAL_INCLUDE_DIRS})
TARGET_INCLUDE_DIRECTORIES(${JIPLIB_LIB_NAME} PUBLIC ${GSL_INCLUDE_DIRS})
TARGET_INCLUDE_DIRECTORIES(${JIPLIB_LIB_NAME} PUBLIC ${JSONCPP_INCLUDE_DIRS})
TARGET_INCLUDE_DIRECTORIES(${JIPLIB_LIB_NAME} PUBLIC "${PROJECT_BINARY_DIR}")# needed for config_jiplib.h
TARGET_INCLUDE_DIRECTORIES(${JIPLIB_LIB_NAME} PUBLIC "${PROJECT_BINARY_DIR}/src")#  this is where sources will go
TARGET_INCLUDE_DIRECTORIES(${JIPLIB_LIB_NAME} PUBLIC ${FANN_INCLUDE_DIRS})
TARGET_INCLUDE_DIRECTORIES(${JIPLIB_LIB_NAME} PUBLIC ${MIALLIB_INCLUDE_DIRS})

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

if(BUILD_WITH_MIALLIB)
  message(STATUS "Found MIALLIB_LIBRARY: ${MIALLIB_LIBRARIES}")
  TARGET_INCLUDE_DIRECTORIES(${JIPLIB_LIB_NAME} PUBLIC
    $<BUILD_INTERFACE:${MIALLIB_INCLUDE_DIR}
    )
  target_link_libraries(
    ${JIPLIB_LIB_NAME}
    miallib::miallib
    )
endif(BUILD_WITH_MIALLIB)

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

if(BUILD_WITH_PYTHON)
  ########## SWIG #############
  SET(PYTHON_SRC_DIR src/swig)
  # SET(PYTHON_CC
  #   ${PYTHON_SRC_DIR}/jiplib_wrap.cc
  #   )
  SET(JIPLIB_I ${PYTHON_SRC_DIR}/jiplib.i)
  FIND_PACKAGE(SWIG REQUIRED)
  INCLUDE(${SWIG_USE_FILE})
  TARGET_INCLUDE_DIRECTORIES(${JIPLIB_LIB_NAME} PUBLIC ${PYTHON_INCLUDE_PATH})
endif(BUILD_WITH_PYTHON)
#we need INCLUDE_DIRECTORIES here instead of TARGET_INCLUDE_DIRECTORIES
# for subfolder WORKING_DIRECTORY
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}/src)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES(${MIALLIB_INCLUDE_DIRS})

SET(CMAKE_SWIG_FLAGS "-keyword")
SET(CMAKE_SWIG_FLAGS "-DMIALLIB=${MIAL_BOOL}")
SET_PROPERTY(SOURCE ${JIPLIB_I} PROPERTY CPLUSPLUS ON)

if(${CMAKE_VERSION} VERSION_LESS "3.8.0")
  SWIG_ADD_MODULE(jiplib python ${JIPLIB_I})
else(${CMAKE_VERSION} VERSION_LESS "3.8.0")
SWIG_ADD_LIBRARY(
  jiplib
  TYPE USE_BUILD_SHARED_LIBS
  LANGUAGE python
  SOURCES ${JIPLIB_I}
  )
endif(${CMAKE_VERSION} VERSION_LESS "3.8.0")

SET_SOURCE_FILES_PROPERTIES(${JIPLIB_I} PROPERTIES SWIG_FLAGS "-W2")
SWIG_LINK_LIBRARIES(jiplib ${PYTHON_LIBRARIES} ${JIPLIB_LIB_NAME} ${BOOST_LIBRARIES})

if(BUILD_WITH_PYTHON)
  SET(PYTHON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/python)
  SET(JIPLIB_INIT ${PYTHON_SOURCE_DIR}/__init__.py)

  if(CREATE_WHEEL)
    MESSAGE(STATUS "creating python wheel: ${CREATE_WHEEL}")
    MESSAGE(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
    MESSAGE(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
    MESSAGE(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
    MESSAGE(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")
    MESSAGE(STATUS "CMAKE_INSTALL_LIBDIR: ${CMAKE_INSTALL_LIBDIR}")
    MESSAGE(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
    SET(WHEEL_DIR ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME})

    add_custom_command(TARGET _${JIPLIB_LIB_NAME}
      POST_BUILD
      COMMAND mkdir -p ${WHEEL_DIR}
      COMMAND cp ${JIPLIB_INIT} ${WHEEL_DIR}
      COMMAND cp ${PYTHON_SOURCE_DIR}/setup.py ${CMAKE_CURRENT_BINARY_DIR}
      COMMAND cp ${CMAKE_BINARY_DIR}/_${JIPLIB_LIB_NAME}.so ${WHEEL_DIR}
      COMMAND cp ${CMAKE_BINARY_DIR}/${JIPLIB_LIB_NAME}.py ${WHEEL_DIR}
      #todo: create links for jiplib?
      # ${JIPLIB_VERSION} #is the full version
      # ${JIPLIB_SO_VERSION} #is the major version
      COMMAND cp ${CMAKE_BINARY_DIR}/lib${JIPLIB_LIB_NAME}.so.${JIPLIB_VERSION} ${WHEEL_DIR}
      COMMAND ln -s ${WHEEL_DIR}/lib${JIPLIB_LIB_NAME}.so.${JIPLIB_VERSION} ${WHEEL_DIR}/lib${JIPLIB_LIB_NAME}.so.${JIPLIB_VERSION_MAJOR}
      COMMAND ln -s ${WHEEL_DIR}/lib${JIPLIB_LIB_NAME}.so.${JIPLIB_VERSION_MAJOR} ${WHEEL_DIR}/lib${JIPLIB_LIB_NAME}.so
      #todo: get major version from miallib config file
      COMMAND cp ${MIALLIB_LIBRARIES}.${MIALLIB_VERSION} ${WHEEL_DIR}
      #todo: create links for miallib?
      COMMAND pip3 wheel ${CMAKE_CURRENT_BINARY_DIR}
      COMMAND rm -rf ${WHEEL_DIR}
      COMMENT "create wheel directory"
      # INSTALL (TARGETS *.whl LIBRARY DESTINATION lib)
      # INSTALL(FILES *.whl
      #   DESTINATION ${CMAKE_INSTALL_LIBDIR}/${JIPLIB_LIB_NAME}
      #   )
      COMMAND mkdir ${CMAKE_INSTALL_PREFIX}/${JIPLIB_LIB_NAME}
    )
  endif(CREATE_WHEEL)
endif(BUILD_WITH_PYTHON)

########## INSTALL ##########
# MESSAGE(STATUS "install library in: ${CMAKE_INSTALL_PREFIX}")
INSTALL (TARGETS ${JIPLIB_LIB_NAME} LIBRARY DESTINATION lib)
# INSTALL (FILES ${IMGCLASS_H} DESTINATION include/jiplib/imageclasses)
# INSTALL (FILES ${APP_H} DESTINATION include/jiplib/apps)
# INSTALL (FILES ${BASE_H} DESTINATION include/jiplib/base)
# INSTALL (FILES ${ALGOR_H} DESTINATION include/jiplib/algorithms)
# INSTALL (FILES ${FILECLASS_H} DESTINATION include/jiplib/fileclasses)
# INSTALL (FILES ${LASCLASS_H} DESTINATION include/jiplib/lasclasses)


# if(BUILD_WITH_PYTHON)
#   EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])" OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
#   # SET(PYTHON_INSTALL_DIR ${PYTHON_SITE_PACKAGES}/${JIPLIB_LIB_NAME} CACHE PATH "Install directory for Python library")
#   #test
#   SET("PYTHON_INSTALL_DIR /home/kempepi/repositories/jeolib-jiplib/build/_skbuild/linux-x86_64-3.9/cmake-install/"/${JIPLIB_LIB_NAME} CACHE PATH "Install directory for Python library")
#   INSTALL(TARGETS ${SWIG_MODULE_JIPLIB_REAL_NAME} LIBRARY DESTINATION ${PYTHON_INSTALL_DIR})
#   MESSAGE(STATUS "Python install directory: ${PYTHON_INSTALL_DIR}")
#   MESSAGE(STATUS "Python executable: ${PYTHON_EXECUTABLE}")
#   INSTALL(FILES ${CMAKE_BINARY_DIR}/${JIPLIB_LIB_NAME}.py DESTINATION ${PYTHON_INSTALL_DIR})
#   INSTALL(FILES ${CMAKE_BINARY_DIR}/_${JIPLIB_LIB_NAME}.so DESTINATION ${PYTHON_INSTALL_DIR})
#   INSTALL(FILES ${JIPLIB_INIT} DESTINATION ${PYTHON_INSTALL_DIR})
# endif(BUILD_WITH_PYTHON)
# INSTALL (FILES "${CMAKE_CURRENT_BINARY_DIR}/config_jiplib.h" DESTINATION ${PROJECT_INCLUDE_DIR})
###############################################################################
