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
cmake_minimum_required(VERSION 2.6.0)
#cmake_minimum_required(VERSION 2.8)
###############################################################################

# The project's name and version
project(jiplib)

include(CTest)

set (JIPLIB_VERSION_MAJOR 0)
set (JIPLIB_VERSION_MINOR 1)
set (JIPLIB_VERSION_PATCH 0)
set (JIPLIB_VERSION "${JIPLIB_VERSION_MAJOR}.${JIPLIB_VERSION_MINOR}.${JIPLIB_VERSION_PATCH}")
set (JIPLIB_SOVERSION "${JIPLIB_VERSION_MAJOR}")

set(CLASSIF 0 CACHE STRING "set a variable that can be used in jim.cc")
set(ODOITHIN 0 CACHE STRING "set a variable that can be used in jim.cc")

set (PROJECT_BINARY_DIR bin)
set (PROJECT_LIBRARY_DIR lib)
set (PROJECT_INCLUDE_DIR include/jiplib)
set (PROJECT_DOC_DIR doc)
set (PROJECT_SOURCE_DIR src)
set (PROJECT_TEST_DIR test)

# configure a header file to pass some of the CMake settings to the source code
configure_file (
  "cmake-config.h.in"
  "config.h"
  )

option (BUILD_SHARED_LIBS "Build with shared library" ON)

# Platform and compiler specific settings

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
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

# Make sure pktools version 3.0.0 is found
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/modules")
find_package(PKTOOLS CONFIG REQUIRED)
if(PKTOOLS_FOUND)
  include_directories(${PKTOOLS_INCLUDE_DIRS})
else(PKTOOLS_FOUND)
  message(WARNING "pktools not found, set manually")
  set(PKTOOLS_INCLUDE_DIR /usr/local/include/pktools CACHE PATH "Include PATH for PKTOOLS")
  include_directories(${PKTOOLS_INCLUDE_DIR})
  # set(PKTOOLS_LIB_PATH /usr/local/lib CACHE PATH "Library PATH for PKTOOLS")
endif()


#should replace following line
# EXECUTE_PROCESS(COMMAND pkg-config --cflags pktools OUTPUT_VARIABLE PKTOOLS_INCLUDE_DIRS OUTPUT_STRIP_TRAILING_WHITESPACE)
# set(PKTOOLS_INCLUDE_DIR ${PKTOOLS_INCLUDE_DIRS} CACHE PATH "Include PATH for PKTOOLS")
# #should not be needed with FindPktools


# Make sure mia is found
set(MIA_LIB_NAME mialib_python)
set(MIA_INCLUDE_DIR /usr/local/include/mialib CACHE PATH "Include PATH for MIA")
set(MIA_LIB_PATH /usr/local/lib CACHE PATH "Library PATH for MIA")

INCLUDE_DIRECTORIES(${MIA_INCLUDE_DIR})
LINK_DIRECTORIES(${MIA_LIB_PATH})

find_package(GDAL 2.00 REQUIRED)
if(GDAL_FOUND)
  include_directories(${GDAL_INCLUDE_DIRS})
endif()

find_package(GSL REQUIRED)
if(GSL_FOUND)
  include_directories(${GSL_INCLUDE_DIRS})
  add_definitions(-DGSL_DLL)
endif()

#todo: use find_library pktools instead of setting target_link_libraries
# set(PKTOOLS_INCLUDE_DIRS /usr/local/include/pktools)
# set(PKTOOLS_LIBRARY_DIR /usr/local/lib)

# find_library(PKTOOLS
#     NAMES libimageClasses libfileClasses liblasClasses libalgorithms
#     HINTS "${CMAKE_PREFIX_PATH}/lib"
# )

###############################################################################
# Setup configure file

# configure_file ( "cmake-config.in" "${CMAKE_CURRENT_BINARY_DIR}/config" @ONLY )
configure_file ( "cmake-config.h.in" "${CMAKE_SOURCE_DIR}/config.h" @ONLY )
# configure_file ( "cmake.pc.in" "jiplib.pc"  @ONLY)
add_definitions(-DHAVE_CONFIG_H)


configure_file ( "cmake-config.in" "${CMAKE_CURRENT_BINARY_DIR}/jiplib-config" @ONLY )
configure_file ( "cmake.pc.in" "jiplib.pc"  @ONLY)

# add the binary tree to the search path for include files so that we will find config.h
include_directories("${PROJECT_BINARY_DIR}")

# Name of C++ library
set(JIPLIB_LIB_NAME jiplib)
# set(JIPLIB_LIB_NAME jiplib-${JIPLIB_VERSION_MAJOR}.${JIPLIB_VERSION_MINOR}.${JIPLIB_VERSION_PATCH})

# Include directories
###############################################################################
include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${CMAKE_SOURCE_DIR}")
###############################################################################
# Instruct CMake to inspect the following subfolders (with CMakeLists.txt in each subfolder)
add_subdirectory ("${PROJECT_DOC_DIR}")
add_subdirectory ("${PROJECT_TEST_DIR}")

set(JIPLIB_CC ${PROJECT_SOURCE_DIR}/jim.cc ${PROJECT_SOURCE_DIR}/jimlist.cc)
set(JIPLIB_H ${PROJECT_SOURCE_DIR}/jim.h ${PROJECT_SOURCE_DIR}/jimlist.h)
# Define the jiplib library.
add_library( ${JIPLIB_LIB_NAME} ${JIPLIB_H} ${JIPLIB_CC})

SET_TARGET_PROPERTIES(${JIPLIB_LIB_NAME}
  PROPERTIES
  SOVERSION ${JIPLIB_SOVERSION}
  VERSION ${JIPLIB_VERSION}
)

target_link_libraries(
  ${JIPLIB_LIB_NAME}
  ${MIA_LIB_NAME}
  gomp
  ${GDAL_LIBRARIES}
  # base
  # algorithms
  # imageClasses
  ${PKTOOLS_LIBRARIES}
  )
# add_subdirectory ("${PROJECT_SOURCE_DIR}")
###############################################################################
find_package(PythonInterp)
if(PYTHONINTERP_FOUND)
  MESSAGE(STATUS "PythonInterp was found")
  file(GLOB MIA_HEADER_FILES ${MIA_INCLUDE_DIR}/mialib_*.h)
  MESSAGE(STATUS "MIA_INCLUDE_DIR: ${MIA_INCLUDE_DIR}")
  MESSAGE(STATUS "MIA_HEADER_FILES: ${MIA_HEADER_FILES}")
  file(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/src/mialib_imagetype)
  file(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/src/mialib_errortype)
  file(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/src/mialib_errortype_d)
  FOREACH(infileName ${MIA_HEADER_FILES})
    MESSAGE(STATUS "Process file: ${infileName}")
    file(STRINGS ${infileName} FUN_IMAGETYPE REGEX "^extern IMAGE \\*[^*]")
    FOREACH(fun ${FUN_IMAGETYPE})
      file(APPEND ${CMAKE_CURRENT_SOURCE_DIR}/src/mialib_imagetype "${fun}\n")
    ENDFOREACH(fun)
    file(STRINGS ${infileName} FUN_ERRORTYPE REGEX "^extern ERROR")
    FOREACH(fun ${FUN_ERRORTYPE})
      file(APPEND ${CMAKE_CURRENT_SOURCE_DIR}/src/mialib_errortype "${fun}\n")
    ENDFOREACH(fun)
  ENDFOREACH(infileName)

  EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src INPUT_FILE ${CMAKE_CURRENT_SOURCE_DIR}/src/fun2method_imagetype.py OUTPUT_QUIET)
  EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src INPUT_FILE ${CMAKE_CURRENT_SOURCE_DIR}/src/fun2method_errortype.py OUTPUT_QUIET)
  EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src INPUT_FILE ${CMAKE_CURRENT_SOURCE_DIR}/src/fun2method_errortype_d.py OUTPUT_QUIET)

  # ## todo: remove all lines in between start and end (check http://stackoverflow.com/questions/6287755/using-sed-to-delete-all-lines-between-two-matching-patterns)
  execute_process(COMMAND bash "-c" "sed -i '/\\/\\/start insert from fun2method_imagetype/,/\\/\\/end insert from fun2method_imagetype/{/\\/\\/start insert from fun2method_imagetype/!{/\\/\\/end insert from fun2method_imagetype/!d}}' jim.h" WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src)
  execute_process(COMMAND bash "-c" "sed -i -ne '/\\/\\/start insert from fun2method_imagetype/ {p; r fun2method_imagetype.h' -e ':a; n; /\\/\\/end insert from fun2method_imagetype/ {p; b}; ba}; p' jim.h" WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src)
  execute_process(COMMAND bash "-c" "sed -i '/\\/\\/start insert from fun2method_errortype/,/\\/\\/end insert from fun2method_errortype/{/\\/\\/start insert from fun2method_errortype/!{/\\/\\/end insert from fun2method_errortype/!d}}' jim.h" WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src)
  execute_process(COMMAND bash "-c" "sed -i -ne '/\\/\\/start insert from fun2method_errortype/ {p; r fun2method_errortype.h' -e ':a; n; /\\/\\/end insert from fun2method_errortype/ {p; b}; ba}; p' jim.h" WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src)
  execute_process(COMMAND bash "-c" "sed -i '/\\/\\/start insert from fun2method_errortype_d/,/\\/\\/end insert from fun2method_errortype_d/{/\\/\\/start insert from fun2method_errortype_d/!{/\\/\\/end insert from fun2method_errortype_d/!d}}' jim.h" WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src)
  execute_process(COMMAND bash "-c" "sed -i -ne '/\\/\\/start insert from fun2method_errortype_d/ {p; r fun2method_errortype_d.h' -e ':a; n; /\\/\\/end insert from fun2method_errortype_d/ {p; b}; ba}; p' jim.h" WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src)
  # execute_process(COMMAND bash "sed -i -ne '/\\/\\/start insert from fun2method_errortype/ {p; r fun2method_errortype.h' -e ':a; n; /\\/\\/end insert from fun2method_errortype/ {p; b}; ba}; p' jim.h")
  # execute_process(COMMAND bash "sed -i -ne '/\\/\\/start insert from fun2method_errortype_d/ {p; r fun2method_errortype_d.h' -e ':a; n; /\\/\\/end insert from fun2method_errortype_d/ {p; b}; ba}; p' jim.h")
else(PYTHONINTERP_FOUND)
  MESSAGE(STATUS "Error: PythonInterp not found")
endif(PYTHONINTERP_FOUND)
########## SWIG #############

FIND_PACKAGE(SWIG REQUIRED)
FIND_PACKAGE(PythonLibs)
SET(PYTHON_SRC_DIR ${PROJECT_SOURCE_DIR}/swig)
FILE(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/swig)
INCLUDE(${SWIG_USE_FILE})
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR})
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
SET(CMAKE_SWIG_FLAGS "")
SET_PROPERTY(SOURCE ${PYTHON_SRC_DIR}/jiplib.i PROPERTY CPLUSPLUS ON)
SWIG_ADD_MODULE(jiplib python ${PYTHON_SRC_DIR}/jiplib.i)
SET_SOURCE_FILES_PROPERTIES(${PYTHON_SRC_DIR}/jiplib.i PROPERTIES SWIG_FLAGS "-Wall")
SWIG_LINK_LIBRARIES(jiplib ${PKTOOLS_LIBRARIES} ${PYTHON_LIBRARIES} ${JIPLIB_LIB_NAME})

########## INSTALL ##########
INSTALL (TARGETS ${JIPLIB_LIB_NAME} LIBRARY DESTINATION lib)
INSTALL (FILES ${JIPLIB_H} DESTINATION include/jiplib)
EXECUTE_PROCESS(COMMAND python -c "from distutils.sysconfig import get_python_lib; print get_python_lib()" OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
SET(PYTHON_INSTALL_DIR ${PYTHON_SITE_PACKAGES} CACHE PATH "Install directory for Python library")
INSTALL(TARGETS ${SWIG_MODULE_jiplib_REAL_NAME} LIBRARY DESTINATION ${PYTHON_INSTALL_DIR})
INSTALL(FILES ${CMAKE_BINARY_DIR}/jiplib.py DESTINATION ${PYTHON_INSTALL_DIR})
###############################################################################
