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

# The project's name
project(jiplib)

include(CTest)

set (JIPLIB_VERSION_MAJOR 1)
set (JIPLIB_VERSION_MINOR 0)
set (JIPLIB_VERSION_PATCH 0)

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

########## INSTALL ##########
install (TARGETS ${JIPLIB_LIB_NAME} DESTINATION lib)
install (FILES ${JIPLIB_H} DESTINATION include/jiplib)
########## CTEST ############
# add_test(test1Run test1)
# set_tests_properties (test1Run
#     PROPERTIES PASS_REGULAR_EXPRESSION "success")

########## BUILDING INSTALLLER ##########
# build a CPack driven installer package
include (InstallRequiredSystemLibraries)
set (CPACK_RESOURCE_FILE_LICENSE  
     "${CMAKE_CURRENT_SOURCE_DIR}/License.txt")
set (CPACK_PACKAGE_VERSION_MAJOR "${JIPLIB_VERSION_MAJOR}")
set (CPACK_PACKAGE_VERSION_MINOR "${JIPLIB_VERSION_MINOR}")
include (CPack)

# Make sure pktools version 3.0.0 is found
set(PKTOOLS_INCLUDE_DIR /usr/local/include/pktools CACHE PATH "Include PATH for PKTOOLS")
set(PKTOOLS_LIB_PATH /usr/local/lib CACHE PATH "Library PATH for PKTOOLS")



# Make sure mia is found
# fix: need to copy jip/swig/jipl_glue.h to jip/ljip/src
set(MIA_LIB_NAME mia_generic)
set(MIA_INCLUDE_DIR /usr/local/include/mia CACHE PATH "Include PATH for MIA")
set(MIA_LIB_PATH /usr/local/lib CACHE PATH "Library PATH for MIA")

INCLUDE_DIRECTORIES(${PKTOOLS_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${MIA_INCLUDE_DIR})
LINK_DIRECTORIES(${MIA_LIB_PATH})

find_package(GDAL 2.00 REQUIRED)
if(GDAL_FOUND)
	include_directories(${GDAL_INCLUDE_DIRS})
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
configure_file ( "cmake.pc.in" "jiplib.pc"  @ONLY)
add_definitions(-DHAVE_CONFIG_H)


configure_file ( "cmake-config.in" "${CMAKE_CURRENT_BINARY_DIR}/jiplib-config" @ONLY )
configure_file ( "cmake.pc.in" "jiplib.pc"  @ONLY)

# add the binary tree to the search path for include files so that we will find config.h
include_directories("${PROJECT_BINARY_DIR}")

# Name of C++ library
set(JIPLIB_LIB_NAME jiplib-${JIPLIB_VERSION_MAJOR}.${JIPLIB_VERSION_MINOR})
# Build libraries
###############################################################################
include_directories("${PROJECT_SOURCE_DIR}")
# Instruct CMake to inspect the following subfolders (with CMakeLists.txt in each subfolder)

set(JIPLIB_IMG_CC src/jim.cc)
set(JIPLIB_IMG_H src/jim.h)
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
# add_subdirectory ("${PROJECT_SOURCE_DIR}")
###############################################################################

###############################################################################
# Build executables

# add_executable(test1 ${PROJECT_TEST_DIR}/test1.cc)
# target_link_libraries(
#   ${JIPLIB_LIB_NAME}
#   )


########## SWIG #############
FIND_PACKAGE(SWIG REQUIRED)
FIND_PACKAGE(PythonLibs)
set(PYTHON_SRC_DIR ${PROJECT_SOURCE_DIR}/swig)
FILE(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/swig)

INCLUDE(${SWIG_USE_FILE})
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR})
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

SET(CMAKE_SWIG_FLAGS "")
set_property(SOURCE ${PYTHON_SRC_DIR}/jiplib.i PROPERTY CPLUSPLUS ON)
swig_add_module(jiplib python ${PYTHON_SRC_DIR}/jiplib.i)


SET_SOURCE_FILES_PROPERTIES(${PYTHON_SRC_DIR}/jiplib.i PROPERTIES SWIG_FLAGS "-Wall")
SWIG_LINK_LIBRARIES(jiplib ${PYTHON_LIBRARIES} ${JIPLIB_LIB_NAME})
