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

set (PROJECT_BINARY_DIR bin)
set (PROJECT_LIBRARY_DIR lib)
set (PROJECT_INCLUDE_DIR include/jiplib)
set (PROJECT_DOC_DIR doc)
set (PROJECT_SOURCE_DIR src)

# Name of C++ library
set(JIPLIB_LIB_NAME jiplib)
set(MIA_LIB_NAME mia_generic)

option (BUILD_SHARED_LIBS "Build with shared library" ON)

# Platform and compiler specific settings

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has C++11 support.")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else(COMPILER_SUPPORTS_CXX11)
  message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

if(WIN32)
else()
  # Recommended C++ compilation flags
  # -Weffc++
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -pg")
  set(JIPLIB_COMMON_CXX_FLAGS
    "-Wall -Wpointer-arith -Wcast-align -Wcast-qual -Wredundant-decls -Wno-long-long")

  if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC ${JIPLIB_COMMON_CXX_FLAGS}")

  elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" OR "${CMAKE_CXX_COMPILER}" MATCHES "clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${JIPLIB_COMMON_CXX_FLAGS}")
  endif()
endif(WIN32)

# Make sure pktools version 3.0.0 is found
set(PKTOOLS_INCLUDE_DIR /usr/local/include/pktools CACHE PATH "Include PATH for PKTOOLS")
set(PKTOOLS_LIB_PATH /usr/local/lib CACHE PATH "Library PATH for PKTOOLS")



# Make sure mia is found
# fix: need to copy jip/swig/jipl_glue.h to jip/ljip/src
set(MIA_INCLUDE_DIR /usr/local/include/mia CACHE PATH "Include PATH for MIA")
set(MIA_LIB_PATH /usr/local/lib CACHE PATH "Library PATH for MIA")

#set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /home/kempepi/mia/python/jip/ljip/lib)

INCLUDE_DIRECTORIES(${MIA_INCLUDE_DIR})
LINK_DIRECTORIES(${MIA_LIB_PATH})

INCLUDE_DIRECTORIES(${PKTOOLS_INCLUDE_DIR})

find_package(GDAL 1.10 REQUIRED)
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

#find_package(pktools 3.0)
#uncomment to enable testing
enable_testing()

add_test (NAME MyTest COMMAND Test)
###############################################################################
# Setup configure file
configure_file ( "cmake-config.h.in" "${CMAKE_CURRENT_BINARY_DIR}/config.h" @ONLY)
include_directories("${CMAKE_CURRENT_BINARY_DIR}")
add_definitions(-DHAVE_CONFIG_H)
###############################################################################

# Build libraries
###############################################################################
include_directories("${PROJECT_SOURCE_DIR}")
# Instruct CMake to inspect the following subfolders (with CMakeLists.txt in each subfolder)
add_subdirectory ("${PROJECT_SOURCE_DIR}")
#uncomment to enable testing
add_subdirectory(test)
###############################################################################

###############################################################################
# Build executables

configure_file ( "cmake-config.in" "${CMAKE_CURRENT_BINARY_DIR}/jiplib-config" @ONLY )
configure_file ( "cmake.pc.in" "jiplib.pc"  @ONLY)

