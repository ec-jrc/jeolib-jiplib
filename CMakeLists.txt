###############################################################################

# CMake build scripts for JIPLIB
#
###############################################################################
cmake_minimum_required(VERSION 2.6.0)
#cmake_minimum_required(VERSION 2.8)
###############################################################################
# Set Project name and version
project (JIPLIB)

include(CTest)

set (PROJECT_BINARY_DIR bin)
set (PROJECT_LIBRARY_DIR lib)
set (PROJECT_INCLUDE_DIR include/jiplib)
set (PROJECT_DOC_DIR doc)
set (PROJECT_SOURCE_DIR src)
set (PROJECT_TEST_DIR test)

# The version number.
set (JIPLIB_VERSION_MAJOR 1)
set (JIPLIB_VERSION_MINOR 0)
set (JIPLIB_VERSION_PATCH 0)
set (JIPLIB_VERSION "${JIPLIB_VERSION_MAJOR}.${JIPLIB_VERSION_MINOR}.${JIPLIB_VERSION_PATCH}")
set (JIPLIB_PACKAGE_VERSION "${JIPLIB_VERSION_MAJOR}.${JIPLIB_VERSION_MINOR}.${JIPLIB_VERSION_PATCH}")
set (JIPLIB_PACKAGE_STRING "JIPLIB ${JIPLIB_VERSION_MAJOR}.${JIPLIB_VERSION_MINOR}.${JIPLIB_VERSION_PATCH}")
set (JIPLIB_PACKAGE_BUGREPORT "pieter.kempeneers@ec.europa.eu")
set (JIPLIB_PACKAGE "jiplib")
set (JIPLIB_COPYRIGHT_YEAR 2018)

set (JIPLIB_LIB_VERSION_MAJOR 1)
set (JIPLIB_LIB_VERSION_MINOR 0)
set (JIPLIB_LIB_VERSION_PATCH 0)
set (JIPLIB_LIB_VERSION "${JIPLIB_LIB_VERSION_MAJOR}.${JIPLIB_LIB_VERSION_MINOR}.${JIPLIB_LIB_VERSION_PATCH}")
set (JIPLIB_LIB_SOVERSION 1)

# set (JIPLIB_PYTHON_VERSION_MAJOR 2)
# set (JIPLIB_PYTHON_VERSION_MINOR 0)
# set (JIPLIB_PYTHON_VERSION_PATCH 0)
# set (JIPLIB_PYTHON_VERSION "${JIPLIB_PYTHON_VERSION_MAJOR}.${JIPLIB_PYTHON_VERSION_MINOR}.${JIPLIB_PYTHON_VERSION_PATCH}")
# set (JIPLIB_PYTHON_SOVERSION 2)

# Name of C++ library
set(JIPLIB_LIB_NAME jiplib)

option (BUILD_SHARED_LIBS "Build with shared library" ON)
set(BUILD_WITH_LIBLAS FALSE CACHE BOOL "Choose if jiplib is to be built with liblas")
set(BUILD_WITH_FANN TRUE CACHE BOOL "Choose if jiplib is to be built with fann")
set(BUILD_WITH_PYTHON TRUE CACHE BOOL "Choose if jiplib is to be built with PYTHON support")
set(PROCESS_IN_PARALLEL TRUE CACHE BOOL "Choose if jiplib should be run in parallel")
set(BUILD_WITH_MIALIB TRUE CACHE BOOL "Choose if jiplib should be compiled with mialib")

set(INSTALL_LIBRARY_DIR ${PROJECT_LIBRARY_DIR} CACHE PATH "Install path for libraries")
# option (INSTALL_LIBRARY_DIR "Install path for libraries" ${PROJECT_LIBRARY_DIR})

set(GSL_INCLUDE_DIR /usr/include CACHE PATH "Include PATH for GSL")
set(GSL_LIB_PATH /usr/lib CACHE PATH "Library PATH for GSL")

set(JIPLIB_INCLUDE_DIRS ${PREFIX}/include/jiplib)
# set(JIPLIB_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}")

###############################################################################

###############################################################################
# CMake settings
# cmake_minimum_required(VERSION 2.6.0)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "RELEASE")
ENDIF()

set(CMAKE_COLOR_MAKEFILE ON)

# Allow advanced users to generate Makefiles printing detailed commands
mark_as_advanced(CMAKE_VERBOSE_MAKEFILE)

# Path to additional CMake modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)
###############################################################################

###############################################################################
# Documentation
#file(READ "doc/index.txt" README )
file(WRITE "README.txt" "${README}")
###############################################################################

###############################################################################
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
    if (MSVC)
        if (MSVC80 OR MSVC90 OR MSVC10)
            add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
            add_definitions(-D_CRT_SECURE_NO_WARNINGS)
            add_definitions(-D_CRT_NONSTDC_NO_WARNING)
            add_definitions(-D_SCL_SECURE_NO_WARNINGS)
        endif()

        if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
            string(REGEX REPLACE "/W[0-4]" "/W4"
                   CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        else()
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
        endif()

        # Generate dot-user file with user-specific settings for Visual Studio project
        set(MSVC_ENVIRONMENT_PATH "" CACHE STRING
            "Custom PATH for Environment property in Visual Studio project configuration")
        mark_as_advanced(MSVC_ENVIRONMENT_PATH)
        set(VCPROJ_USER_ENVIRONMENT_PATH_DEBUG "${ENVIRONMENT_PATH}")
        mark_as_advanced(VCPROJ_USER_ENVIRONMENT_PATH_DEBUG)

    endif()

  set(CMAKE_INCLUDE_PATH c:/osgeo4w/include;$ENV{CMAKE_INCLUDE_PATH})
    set(CMAKE_LIBRARY_PATH c:/osgeo4w/lib;$ENV{CMAKE_LIBRARY_PATH})
else()
  # Recommended C++ compilation flags
  # -Weffc++
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -pg")
  set(JIPLIB_COMMON_CXX_FLAGS
    "-Wpointer-arith -Wcast-align -Wcast-qual -Wredundant-decls -Wno-long-long")
    # "-Wall -Wpointer-arith -Wcast-align -Wcast-qual -Wredundant-decls -Wno-long-long")

  if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC ${JIPLIB_COMMON_CXX_FLAGS}")
  elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" OR "${CMAKE_CXX_COMPILER}" MATCHES "clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${JIPLIB_COMMON_CXX_FLAGS}")
  endif()
endif(WIN32)

if (APPLE)
  set(SO_EXT dylib)
elseif(WIN32)
  set(SO_EXT dll)
else()
  set(SO_EXT so)
endif(APPLE)
###############################################################################

###############################################################################
# Check the required libraries are present

find_package(GDAL 1.10 REQUIRED)
if(GDAL_FOUND)
  include_directories(${GDAL_INCLUDE_DIRS})
endif()

find_package(GSL REQUIRED)
if(GSL_FOUND)
  include_directories(${GSL_INCLUDE_DIRS})
  add_definitions(-DGSL_DLL)
endif()

if (PROCESS_IN_PARALLEL)
  find_package(OpenMP)
  if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(PARALLEL_BOOL 1)
    message(STATUS "parallel processing: ${PARALLEL_BOOL}")
  endif (OPENMP_FOUND)
else (PROCESS_IN_PARALLEL)
  set(PARALLEL_BOOL 0)
  message(STATUS "parallel processing: ${PARALLEL_BOOL}")
endif (PROCESS_IN_PARALLEL)
# find_package(Armadillo REQUIRED)
# if(ARMADILLO_FOUND)
#   include_directories(${ARMADILLO_INCLUDE_DIRS})
# endif()

if (BUILD_WITH_MIALIB)
  set(MIA_BOOL 1)
  message(STATUS "build with mialib: ${MIA_BOOL}")
  # Make sure mia is found
  set(MIA_LIB_NAME mialib_python)
  set(MIA_INCLUDE_DIR /usr/local/include/mialib CACHE PATH "Include PATH for MIA")
  set(MIA_LIB_PATH /usr/local/lib CACHE PATH "Library PATH for MIA")

  INCLUDE_DIRECTORIES(${MIA_INCLUDE_DIR})
  LINK_DIRECTORIES(${MIA_LIB_PATH})

  file(COPY ${JIM_H} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/src)
  file(COPY ${JIM_CC} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/src)
  file(COPY ${JIMLIST_H} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/src)
  file(COPY ${JIMLIST_CC} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/src)
  set(JIM_H ${CMAKE_CURRENT_BINARY_DIR}/src/jim.h)
  set(JIM_CC ${CMAKE_CURRENT_BINARY_DIR}/src/jim.cc)
  set(JIMLIST_H ${CMAKE_CURRENT_BINARY_DIR}/src/jimlist.h)
  set(JIMLIST_CC ${CMAKE_CURRENT_BINARY_DIR}/src/jimlist.cc)
else(BUILD_WITH_MIALIB)
  set(MIA_BOOL 0)
  message(STATUS "build with mialib: ${MIA_BOOL}")
endif(BUILD_WITH_MIALIB)

find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)
include_directories(${JSONCPP_INCLUDE_DIRS})


set(Boost_DEBUG 0)
find_package(Boost COMPONENTS filesystem system serialization)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
  add_definitions("-DBOOST_ALL_NO_LIB")
endif()

if (BUILD_WITH_LIBLAS)
  find_package(LIBLAS)
  if(LIBLAS_FOUND)
    include_directories(${LIBLAS_INCLUDE_DIRS})
  endif()

#	include_directories(${BOOST_INCLUDE_DIR})
#	if (MSVC)
#		set(BOOST_LIBRARIES -LIBPATH:${BOOST_LIB_PATH} libboost_filesystem-vc100-mt-1_56.lib libboost_system-vc100-mt-1_56.lib)
#	else()
#		set(BOOST_LIBRARIES -L${BOOST_LIB_PATH} -lboost_filesystem -lboost_system)
#	endif(MSVC)
endif(BUILD_WITH_LIBLAS)

if (BUILD_WITH_FANN)
  set(FANN_VERSION_MINOR 2 CACHE STRING "LIBFANN version")
  FIND_PACKAGE(FANN 2.${FANN_VERSION_MINOR} EXACT)
  if(FANN_FOUND)
    message(STATUS "Found FANN exact version 2.${FANN_VERSION_MINOR}")
    include_directories(${FANN_INCLUDE_DIRS})
    add_definitions(-DFANN_DLL)
  else(FANN_FOUND)
    message(STATUS "Error: FANN exact version 2.${FANN_VERSION_MINOR} not found, trying default")
    find_package(FANN)
  endif()
endif(BUILD_WITH_FANN)

if (BUILD_WITH_PYTHON)
  set(PYTHON_BOOL 1)
  set(PYTHON_VERSION "2.7.6" CACHE STRING "Specify which Python version to use")
  FIND_PACKAGE(PythonLibs ${PYTHON_VERSION} EXACT)
  if(PYTHONLIBS_FOUND)
    message(STATUS "Python version found: ${PYTHONLIBS_VERSION_STRING}")
    message(STATUS "Found Python library: ${PYTHON_LIBRARY}")
    message(STATUS "Found Python include dir: ${PYTHON_INCLUDE_DIR}")
  else(PYTHONLIBS_FOUND)
    message(STATUS "Error: PYTHON exact version ${PYTHON_VERSION} not found, trying default")
    FIND_PACKAGE(PythonLibs REQUIRED)
  endif(PYTHONLIBS_FOUND)
else (BUILD_WITH_PYTHON)
  set(PYTHON_BOOL 0)
endif (BUILD_WITH_PYTHON)

###############################################################################
###############################################################################
# Setup configure file
configure_file ( "cmake-config.h.in" "${CMAKE_CURRENT_BINARY_DIR}/config_jiplib.h" @ONLY)
include_directories("${CMAKE_CURRENT_BINARY_DIR}")
# configure_file ( "cmake-config.h.in" "${CMAKE_CURRENT_BINARY_DIR}/base/config_jiplib.h" @ONLY)
# include_directories("${CMAKE_CURRENT_BINARY_DIR}/base")
add_definitions(-DHAVE_CONFIG_H)
###############################################################################

###############################################################################
# Build libraries
include_directories("${PROJECT_SOURCE_DIR}")
add_subdirectory ("${PROJECT_SOURCE_DIR}")
add_subdirectory ("${PROJECT_DOC_DIR}")
add_subdirectory ("${PROJECT_TEST_DIR}")
###############################################################################

###############################################################################
# Build executables

#test
# configure_file ( "cmake-config.in" "${CMAKE_CURRENT_BINARY_DIR}/config-jiplib" @ONLY )
# configure_file ( "cmake-config.in" "${CMAKE_CURRENT_BINARY_DIR}/jiplib-config" @ONLY )
configure_file ( "cmake.pc.in" "jiplib.pc"  @ONLY)
configure_file ( "jiplib-config.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/jiplib-config.cmake" @ONLY )


set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(PKLIBS ${JSONCPP_LIBRARIES} ${GSL_LIBRARIES} ${JIPLIB_LIB_NAME} ${GDAL_LIBRARIES} ${GSL_LIBRARIES} ${BOOST_LIBRARIES} ${GSL_LIBRARIES} )
if (BUILD_WITH_FANN)
  set(PKLIBS ${PKLIBS} ${FANN_LIBRARIES})
endif (BUILD_WITH_FANN)
if (BUILD_WITH_PYTHON)
  set(PKLIBS ${PKLIBS} ${PYTHON_LIBRARIES})
endif (BUILD_WITH_PYTHON)

###############################################################################
# Installation
#install (FILES "${PROJECT_BINARY_DIR}/jiplib-config" DESTINATION bin PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
#install (FILES "${PROJECT_BINARY_DIR}/jiplib-config" DESTINATION bin PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
#test
install (FILES "${CMAKE_CURRENT_BINARY_DIR}/jiplib-config" DESTINATION bin PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
install (FILES "jiplib.pc" DESTINATION ${INSTALL_LIBRARY_DIR}/pkgconfig PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/jiplib-config.cmake" DESTINATION lib/cmake/jiplib)

install (FILES "${CMAKE_CURRENT_BINARY_DIR}/config_jiplib.h" DESTINATION ${PROJECT_INCLUDE_DIR})
if (JIPLIB_WITH_APPS)
  install (TARGETS ${PKAPPS} DESTINATION bin PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
  install (TARGETS ${PKUTILS} DESTINATION bin PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
  if (BUILD_WITH_LIBLAS)
    install (TARGETS ${PKLASAPPS} DESTINATION bin PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
  endif(BUILD_WITH_LIBLAS)
  if (BUILD_WITH_FANN)
    install (TARGETS ${PKFANNAPPS} DESTINATION bin PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
  endif(BUILD_WITH_FANN)
endif(JIPLIB_WITH_APPS)

###############################################################################
# build a CPack driven installer package
include (InstallRequiredSystemLibraries)
set (CPACK_RESOURCE_FILE_LICENSE
     "${CMAKE_CURRENT_SOURCE_DIR}/License.txt")
set (CPACK_PACKAGE_VERSION_MAJOR "${JIPLIB_VERSION_MAJOR}")
set (CPACK_PACKAGE_VERSION_MINOR "${JIPLIB_VERSION_MINOR}")
set (CPACK_PACKAGE_VERSION_PATCH "${JIPLIB_VERSION_PATCH}")
# set(CPACK_SOURCE_PACKAGE_FILE_NAME
#   "${CMAKE_PROJECT_NAME}-${JIPLIB_VERSION_MAJOR}.${JIPLIB_VERSION_MINOR}.${JIPLIB_VERSION_PATCH}")

# Create Distribution
SET(CPACK_SOURCE_GENERATOR "TGZ")
SET(CPACK_CMAKE_GENERATOR "Unix Makefiles")

# Set files to ignore
list(APPEND CPACK_SOURCE_IGNORE_FILES "_CPack_Packages")
list(APPEND CPACK_SOURCE_IGNORE_FILES ".tgz")
list(APPEND CPACK_SOURCE_IGNORE_FILES ".git")
list(APPEND CPACK_SOURCE_IGNORE_FILES "README")
list(APPEND CPACK_SOURCE_IGNORE_FILES "HOWTORELEASE.txt")
list(APPEND CPACK_SOURCE_IGNORE_FILES "CMakeCache.txt")
list(APPEND CPACK_SOURCE_IGNORE_FILES "CPackConfig.cmake")
list(APPEND CPACK_SOURCE_IGNORE_FILES "schemas")

include(CPack)

add_custom_target(dist COMMAND ${CMAKE_MAKE_PROGRAM} package_source)
###############################################################################
