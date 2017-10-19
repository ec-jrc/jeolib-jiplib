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
set(MCISRG 0 CACHE STRING "set a variable that can be used in jim.cc")

set (PROJECT_BINARY_DIR bin)
set (PROJECT_LIBRARY_DIR lib)
set (PROJECT_INCLUDE_DIR include/jiplib)
set (PROJECT_DOC_DIR doc)
set (PROJECT_SOURCE_DIR src)
set (PROJECT_TEST_DIR test)

# configure a header file to pass some of the CMake settings to the source code
configure_file (
  "cmake-config.h.in"
  "config_jiplib.h"
  )

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

# Make sure pktools version 3.0.0 is found
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/modules")
find_package(PKTOOLS CONFIG REQUIRED)
if(PKTOOLS_FOUND)
  message(STATUS "Found pktools: ${PKTOOLS_LIBRARIES}")
  include_directories(${PKTOOLS_INCLUDE_DIRS})
else(PKTOOLS_FOUND)
  message(WARNING "pktools not found, set manually")
  set(PKTOOLS_INCLUDE_DIR /usr/local/include/pktools CACHE PATH "Include PATH for PKTOOLS")
  include_directories(${PKTOOLS_INCLUDE_DIR})
  # set(PKTOOLS_LIB_PATH /usr/local/lib CACHE PATH "Library PATH for PKTOOLS")
endif()


# Make sure Python libraries are found
FIND_PACKAGE(PythonLibs REQUIRED)

#should replace following line
# EXECUTE_PROCESS(COMMAND pkg-config --cflags pktools OUTPUT_VARIABLE PKTOOLS_INCLUDE_DIRS OUTPUT_STRIP_TRAILING_WHITESPACE)
# set(PKTOOLS_INCLUDE_DIR ${PKTOOLS_INCLUDE_DIRS} CACHE PATH "Include PATH for PKTOOLS")
# #should not be needed with FindPktools


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
# link_libraries(${JSONCPP_LIBRARIES})
include_directories(${JSONCPP_INCLUDE_DIRS})

#todo: use find_library pktools instead of setting target_link_libraries
# set(PKTOOLS_INCLUDE_DIRS /usr/local/include/pktools)
# set(PKTOOLS_LIBRARY_DIR /usr/local/lib)

# find_library(PKTOOLS
#     NAMES libimageClasses libfileClasses liblasClasses libalgorithms
#     HINTS "${CMAKE_PREFIX_PATH}/lib"
# )

###############################################################################
# Setup configure file

configure_file ( "cmake-config.in" "${CMAKE_CURRENT_BINARY_DIR}/jiplib-config" @ONLY )
# configure_file ( "cmake-config.h.in" "${CMAKE_SOURCE_DIR}/config_jiplib.h" @ONLY )
# configure_file ( "cmake.pc.in" "jiplib.pc"  @ONLY)
add_definitions(-DHAVE_CONFIG_H)


# configure_file ( "cmake-config.in" "${CMAKE_CURRENT_BINARY_DIR}/jiplib-config" @ONLY )
# configure_file ( "cmake-config.in" "${CMAKE_CURRENT_BINARY_DIR}/config_jiplib" @ONLY )
configure_file ( "cmake.pc.in" "jiplib.pc"  @ONLY)



# Name of C++ library
set(JIPLIB_LIB_NAME jiplib)
# set(JIPLIB_LIB_NAME jiplib-${JIPLIB_VERSION_MAJOR}.${JIPLIB_VERSION_MINOR}.${JIPLIB_VERSION_PATCH})

set(BUILD_WITH_MIALIB TRUE CACHE BOOL "JIPlib needs to be built with mialiab")
# Include directories
###############################################################################
# include_directories("${PROJECT_SOURCE_DIR}")
# include_directories("${CMAKE_SOURCE_DIR}")
include_directories("${PROJECT_BINARY_DIR}")# needed for config_jiplib.h
include_directories("${CMAKE_CURRENT_BINARY_DIR}")# this is where jim.h will go
include_directories("${CMAKE_CURRENT_BINARY_DIR}/src/")
include_directories("${CMAKE_CURRENT_BINARY_DIR}/doc")#needed for jiplib_doc.i
###############################################################################
# Instruct CMake to inspect the following subfolders (with CMakeLists.txt in each subfolder)
add_subdirectory ("${PROJECT_DOC_DIR}")
add_subdirectory ("${PROJECT_TEST_DIR}")

set(JIM_H ${PROJECT_SOURCE_DIR}/jim.h)
set(JIM_CC ${PROJECT_SOURCE_DIR}/jim.cc)
set(JIMLIST_H ${PROJECT_SOURCE_DIR}/jimlist.h)
set(JIMLIST_CC ${PROJECT_SOURCE_DIR}/jimlist.cc)
if (BUILD_WITH_MIALIB)
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
endif(BUILD_WITH_MIALIB)
set(JIPLIB_H ${JIM_H} ${JIMLIST_H})
set(JIPLIB_CC ${JIM_CC} ${JIMLIST_CC})
set(JIPLIB_DOC ${CMAKE_CURRENT_BINARY_DIR}/doc/jiplib_doc.i)

# Define the jiplib library.
add_library( ${JIPLIB_LIB_NAME} ${JIPLIB_H} ${JIPLIB_CC} ${JIPLIB_DOC})

SET_TARGET_PROPERTIES(${JIPLIB_LIB_NAME}
  PROPERTIES
  SOVERSION ${JIPLIB_SOVERSION}
  VERSION ${JIPLIB_VERSION}
)

if(BUILD_WITH_MIALIB)
  target_link_libraries(
    ${JIPLIB_LIB_NAME}
    ${MIA_LIB_NAME}
    ${PYTHON_LIBRARIES}
    gomp
    ${GDAL_LIBRARIES}
    ${JSONCPP_LIBRARIES}
    ${PKTOOLS_LIBRARIES}
    )
else(BUILD_WITH_MIALIB)
  target_link_libraries(
    ${JIPLIB_LIB_NAME}
    ${PYTHON_LIBRARIES}
    gomp
    ${GDAL_LIBRARIES}
    ${JSONCPP_LIBRARIES}
    ${PKTOOLS_LIBRARIES}
    )
endif(BUILD_WITH_MIALIB)
###############################################################################
find_package(PythonInterp)
if(PYTHONINTERP_FOUND)
  file(WRITE ${JIPLIB_DOC} "#This is an automatically generated doc file, do not edit")
  #todo: treat mialib functions that return: int, long int, double, G_TYPE
  # G_TYPE: getpixval
  # G_TYPE* min_max
  # long int thresholdRegion_Size
  # long int thresholdRegion_Contrast
  # double julian_date
  if(BUILD_WITH_MIALIB)
    MESSAGE(STATUS "PythonInterp was found")
    #file(COPY ${PROJECT_SOURCE_DIR}/old2NewNames.json DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/src)
    file(COPY ${PROJECT_SOURCE_DIR}/rename.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/src)
    file(COPY ${PROJECT_SOURCE_DIR}/rename.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/src)
    file(COPY ${PROJECT_SOURCE_DIR}/fun2method_imagetype.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/src)
    file(COPY ${PROJECT_SOURCE_DIR}/fun2method_imagetype_multi.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/src)
    file(COPY ${PROJECT_SOURCE_DIR}/fun2method_imagetype_jimlist.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/src)
    file(COPY ${PROJECT_SOURCE_DIR}/fun2method_imagelisttype.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/src)
    file(COPY ${PROJECT_SOURCE_DIR}/fun2method_errortype.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/src)
    file(COPY ${PROJECT_SOURCE_DIR}/fun2method_errortype_d.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/src)
    file(COPY ${PROJECT_SOURCE_DIR}/fun2method_errortype_nd.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/src)

    file(GLOB MIA_HEADER_FILES ${MIA_INCLUDE_DIR}/mialib_*.h)
    MESSAGE(STATUS "MIA_INCLUDE_DIR: ${MIA_INCLUDE_DIR}")
    MESSAGE(STATUS "MIA_HEADER_FILES: ${MIA_HEADER_FILES}")
    file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/src/mialib_imagetype)
    file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/src/mialib_imagelisttype)
    file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/src/mialib_errortype_nd)
    file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/src/mialib_errortype)
    #Mialib functions have been designed as destructive. We will therefore create a copy to avoid the destructive behaviour. The destructive method will still be available with the prefix d_
    #The following functions F_ND from mialib do not modify the input image and should therefore not create a copy. They will be treated in the same way as the destructive functions (without the prefix d_)
    SET(F_ND "dumpxyz|szcompat|szgeocompat|iminfo|tiffinfo|writeGnuPlot3D|vectorizeImage|IsPartitionEqual|IsPartitionFiner|dendro|getfirstmaxpos|volume|imequalp|getmax|getminmax")
    #These functions must only be included if MCISRG is set
    SET(F_MCISRG "labelImage|segmentImage")
    #Create a text file with the list of headers to MIA functions, e.g., mialib_imagetype for functions returning IMAGE *
    FOREACH(infileName ${MIA_HEADER_FILES})
      MESSAGE(STATUS "Process file: ${infileName}")
      file(STRINGS ${infileName} FUN_IMAGETYPE REGEX "^extern IMAGE \\*[^*]")
      FOREACH(fun ${FUN_IMAGETYPE})
        IF("${fun}" MATCHES "${F_MCISRG}")
          IF(MCISRG)
            file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/src/mialib_imagetype "${fun}\n")
          ENDIF(MCISRG)
        ELSE("${fun}" MATCHES "${F_MCISRG}")
          file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/src/mialib_imagetype "${fun}\n")
        ENDIF("${fun}" MATCHES "${F_MCISRG}")
      ENDFOREACH(fun)
      file(STRINGS ${infileName} FUN_IMAGELISTTYPE REGEX "^extern IMAGE \\*\\*[^*]")
      FOREACH(fun ${FUN_IMAGELISTTYPE})
        file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/src/mialib_imagelisttype "${fun}\n")
      ENDFOREACH(fun)
      file(STRINGS ${infileName} FUN_ERRORTYPE REGEX "^extern ERROR")
      FOREACH(fun ${FUN_ERRORTYPE})
        IF("${fun}" MATCHES "${F_ND}")
          file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/src/mialib_errortype_nd "${fun}\n")
        ELSE("${fun}" MATCHES "${F_ND}")
          file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/src/mialib_errortype "${fun}\n")
        ENDIF("${fun}" MATCHES "${F_ND}")
      ENDFOREACH(fun)
    ENDFOREACH(infileName)

    #Create headers with new names in build directory
    EXECUTE_PROCESS(COMMAND ${CMAKE_CURRENT_BINARY_DIR}/src/rename.sh ${CMAKE_CURRENT_BINARY_DIR}/src WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src OUTPUT_QUIET)
    EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src INPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/src/fun2method_imagetype.py OUTPUT_QUIET)
    EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src INPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/src/fun2method_imagetype_multi.py OUTPUT_QUIET)
    EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src INPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/src/fun2method_imagetype_jimlist.py OUTPUT_QUIET)
    EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src INPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/src/fun2method_imagelisttype.py OUTPUT_QUIET)
    EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src INPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/src/fun2method_errortype.py OUTPUT_QUIET)
    EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src INPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/src/fun2method_errortype_d.py OUTPUT_QUIET)
    EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src INPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/src/fun2method_errortype_nd.py OUTPUT_QUIET)

    # remove all lines in between start and end in jim.cc
    execute_process(COMMAND bash "-c" "sed -i '/\\/\\/start insert from fun2method_imagetype/,/\\/\\/end insert from fun2method_imagetype/{/\\/\\/start insert from fun2method_imagetype/!{/\\/\\/end insert from fun2method_imagetype/!d}}' jim.h" WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src)
    execute_process(COMMAND bash "-c" "sed -i -ne '/\\/\\/start insert from fun2method_imagetype/ {p; r fun2method_imagetype.h' -e ':a; n; /\\/\\/end insert from fun2method_imagetype/ {p; b}; ba}; p' jim.h" WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src)
    execute_process(COMMAND bash "-c" "sed -i '/\\/\\/start insert from fun2method_imagetype_multi/,/\\/\\/end insert from fun2method_imagetype_multi/{/\\/\\/start insert from fun2method_imagetype_multi/!{/\\/\\/end insert from fun2method_imagetype_multi/!d}}' jim.h" WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src)
    execute_process(COMMAND bash "-c" "sed -i -ne '/\\/\\/start insert from fun2method_imagetype_multi/ {p; r fun2method_imagetype_multi.h' -e ':a; n; /\\/\\/end insert from fun2method_imagetype_multi/ {p; b}; ba}; p' jim.h" WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src)
    execute_process(COMMAND bash "-c" "sed -i '/\\/\\/start insert from fun2method_imagetype_jimlist/,/\\/\\/end insert from fun2method_imagetype_jimlist/{/\\/\\/start insert from fun2method_imagetype_jimlist/!{/\\/\\/end insert from fun2method_imagetype_jimlist/!d}}' jimlist.h" WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src)
    execute_process(COMMAND bash "-c" "sed -i -ne '/\\/\\/start insert from fun2method_imagetype_jimlist/ {p; r fun2method_imagetype_jimlist.h' -e ':a; n; /\\/\\/end insert from fun2method_imagetype_jimlist/ {p; b}; ba}; p' jimlist.h" WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src)
    execute_process(COMMAND bash "-c" "sed -i '/\\/\\/start insert from fun2method_imagelisttype/,/\\/\\/end insert from fun2method_imagelisttype/{/\\/\\/start insert from fun2method_imagelisttype/!{/\\/\\/end insert from fun2method_imagelisttype/!d}}' jimlist.h" WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src)
    execute_process(COMMAND bash "-c" "sed -i -ne '/\\/\\/start insert from fun2method_imagelisttype/ {p; r fun2method_imagelisttype.h' -e ':a; n; /\\/\\/end insert from fun2method_imagelisttype/ {p; b}; ba}; p' jim.h" WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src)
    execute_process(COMMAND bash "-c" "sed -i '/\\/\\/start insert from fun2method_errortype/,/\\/\\/end insert from fun2method_errortype/{/\\/\\/start insert from fun2method_errortype/!{/\\/\\/end insert from fun2method_errortype/!d}}' jim.h" WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src)
    execute_process(COMMAND bash "-c" "sed -i -ne '/\\/\\/start insert from fun2method_errortype/ {p; r fun2method_errortype.h' -e ':a; n; /\\/\\/end insert from fun2method_errortype/ {p; b}; ba}; p' jim.h" WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src)
    execute_process(COMMAND bash "-c" "sed -i '/\\/\\/start insert from fun2method_errortype_d/,/\\/\\/end insert from fun2method_errortype_d/{/\\/\\/start insert from fun2method_errortype_d/!{/\\/\\/end insert from fun2method_errortype_d/!d}}' jim.h" WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src)
    execute_process(COMMAND bash "-c" "sed -i -ne '/\\/\\/start insert from fun2method_errortype_d/ {p; r fun2method_errortype_d.h' -e ':a; n; /\\/\\/end insert from fun2method_errortype_d/ {p; b}; ba}; p' jim.h" WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src)
    execute_process(COMMAND bash "-c" "sed -i '/\\/\\/start insert from fun2method_errortype_nd/,/\\/\\/end insert from fun2method_errortype_nd/{/\\/\\/start insert from fun2method_errortype_nd/!{/\\/\\/end insert from fun2method_errortype_nd/!d}}' jim.h" WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src)
    execute_process(COMMAND bash "-c" "sed -i -ne '/\\/\\/start insert from fun2method_errortype_nd/ {p; r fun2method_errortype_nd.h' -e ':a; n; /\\/\\/end insert from fun2method_errortype_nd/ {p; b}; ba}; p' jim.h" WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src)
    #Append the source files to jim.cc
    file(APPEND ${JIM_CC} "#include \"${CMAKE_CURRENT_BINARY_DIR}/src/fun2method_imagetype.cc\"\n")
    file(APPEND ${JIM_CC} "#include \"${CMAKE_CURRENT_BINARY_DIR}/src/fun2method_imagetype_multi.cc\"\n")
    file(APPEND ${JIM_CC} "#include \"${CMAKE_CURRENT_BINARY_DIR}/src/fun2method_imagetype_jimlist.cc\"\n")
    file(APPEND ${JIM_CC} "#include \"${CMAKE_CURRENT_BINARY_DIR}/src/fun2method_imagelisttype.cc\"\n")
    file(APPEND ${JIM_CC} "#include \"${CMAKE_CURRENT_BINARY_DIR}/src/fun2method_errortype.cc\"\n")
    file(APPEND ${JIM_CC} "#include \"${CMAKE_CURRENT_BINARY_DIR}/src/fun2method_errortype_d.cc\"\n")
    file(APPEND ${JIM_CC} "#include \"${CMAKE_CURRENT_BINARY_DIR}/src/fun2method_errortype_nd.cc\"\n")
  else(PYTHONINTERP_FOUND)
    MESSAGE(STATUS "Error: PythonInterp not found")
  endif(BUILD_WITH_MIALIB)
endif(PYTHONINTERP_FOUND)
########## SWIG #############

FIND_PACKAGE(SWIG REQUIRED)
SET(PYTHON_SRC_DIR ${PROJECT_SOURCE_DIR}/swig)
FILE(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/swig)
INCLUDE(${SWIG_USE_FILE})
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR})
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR}/doc)
FILE(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/swig)
SET(CMAKE_SWIG_FLAGS "")
SET_PROPERTY(SOURCE ${PYTHON_SRC_DIR}/jiplib.i PROPERTY CPLUSPLUS ON)
SWIG_ADD_MODULE(jiplib python ${PYTHON_SRC_DIR}/jiplib.i)
# SET_SOURCE_FILES_PROPERTIES(${PYTHON_SRC_DIR}/jiplib.i PROPERTIES SWIG_FLAGS "-Wall")
SET_SOURCE_FILES_PROPERTIES(${PYTHON_SRC_DIR}/jiplib.i PROPERTIES SWIG_FLAGS "-W2")
SWIG_LINK_LIBRARIES(jiplib ${PKTOOLS_LIBRARIES} ${PYTHON_LIBRARIES} ${JIPLIB_LIB_NAME})

########## INSTALL ##########
INSTALL (TARGETS ${JIPLIB_LIB_NAME} LIBRARY DESTINATION lib)
INSTALL (FILES ${JIPLIB_H} DESTINATION include/jiplib)
EXECUTE_PROCESS(COMMAND python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())" OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
SET(PYTHON_INSTALL_DIR ${PYTHON_SITE_PACKAGES} CACHE PATH "Install directory for Python library")
INSTALL(TARGETS ${SWIG_MODULE_jiplib_REAL_NAME} LIBRARY DESTINATION ${PYTHON_INSTALL_DIR})
INSTALL(FILES ${CMAKE_BINARY_DIR}/jiplib.py DESTINATION ${PYTHON_INSTALL_DIR})
install (FILES "${CMAKE_CURRENT_BINARY_DIR}/config_jiplib.h" DESTINATION ${PROJECT_INCLUDE_DIR})
###############################################################################
