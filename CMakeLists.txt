###############################################################################
# CMakeLists build script for cmake
# Author(s): Pieter.Kempeneers@ec.europa.eu
# Copyright (C) 2016-2026 European Union (Joint Research Centre)
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

cmake_minimum_required(VERSION 3.15)
project(jiplib LANGUAGES CXX)

# --- 1. Project Metadata ---
set(JIPLIB_VERSION_MAJOR 1)
set(JIPLIB_VERSION_MINOR 2)
set(JIPLIB_VERSION_PATCH 0)
set(JIPLIB_VERSION "${JIPLIB_VERSION_MAJOR}.${JIPLIB_VERSION_MINOR}.${JIPLIB_VERSION_PATCH}")
set(JIPLIB_SOVERSION "${JIPLIB_VERSION_MAJOR}")

# --- 2. Build Settings ---
option(BUILD_SHARED_LIBS "Build with shared library" ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)

if(TARGET miallib)
    # 1. If we are building inside pyjeo, the 'miallib' target already exists!
    message(STATUS "****** MIALLIB detected as an internal target. Skipping find_package.")
    set(MIALLIB_FOUND TRUE)
    set(MIALLIB_TARGET miallib)
else()
    # 2. If building jiplib STANDALONE, try to find it on the system
    find_package(MIALLIB 1.2.0 QUIET)

    message(STATUS "Found MIALLIB_LIBRARY: ${MIALLIB_LIBRARIES}")
    get_cmake_property(_vars VARIABLES)
    foreach(_var ${_vars})
        if(_var MATCHES "^MIALLIB")
            message(STATUS "${_var} = ${${_var}}")
        endif()
    endforeach()
    
    if(TARGET miallib::miallib)
	message(STATUS "Found target: miallib::miallib")
	#If location not found, set path
        set_target_properties(miallib::miallib PROPERTIES
            IMPORTED_LOCATION "${MIALLIB_LIBRARIES}"
	)
	get_target_property(LOC miallib::miallib IMPORTED_LOCATION)
        message(STATUS "DEBUG: miallib location is: ${LOC}")
    elseif(TARGET miallib)
	message(STATUS "Found target: miallib")
    else()
	message(STATUS "No target found, set miallib path variables")
    endif()

    if(NOT MIALLIB_FOUND)
        # 3. Last resort: check if it's a sibling directory (common in dev)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../miallib")
             add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../miallib" "${CMAKE_CURRENT_BINARY_DIR}/miallib_build")
             set(MIALLIB_TARGET miallib)
             set(MIALLIB_FOUND TRUE)
        else()
             message(FATAL_ERROR "MIALLIB not found. Please install it or provide it in external/miallib")
        endif()
    else()
        set(MIALLIB_TARGET MIALLIB::MIALLIB)
    endif()
endif()
# 1. Locate MIALLIB headers (as siblings in the pyjeo build)
if(TARGET miallib)
    get_target_property(MIALLIB_BASE_DIR miallib SOURCE_DIR)
    set(MIALLIB_INCLUDE_DIR "${MIALLIB_BASE_DIR}/core/c")
    MESSAGE(STATUS "MIALLIB_INCLUDE_DIR with target miallib: ${MIALLIB_INCLUDE_DIR}")
else()
    # Fallback for standalone jiplib build
    set(MIALLIB_INCLUDE_DIR "${MIALLIB_INCLUDE_DIRS}")
    MESSAGE(STATUS "MIALLIB_INCLUDE_DIR in standalone: ${MIALLIB_INCLUDE_DIR}")
endif()

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

set(GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses")
file(MAKE_DIRECTORY "${GEN_DIR}")

# --- A. Copy rename.txt and scripts ---
# rename.txt is the critical input for rename.sh
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/src/imageclasses/rename.txt" "${GEN_DIR}/rename.txt" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/src/imageclasses/rename.sh" "${GEN_DIR}/rename.sh" COPYONLY)

# --- B. Copy the Python generators ---
set(PY_GENERATORS
    imagetype imagetype_multi imagetype_jimlist imagelisttype
    errortype errortype_d errortype_nd errortype_nm errortype_d_nm
)

message(STATUS "****** Running Transformation Pipeline in ${GEN_DIR}")

#test
# 1. Standardize the paths
get_filename_component(GEN_DIR_ABS "${GEN_DIR}" ABSOLUTE)
set(MY_JSON_FILE "${GEN_DIR_ABS}/old2NewNames.json")

message(STATUS "****** Preparing JSON mapping (replacing rename.sh)")
message(STATUS "****** GEN_DIR_ABS: ${GEN_DIR_ABS}")


#Read the source file directly
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/src/imageclasses/rename.txt" rename_lines)

set(json_entries "")
list(LENGTH rename_lines total_lines)
message(STATUS "DEBUG: Found ${total_lines} lines in rename.txt. Generated ${len_c} entries for JSON.")
math(EXPR last_idx "${total_lines} - 1")

#Process each line
foreach(i RANGE ${last_idx})
    list(GET rename_lines ${i} line)

    # Extract C Name: Look for word before the '('
    # This replaces the logic that used to be in c_names.txt
    string(REGEX MATCH "[a-zA-Z0-9_]+\\(" c_name_raw "${line}")
    string(REPLACE "(" "" c_name "${c_name_raw}")

    # Extract Method Name: Everything after the last space
    # This replaces the 'sed' logic
    string(REGEX REPLACE "^.* " "" m_name "${line}")

    # Build the JSON entry in memory (This replaces 'paste')
    if(NOT c_name STREQUAL "" AND NOT m_name STREQUAL "")
        if(i LESS last_idx)
            list(APPEND json_entries "  \"${c_name}\": \"${m_name}\",")
        else()
            list(APPEND json_entries "  \"${c_name}\": \"${m_name}\"")
        endif()
    endif()
endforeach()

#Write the final result
string(REPLACE ";" "\n" json_body "${json_entries}")
file(WRITE "${MY_JSON_FILE}" "{\n${json_body}\n}")
if(NOT EXISTS "${MY_JSON_FILE}")
    message(FATAL_ERROR
        "CRITICAL ERROR: old2NewNames.json was not generated!\n"
        "Expected at: ${MY_JSON_FILE}\n"
        "Please check the previous regex extraction logic."
    )
else()
    # Optional: Verify it isn't an empty file
    file(SIZE "${MY_JSON_FILE}" _json_size)
    if(_json_size LESS 5)
        message(FATAL_ERROR "JSON file exists but appears to be empty or invalid.")
    endif()
    message(STATUS "SUCCESS: Confirmed old2NewNames.json exists and has data.")
endif()
# Run the Python loop
foreach(gen ${PY_GENERATORS})
    message(STATUS "****** Running Transformation Pipeline: fun2method_${gen}")
    message(STATUS "   JSON:   '${MY_JSON_FILE}'")
    execute_process(
        COMMAND "${Python_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/imageclasses/fun2method_${gen}.py"
                "-o"
                "fun2method_${gen}"
                "-j"
                "${MY_JSON_FILE}"
        WORKING_DIRECTORY "${GEN_DIR_ABS}"
        RESULT_VARIABLE gen_res
	OUTPUT_QUIET
    )
    if(NOT gen_res EQUAL 0)
        message(FATAL_ERROR "Generation failed for ${gen}. Exit code: ${gen_res}")
    endif()
    if(NOT EXISTS "${GEN_DIR_ABS}/fun2method_${gen}.cc")
        message(FATAL_ERROR
            "CRITICAL ERROR: output file was not generated!"
        )
    endif()
    if(NOT EXISTS "${GEN_DIR_ABS}/fun2method_${gen}.h")
        message(FATAL_ERROR
            "CRITICAL ERROR: output file was not generated!"
        )
    endif()

    if(NOT gen_res EQUAL 0)
        message(FATAL_ERROR "Generator fun2method_${gen}.py failed")
    endif()
endforeach()

# Helper Macro for Code Injection ---
macro(inject_code_block FILE_PATH MARKER_ID INJECT_FILE_NAME)
    if(EXISTS "${FILE_PATH}")
        file(READ "${FILE_PATH}" CURRENT_CONTENTS)

        # Try finding the file in Source first, then Binary
        set(PATH_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/imageclasses/${INJECT_FILE_NAME}")
        set(PATH_BIN "${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/${INJECT_FILE_NAME}")

        if(EXISTS "${PATH_SRC}")
            set(INJECT_FILE_PATH "${PATH_SRC}")
        elseif(EXISTS "${PATH_BIN}")
            set(INJECT_FILE_PATH "${PATH_BIN}")
        else()
            set(INJECT_FILE_PATH "NOTFOUND")
        endif()

        if(NOT "${INJECT_FILE_PATH}" STREQUAL "NOTFOUND")
            file(READ "${INJECT_FILE_PATH}" INJECT_DATA)

            string(REGEX REPLACE
                "(//start insert from ${MARKER_ID}\n)(.*)(\n//end insert from ${MARKER_ID})"
                "\\1${INJECT_DATA}\\3"
                UPDATED_CONTENTS
                "${CURRENT_CONTENTS}")

            message(STATUS "CHECK: Creating JSON for ${gen} with ${len_c} entries.")
            file(WRITE "${FILE_PATH}" "${UPDATED_CONTENTS}")
        else()
            message(WARNING "Skipping injection: ${INJECT_FILE_NAME} not found in Source or Binary dirs.")
        endif()
    endif()
endmacro()

# Initialize the Adapted Files ---
set(JIM_DEST "${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/Jim.h")
set(JIMLIST_DEST "${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/JimList.h")

# Copy the originals to the build dir first so we have a clean slate to modify
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/src/imageclasses/Jim.h" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/src/imageclasses/JimList.h" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses")

# Run all Injections for Jim.h ---
set(JIM_BLOCKS
    "imagetype" "imagetype_multi" "imagelisttype" "errortype"
    "errortype_d" "errortype_nd" "errortype_nm" "errortype_d_nm"
)

message(STATUS "DEBUG: Looking for injections in ${CMAKE_CURRENT_SOURCE_DIR}/src/imageclasses")

foreach(block ${JIM_BLOCKS})
    inject_code_block("${JIM_DEST}" "fun2method_${block}" "fun2method_${block}.h")
endforeach()

# Run all Injections for JimList.h ---
inject_code_block("${JIMLIST_DEST}" "fun2method_imagetype_jimlist" "fun2method_imagetype_jimlist.h")
inject_code_block("${JIMLIST_DEST}" "fun2method_imagelisttype" "fun2method_imagelisttype.h")

message(STATUS "****** Jim.h and JimList.h have been successfully adapted via CMake.")

# Prepare Jim.cc and JimList.cc in the build folder ---
# Copy the original files if they haven't been copied yet
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/src/imageclasses/Jim.cc" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/src/imageclasses/JimList.cc" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses")

set(JIM_CC_DEST "${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/Jim.cc")
set(JIMLIST_CC_DEST "${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/JimList.cc")

# --- 2. Define the includes we want to add ---
set(JIM_INCLUDES
    "fun2method_imagetype.cc"
    "fun2method_imagetype_multi.cc"
    "fun2method_imagelisttype.cc"
    "fun2method_errortype.cc"
    "fun2method_errortype_d.cc"
    "fun2method_errortype_nd.cc"
    "fun2method_errortype_nm.cc"
    "fun2method_errortype_d_nm.cc"
)

# Robust Append for Jim.cc ---
# We read the file first to check if we've already added the includes
file(READ "${JIM_CC_DEST}" JIM_CC_CONTENT)

set(NEW_INCLUDES "")
foreach(inc_file ${JIM_INCLUDES})
    set(LINE_TO_ADD "#include \"${inc_file}\"")
    # Only add if the line isn't already there
    if(NOT JIM_CC_CONTENT MATCHES "${LINE_TO_ADD}")
        string(APPEND NEW_INCLUDES "${LINE_TO_ADD}\n")
    endif()
endforeach()

if(NEW_INCLUDES)
    file(APPEND "${JIM_CC_DEST}" "${NEW_INCLUDES}")
endif()

# Robust Append for JimList.cc ---
file(READ "${JIMLIST_CC_DEST}" JIMLIST_CC_CONTENT)
set(JIMLIST_LINE "#include \"fun2method_imagetype_jimlist.cc\"")

if(NOT JIMLIST_CC_CONTENT MATCHES "${JIMLIST_LINE}")
    file(APPEND "${JIMLIST_CC_DEST}" "${JIMLIST_LINE}\n")
endif()

message(STATUS "****** Jim.cc and JimList.cc include lines verified.")
find_package(Boost REQUIRED CONFIG COMPONENTS filesystem system serialization)
find_package(GDAL 3.00 REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(OpenMP REQUIRED)

message(STATUS "Searching (1) for FANN in: ${CMAKE_MODULE_PATH}")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
message(STATUS "Searching (2) for FANN in: ${CMAKE_MODULE_PATH}")
FIND_PACKAGE(FANN 2.2 REQUIRED)
message(STATUS "FANN_INCLUDE_DIRS: ${FANN_INCLUDE_DIRS}")
message(STATUS "FANN_LIBRARIES: ${FANN_LIBRARIES}")

find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

# Source Preparation ---
file(COPY ${PROJECT_SOURCE_DIR}/src DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
set(SRC_ROOT ${CMAKE_CURRENT_BINARY_DIR}/src)

# The project's name and version
project(jiplib)

# Name of C++ library
SET(JIPLIB_LIB_NAME jiplib)
SET(JIPLIB_PYTHON_LIB_NAME jiplib)

# Create the target here so all subsequent target_... commands work
add_library(${JIPLIB_LIB_NAME} SHARED)
enable_testing()
INCLUDE(CTest)

SET(JIPLIB_VERSION_MAJOR 1)
SET(JIPLIB_VERSION_MINOR 2)
SET(JIPLIB_VERSION_PATCH 0)
SET(JIPLIB_VERSION "${JIPLIB_VERSION_MAJOR}.${JIPLIB_VERSION_MINOR}.${JIPLIB_VERSION_PATCH}")
SET(PACKAGE_VERSION "${JIPLIB_VERSION}")
SET(JIPLIB_SOVERSION "${JIPLIB_VERSION_MAJOR}")
SET(JIPLIB_PACKAGE_BUGREPORT "pieter.kempeneers@ec.europa.eu")

SET(CLASSIF 0 CACHE STRING "set a variable that can be used in Jim.cc")
SET(ODOITHIN 0 CACHE STRING "set a variable that can be used in Jim.cc")
SET(MCISRG 1 CACHE STRING "set a variable that can be used in Jim.cc")

SET(BUILD_WITH_LIBLAS FALSE CACHE BOOL "Choose if jiplib is to be built with liblas")
SET(BUILD_WITH_FANN TRUE CACHE BOOL "Choose if jiplib is to be built with fann")
SET(BUILD_WITH_PYTHON TRUE CACHE BOOL "Choose if jiplib is to be built with PYTHON support")
SET(PROCESS_IN_PARALLEL TRUE CACHE BOOL "Choose if jiplib should be run in parallel")
#from https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling
#By default if you don't change any RPATH related settings, CMake will
# link the executables and shared libraries with full RPATH to all used
# libraries in the build tree. When installing, it will clear the RPATH of
# these targets so they are installed with an empty RPATH
set_target_properties(jiplib PROPERTIES INSTALL_RPATH "$ORIGIN")

SET(PROJECT_INCLUDE_DIR include/jiplib)
SET(PROJECT_TEST_DIR test)

SET(CMAKE_COLOR_MAKEFILE ON)

SET(MIAL_BOOL 1)

option (BUILD_SHARED_LIBS "Build with shared library" ON)

# Platform and compiler specific settings

# Elevate to C++17 to match Conda dependencies
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

########## BUILDING INSTALLLER ##########
# build a CPack driven installer package
INCLUDE (InstallRequiredSystemLibraries)
SET(CPACK_RESOURCE_FILE_LICENSE
     "${CMAKE_CURRENT_SOURCE_DIR}/COPYING")
SET(CPACK_PACKAGE_VERSION_MAJOR "${JIPLIB_VERSION_MAJOR}")
SET(CPACK_PACKAGE_VERSION_MINOR "${JIPLIB_VERSION_MINOR}")
SET(CPACK_PACKAGE_VERSION_PATCH "${JIPLIB_VERSION_PATCH}")
SET(CPACK_GENERATOR "DEB")
SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "Pieter Kempeneers")
SET(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
# Set files to ignore
list(APPEND CPACK_SOURCE_IGNORE_FILES "_CPack_Packages")
list(APPEND CPACK_SOURCE_IGNORE_FILES ".gz")
list(APPEND CPACK_SOURCE_IGNORE_FILES ".GZ")
list(APPEND CPACK_SOURCE_IGNORE_FILES ".tgz")
list(APPEND CPACK_SOURCE_IGNORE_FILES ".TGZ")
list(APPEND CPACK_SOURCE_IGNORE_FILES ".bz2")
list(APPEND CPACK_SOURCE_IGNORE_FILES ".BZ2")
list(APPEND CPACK_SOURCE_IGNORE_FILES ".zip")
list(APPEND CPACK_SOURCE_IGNORE_FILES ".ZIP")
list(APPEND CPACK_SOURCE_IGNORE_FILES "test")
list(APPEND CPACK_SOURCE_IGNORE_FILES ".svn")
list(APPEND CPACK_SOURCE_IGNORE_FILES ".git")
list(APPEND CPACK_SOURCE_IGNORE_FILES "HOWTORELEASE.txt")
list(APPEND CPACK_SOURCE_IGNORE_FILES "CMakeCache.txt")
list(APPEND CPACK_SOURCE_IGNORE_FILES "CPackConfig.cmake")
list(APPEND CPACK_SOURCE_IGNORE_FILES "schemas")
list(APPEND CPACK_SOURCE_IGNORE_FILES "/build/;~$;${CPACK_SOURCE_IGNORE_FILES}")

INCLUDE(CPack)
CPACK_ADD_COMPONENT(libraries)
CPACK_ADD_COMPONENT(headers)
CPACK_ADD_COMPONENT(wheels)
SET(CPACK_DEB_COMPONENT_INSTALL YES)
SET(CPACK_COMPONENTS_ALL libraries headers wheels)

add_custom_target(dist COMMAND ${CMAKE_MAKE_PROGRAM} package_source)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

# Make sure Boost libraries are found
SET(Boost_DEBUG 0)

if(BUILD_WITH_PYTHON)
  SET(CREATE_WHEEL TRUE CACHE BOOL "Choose if jiplib PYTHON wheel is to be created")
endif(BUILD_WITH_PYTHON)

MESSAGE(STATUS "CREATE_WHEEL: ${CREATE_WHEEL}")

SET(PYTHON3 ON CACHE BOOL "set ON for PYTHON3")

find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)
MESSAGE(STATUS "Python_NumPy_FOUND: ${Python_NumPy_FOUND}")

message(STATUS "Python version found: ${Python_VERSION}")
MESSAGE(STATUS "Python_LIBRARIES: ${Python_LIBRARIES}")
MESSAGE(STATUS "Python_INCLUDE_DIRS: ${Python_INCLUDE_DIRS}")
message(STATUS "Python package will be installed in ${Python_SITELIB}")

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
  message(STATUS "GDAL version found: ${GDAL_VERSION}")
  message(STATUS "GDAL version major: ${GDAL_VERSION_MAJOR}")
  message(STATUS "GDAL version minor: ${GDAL_VERSION_MINOR}")
endif()

find_package(GSL REQUIRED)
target_link_libraries(jiplib PRIVATE GSL::gsl)
if(GSL_FOUND)
  message(STATUS "Found GSL: ${GSL_LIBRARIES}")
endif()

#Find the module and create a modern target
pkg_check_modules(JSONCPP REQUIRED IMPORTED_TARGET jsoncpp)

if(PROCESS_IN_PARALLEL)
	find_package(OpenMP REQUIRED)
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

INCLUDE(GNUInstallDirs)
SET(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR}/jiplib
  CACHE PATH "Location of header files" )
SET(SYSCONFIG_INSTALL_DIR ${CMAKE_INSTALL_SYSCONFDIR}/jiplib
  CACHE PATH "Location of configuration files" )

# configure a header file to pass some of the CMake settings to the source code
configure_file (
  "cmake-config.h.in"
  "config_jiplib.h"
  )

#Generating Package Files
INCLUDE(CMakePackageConfigHelpers)

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


# Instruct CMake to inspect the following subfolders (with CMakeLists.txt in each subfolder)
add_subdirectory ("${PROJECT_TEST_DIR}")

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
  ${IMGCLASS_SRC_DIR}/Json_compat.h
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
  ${IMGCLASS_SRC_DIR}/Json_compat.cc
  )

if(BUILD_WITH_FANN)
  SET(IMGCLASS_CC ${IMGCLASS_CC}
  ${IMGCLASS_SRC_DIR}/jlann_lib.cc
  ${IMGCLASS_SRC_DIR}/jlannogr_lib.cc
  )
endif(BUILD_WITH_FANN)

if(TARGET miallib)
  set(MIALLIB_TARGET miallib)
else()
  set(MIALLIB_TARGET MIALLIB::MIALLIB)
endif()

SET(MIAL_BOOL 1)
message(STATUS "build with miallib: ${MIAL_BOOL}")

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
EXECUTE_PROCESS(COMMAND ${Python_EXECUTABLE} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses INPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/fun2method_imagetype.py OUTPUT_QUIET)
EXECUTE_PROCESS(COMMAND ${Python_EXECUTABLE} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses INPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/fun2method_imagetype_multi.py OUTPUT_QUIET)
EXECUTE_PROCESS(COMMAND ${Python_EXECUTABLE} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses INPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/fun2method_imagetype_jimlist.py OUTPUT_QUIET)
EXECUTE_PROCESS(COMMAND ${Python_EXECUTABLE} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses INPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/fun2method_imagelisttype.py OUTPUT_QUIET)
EXECUTE_PROCESS(COMMAND ${Python_EXECUTABLE} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses INPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/fun2method_errortype.py OUTPUT_QUIET)
EXECUTE_PROCESS(COMMAND ${Python_EXECUTABLE} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses INPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/fun2method_errortype_d.py OUTPUT_QUIET)
EXECUTE_PROCESS(COMMAND ${Python_EXECUTABLE} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses INPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/fun2method_errortype_nd.py OUTPUT_QUIET)
EXECUTE_PROCESS(COMMAND ${Python_EXECUTABLE} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses INPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/fun2method_errortype_nm.py OUTPUT_QUIET)
EXECUTE_PROCESS(COMMAND ${Python_EXECUTABLE} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses INPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/src/imageclasses/fun2method_errortype_d_nm.py OUTPUT_QUIET)

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

###############################################################################
# Define the jiplib library.
if (BUILD_WITH_LIBLAS)
  SET(JIPLIB_HEADER_FILES ${LASCLASS_H} ${IMGCLASS_H} ${ALGOR_H} ${FILECLASS_H} ${BASE_H})
  SET(JIPLIB_SRC_FILES ${LASCLASS_CC} ${IMGCLASS_CC} ${ALGOR_CC} ${FILECLASS_CC} ${BASE_CC})
else(BUILD_WITH_LIBLAS)
  SET(JIPLIB_HEADER_FILES ${IMGCLASS_H} ${ALGOR_H} ${FILECLASS_H} ${BASE_H})
  SET(JIPLIB_SRC_FILES ${IMGCLASS_CC} ${ALGOR_CC} ${FILECLASS_CC} ${BASE_CC})
endif(BUILD_WITH_LIBLAS)
message(STATUS "JIPLIB_SRC_FILES: ${JIPLIB_SRC_FILES}")

message(STATUS "${CMAKE_CURRENT_BINARY_DIR}: ${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "${SRC_ROOT}: ${SRC_ROOT}")

target_include_directories(jiplib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<BUILD_INTERFACE:${SRC_ROOT}>
    $<INSTALL_INTERFACE:include>
)
target_sources(jiplib PRIVATE ${JIPLIB_SRC_FILES})
SET_TARGET_PROPERTIES(${JIPLIB_LIB_NAME} PROPERTIES PUBLIC_HEADER "${JIPLIB_HEADER_FILES}")

###############################################################################

TARGET_COMPILE_DEFINITIONS(${JIPLIB_LIB_NAME} PUBLIC MIALLIB_DLL)
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

get_target_property(JSON_LOC PkgConfig::JSONCPP INTERFACE_LINK_LIBRARIES)
message(STATUS "JSONCPP Target Location: ${JSON_LOC}")
target_link_libraries(${JIPLIB_LIB_NAME}
    PUBLIC
        miallib::miallib
    PRIVATE
        Python::Module
        GDAL::GDAL
        GSL::gsl
	PkgConfig::JSONCPP
        OpenMP::OpenMP_CXX
        Boost::filesystem
        Boost::system
        Boost::serialization
)

target_include_directories(jiplib PUBLIC
 ${MIALLIB_INCLUDE_DIR}
 )

message(STATUS "JSONCPP LIBRARIES: ${JSONCPP_LIBRARIES}")
message(STATUS "JSONCPP INCLUDE DIRS: ${JSONCPP_INCLUDE_DIRS}")

if(BUILD_WITH_FANN)
  target_link_libraries(
    ${JIPLIB_LIB_NAME}
    PRIVATE
    ${FANN_LIBRARIES}
    )
endif(BUILD_WITH_FANN)


if(BUILD_WITH_PYTHON)
  ########## SWIG #############
  SET(PYTHON_SRC_DIR src/swig)
  FIND_PACKAGE(SWIG REQUIRED)
  MESSAGE(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")
  # todo adapt to new policies:
  #Changed in version 3.13: This command creates a target with the specified <name> when policy CMP0078 is set to NEW. Otherwise, the legacy behavior will choose a different target name and store it in the SWIG_MODULE_<name>_REAL_NAME variable.
  #Changed in version 3.15: Alternate library name (set with the OUTPUT_NAME property, for example) will be passed on to Python and CSharp wrapper libraries.
  cmake_policy(SET CMP0078 NEW)
  cmake_policy(SET CMP0086 NEW)
  INCLUDE(${SWIG_USE_FILE})
  MESSAGE(STATUS "Python_INCLUDE_DIRS: ${Python_INCLUDE_DIRS}")
  MESSAGE(STATUS "Python_NumPy_INCLUDE_DIRS: ${Python_NumPy_INCLUDE_DIRS}")
  MESSAGE(STATUS "Python_LIBRARIES: ${Python_LIBRARIES}")
  TARGET_INCLUDE_DIRECTORIES(${JIPLIB_LIB_NAME} PUBLIC ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})
endif(BUILD_WITH_PYTHON)

#we need INCLUDE_DIRECTORIES here instead of TARGET_INCLUDE_DIRECTORIES
# for subfolder WORKING_DIRECTORY
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}/src)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES(${MIALLIB_INCLUDE_DIRS})

SET(CMAKE_SWIG_FLAGS "-keyword")
SET(CMAKE_SWIG_FLAGS "-DMIALLIB=${MIAL_BOOL}")
SET_PROPERTY(SOURCE ${JIPLIB_I} PROPERTY CPLUSPLUS ON)

SET_SOURCE_FILES_PROPERTIES(${JIPLIB_I} PROPERTIES SWIG_FLAGS "-W2")
TARGET_INCLUDE_DIRECTORIES(jiplib PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# SWIG Python Wrapper ---
set_property(SOURCE src/swig/jiplib.i PROPERTY CPLUSPLUS ON)
set_property(TARGET jiplib PROPERTY SWIG_USE_TARGET_INTERFACE_INCLUDE_DIRECTORIES ON)

# Define Unique Names ---
set(JIPLIB_LIB_NAME "jiplib")           # The C++ shared library (The Engine)
set(JIPLIB_PYTHON_TARGET "jiplib_py")   # The SWIG target name (The Wrapper)

# Create the SWIG Library ---
# Use the unique 'jiplib_py' name to avoid CMP0002 errors
# Clear out the old version-check block and use this:

# Use the current source directory as a base to avoid the "//" issue
set(JIPLIB_I "${CMAKE_CURRENT_SOURCE_DIR}/src/swig/jiplib.i")
swig_add_library(jiplib_python
    TYPE MODULE
    LANGUAGE python
    SOURCES "${JIPLIB_I}"
)

if(NOT TARGET jiplib_python)
    message(FATAL_ERROR "Critical Error: jiplib_python target was not created. Check if JIPLIB_I is defined.")
endif()


message(STATUS "${PROJECT_SOURCE_DIR}/external/miallib/core/c")
if(TARGET miallib::miallib)
    get_target_property(MIALLIB_INCLUDES miallib::miallib INTERFACE_INCLUDE_DIRECTORIES)
elseif(TARGET miallib)
    get_target_property(MIALLIB_INCLUDES miallib INTERFACE_INCLUDE_DIRECTORIES)
else()
    message(WARNING "Target miallib not found")
endif()

# Update the SWIG properties
set_property(TARGET jiplib_python PROPERTY SWIG_USE_TARGET_INCLUDE_DIRECTORIES TRUE)
#Add only the "internal" generated paths that aren't in targets
# Usually, SWIG needs the directory where the generated .cxx files live
set_property(TARGET jiplib_python APPEND PROPERTY
    SWIG_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_BINARY_DIR}"
)
# To fix the GDAL version warning, we can pre-define the version for SWIG
# Convert CMake GDAL variables into a format SWIG understands
math(EXPR GDAL_VERSION_NUM_CALC "${GDAL_VERSION_MAJOR} * 1000000 + ${GDAL_VERSION_MINOR} * 10000")

set_property(TARGET jiplib_python APPEND PROPERTY
    SWIG_FLAGS
    "-DGDAL_VERSION_NUM=${GDAL_VERSION_NUM_CALC}"
)
# Set the output name so 'import jiplib' works
# This keeps the internal target name unique while the file name stays 'jiplib'
set_target_properties(jiplib_python PROPERTIES
    OUTPUT_NAME "jiplib"
    PREFIX "_"
)

target_link_libraries(jiplib_python
    PRIVATE
    jiplib
    Python::Module
    Python::NumPy
)
set(JIPLIB_PYTHON_LIB_NAME ${JIPLIB_PYTHON_TARGET})

set_target_properties(jiplib PROPERTIES OUTPUT_NAME "jiplib")

# Installation (Crucial for Wheels) ---
include(GNUInstallDirs)

set_target_properties(jiplib_python PROPERTIES INSTALL_RPATH "$ORIGIN")
