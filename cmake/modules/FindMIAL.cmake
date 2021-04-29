include(GNUInstallDirs)

MESSAGE(STATUS "PROJECT_BINARY_DIR: ${PROJECT_BINARY_DIR}")
find_library(
    MIAL_LIBRARY
    NAMES miallib_generic
    HINTS ${PROJECT_BINARY_DIR}/../../jeolib-miallib/core/build/lib ${CMAKE_INSTALL_LIBDIR}
    PATH_SUFFIXES mial/native-linux-x64/)

find_path(MIAL_INCLUDE_DIR
  NAMES mialtypes.h
  HINTS ${PROJECT_BINARY_DIR}/../../jeolib-miallib/core/c ${CMAKE_INSTALL_INCLUDEDIR}
  PATH_SUFFIXES native-linux-api/miallib/ miallib)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(mial DEFAULT_MSG
                                  MIAL_LIBRARY
                                  MIAL_INCLUDE_DIR)

mark_as_advanced(MIAL_LIBRARY MIAL_INCLUDE_DIR)

if(MIAL_FOUND AND NOT TARGET mial::mial)
  add_library(mial::mial SHARED IMPORTED)
  set_target_properties(
    mial::mial
    PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${MIAL_INCLUDE_DIR}"
      IMPORTED_LOCATION ${MIAL_LIBRARY})
endif()
