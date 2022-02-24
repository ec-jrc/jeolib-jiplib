include(GNUInstallDirs)

find_library(
    MIAL_LIBRARY
    miallib_generic
    HINTS ${CMAKE_INSTALL_FULL_LIBDIR}
    PATH_SUFFIXES mial/native-linux-x64/)

find_path(MIAL_INCLUDE_DIR
  NAMES mialtypes.h
  HINTS ${CMAKE_INSTALL_INCLUDEDIR}
  PATH_SUFFIXES native-linux-api/miallib/ miallib)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(mial DEFAULT_MSG
                                  MIAL_LIBRARY
                                  MIAL_INCLUDE_DIR)

mark_as_advanced(MIAL_LIBRARY MIAL_INCLUDE_DIR)

set(MIAL_LIBRARIES ${MIAL_LIBRARY} )
set(MIAL_INCLUDE_DIRS ${MIAL_INCLUDE_DIR} )

if(MIAL_FOUND AND NOT TARGET mial::mial)
  add_library(mial::mial SHARED IMPORTED)
  set_target_properties(
    mial::mial
    PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${MIAL_INCLUDE_DIRS}"
      IMPORTED_LOCATION ${MIAL_LIBRARY})
endif()
