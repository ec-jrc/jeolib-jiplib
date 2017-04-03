###############################################################################
# add a target to generate API documentation with Doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
  # SET(DOXY2SWIG ${CMAKE_CURRENT_SOURCE_DIR}/doxy2swig.py)
  find_program(DOXY2SWIG doxy2swig.py HINTS ${CMAKE_CURRENT_SOURCE_DIR})
  if(NOT DOXY2SWIG)
    message(WARNING "doxy2swig.py not found!")
  endif()

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
  add_custom_target(doc
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    COMMAND ${PYTHON_EXECUTABLE} ${DOXY2SWIG} ${CMAKE_CURRENT_BINARY_DIR}/xml/index.xml ${CMAKE_CURRENT_BINARY_DIR}/jiplib_doc.i
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating jiplib API documentation with Doxygen" VERBATIM
    )
endif(DOXYGEN_FOUND)

# cd ~/jiplib/build
# make doc
# cd ~/jiplib/build/doc/xml
# python ~/jiplib/doxy2swig.py index.xml ~/jiplib/src/swig/jiplib_doc.i
# cd ~/jiplib/build
# touch ~/jiplib/src/swig/jiplib.i
# make -j
