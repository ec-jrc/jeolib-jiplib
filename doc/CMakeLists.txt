###############################################################################
# add a target to generate API documentation with Doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
  SET(DOXY2SWIG ${CMAKE_CURRENT_SOURCE_DIR}/doxy2swig.py)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
  # find_package(PythonInterp 2.6.0 REQUIRED)
  add_custom_target(doc
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    # COMMAND ${PYTHON_EXECUTABLE} ${DOXY2SWIG} ${CMAKE_CURRENT_BINARY_DIR}/xml/index.xml ${CMAKE_CURRENT_BINARY_DIR}/jiplib_doc.i
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating jiplib API documentation with Doxygen" VERBATIM
    )
endif(DOXYGEN_FOUND)

# cd ~/jiplib/build
# make doc
# cd ~/jiplib/doc/xml
# python ~/jiplib/doxy2swig.py index.xml ~/jiplib/src/swig/jiplib_doc.i
# cd ~/jiplib/build
# touch ~/jiplib/src/swig/jiplib.i
# make -j
