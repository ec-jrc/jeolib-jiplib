//SWIG interface for jiplib
%include <std_string.i>
%include <std_vector.i>
%include <std_list.i>
%include <std_iostream.i>
%include <std_shared_ptr.i>
%shared_ptr(ImgRaster)
%shared_ptr(jiplib::Jim)

%include "exception.i"
%exception {
  try {
    $action
      }
  catch (const std::string errorString) {
    PyErr_SetString(PyExc_SystemError,errorString.c_str());
    SWIG_fail;
  }
}
%define DOCSTRING
"Joint image processing library (jiplib)
developed in the framework of the JEODPP of the EO&SS@BD pilot project."
%enddef


%feature("autodoc", "2");

%module(docstring=DOCSTRING) jiplib
/* %module(docstring="Joint image processing library (jiplib)") jiplib */

 //almost works, but need to introduce typecheck for overloaded functions
 //then replace PyList with kwargs 
 /* %typemap(in) (const app::AppFactory&) { */
 /*   std::cout << "we are in typemap" << std::endl; */
 /*   int argc=PyList_Size($input)+1; */
 /*   std::vector<std::string> argv; */
 /*   argv.push_back("function"); */
 /*   int i = 0; */
 /*   for (i = 0; i < PyList_Size($input); i++) { */
 /*     PyObject *s = PyList_GetItem($input,i); */
 /*     if (!PyString_Check(s)) { */
 /*       PyErr_SetString(PyExc_ValueError, "List items must be strings"); */
 /*       return NULL; */
 /*     } */
 /*     std::string inputString=PyString_AsString(s); */
 /*     argv.push_back(inputString); */
 /*   } */
 /*   $1=new app::AppFactory(); */
 /*   $1->setOptions(argc,argv); */
 /* } */

/* %typemap(freearg)  (const app::AppFactory&){ */
/*   if ($1) free($1); */
/*  } */

%{
#include <memory>
#include "config.h"
#include "imageclasses/ImgRaster.h"
#include "imageclasses/ImgCollection.h"
#include "apps/AppFactory.h"
#include "algorithms/Filter2d.h"
#include "jim.h"
#include "mialib_swig.h"
#include <cpl_error.h>
  %}


%template(ImgVectorRaster) std::vector< std::shared_ptr< ImgRaster > >;
%template(ImgVectorJim) std::vector< std::shared_ptr< jiplib::Jim > >;

//Parse the header file
%include "swig/pktools.i"
%include "imageclasses/ImgCollection.h"
%include "imageclasses/ImgRaster.h"
%include "apps/AppFactory.h"
%include "algorithms/Filter2d.h"
%include "jim.h"

// Instantiate templates for vector
/* %template(ByteVector) std::vector<char>; */
/* %template(Int16Vector) std::vector<short>; */
/* %template(UInt16Vector) std::vector<unsigned short>; */
/* %template(Int32Vector) std::vector<int>; */
/* %template(UInt32Vector) std::vector<unsigned int>; */
/* %template(Float32Vector) std::vector<float>; */
/* %template(Float64Vector) std::vector<double>; */
/* %template(StringVector) std::vector<std::string>; */

enum CPLErr {CE_None = 0, CE_Debug = 1, CE_Warning = 2, CE_Failure = 3, CE_Fatal = 4};
enum GDALDataType {GDT_Unknown = 0, GDT_Byte = 1, GDT_UInt16 = 2, GDT_Int16 = 3, GDT_UInt32 = 4, GDT_Int32 = 5, GDT_Float32 = 6, GDT_Float64 = 7, GDT_CInt16 = 8, GDT_CInt32 = 9, GDT_CFloat32 = 10, GDT_CFloat64 = 11, GDT_TypeCount = 12}; 

//from :  http://stackoverflow.com/questions/39436632/wrap-a-function-that-takes-a-struct-of-optional-arguments-using-kwargs
