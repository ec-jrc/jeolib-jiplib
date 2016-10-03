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

 //working
/* %typemap(in) const app::AppFactory& { */
/*    std::cout << "we are in typemap AppFactory" << std::endl; */
/*    void *argp2 = 0 ; */
/*    /\* $1=new app::AppFactory(); *\/ */
/*    int res2 = 0 ; */
/*    res2 = SWIG_ConvertPtr($input, &argp2, SWIGTYPE_p_app__AppFactory,  0  | 0); */
/*    if (!SWIG_IsOK(res2)) { */
/*      SWIG_exception_fail(SWIG_ArgError(res2), "in method '" "Derived_printKeyValues" "', argument " "2"" of type '" "app::AppFactory const &""'"); */
/*    } */
/*    if (!argp2) { */
/*      SWIG_exception_fail(SWIG_ValueError, "invalid null reference " "in method '" "Derived_printKeyValues" "', argument " "2"" of type '" "app::AppFactory const &""'"); */
/*    } */
/*    $1 = reinterpret_cast< app::AppFactory * >(argp2); */
/*    $1->showOptions(); */
/*    } */

%typemap(in) const std::vector<double>& {
  if(PyList_Check($input)){
    $1=new std::vector<double>(PyList_Size($input));
    Py_ssize_t ppos=0;
    for(Py_ssize_t i=0;i<PyList_Size($input);++i){
      PyObject *rValue;
      rValue=PyList_GetItem($input,i);
      $1->at(i)=PyFloat_AsDouble(rValue);
    }
  } else {
    SWIG_exception(SWIG_TypeError, "PyList expected");
  }
 }

%typemap(freearg)  (const std::vector<double>&){
  if ($1) free($1);
 }

%typemap(in) const app::AppFactory& {
  std::cout << "we are in typemap AppFactory" << std::endl;
  if(PyDict_Check($input)){
    PyObject *pKey, *pValue;
    Py_ssize_t ppos=0;
    $1=new app::AppFactory();
    while (PyDict_Next($input, &ppos, &pKey, &pValue)) {
      std::string theKey=PyString_AsString(pKey);
      std::string theValue;
      if(PyList_Check(pValue)){
        for(Py_ssize_t i=0;i<PyList_Size(pValue);++i){
          PyObject *rValue;
          rValue=PyList_GetItem(pValue,i);
          if(PyString_Check(rValue))
            theValue=PyString_AsString(rValue);
          else
            theValue=PyString_AsString(PyObject_Repr(rValue));
          $1->pushOption(theKey,theValue);
        }
        continue;
      }
      else if(PyString_Check(pValue))
        theValue=PyString_AsString(pValue);
      else
        theValue=PyString_AsString(PyObject_Repr(pValue));
      $1->pushOption(theKey,theValue);
    }
    $1->showOptions();
  } else {
    SWIG_exception(SWIG_TypeError, "Python dictionary expected");
  }
 }

%typemap(freearg)  (const app::AppFactory&){
  if ($1) free($1);
 }

/* !!! from: http://svn.salilab.org/imp/branches/1.0/kernel/pyext/IMP_streams.i */
/* to allow overloading and select the appropriate typemap when a Python object is provided */
%typemap(typecheck) (const app::AppFactory&) = PyObject *;
/* %typemap(typecheck) (const app::AppFactory&) = PyDict *; */

%typemap(typecheck) (const app:AppFactory& app) {
  $1 = PyDict_Check($input) ? 1 : 0;
 }

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
//%include "swig/pktools.i"
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

/* %typemap(in) jiplib::Jim& { */
/*   std::cout << "we are in typemap AppFactory" << std::endl; */
/*   void *argp2 = 0 ; */
/*   int res2 = 0 ; */
/*   res2 = SWIG_ConvertPtr($input, &argp2, SWIGTYPE_p_std__shared_ptrT_jiplib__Jim_,  0  | 0); */
/*   if (!SWIG_IsOK(res2)) { */
/*     SWIG_exception_fail(SWIG_ArgError(res2), "in method '" "Derived_printKeyValues" "', argument " "2"" of type '" "jiplib::Jim&""'"); */
/*   } */
/*   if (!argp2) { */
/*     SWIG_exception_fail(SWIG_ValueError, "invalid null reference , argument " "2"" of type '" "jiplib::Jim&""'"); */
/*   } */
/*   $1 = reinterpret_cast< jiplib::Jim* >(argp2); */

/*   $1=*() */
/*     } */
