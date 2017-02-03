//SWIG interface for jiplib
%include <std_string.i>
%include <std_vector.i>
%include <std_list.i>
%include <std_iostream.i>
%include <std_shared_ptr.i>
%shared_ptr(ImgRaster)
%shared_ptr(jiplib::Jim)

 //catch all exceptions thrown in C++
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

%define DOCJIPLIB
"Joint image processing library (jiplib)
developed in the framework of the JEODPP of the EO&SS@BD pilot project."
%enddef


%feature("autodoc", "2");

%import "jiplib_doc.i"

%module(docstring=DOCJIPLIB) jiplib

 /* %template(Float64Vector) std::vector<double>; */

/* %apply int &INOUT{ int &nbin }; */
/* %apply double &INOUT{ double &min }; */
/* %apply double &INOUT{ double &max }; */
/* %apply Float64Vector &INOUT{ std::vector<double>& }; */

/* to resolve naming conflicts with mialib library*/
%rename(filter2d_erode) filter2d::erode;
%rename(filter2d_dilate) filter2d::dilate;
%rename(filter2d_shift) filter2d::shift;

/* %typemap(in) OGRLayerH { */
/*              std::cout << "we are in typemap OGRLayer" << std::endl; */
/*              $1=$self->swigCPtr; */
/*              } */
/* %typemap(in) ImgRaster::rasterizeBuf(const std::string& ogrFilename){ */
/*   std::cout << "we are in typemap ImgReaderOgr&" << std::endl; */
/*   if(PyString_Check($input)){ */
/*     ogrReader.open(PyString_AsString($input)); */
/*     $1=&ogrReader; */
/*   } else { */
/*    SWIG_exception(SWIG_TypeError, "Python string expected"); */
/*   } */
/*  } */

/* !!! from: http://svn.salilab.org/imp/branches/1.0/kernel/pyext/IMP_streams.i */
/* to allow overloading and select the appropriate typemap when a Python object is provided */
%typemap(typecheck) (ImgReaderOgr&) = PyObject *;
/* %typemap(typecheck) (app::AppFactory&) = PyDict *; */

%typemap(typecheck) (ImgReaderOgr& ogrReader, double burnValue=1.0, const std::vector<std::string>& layernames=std::vector<std::string>()) {
  $1 = PyDict_Check($input) ? 1 : 0;
 }

%typemap(typecheck) (ImgReaderOgr& ogrReader, const std::vector<std::string>& controlOptions, const std::vector<std::string>& layernames=std::vector<std::string>()) {
  $1 = PyDict_Check($input) ? 1 : 0;
 }
%typemap(in) app::AppFactory& (app::AppFactory tempFactory){
  std::cout << "we are in typemap AppFactory" << std::endl;
  if(PyDict_Check($input)){
    PyObject *pKey, *pValue;
    Py_ssize_t ppos=0;
    /* $1=new app::AppFactory(); */
    $1=&tempFactory;
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
          $1->pushLongOption(theKey,theValue);
        }
        continue;
      }
      else if(PyString_Check(pValue)){
        theValue=PyString_AsString(pValue);
        $1->pushLongOption(theKey,theValue);
      }
      else if(PyBool_Check(pValue)){
        if(pValue==Py_True){
          if(theKey=="help")
            $1->pushLongOption("dict");
          else
            $1->pushLongOption(theKey);
        }
      }
      else{
        theValue=PyString_AsString(PyObject_Repr(pValue));
        $1->pushLongOption(theKey,theValue);
      }
    }
    $1->showOptions();
  } else {
    SWIG_exception(SWIG_TypeError, "Python dictionary expected");
  }
 }

/* !!! from: http://svn.salilab.org/imp/branches/1.0/kernel/pyext/IMP_streams.i */
/* to allow overloading and select the appropriate typemap when a Python object is provided */
%typemap(typecheck) (app::AppFactory&) = PyObject *;
/* %typemap(typecheck) (app::AppFactory&) = PyDict *; */

/* we provide SWIG with a typecheck to determine which of the C++ functions to use. */
/* if Python function argument $input is a PyDict, then use the C++ function with the AppFactory */
/* Note:  %typecheck(X) is a macro for %typemap(typecheck,precedence=X) so we might want to include a second argument precedence=X after app?*/
%typemap(typecheck) (const app::AppFactory& app) {
  $1 = PyDict_Check($input) ? 1 : 0;
 }

%typemap(typecheck) (app::AppFactory& app) {
  $1 = PyDict_Check($input) ? 1 : 0;
 }

/* %apply std::vector<double> & OUTPUT { std::vector<double>& dVector }; */
/* %typemap(argout) const std::vector<double>& dVector ""; */
/* %apply const std::vector<double> & { const std::vector<double>& dVector }; */

//$input is the input Python object
//$1 is the (c/c++) function call argument
/* %typemap(argout) std::vector<double>& { */
/*   std::cout << "we are in typemap(argout) std::vector<double>& for $symname" << std::endl; */
/*   PyObject * o = 0 ; */
/*   for(int i=0;i<$1->size();++i){ */
/*     o=PyFloat_FromDouble($1->at(i)); */
/*     PyList_SetItem($result,i,o); */
/*   } */
/*  } */

//typemaps for jiplib::Jim
%{
#include <memory>
 %}
namespace jiplib{
  //return the object itself for all functions returning CPLErr
  %typemap(out) CPLErr {
    std::cout << "we are in typemap(out) CPLErr for jiplib::Jim::$symname" << std::endl;
    if($1==CE_Failure)
      std::cout << "Warning: CE_Failure" << std::endl;
    void *argp2;
    int res2=0;
    res2 = SWIG_ConvertPtr($self, &argp2, SWIGTYPE_p_std__shared_ptrT_jiplib__Jim_t,  0  | 0);
    if (!SWIG_IsOK(res2)) {
      SWIG_exception_fail(SWIG_ArgError(res2), "in method " "$symname");
    }
    if (!argp2) {
      SWIG_exception_fail(SWIG_ValueError, "invalid null reference , argument " "2"" of type '" "shared_ptr<const jiplib::Jim&>""'");
    }
    std::shared_ptr<jiplib::Jim> result=(*(reinterpret_cast< std::shared_ptr< jiplib::Jim > * >(argp2)))->getShared();
    PyObject* o=0;
    std::shared_ptr<  jiplib::Jim > *smartresult = result ? new std::shared_ptr<  jiplib::Jim >(result) : 0;
    o = SWIG_NewPointerObj(SWIG_as_voidptr(smartresult), SWIGTYPE_p_std__shared_ptrT_jiplib__Jim_t, SWIG_POINTER_OWN | 0);
    if(o)
      $result=o;
    else
      SWIG_exception_fail(SWIG_ArgError(res2), "in method " "$symname");
    /* $result=$self; */
  }
}

%{
#include <memory>
#include "config.h"
#include "imageclasses/ImgRaster.h"
#include "imageclasses/ImgReaderOgr.h"
#include "imageclasses/ImgCollection.h"
#include "apps/AppFactory.h"
#include "algorithms/Filter2d.h"
#include "jim.h"
#include "jimlist.h"
#include "mialib_swig.h"
#include <cpl_error.h>
  %}

%template(ImgVectorJim) std::vector< std::shared_ptr< jiplib::Jim > >;
%template(ImgListJim) std::list< std::shared_ptr< jiplib::Jim > >;

//Parse the header file
//%include "swig/pktools.i"
%include "swig/mialib_tmp.i"
%include "swig/jiplib_python.i"
%include "imageclasses/ImgCollection.h"
%include "imageclasses/ImgRaster.h"
%include "imageclasses/ImgReaderOgr.h"
%include "apps/AppFactory.h"
%include "algorithms/Filter2d.h"
%include "jim.h"
%include "jimlist.h"



enum CPLErr {CE_None = 0, CE_Debug = 1, CE_Warning = 2, CE_Failure = 3, CE_Fatal = 4};
enum GDALDataType {GDT_Unknown = 0, GDT_Byte = 1, GDT_UInt16 = 2, GDT_Int16 = 3, GDT_UInt32 = 4, GDT_Int32 = 5, GDT_Float32 = 6, GDT_Float64 = 7, GDT_CInt16 = 8, GDT_CInt32 = 9, GDT_CFloat32 = 10, GDT_CFloat64 = 11, GDT_TypeCount = 12};
