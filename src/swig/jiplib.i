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

//Forgetting to place these commands will show itself as an ugly segmentation fault (crash) as soon as any C-API subroutine is actually called
//see also: https://docs.scipy.org/doc/numpy-1.10.0/user/c-info.how-to-extend.html
%init %{
  import_array();
  %}

//typemaps for jiplib::Jim
%{
#include <memory>
  %}

namespace jiplib{
#    NPY_INT8, NPY_INT16, NPY_INT32, NPY_INT64, NPY_UINT8, NPY_UINT16, NPY_UINT32, NPY_UINT64, NPY_FLOAT32, NPY_FLOAT64, NPY_COMPLEX64, NPY_COMPLEX128.

  %extend Jim {
    static std::shared_ptr<jiplib::Jim> np2jim(PyObject* npArray) {
      if(PyArray_Check(npArray)){
        PyArrayObject *obj=(PyArrayObject *)npArray;
        GDALDataType jDataType;
        int typ=PyArray_TYPE((PyArrayObject *)npArray);

        switch(PyArray_TYPE((PyArrayObject*)npArray)){
        case NPY_UINT8:
          jDataType=GDT_Byte;
          break;
        case NPY_UINT16:
          jDataType=GDT_UInt16;
          break;
        case NPY_INT16:
          jDataType=GDT_Int16;
          break;
        case NPY_UINT32:
          jDataType=GDT_UInt32;
          break;
        case NPY_INT32:
          jDataType=GDT_Int16;
          break;
        case NPY_FLOAT32:
          jDataType=GDT_Float32;
          break;
        case NPY_FLOAT64:
          jDataType=GDT_Float64;
          break;
          // case NPY_UINT64:
          //   jDataType=;
          // break;
          // case NPY_INT64:
          //   jDataType=;
          // break;
        default:
          std::string errorString="Error: Unknown data type";
          throw(errorString);
        }
        int dim=(PyArray_NDIM((PyArrayObject*)npArray))? 3 : 2;
        int nplane=(PyArray_NDIM((PyArrayObject*)npArray)==3) ? PyArray_DIM((PyArrayObject*)npArray,2): 1;//todo: check if nth dim starts from 0
        int nrow=PyArray_DIM((PyArrayObject*)npArray,0);
        int ncol=PyArray_DIM((PyArrayObject*)npArray,1);
        int nband=1;//only single band supported for now
        std::shared_ptr<jiplib::Jim> imgWriter=std::make_shared<jiplib::Jim>(((PyArrayObject*)npArray)->data,ncol,nrow,nplane,jDataType);
        return(imgWriter);
      }
      else{
        std::cerr << "Error: expected a numpy array as input" << std::endl;
        return(0);
      }
    }

    PyObject* jim2np(int band=0) {
      if(band>=$self->nrOfBand()){
        std::string errorString="Error: band out of range";
        throw(errorString);
      }
      int npDataType;
      switch ($self->getDataType()){
      case GDT_Byte:
        npDataType=NPY_UINT8;
        break;
      case GDT_UInt16:
        npDataType=NPY_UINT16;
        break;
      case GDT_Int16:
        npDataType=NPY_INT16;
        break;
      case GDT_UInt32:
        npDataType=NPY_UINT32;
        break;
      case GDT_Int32:
        npDataType=NPY_INT32;
        break;
      case GDT_Float32:
        npDataType=NPY_FLOAT32;
        break;
      case GDT_Float64:
        npDataType=NPY_FLOAT64;
        break;
        // case JDT_UInt64:
        //   npDataType=NPY_UINT64;
        // break;
        // case JDT_Int64:
        //   npDataType=NPY_INT64;
        // break;
      case GDT_Unknown:
      default:
        std::string errorString="Error: Unknown data type";
        throw(errorString);
      }
      int dim=($self->nrOfPlane()>1)? 3 : 2;
      if($self->nrOfPlane()>1){
        npy_intp dims[3];
        dims[0]=$self->nrOfRow();
        dims[1]=$self->nrOfCol();
        dims[2]=$self->nrOfPlane();
        PyArrayObject *npArray=(PyArrayObject*)PyArray_SimpleNewFromData(dim,dims,npDataType,(void*)$self->getDataPointer(band));
        if(npArray)
          return(PyArray_Return(npArray));
        else
          return(0);
      }
      else{
        npy_intp dims[2];
        dims[0]=$self->nrOfRow();
        dims[1]=$self->nrOfCol();
        PyArrayObject *npArray=(PyArrayObject*)PyArray_SimpleNewFromData(dim,dims,npDataType,(void*)$self->getDataPointer(band));
        if(npArray)
          return(PyArray_Return(npArray));
        else
          return(0);
      }
    }
  }
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
  //convert std::string to Python string or PyList of strings if multi-line string
  /* %typemap(out) std::string { */
    /* PySys_WriteStdout($1.c_str()); */
    /* Py_RETURN_NONE; */

    /* std::cout << "we are in typemap(out) std::string for jiplib::Jim::$symname" << std::endl; */
    /* std::string::size_type prevpos = 0; // Must initialize */
    /* std::string::size_type pos = 0; // Must initialize */
    /* std::string theString=$1; */
    /* $result = PyList_New(0); */
    /* while ( ( pos = theString.find ("\n",pos) ) != std::string::npos ){ */
    /*   std::string astring=theString.substr(prevpos,pos-prevpos); */
    /*   prevpos=pos; */
    /*   theString.erase ( pos, 1 ); */
    /*   //test */
    /*   std::cout << aString << std::endl; */
    /*   PyList_Append($result,PyString_FromString(astring.c_str())); */
    /* } */
    /* if(PyList_Size($result)<2) */
    /*   $result=PyString_FromString($1.c_str()); */
  /* } */
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

