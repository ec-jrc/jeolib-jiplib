/**********************************************************************
Interface file for SWIG
Author(s): Pieter.Kempeneers@ec.europa.eu, Pierre.Soille@ec.europa.eu
Copyright (C) 2016-2020 European Union (Joint Research Centre)

This file is part of jiplib.

jiplib is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

jiplib is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with jiplib.  If not, see <https://www.gnu.org/licenses/>.
***********************************************************************/
%include <std_string.i>
%include <std_vector.i>
%include <std_map.i>
%include <std_list.i>
%include <std_iostream.i>
%include <std_shared_ptr.i>
%shared_ptr(Jim)
/* %shared_ptr(jiplib::Jim) */
%shared_ptr(VectorOgr)

/* http://www.swig.org/Doc1.3/Arguments.html */
%include "typemaps.i"

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
"HELP.MODULE"
%enddef

%module(docstring=DOCJIPLIB) jiplib
#pragma SWIG nowarn=303,362,401,467,503,509,

/* %define DOCJIPLIB */
/* "Joint image processing library (jiplib) */
/* developed in the framework of the JEODPP of the EO&SS@BD pilot project." */
/* %enddef */


/* %feature("autodoc", "2"); */

//uncomment if documentation is needed (make sure to execute make doc before make!!!)
/* %pythoncode %{ */
/* %import "jiplib_doc.i" */
     /* %} */

/* %module(docstring=DOCJIPLIB) jiplib */

 /* %template(Float64Vector) std::vector<double>; */

 /* %apply int &INOUT{ int &nbin }; */
 /* %apply double &INOUT{ double &min }; */
 /* %apply double &INOUT{ double &max }; */
 /* %apply Float64Vector &INOUT{ std::vector<double>& }; */

 /* to resolve naming conflicts with miallib library: rename(to) from*/
%rename(filter2d_erode) filter2d::erode;
%rename(filter2d_dilate) filter2d::dilate;
%rename(filter2d_shift) filter2d::shift;

%typemap(in) app::AppFactory& (app::AppFactory tempFactory){
  if(PyDict_Check($input)){
    PyObject *pKey, *pValue;
    Py_ssize_t ppos=0;
    /* $1=new app::AppFactory(); */
    $1=&tempFactory;
    while (PyDict_Next($input, &ppos, &pKey, &pValue)) {
      std::string theKey;
      std::string theValue;
      if(PyString_Check(pKey)){
        theKey=PyString_AsString(pKey);
      }
      else if(PyUnicode_Check(pKey)){
        theKey=PyString_AsString(PyUnicode_AsUTF8String(pKey));
      }
      else{
        PyErr_SetString(PyExc_TypeError,"Expected a string.");
        std::string errorString="Error: Expected a string.";
        throw(errorString);
        return NULL;
      }
      if(PyList_Check(pValue)){
        for(Py_ssize_t i=0;i<PyList_Size(pValue);++i){
          PyObject *rValue;
          rValue=PyList_GetItem(pValue,i);
          if (PyString_Check(rValue))
            theValue=PyString_AsString(rValue);
          else if (PyUnicode_Check(rValue))
            theValue = PyString_AsString(PyUnicode_AsUTF8String(rValue));
          else if(rValue != Py_None){
            PyObject *tmp_obj = PyObject_Repr(rValue);
            if (PyString_Check(tmp_obj))
              theValue=PyString_AsString(tmp_obj);
            else if (PyUnicode_Check(tmp_obj))
              theValue=PyString_AsString(PyUnicode_AsUTF8String(tmp_obj));
          }
          else{
            PyErr_SetString(PyExc_TypeError,"Expected a string.");
            return NULL;
          }
          $1->pushLongOption(theKey,theValue);
        }
        continue;
      }
      else if(PyString_Check(pValue)){
        theValue=PyString_AsString(pValue);
        $1->pushLongOption(theKey,theValue);
      }
      else if (PyUnicode_Check(pValue)){
        theValue = PyString_AsString(PyUnicode_AsUTF8String(pValue));
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
      else if(pValue != Py_None){
        PyObject *tmp_obj = PyObject_Repr(pValue);
        if (PyString_Check(tmp_obj))
          theValue=PyString_AsString(tmp_obj);
        else if (PyUnicode_Check(tmp_obj))
          theValue=PyString_AsString(PyUnicode_AsUTF8String(tmp_obj));
        $1->pushLongOption(theKey,theValue);
      }
    }
  }
  else{
    PyObject* kwargs;
    char *theString;
    char *theOptInt;
    strcpy(theString, "theString");
    strcpy(theOptInt, "theOptInt");
    static char *kwlist[] = {
      theString,
      theOptInt,
      NULL
    };
    PyArg_ParseTupleAndKeywords($input,kwargs,"S|i",kwlist);
    /* SWIG_exception(SWIG_TypeError, "Python dictionary expected"); */
  }
  if(!($1))
    SWIG_exception(SWIG_TypeError, "Error: could not parse argument for " "$symname");
  //test
  /* $1->showOptions(); */
 }

/* %ignore createJim(); */
/* %ignore createJim(const std::shared_ptr<Jim>, bool); */
/* %ignore createJim(const std::string&, bool); */
%pythoncode %{

  t_UCHAR    =  3
  t_SHORT    =  4
  t_USHORT   =  5
  t_INT32    =  6
  t_UINT32   =  7
  t_INT64    =  8
  t_UINT64   =  9
  t_FLOAT    = 10
  _MIAFLOAT = 10
  t_DOUBLE   = 11
  GDT_Byte = 1
  GDT_UInt16 = 2
  GDT_Int16 = 3
  GDT_UInt32 = 4
  GDT_Int32 = 5
  GDT_Float32 = 6
  GDT_Float64 = 7
  JDT_UInt64 = 24
  JDT_Int64 = 25

  import os.path
  def createJim(arg1=None,**kwargs):
      appDict={}
      if arg1:
          if isinstance(arg1,Jim):
              if 'copyData' in kwargs.keys():
                  return Jim_createImg(arg1,kwargs['copyData'])
              else:
                  return Jim_createImg(arg1)
          elif isinstance(arg1,str):
              if os.path.isfile(arg1):
                  appDict.update({'filename':arg1})
              else:
                raise IOError("Error: {} is not a regular file".format(arg1))
      for key, value in kwargs.items():
          appDict.update({key:value})
      if appDict:
          # SWIG generates wrappers that try to work around calling static member functions, replaceing :: with _ (underscore)
          return Jim_createImg(appDict)
      else:
          return Jim_createImg()

  def createVector(arg1=None,**kwargs):
      appDict={}
      for key, value in kwargs.items():
          appDict.update({key:value})
      if arg1:
          if isinstance(arg1,VectorOgr):
              if appDict:
                  return VectorOgr_createVector(arg1,appDict)
              else:
                  return VectorOgr_createVector(arg1)
          elif isinstance(arg1,str):
              if os.path.isfile(arg1):
                  appDict.update({'filename':arg1})
              else:
                  print("Warning: {} is not a regular file".format(arg1))
                  appDict.update({'filename':arg1})
          else:
              raise TypeError("Error: bad argument type for createVector, arguments without names should be a path or of VectorOgr type")
      if appDict:
          # SWIG generates wrappers that try to work around calling static member functions, replaceing :: with _ (underscore)
          return VectorOgr_createVector(appDict)
      else:
          return VectorOgr_createVector()
    %}

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

/* %typemap(in) const Vector2d<double> &taps */
/* { */
/*   std::cout << "we are in typemap(in) std::vector<double>& for $symname with taps" << std::endl; */
/*   if ($input != NULL && PyArray_Check($input)) */
/*     { */
/*       /\* $1=&tmpTaps; *\/ */
/*       int nrow=PyArray_DIM((PyArrayObject*)($input),0); */
/*       int ncol=PyArray_DIM((PyArrayObject*)($input),1); */
/*       $1->resize(nrow,ncol); */
/*       for(size_t irow=0;irow<nrow;++irow){ */
/*         for(size_t icol=0;icol<ncol;++icol){ */
/*           //todo: check if row and col should be reversed? */
/*           ($1->at(irow)).at(icol)=*((double*)(PyArray_GETPTR2($input, irow, icol))); */
/*         } */
/*       } */
/*     } */
/*   else */
/*     { */
/*       PyErr_SetString(PyExc_TypeError, "not a numpy array"); */
/*       SWIG_fail; */
/*     } */
/* } */

/* https://stackoverflow.com/questions/15185010/how-to-receive-reference-and-pointer-arguments-in-python-swig */
%typemap(in,numinputs=0) double& ulx (double temp) "$1 = &temp;"
%typemap(in,numinputs=0) double& uly (double temp) "$1 = &temp;"
%typemap(in,numinputs=0) double& lrx (double temp) "$1 = &temp;"
%typemap(in,numinputs=0) double& lry (double temp) "$1 = &temp;"

%typemap(argout) double& ulx {
  %append_output(PyFloat_FromDouble(*$1));
 }

%typemap(argout) double& uly {
  %append_output(PyFloat_FromDouble(*$1));
 }

%typemap(argout) double& lrx {
  %append_output(PyFloat_FromDouble(*$1));
 }

%typemap(argout) double& lry {
  %append_output(PyFloat_FromDouble(*$1));
}

%typemap(in,numinputs=0) double& gt0 (double temp) "$1 = &temp;"
%typemap(in,numinputs=0) double& gt1 (double temp) "$1 = &temp;"
%typemap(in,numinputs=0) double& gt2 (double temp) "$1 = &temp;"
%typemap(in,numinputs=0) double& gt3 (double temp) "$1 = &temp;"
%typemap(in,numinputs=0) double& gt4 (double temp) "$1 = &temp;"
%typemap(in,numinputs=0) double& gt5 (double temp) "$1 = &temp;"

%typemap(argout) double& gt0 {
  %append_output(PyFloat_FromDouble(*$1));
 }

%typemap(argout) double& gt1 {
  %append_output(PyFloat_FromDouble(*$1));
 }

%typemap(argout) double& gt2 {
  %append_output(PyFloat_FromDouble(*$1));
 }

%typemap(argout) double& gt3 {
  %append_output(PyFloat_FromDouble(*$1));
 }

%typemap(argout) double& gt4 {
  %append_output(PyFloat_FromDouble(*$1));
 }

%typemap(argout) double& gt5 {
  %append_output(PyFloat_FromDouble(*$1));
 }

%typemap(in,numinputs=0) double& centerX (double temp) "$1 = &temp;"
%typemap(in,numinputs=0) double& centerY (double temp) "$1 = &temp;"

%typemap(argout) double& centerX {
  %append_output(PyFloat_FromDouble(*$1));
 }

%typemap(argout) double& centerY {
  %append_output(PyFloat_FromDouble(*$1));
 }

%typemap(in,numinputs=0) double& refX (double temp) "$1 = &temp;"
%typemap(in,numinputs=0) double& refY (double temp) "$1 = &temp;"

%typemap(argout) double& refX {
  %append_output(PyFloat_FromDouble(*$1));
 }

%typemap(argout) double& refY {
  %append_output(PyFloat_FromDouble(*$1));
 }

/* Set the input argument to point to a temporary variable */
/* %typemap(in, numinputs=0) double *gt (double temp) { */
/*   $1 = &temp; */
/*  } */

 /*  %typemap(argout) double* { */
 /*  // Append output value $1 to $result */
 /*  int n=6; */
 /*  PyObject *pyList; */
 /*  pyList=PyList_New(n); */
 /*  for(int i=0; i<n; ++i){ */
 /*    PyList_SET_ITEM(pyList, i, PyFloat_FromDouble($1[i])); */
 /*  } */
 /*  $result = pyList; */
 /* } */

%typemap(in, numinputs=0) double *minval (double temp) {
  $1 = &temp;
 }

%typemap(argout) double* minval {
  %append_output(PyFloat_FromDouble(*$1));
 }

%typemap(in, numinputs=0) double *maxval (double temp) {
  $1 = &temp;
 }

%typemap(argout) double* maxval {
  %append_output(PyFloat_FromDouble(*$1));
 }

%typemap(argout) unsigned long int* ofs{
  %append_output(PyInt_FromLong(*$1));
 }

%typemap(in) double* gt(double temp[6]){
  int i;
  if (!PySequence_Check($input)) {
    PyErr_SetString(PyExc_ValueError,"Expected a sequence");
    return NULL;
  }
  if (PySequence_Length($input) != 6) {
    PyErr_SetString(PyExc_ValueError,"Size mismatch. Expected 6 elements");
    return NULL;
  }
  for (i = 0; i < 6; i++) {
    PyObject *o = PySequence_GetItem($input,i);
    if (PyNumber_Check(o)) {
      temp[i] = PyFloat_AsDouble(o);
    } else {
      PyErr_SetString(PyExc_ValueError,"Sequence elements must be numbers");
      return NULL;
    }
  }
  $1 = temp;
 }

/* From: http://biomol.bme.utexas.edu/~mh43854/openmm/archive/openmm-master/wrappers/python/src/swig_doxygen/swig_lib/python/typemaps.i */
/* The following two typemaps cause a non-const vector<string>& named fields to become a return list of values. */

%typemap(in, numinputs=0) std::vector<double>& noDataValues (std::vector<double> temp) {
  $1 = &temp;
 }

%typemap(argout) std::vector<double>& noDataValues{
  int i, n;
  PyObject *pyList;

  n=(*$1).size();
  pyList=PyList_New(n);
  for (i=0; i<n; i++) {
    double theValue=(*$1).at(i);
    PyList_SET_ITEM(pyList, i, PyFloat_FromDouble(theValue));
  }
  $result = pyList;
 }

%typemap(in, numinputs=0) std::vector<std::string>& fields (std::vector<std::string> temp) {
  $1 = &temp;
 }

%typemap(argout) std::vector<std::string>& fields{
  int i, n;
  PyObject *pyList;

  n=(*$1).size();
  pyList=PyList_New(n);
  for (i=0; i<n; i++) {
    std::string theField=(*$1).at(i);
    PyList_SET_ITEM(pyList, i, PyString_FromString(theField.c_str()));
  }
  $result = pyList;
}

%typemap(in, numinputs=0) std::vector<double>& bbvector (std::vector<double> temp) {
  $1 = &temp;
 }

%typemap(argout) std::vector<double>& bbvector{
  int i, n;
  PyObject *pyList;

  n=(*$1).size();
  pyList=PyList_New(n);
  for (i=0; i<n; i++) {
    double theValue=(*$1).at(i);
    PyList_SET_ITEM(pyList, i, PyFloat_FromDouble(theValue));
  }
  $result = pyList;
 }

/* %typemap(in, numinputs=0) std::vector<double>& field (std::vector<double> temp) { */
/*   $1 = &temp; */
/*  } */

%typemap(in, numinputs=0) (double& i, double& j) (double i, double j){
  $1 = &i;
  $2 = &j;
 }

%typemap(argout) (double& i, double& j){
  PyObject *pyList;

  pyList=PyList_New(2);
  double theValue1=*$1;
  double theValue2=*$2;
  PyList_SET_ITEM(pyList, 0, PyFloat_FromDouble(theValue1));
  PyList_SET_ITEM(pyList, 1, PyFloat_FromDouble(theValue2));
  $result = pyList;
 }

%typemap(in, numinputs=0) (double& x, double& y) (double x, double y){
  $1 = &x;
  $2 = &y;
 }

%typemap(argout) (double& x, double& y){
  PyObject *pyList;

  pyList=PyList_New(2);
  double theValue1=*$1;
  double theValue2=*$2;
  PyList_SET_ITEM(pyList, 0, PyFloat_FromDouble(theValue1));
  PyList_SET_ITEM(pyList, 1, PyFloat_FromDouble(theValue2));
  $result = pyList;
 }

/* %typemap(typecheck) (std::vector<double>&) = PyObject *; */
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
///SWIG applies typemaps in the order in which they are defined in the interface file. This means that typemaps only take effect for declarations that follow the typemap definition
///therefore: define typemaps before including the classes with %include!!! (https://github.com/swig/swig/wiki/FAQ)
%{
#include <memory>
  %}

 /* %rename jiplib::Jim::crop jiplib::Jim::geospatial::crop; */

%{
#include <memory>
#include <algorithm>
#include "imageclasses/Jim.h"
#include "imageclasses/JimList.h"
/* #include "../../build/src/imageclasses/Jim.h" */
/* #include "../../build/src/imageclasses/JimList.h" */
/* #include "../../build/src/imageclasses/VectorOgr.h" */
#include "apps/AppFactory.h"
#include "algorithms/Filter2d.h"
#if MIALLIB == 1
#include "miallib_swig.h"
#include "op.h"
#endif
#include <cpl_error.h>
  %}

/* namespace jiplib{ */
/*      NPY_INT8, NPY_INT16, NPY_INT32, NPY_INT64, NPY_UINT8, NPY_UINT16, NPY_UINT32, NPY_UINT64, NPY_FLOAT32, NPY_FLOAT64, NPY_COMPLEX64, NPY_COMPLEX128. */


  //return the object itself for all functions returning CPLErr
  %typemap(out) CPLErr {
    //std::cout << "we are in typemap(out) CPLErr for jiplib::Jim::$symname" << std::endl;
    if($1==CE_Failure)
      std::cout << "Warning: CE_Failure in method $symname" << std::endl;
    void *argp2;
    int res2=0;
    res2 = SWIG_ConvertPtr($self, &argp2, SWIGTYPE_p_std__shared_ptrT_Jim_t,  0  | 0);
    if (!SWIG_IsOK(res2)) {
      SWIG_exception_fail(SWIG_ArgError(res2), "in method " "$symname");
    }
    if (!argp2) {
      SWIG_exception_fail(SWIG_ValueError, "invalid null reference , argument " "2"" of type '" "shared_ptr<const Jim&>""'");
    }
    std::shared_ptr<Jim> result=(*(reinterpret_cast< std::shared_ptr< Jim > * >(argp2)))->getShared();
    PyObject* o=0;
    std::shared_ptr<  Jim > *smartresult = result ? new std::shared_ptr<  Jim >(result) : 0;
    o = SWIG_NewPointerObj(SWIG_as_voidptr(smartresult), SWIGTYPE_p_std__shared_ptrT_Jim_t, SWIG_POINTER_OWN | 0);
    if(o)
      $result=o;
    else
      SWIG_exception_fail(SWIG_ArgError(res2), "in method " "$symname");
  }
  //return the object itself for all functions returning OGRErr
  %typemap(out) OGRErr {
    //std::cout << "we are in typemap(out) OGRErr for jiplib::VectorOgr::$symname" << std::endl;
    if($1==OGRERR_FAILURE)
      std::cout << "Warning: OGRERR_FAILURE" << std::endl;
    void *argp2;
    int res2=0;
    res2 = SWIG_ConvertPtr($self, &argp2, SWIGTYPE_p_std__shared_ptrT_VectorOgr_t,  0  | 0);
    if (!SWIG_IsOK(res2)) {
      SWIG_exception_fail(SWIG_ArgError(res2), "in method " "$symname");
    }
    if (!argp2) {
      SWIG_exception_fail(SWIG_ValueError, "invalid null reference , argument " "2"" of type '" "shared_ptr<const VectorOgr&>""'");
    }
    std::shared_ptr<VectorOgr> result=(*(reinterpret_cast< std::shared_ptr< VectorOgr > * >(argp2)))->getShared();
    PyObject* o=0;
    std::shared_ptr<  VectorOgr > *smartresult = result ? new std::shared_ptr<  VectorOgr >(result) : 0;
    o = SWIG_NewPointerObj(SWIG_as_voidptr(smartresult), SWIGTYPE_p_std__shared_ptrT_VectorOgr_t, SWIG_POINTER_OWN | 0);
    if(o)
      $result=o;
    else
      SWIG_exception_fail(SWIG_ArgError(res2), "in method " "$symname");
  }

  //convert multimap to PyDict
  %typemap(out) std::multimap<std::string,std::string> getStats {
    PyObject *d = PyDict_New();
    std::multimap<std::string,std::string>::const_iterator mit=$1.begin();
    while(mit!=$1.end()){
      std::string key=mit->first;
      std::string val=mit->second;
      if(key.find("histogram")!=std::string::npos){
        //for creating PyArrayObject from c array, check https://codereview.stackexchange.com/questions/92266/sending-a-c-array-to-python-numpy-and-back/92353#92353
        PyObject *lb = PyList_New(0);
        PyObject *lh = PyList_New(0);
        //construct dictionary as list of values
        std::istringstream buf(val);
        std::istream_iterator<std::string> beg(buf), end;
        std::vector<std::string> tokens(beg, end); // done!
        for(int ibin=0;ibin<tokens.size()-1;ibin=ibin+2){
          double binValue=std::stod(tokens[ibin].c_str());
          std::string histString=tokens[ibin+1];
          histString.erase(std::remove(histString.begin(), histString.end(), '\n'), histString.end());
          double histValue=std::stod(histString.c_str());
          PyList_Append(lb,PyFloat_FromDouble(binValue));
          PyList_Append(lh,PyFloat_FromDouble(histValue));
        }
        std::string keyBin="bin";
        PyDict_SetItem(d, PyString_FromString(keyBin.c_str()), lb);
        PyDict_SetItem(d, PyString_FromString(key.c_str()), lh);
      }
      else if(val.find("[")==std::string::npos)
          PyDict_SetItem(d, PyString_FromString(key.c_str()), PyFloat_FromDouble(std::stod(val.c_str())));
      else{
        PyObject *lb = PyList_New(0);
        val.erase(std::remove(val.begin(), val.end(), '['), val.end());
        val.erase(std::remove(val.begin(), val.end(), ']'), val.end());

        std::stringstream ss(val);

        for (double dv; ss >> dv;) {

          PyList_Append(lb,PyFloat_FromDouble(dv));
          if (ss.peek() == ',')
            ss.ignore();
        }
        PyDict_SetItem(d, PyString_FromString(key.c_str()), lb);
      }
      ++mit;
    }
    $result=d;
  }
/* } */

/* %typemap(out) std::string getUniquePixels { */
/*   $result=PyString_FromString($1.c_str()); */
/*  } */
/* %typemap(out) std::map<std::vector<unsigned short>,unsigned short>& getUniquePixels { */
/*     $result=PyString_FromString("hello"); */
/* } */
/* %typemap(out) std::map<unsigned short,unsigned short> getUniquePixels { */
/*     $result=PyString_FromString("hello"); */
/* } */
/* %typemap(out) std::vector<unsigned short> getUniquePixels { */
/*   $result=PyString_FromString("hello"); */
/* } */
//convert map to PyDict
/* %typemap(out) std::map<std::vector<unsigned short>,std::vector<std::pair<unsigned short,unsigned short> > > getUniquePixels { */
/*   PyObject *d = PyDict_New(); */
/*   std::map<std::vector<unsigned short>,std::vector<std::pair<unsigned short,unsigned short> > >::const_iterator mit=$1.begin(); */
/*   while(mit!=$1.end()){ */
/*     std::vector<unsigned short> pv=mit->first; */
/*     PyObject *pyv = PyList_New(mit->first.size()); */
/*     for(int index=0;index<mit->first.size();++index) */
/*       PyList_SetItem(pyv, index,PyInt_FromLong(mit->first[index])); */
/*     PyObject *coordinates = PyList_New(mit->second.size()); */
/*     //construct dictionary as list of values */
/*     for(int icoord=0;icoord<mit->second.size();++icoord){ */
/*       PyObject *lc = PyList_New(2); */
/*       unsigned int xcoord=mit->second[icoord].first; */
/*       unsigned int ycoord=mit->second[icoord].second; */
/*       PyList_SetItem(lc,0,PyInt_FromLong(xcoord)); */
/*       PyList_SetItem(lc,1,PyInt_FromLong(ycoord)); */
/*       PyList_SetItem(coordinates,icoord,lc); */
/*     } */
/*     PyDict_SetItem(d, pyv, coordinates); */
/*     ++mit; */
/*   } */
/*   $result=d; */
/* } */



%template(ImgVectorJim) std::vector< std::shared_ptr< Jim > >;
%template(JimListJim) std::list< std::shared_ptr< Jim > >;
%template(VectorDouble) std::vector<double>;
%template(VectorString) std::vector<std::string>;
/* %template(ImgVectorJim) std::vector< std::shared_ptr< jiplib::Jim > >; */
/* %template(JimListJim) std::list< std::shared_ptr< jiplib::Jim > >; */
/* %template(VectorDouble) std::vector<double>; */

//Parse the header file
%include "swig/miallib_tmp.i"

%include "op.h"
/* %include "swig/jiplib_python.i" */

%{
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <limits>
#include "Python.h"
#include "numpy/arrayobject.h"
#include "config_jiplib.h"
%}

%typemap(in,numinputs=1) (PyArrayObject *psArray)
{
  if ($input != NULL && PyArray_Check($input))
    {
      $1 = (PyArrayObject*)($input);
    }
  else
    {
      PyErr_SetString(PyExc_TypeError, "not a numpy array");
      SWIG_fail;
    }
}

   %inline %{
     std::shared_ptr<Jim> np2jim(PyObject* npArray) {
       app::AppFactory afactory;
    /* jim.np2jim(npArray); */
    if(PyArray_Check(npArray)){
      PyArrayObject *obj=(PyArrayObject *)npArray;
      int typ=PyArray_TYPE((PyArrayObject *)npArray);

      switch(PyArray_TYPE((PyArrayObject*)npArray)){
      case NPY_UINT8:
        afactory.pushLongOption("otype","GDT_Byte");
        break;
      case NPY_UINT16:
        afactory.pushLongOption("otype","GDT_UInt16");
        break;
      case NPY_INT16:
        afactory.pushLongOption("otype","GDT_Int16");
        break;
      case NPY_UINT32:
        afactory.pushLongOption("otype","GDT_UInt32");
        break;
      case NPY_INT32:
        afactory.pushLongOption("otype","GDT_Int32");
        break;
      case NPY_FLOAT32:
        afactory.pushLongOption("otype","GDT_Float32");
        break;
      case NPY_FLOAT64:
        afactory.pushLongOption("otype","GDT_Float64");
        break;
        case NPY_UINT64:
          afactory.pushLongOption("otype","JDT_UInt64");
        break;
        case NPY_INT64:
          afactory.pushLongOption("otype","JDT_Int64");
        break;
      default:
        std::string errorString="Error: Unknown data type";
        throw(errorString);
      }
      int ndim=(PyArray_NDIM((PyArrayObject*)npArray)==3)? 3 : 2;
      int nplane=(PyArray_NDIM((PyArrayObject*)npArray)==3) ? PyArray_DIM((PyArrayObject*)npArray,0): 1;
      int nrow=(PyArray_NDIM((PyArrayObject*)npArray)==3) ? PyArray_DIM((PyArrayObject*)npArray,1): PyArray_DIM((PyArrayObject*)npArray,0);
      int ncol=(PyArray_NDIM((PyArrayObject*)npArray)==3) ? PyArray_DIM((PyArrayObject*)npArray,2): PyArray_DIM((PyArrayObject*)npArray,1);
      /* int nrow=PyArray_DIM((PyArrayObject*)npArray,1); */
      /* int ncol=PyArray_DIM((PyArrayObject*)npArray,2); */
      int nband=1;//only single band supported for now
      afactory.pushLongOption("ncol",ncol);
      afactory.pushLongOption("nrow",nrow);
      afactory.pushLongOption("nplane",nplane);
      afactory.pushLongOption("nband",nband);
      std::shared_ptr<Jim> jim=Jim::createImg();
      jim->open(afactory);
      memcpy(jim->getDataPointer(),(void*)(PyArray_DATA((PyArrayObject*)npArray)),jim->getDataTypeSizeBytes()*jim->nrOfCol()*jim->nrOfRow()*jim->nrOfPlane());
      return(jim);
      /* return(jim->getShared()); */
    }
    else{
      std::cerr << "Error: expected a numpy array as input, returning empty Jim image" << std::endl;
      /* return create; */
      return 0;
    }
  }

  //todo: return multi-plane/band images according to: http://scikit-image.org/docs/dev/user_guide/numpy_images.html
     PyObject* jim2np(std::shared_ptr<Jim> aJim, size_t band=0, bool copyData=true) {
    if(band>=aJim->nrOfBand()){
      std::string errorString="Error: band out of range";
      throw(errorString);
    }
    int npDataType;
    switch (aJim->getDataType()){
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
    case JDT_UInt64:
      npDataType=NPY_UINT64;
      break;
    case JDT_Int64:
      npDataType=NPY_INT64;
      break;
    case GDT_Unknown:
    default:
      std::string errorString="Error: Unknown data type";
      throw(errorString);
    }
    void *npdata=0;
    if(copyData){
      //alllocate memory for npdata
      npdata=(void *) calloc(static_cast<size_t>(aJim->nrOfPlane()*aJim->nrOfCol()*aJim->nrOfRow()),aJim->getDataTypeSizeBytes());
      //copy data from jim to npdata
      aJim->copyData(npdata,band);
    }
    else
      npdata=(void*)(aJim->getDataPointer(band));
    int ndim=(aJim->nrOfPlane()>1)? 3 : 2;
    if(aJim->nrOfPlane()>1){
      npy_intp dims[3];
      dims[0]=aJim->nrOfPlane();
      dims[1]=aJim->nrOfRow();
      dims[2]=aJim->nrOfCol();
      PyArrayObject *npArray=(PyArrayObject*)PyArray_SimpleNewFromData(ndim,dims,npDataType,npdata);
      if(npArray)
        return(PyArray_Return(npArray));
      else
        return(0);
    }
    else{
      npy_intp dims[2];
      dims[0]=aJim->nrOfRow();
      dims[1]=aJim->nrOfCol();
      PyArrayObject *npArray=(PyArrayObject*)PyArray_SimpleNewFromData(ndim,dims,npDataType,npdata);
      if(npArray)
        return(PyArray_Return(npArray));
      else
        return(0);
    }
  }

  PyObject* jimvect2np(std::shared_ptr<VectorOgr> aJimVect, std::vector<std::string> fname, size_t ilayer) {
    size_t ntotalfeatures=0;
    size_t nfeatures=0;
    size_t nfields=0;
    int npDataType=NPY_FLOAT64;
    if(aJimVect->isEmpty())
      return(0);
    if(ilayer>=aJimVect->getLayerCount())
      return(0);
    //todo: destroy empty features?

    ntotalfeatures=aJimVect->getFeatureCount(ilayer);
    std::vector<OGRFieldDefn*> fields;
    std::vector<size_t> fieldindexes;
    aJimVect->getFields(fields,ilayer);
    for(unsigned int ifield=0;ifield<fields.size();++ifield){
      if(fname.size()){
        if(std::find(fname.begin(),fname.end(),fields[ifield]->GetNameRef())!=fname.end())
          fieldindexes.push_back(ifield);
      }
      else
        fieldindexes.push_back(ifield);
    }
    nfields=fieldindexes.size();
    int ndim=2;
    npy_intp dims[2]{static_cast<int>(ntotalfeatures),static_cast<int>(nfields)};
    PyArrayObject *npArray =(PyArrayObject*) PyArray_SimpleNew(ndim, dims, npDataType);
    double *ad = (double *) PyArray_DATA((PyArrayObject *) npArray);
    OGRFeatureDefn *poFDefn = aJimVect->getLayer(ilayer)->GetLayerDefn();
    for(size_t ifeature = 0; ifeature < aJimVect->getFeatureCount(ilayer); ++ifeature) {
      OGRFeature *thisFeature=aJimVect->getFeatureRef(ifeature,ilayer);
      if(!thisFeature){
        // std::cerr << "Warning: " << ifeature << " is NULL" << std::endl;
        continue;
      }
      for(std::vector<size_t>::const_iterator fit=fieldindexes.begin();fit!=fieldindexes.end();++fit)
        *ad++=thisFeature->GetFieldAsDouble(*fit);
    }
    if(npArray){
      return(PyArray_Return(npArray));
    }
    else{
      return(0);
    }
  }

  PyObject* dict(std::shared_ptr<VectorOgr> aJimVect, std::vector<std::string> fname, size_t ilayer) {
    size_t ntotalfeatures=0;
    size_t nfeatures=0;
    size_t nfields=0;
    int npDataType=NPY_FLOAT64;
    if(aJimVect->isEmpty())
      return(0);
    if(ilayer>=aJimVect->getLayerCount())
      return(0);
    //todo: destroy empty features?

    ntotalfeatures=aJimVect->getFeatureCount(ilayer);
    std::vector<OGRFieldDefn*> fields;
    std::vector<size_t> fieldindexes;
    aJimVect->getFields(fields,ilayer);
    std::vector<PyObject*> pyListVector;
    for(unsigned int ifield=0;ifield<fields.size();++ifield){
      if(fname.size()){
        if(std::find(fname.begin(),fname.end(),fields[ifield]->GetNameRef())==fname.end())
          continue;
      }
      fieldindexes.push_back(ifield);
      pyListVector.push_back(PyList_New(ntotalfeatures));
    }
    nfields=fieldindexes.size();

    size_t ifield=0;
    for(size_t ifeature = 0; ifeature < ntotalfeatures; ++ifeature) {
      OGRFeature *thisFeature=aJimVect->getFeatureRef(ifeature,ilayer);
      if(!thisFeature){
        // std::cerr << "Warning: " << ifeature << " is NULL" << std::endl;
        continue;
      }
      ifield=0;
      for(std::vector<size_t>::const_iterator fit=fieldindexes.begin();fit!=fieldindexes.end();++fit){
        //get field type
        switch(fields[*fit]->GetType()){
        case(OFTReal):
          PyList_SET_ITEM(pyListVector[ifield], ifeature, PyFloat_FromDouble(thisFeature->GetFieldAsDouble(*fit)));
          break;
        case(OFTString):
          PyList_SET_ITEM(pyListVector[ifield], ifeature, PyString_FromString(thisFeature->GetFieldAsString(*fit)));
          break;
        case(OFTInteger):
        case(OFTInteger64):
          PyList_SET_ITEM(pyListVector[ifield], ifeature, PyLong_FromLong(thisFeature->GetFieldAsInteger64(*fit)));
          break;
        case(OFTBinary):
        case(OFTTime):
        case(OFTDateTime):
        case(OFTWideString):
        case(OFTDate):
        case(OFTRealList):
        case(OFTIntegerList):
        case(OFTStringList):
        case(OFTWideStringList):
        case(OFTInteger64List):
        default:{
          std::ostringstream errorStringStream;
          std::string ftypename = OGRFieldDefn::GetFieldTypeName(fields[*fit]->GetType());
          errorStringStream << "Error: field type " << ftypename << " not supported";
          throw(errorStringStream.str());
          break;
        }
        }
        ++ifield;
      }
    }

    ifield=0;
    PyObject *returnDict = PyDict_New();
    for(std::vector<size_t>::const_iterator fit=fieldindexes.begin();fit!=fieldindexes.end();++fit){
      std::string key = fields[*fit]->GetNameRef();
      PyDict_SetItem(returnDict, PyString_FromString(key.c_str()), pyListVector[ifield]);
      ++ifield;
    }
    if(returnDict)
      return returnDict;
    else
      return(0);
  }

%}

%include "imageclasses/Jim.h"
%include "imageclasses/JimList.h"
%include "imageclasses/VectorOgr.h"
%include "apps/AppFactory.h"
%include "algorithms/Filter2d.h"
enum CPLErr {CE_None = 0, CE_Debug = 1, CE_Warning = 2, CE_Failure = 3, CE_Fatal = 4};
enum GDALDataType {GDT_Unknown = 0, GDT_Byte = 1, GDT_UInt16 = 2, GDT_Int16 = 3, GDT_UInt32 = 4, GDT_Int32 = 5, GDT_Float32 = 6, GDT_Float64 = 7, GDT_CInt16 = 8, GDT_CInt32 = 9, GDT_CFloat32 = 10, GDT_CFloat64 = 11, GDT_TypeCount = 12};
