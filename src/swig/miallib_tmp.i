/**********************************************************************
Interface file for SWIG
Author(s): Pierre.Soille@ec.europa.eu
Copyright (c) 2016-2019 European Union (Joint Research Centre)
License EUPLv1.2

This file is part of jiplib
***********************************************************************/
%include constraints.i

%include carrays.i
%array_functions(IMAGE *, imap)
%array_functions(int , intp) // used for example for box array
%array_functions(double , doublep)

// 20170317
// integer box array with 2,4, or 6 size parameters (1-D, 2-D, or 3-D images respectively)
%typemap(in) (int *box) {
  int i, dim;
  $1 =  (int *) calloc(6, sizeof(int));
  if (!PySequence_Check($input)) {
    PyErr_SetString(PyExc_ValueError,"Expected a sequence");
    return NULL;
  }
  dim=PySequence_Length($input);
  if ((dim!=2) || (dim!=4) || (dim!=6)){
    for (i = 0; i < dim; i++) {
      PyObject *o = PySequence_GetItem($input,i);
      // https://docs.python.org/3.5/c-api/long.html
      if (PyInt_Check(o)) {
	$1[i] = (int)PyInt_AsLong(o);
      }
      else {
	PyErr_SetString(PyExc_ValueError,"Sequence elements must be integers");
	free($1);
	return NULL;
      }
    }
  }
  else {
      PyErr_SetString(PyExc_ValueError,"Sequence elements must be equal to 2, 4, or 6 for the size of the [left, right], [left, right, top, bottom], or [left, right, top, bottom, up, down] borders respectively.");
      return NULL;
  }
 }

//allow for overloading (due to int iband)
%typemap(typecheck) (int *box) = PyObject *;
  %typemap(typecheck) (int *box, double d_gval, int iband){
  $1=PySequence_Check($input) ? 1 : 0;
 }
%typemap(typecheck) (int *box, double d_gval){
  $1=PySequence_Check($input) ? 1 : 0;
 }
%typemap(typecheck) (int *box, int iband){
  $1=PySequence_Check($input) ? 1 : 0;
 }
%typemap(typecheck) (int *box){
  $1=PySequence_Check($input) ? 1 : 0;
 }
// Free the box array
%typemap(freearg) (int *box) {
  free($1);
}

// make sure box parameters are non-negative
%typemap(check) int *box {
  int i;
  for (i=0; i<6; i++){
    if ($1[i] < 0) {
      SWIG_exception(SWIG_ValueError, "Expected non-negative value.");
    }
  }
 }

// 20160922
// Allow for automatic garbage collection (no need to patch!)
%extend IMAGE {             // Attach these functions to struct IMAGE
  IMAGE(int type, long int nx, int ny, int nz) {
    return create_image(type, nx,ny,nz);
  }
  ~IMAGE() {
    free_image($self);
  }
  void iminfoMethod() {
    iminfo($self);
  }
};

%typemap(newfree) IMAGE * {
  delete $1;
}

// Addtional code for IMAGE<->NumPy array conversions [20160729]
// adapted from gdal_array.i

%clear IMAGE *;
