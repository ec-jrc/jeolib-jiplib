/**********************************************************************
Interface file for SWIG
Author(s): Pierre.Soille@ec.europa.eu
Copyright (c) 2016-2018 European Union (Joint Research Centre)
License EUPLv1.2

This file is part of jiplib
***********************************************************************/
%include constraints.i

%include carrays.i
%array_functions(IMAGE *, imap)
%array_functions(int , intp) // used for example for box array
%array_functions(double , doublep)

/* %include mialib_newobjects.i */


/* %typemap(in) G_TYPE { */
/*   std::cout << "we are in typemap(in) G_TYPE" << std::endl; */
/*   G_TYPE gt; */
/*   if (!PyFloat_Check($input)) { */
/*     PyErr_SetString(PyExc_ValueError,"Expected a number"); */
/*     return NULL; */
/*   } */
/*   double dval=PyFloat_AsDouble($input); */
/*   gt.generic_val=(unsigned char)dval; */
/*   gt.uc_val=(unsigned char)dval; */
/*   gt.us_val=(unsigned short)dval; */
/*   gt.s_val=(short)dval; */
/*   gt.i32_val=(int)dval; */
/*   gt.u32_val=(unsigned int)dval; */
/*   gt.i64_val=(long int)dval; */
/*   gt.u64_val=(unsigned long int)dval; */
/*   gt.f_val=(float)dval; */
/*   gt.d_val=(double)dval; */
/*   $1=gt; */
/*  } */


/* %typemap(in) IMAGE * (IMAGE *tempMIA){ */
/* /\* %typemap(in) IMAGE * { *\/ */
/*   std::cout << "we are in typemap(in) IMAGE*" << std::endl; */
/*   void *argp2 = 0 ; */
/*   int res2 = 0 ; */
/*   /\* IMAGE *tempMIA=0; *\/ */
/*   res2 = SWIG_ConvertPtr($input, &argp2, SWIGTYPE_p_std__shared_ptrT_jiplib__Jim_t,  0  | 0); */
/*   if (!SWIG_IsOK(res2)) { */
/*     SWIG_exception_fail(SWIG_ArgError(res2), "in method " "$symname"); */
/*   } */
/*   if (!argp2) { */
/*     SWIG_exception_fail(SWIG_ValueError, "invalid null reference , argument " "2"" of type '" "shared_ptr<const jiplib::Jim&>""'"); */
/*   } */
/*   tempMIA = (*(reinterpret_cast< std::shared_ptr< jiplib::Jim > * >(argp2)))->getMIA(); */
/*   //test */
/*   /\* std::cout << "jim before function call:" << std::endl; *\/ */
/*   /\* (*(reinterpret_cast< std::shared_ptr< jiplib::Jim > * >(argp2)))->dumpimg(); *\/ */
/*   if(tempMIA) */
/*     $1=tempMIA; */
/*  } */

/* %typemap(out) IMAGE * { */
/*   std::cout << "we are in typemap(out) IMAGE*" << std::endl; */
/*   void *argp2 = 0 ; */
/*   int res2 = 0 ; */
/*   IMAGE *imp=(IMAGE *)$1; */
/*   std::shared_ptr<jiplib::Jim> imgWriter=std::make_shared<jiplib::Jim>(); */
/*   /\* std::shared_ptr<jiplib::Jim> result=jiplib::Jim::createImg(); *\/ */
/*   std::shared_ptr<jiplib::Jim> result = std::make_shared<jiplib::Jim>(); */
/*   result->setMIA(imp); */
/*   PyObject* o=0; */
/*   std::shared_ptr<  jiplib::Jim > *smartresult = result ? new std::shared_ptr<  jiplib::Jim >(result) : 0; */
/*   o = SWIG_NewPointerObj(SWIG_as_voidptr(smartresult), SWIGTYPE_p_std__shared_ptrT_jiplib__Jim_t, SWIG_POINTER_OWN | 0); */
/*   if(o) */
/*     $result=o; */
/*   else */
/*     SWIG_exception_fail(SWIG_ArgError(res2), "in method " "$symname"); */
/*  } */

/* //$input is the input Python object */
/* //$1 is the (c/c++) function call argument */
/* %typemap(argout) IMAGE * { */
/*   std::cout << "we are in typemap(argout) IMAGE*" << std::endl; */
/*   void *argp2 = 0 ; */
/*   int res2 = 0 ; */
/*   res2 = SWIG_ConvertPtr($input, &argp2, SWIGTYPE_p_std__shared_ptrT_jiplib__Jim_t,  0  | 0); */
/*   IMAGE *imp=(IMAGE *)$1; */
/*   (*(reinterpret_cast< std::shared_ptr< jiplib::Jim > * >(argp2)))->setMIA(imp); */
/*  } */

/* %typemap(in) (IMAGE **tempMIA, int nc){ */
/*   std::cout << "we are in typemap(in) IMAGE**" << std::endl; */
/*   void *argp2 = 0 ; */
/*   int res2 = 0 ; */
/*   res2 = SWIG_ConvertPtr($input, &argp2, SWIGTYPE_p_jiplib__JimList,  0  | 0); */
/*   /\* res2 = SWIG_ConvertPtr($input, &argp2, SWIGTYPE_p_std__shared_ptrT_jiplib__Jim_t,  0  | 0); *\/ */
/*   if (!SWIG_IsOK(res2)) { */
/*     SWIG_exception_fail(SWIG_ArgError(res2), "in method " "$symname"); */
/*   } */
/*   if (!argp2) { */
/*     SWIG_exception_fail(SWIG_ValueError, "invalid null reference , argument " "2"" of type '" "jiplib::JimList""'"); */
/*   } */
/*   jiplib::JimList jimList= (*(reinterpret_cast< jiplib::JimList * >(argp2))); */
/*   $1 = (IMAGE **) malloc(jimList.size()*sizeof(IMAGE **)); */
/*   $2 = jimList.size(); */
/*   for (int i = 0; i < jimList.size(); i++) */
/*     $1[i]=(std::dynamic_pointer_cast< jiplib::Jim >(jimList[i]))->getMIA(); */
/*  } */


// 20170317
// integer box array with 2,4, or 6 size parameters (1-D, 2-D, or 3-D images respectively)
%typemap(in) (int *box) {
    std::cout << "we are in typemap(in) int *box for jiplib::Jim::$symname" << std::endl;
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

/* // These are the headers with the declarations that will be warped */
/* // It needs to be inserted before the extend declaration (but after the typemaps) */
/* %include "mialib_swig.h" */
/* %include "mialib_convolve.h" */
/* %include "mialib_dem.h" */
/* %include "mialib_dist.h" */
/* %include "mialib_erodil.h" */
/* %include "mialib_format.h" */
/* %include "mialib_geodesy.h" */
/* %include "mialib_geometry.h" */
/* %include "mialib_hmt.h" */
/* %include "mialib_imem.h" */
/* %include "mialib_io.h" */
/* %include "mialib_label.h" */
/* %include "mialib_miscel.h" */
/* %include "mialib_opclo.h" */
/* %include "mialib_pointop.h" */
/* %include "mialib_proj.h" */
/* %include "mialib_segment.h" */
/* %include "mialib_stats.h" */
/* %include "op.h" */

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

/* %init %{ */
/*   print_mia_banner(); */
/* %} */

%clear IMAGE *;
