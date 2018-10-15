/* mialib.i */

%include constraints.i



// %feature("docstring");

%define DOCSTRING
"This is an initial test module for the JIPL (Joint Image Processing Library)
developed in the framework of the JEODPP of the EO&SS@BD pilot project.
Emphasis is on morphological image analysis functionalities.
Contact: Pierre.Soille@jrc.ec.europa.eu"
%enddef

/* %feature("autodoc", "1"); */

// It consists of wrappers of C code underlying mialisp orginally developed
// by Pierre Soille over the years since 1988.

%module(docstring=DOCSTRING) mialib


// see https://stackoverflow.com/questions/11435102/is-there-a-good-way-to-produce-documentation-for-swig-interfaces


/* %import "mial_doxy2swig.i" */



%{
/* Put header files here or function declarations like below */
/* #include "mialib_swig.h" */
#include <memory>
#include "mialib/mialib_swig.h"
#include "mialib/mialib_convolve.h"
#include "mialib/mialib_dem.h"
#include "mialib/mialib_dist.h"
#include "mialib/mialib_erodil.h"
#include "mialib/mialib_format.h"
#include "mialib/mialib_geodesy.h"
#include "mialib/mialib_geometry.h"
#include "mialib/mialib_hmt.h"
#include "mialib/mialib_imem.h"
#include "mialib/mialib_io.h"
#include "mialib/mialib_label.h"
#include "mialib/mialib_miscel.h"
#include "mialib/mialib_opclo.h"
#include "mialib/mialib_pointop.h"
#include "mialib/mialib_proj.h"
#include "mialib/mialib_segment.h"
#include "mialib/mialib_stats.h"
#include "mialib/op.h"
#include "imageclasses/Jim.h"
%}




// See info on cpointers
// http://www.swig.org/Doc3.0/SWIGDocumentation.html#Library_nn3


//%include cpointer.i
//%pointer_functions(IMAGE,imagep)
//%pointer_functions(IMAGE *,imap)


// definition of array pointers
// for use, e.g., in writing multiband image
%include carrays.i
%array_functions(IMAGE *, imap)
%array_functions(int , intp) // used for example for box array
%array_functions(double , doublep)


// %nodefaultctor image;      // No default constructor
// %nodefaultdtor image;      // No default destructor




// renaming:

/* imem.c */
/* %rename(imInfo) iminfo; */
/* %rename(imToArray) imtoarray; */
/* %rename(arrayToIm) arraytoim; */
/* %rename(setPixVal) setpixval; */
/* %rename(getPixVal) getpixval; */
/* %rename(createImArray) create_imarray; */

// rename the C declarations
// %rename("%(lowercamelcase)s", %$isfunction) ""; // foo_bar -> fooBar; FooBar -> fooBar


// %rename("nd_%s", regextarget=1, fullname=1) "IMAGE \*\(.*";



// new object with their constructor and destructor
//%newobject *IMAGE();
//%newobject *G_TYPE();

//%typemap(newfree) IMAGE * "free_image($1);";

// 20160922
// define each mialib function returning a new IMAGE as a new object
// this triggers the setting of 'SWIG_POINTER_OWN' for the new IMAGE
// rather than '0' previously
// (note that for the destructor ~IMAGE() the setting is 'SWIG_POINTER_NEW')


%include mialib_newobjects.i


%typemap(in) G_TYPE {
  std::cout << "we are in typemap(in) G_TYPE" << std::endl;
  G_TYPE gt;
  if (!PyFloat_Check($input)) {
    PyErr_SetString(PyExc_ValueError,"Expected a number");
    return NULL;
  }
  double dval=PyFloat_AsDouble($input);
  gt.generic_val=(unsigned char)dval;
  gt.uc_val=(unsigned char)dval;
  gt.us_val=(unsigned short)dval;
  gt.s_val=(short)dval;
  gt.i32_val=(int)dval;
  gt.u32_val=(unsigned int)dval;
  gt.i64_val=(long int)dval;
  gt.u64_val=(unsigned long int)dval;
  gt.f_val=(float)dval;
  gt.d_val=(double)dval;
  $1=gt;
 }


// 20160923
// define a typemap to handle IMAGE arrays as lists in Python
// needed to specify names to have multiple argument working
/* %typemap(in) (IMAGE **imap, int nc) { */
/*   int i,dim; */
/*   int res1; */
/*   void *argp1 = 0 ; */
/*   if (!PySequence_Check($input)) { */
/*     PyErr_SetString(PyExc_ValueError,"Expected a sequence"); */
/*     return NULL; */
/*   } */
/*   dim=PySequence_Length($input); */
/*   $2=dim; */
/*   printf("coucou: dim=%d\n", dim); */
/*   $1 = (IMAGE **) malloc(dim*sizeof(IMAGE **)); */
/*   for (i = 0; i < dim; i++) { */
/*     PyObject *o = PySequence_GetItem($input,i); */
/*     res1 = SWIG_ConvertPtr(o, &argp1,SWIGTYPE_p_IMAGE, 0 |  0 ); */
/*     if (SWIG_IsOK(res1)) { */
/*       $1[i] = (IMAGE *) argp1; */
/*     } */
/*     else { */
/*       PyErr_SetString(PyExc_ValueError,"Sequence elements must be IMAGE pointers");       */
/*       free($1); */
/*       return NULL; */
/*     } */
/*   } */
/*  } */

/* // handling IMAGE array output argument as python list */
/* %typemap(out) IMAGE **rotatecoor { */
/*   int i; */
/*   int nc=2; */
/*   IMAGE **imap=(IMAGE **)$1; */
/*   $result = PyList_New(nc); */
/*   PyObject * o = 0 ; */
/*   for (i = 0; i < nc; i++) { */
/*     o = SWIG_NewPointerObj(SWIG_as_voidptr(imap[i]), SWIGTYPE_p_IMAGE, SWIG_POINTER_OWN |  0 ); */
/*     PyList_SetItem($result,i,o); */
/*   } */
/*   free(imap); */
/*  } */

/* %typemap(out) IMAGE **imrgb2hsx { */
/*   int i; */
/*   int nc=3; */
/*   IMAGE **imap=(IMAGE **)$1; */
/*   $result = PyList_New(nc); */
/*   PyObject * o = 0 ; */
/*   for (i = 0; i < nc; i++) { */
/*     o = SWIG_NewPointerObj(SWIG_as_voidptr(imap[i]), SWIGTYPE_p_IMAGE, SWIG_POINTER_OWN |  0 ); */
/*     PyList_SetItem($result,i,o); */
/*   } */
/*   free(imap); */
/*  } */

/* %typemap(out) IMAGE **PartitionSimilarity { */
/*   int i; */
/*   int nc=4; */
/*   IMAGE **imap=(IMAGE **)$1; */
/*   $result = PyList_New(nc); */
/*   PyObject * o = 0 ; */
/*   for (i = 0; i < nc; i++) { */
/*     o = SWIG_NewPointerObj(SWIG_as_voidptr(imap[i]), SWIGTYPE_p_IMAGE, SWIG_POINTER_OWN |  0 ); */
/*     PyList_SetItem($result,i,o); */
/*   } */
/*   free(imap); */
/*  } */

/* %typemap(out) IMAGE **alphatree { */
/*   int i; */
/*   int nc=5; */
/*   IMAGE **imap=(IMAGE **)$1; */
/*   $result = PyList_New(nc); */
/*   PyObject * o = 0 ; */
/*   for (i = 0; i < nc; i++) { */
/*     o = SWIG_NewPointerObj(SWIG_as_voidptr(imap[i]), SWIGTYPE_p_IMAGE, SWIG_POINTER_OWN |  0 ); */
/*     PyList_SetItem($result,i,o); */
/*   } */
/*   free(imap); */
/*  } */

/* %typemap(out) IMAGE **histrgbmatch { */
/*   int i; */
/*   int nc=3; */
/*   IMAGE **imap=(IMAGE **)$1; */
/*   $result = PyList_New(nc); */
/*   PyObject * o = 0 ; */
/*   for (i = 0; i < nc; i++) { */
/*     o = SWIG_NewPointerObj(SWIG_as_voidptr(imap[i]), SWIGTYPE_p_IMAGE, SWIG_POINTER_OWN |  0 ); */
/*     PyList_SetItem($result,i,o); */
/*   } */
/*   free(imap); */
/*  } */

/* %typemap(out) IMAGE **histrgb3dmatch { */
/*   int i; */
/*   int nc=3; */
/*   IMAGE **imap=(IMAGE **)$1; */
/*   $result = PyList_New(nc); */
/*   PyObject * o = 0 ; */
/*   for (i = 0; i < nc; i++) { */
/*     o = SWIG_NewPointerObj(SWIG_as_voidptr(imap[i]), SWIGTYPE_p_IMAGE, SWIG_POINTER_OWN |  0 ); */
/*     PyList_SetItem($result,i,o); */
/*   } */
/*   free(imap); */
/*  } */


%typemap(in) IMAGE * (IMAGE *tempMIA){
/* %typemap(in) IMAGE * { */
  std::cout << "we are in typemap(in) IMAGE*" << std::endl;
  void *argp2 = 0 ;
  int res2 = 0 ;
  /* IMAGE *tempMIA=0; */
  res2 = SWIG_ConvertPtr($input, &argp2, SWIGTYPE_p_std__shared_ptrT_jiplib__Jim_t,  0  | 0);
  if (!SWIG_IsOK(res2)) {
    SWIG_exception_fail(SWIG_ArgError(res2), "in method " "$symname");
  }
  if (!argp2) {
    SWIG_exception_fail(SWIG_ValueError, "invalid null reference , argument " "2"" of type '" "shared_ptr<const jiplib::Jim&>""'");
  }
  tempMIA = (*(reinterpret_cast< std::shared_ptr< jiplib::Jim > * >(argp2)))->getMIA();
  //test
  /* std::cout << "jim before function call:" << std::endl; */
  /* (*(reinterpret_cast< std::shared_ptr< jiplib::Jim > * >(argp2)))->dumpimg(); */
  if(tempMIA)
    $1=tempMIA;
 }

%typemap(out) IMAGE * {
  std::cout << "we are in typemap(out) IMAGE*" << std::endl;
  void *argp2 = 0 ;
  int res2 = 0 ;
  IMAGE *imp=(IMAGE *)$1;
  std::shared_ptr<jiplib::Jim> imgWriter=std::make_shared<jiplib::Jim>();
  /* std::shared_ptr<jiplib::Jim> result=jiplib::Jim::createImg(); */
  std::shared_ptr<jiplib::Jim> result = std::make_shared<jiplib::Jim>();
  result->setMIA(imp);
  PyObject* o=0;
  std::shared_ptr<  jiplib::Jim > *smartresult = result ? new std::shared_ptr<  jiplib::Jim >(result) : 0;
  o = SWIG_NewPointerObj(SWIG_as_voidptr(smartresult), SWIGTYPE_p_std__shared_ptrT_jiplib__Jim_t, SWIG_POINTER_OWN | 0);
  if(o)
    $result=o;
  else
    SWIG_exception_fail(SWIG_ArgError(res2), "in method " "$symname");
 }

//$input is the input Python object
//$1 is the (c/c++) function call argument
%typemap(argout) IMAGE * {
  std::cout << "we are in typemap(argout) IMAGE*" << std::endl;
  void *argp2 = 0 ;
  int res2 = 0 ;
  res2 = SWIG_ConvertPtr($input, &argp2, SWIGTYPE_p_std__shared_ptrT_jiplib__Jim_t,  0  | 0);
  IMAGE *imp=(IMAGE *)$1;
  (*(reinterpret_cast< std::shared_ptr< jiplib::Jim > * >(argp2)))->setMIA(imp);
 }

%typemap(in) (IMAGE **tempMIA, int nc){
  std::cout << "we are in typemap(in) IMAGE**" << std::endl;
  void *argp2 = 0 ;
  int res2 = 0 ;
  res2 = SWIG_ConvertPtr($input, &argp2, SWIGTYPE_p_jiplib__JimList,  0  | 0);
  /* res2 = SWIG_ConvertPtr($input, &argp2, SWIGTYPE_p_std__shared_ptrT_jiplib__Jim_t,  0  | 0); */
  if (!SWIG_IsOK(res2)) {
    SWIG_exception_fail(SWIG_ArgError(res2), "in method " "$symname");
  }
  if (!argp2) {
    SWIG_exception_fail(SWIG_ValueError, "invalid null reference , argument " "2"" of type '" "jiplib::JimList""'");
  }
  jiplib::JimList jimList= (*(reinterpret_cast< jiplib::JimList * >(argp2)));
  $1 = (IMAGE **) malloc(jimList.size()*sizeof(IMAGE **));
  $2 = jimList.size();
  for (int i = 0; i < jimList.size(); i++)
    $1[i]=(std::dynamic_pointer_cast< jiplib::Jim >(jimList[i]))->getMIA();
 }


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



/* %typemap(out) IMAGE **rotatecoor { */
/*   std::cout << "we are in typemap(out) IMAGE** rotatecoor" << std::endl; */
/*   void *argp2 = 0 ; */
/*   int res2 = 0 ; */
/*   IMAGE **imap=(IMAGE **)$1; */
/*   int nc=2; */
/*   jiplib::JimList result; */
/*   for(int i=0;i<nc;++i){ */
/*     //todo: how will smart pointer be garbage collected in Python (is there a need for smartresult)? */
/*     std::shared_ptr<jiplib::Jim> jimImg=jiplib::Jim::createImg(); */
/*     jimImg->setMIA(imap[i]); */
/*     result.pushImage(jimImg); */
/*   } */
/*   PyObject* o=0; */
/*   o = SWIG_NewPointerObj(SWIG_as_voidptr(&result), SWIGTYPE_p_jiplib__JimList, SWIG_POINTER_OWN | 0); */
/*   if(o) */
/*     $result=o; */
/*   else */
/*     SWIG_exception_fail(SWIG_ArgError(res2), "in method " "$symname"); */
/*   free(imap); */
/*  } */

// These are the headers with the declarations that will be warped
// It needs to be inserted before the extend declaration (but after the typemaps)
%include "mialib_swig.h"
%include "mialib_convolve.h"
%include "mialib_dem.h"
%include "mialib_dist.h"
%include "mialib_erodil.h"
%include "mialib_format.h"
%include "mialib_geodesy.h"
%include "mialib_geometry.h"
%include "mialib_hmt.h"
%include "mialib_imem.h"
%include "mialib_io.h"
%include "mialib_label.h"
%include "mialib_miscel.h"
%include "mialib_opclo.h"
%include "mialib_pointop.h"
%include "mialib_proj.h"
%include "mialib_segment.h"
%include "mialib_stats.h"
%include "op.h"

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

%init %{
  print_mia_banner();
  /* import_array(); */
%}

%clear IMAGE *;

/* #if defined(SWIGPYTHON) */
/* %include "mialib_python.i" */
/* #endif */






// typemap for mialib functions returning a G_TYPE
/* %typemap(out) G_TYPE getPixVal { */
/*   double dval=0.0; */
/*   switch (GetImDataType(arg1)) { */
/*   case t_UCHAR: */
/*     dval=(double)$1.uc_val; */
/*     break; */
/*   case t_SHORT: */
/*     dval=(double)$1.s_val; */
/*     break; */
/*   case t_USHORT: */
/*     dval=(double)$1.us_val; */
/*     break; */
/*   case t_INT32: */
/*     dval=(double)$1.i32_val; */
/*     break; */
/*   case t_UINT32: */
/*     dval=(double)$1.u32_val; */
/*     break; */
/*   case t_INT64: */
/*     dval=(double)$1.i64_val; */
/*     break; */
/*   case t_UINT64: */
/*     dval=(double)$1.u64_val; */
/*     break; */
/*   case t_MIAFLOAT: */
/*     dval=(double)$1.f_val; */
/*     break; */
/*   case t_DOUBLE: */
/*     dval=(double)$1.d_val; */
/*     break; */
/*   default: */
/*     printf("getpixval(): undefined pixel type (%d) !\n)", GetImDataType(arg1)); */
/*   } */
/*   $result=PyFloat_FromDouble(dval); */
/*  } */
