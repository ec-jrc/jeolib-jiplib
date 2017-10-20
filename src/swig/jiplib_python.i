// python specific interface file

// provide support for numpy array with mial


%{
// #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"
#include "numpy/arrayobject.h"
%}

%typemap(in,numinputs=1) (PyArrayObject *psArray)
{
  /* %typemap(in,numinputs=1) (PyArrayObject  *psArray) */
  if ($input != NULL && PyArray_Check($input))
  {
std::cout << "debug0" << std::endl;
      $1 = (PyArrayObject*)($input);
std::cout << "debug1" << std::endl;
  }
  else
  {
    std::cout << "debug2" << std::endl;
      PyErr_SetString(PyExc_TypeError, "not a numpy array");
      SWIG_fail;
  }
}

//    NPY_INT8, NPY_INT16, NPY_INT32, NPY_INT64, NPY_UINT8, NPY_UINT16, NPY_UINT32, NPY_UINT64, NPY_FLOAT32, NPY_FLOAT64, NPY_COMPLEX64, NPY_COMPLEX128.

/* %extend Jim { */
/*   void jim2np() { */
/*     /\* PyObject *PyArray_SimpleNewFromData(int nd, npy_intp* dims, int typenum, void* data); *\/ */
/*     npy_intp dims[2]; */
/*     dims[0]=$self->nrOfRow(); */
/*     dims[1]=$self->nrOfCol(); */
/*     std::cout << "dim: " << dims[0] << ", " << dims[1] << std::endl; */
/*     /\* PyObject *npArray=PyArray_SimpleNewFromData(2,dims,NPY_UINT16,$self->getDataPointer()); *\/ */
/*     /\* PyObject *npArray=PyArray_SimpleNew(2,dims,NPY_UINT16); *\/ */
/*     /\* return PyArray_Return(npArray); *\/ */
/*   } */
  /* PyArrayObject jim2np() { */
  /*   /\* PyObject *PyArray_SimpleNewFromData(int nd, npy_intp* dims, int typenum, void* data); *\/ */
  /*   npy_intp dims[2]; */
  /*   dims[0]=$self->nrOfRow(); */
  /*   dims[1]=$self->nrOfCol(); */
  /*   /\* PyObject *npArray=PyArray_SimpleNewFromData(2,dims,NPY_UINT16,$self->getDataPointer()); *\/ */
  /*   PyObject *npArray=PyArray_SimpleNew(2,dims,NPY_UINT16); */
  /*   return PyArray_Return(npArray); */
  /* } */

%inline %{

  ERROR_TYPE RasterIOMIALib( IMAGE *im, PyArrayObject *psArray) {
    psArray->data = (char *)memcpy((void *)(psArray->data), (void *)GetImPtr(im), GetImNx(im)*GetImNy(im)*(GetImBitPerPixel(im)/8) );
    return NO_ERROR;
  }

  ERROR_TYPE _ConvertNumPyArrayToMIALibIMAGE( PyArrayObject *psArray, IMAGE *im ) {
    im->p_im=memcpy( (void *)GetImPtr(im), (void *)(psArray->data), GetImNx(im)*GetImNy(im)*(GetImBitPerPixel(im)/8));
    return NO_ERROR;
  }

  ERROR_TYPE _ConvertNumPyArrayToJim( PyArrayObject *psArray, std::shared_ptr< jiplib::Jim > ajim){
    IMAGE *im=ajim->getMIA();
    im->p_im=memcpy( (void *)GetImPtr(im), (void *)(psArray->data), GetImNx(im)*GetImNy(im)*(GetImBitPerPixel(im)/8));
    ajim->setMIA();
    return NO_ERROR;
  }

  /* ERROR_TYPE RasterIOJim( std::shared_ptr< jiplib::Jim > ajim, PyArrayObject *psArray) { */
  /* int RasterIOJim( std::shared_ptr< jiplib::Jim > ajim, PyArrayObject *psArray) { */
  /* int RasterIOJim( std::shared_ptr< jiplib::Jim > ajim ) { */
  void RasterIOJim( PyArrayObject *psArray) {
    /* psArray->data = (char *)memcpy((void *)(psArray->data), ajim->getDataPointer(), ajim->nrOfCol()*ajim->nrOfRow()*GDALGetDataTypeSize(ajim->getDataType())>>3 ); */
  }
  %}

%pythoncode %{
import numpy

t_UCHAR    =  3
t_SHORT    =  4
t_USHORT   =  5
t_INT32    =  6
t_UINT32   =  7
t_INT64    =  8
t_UINT64   =  9
t_FLOAT    = 10
t_MIAFLOAT = 10
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



def NumPyToImDataTypeCode(numeric_type):
  """Converts a given numpy array data type code into the correspondent
    MIALib image data type code."""
  if not isinstance(numeric_type, (numpy.dtype,type)):
    raise TypeError("Input must be a valid numpy Array data type")
  if numeric_type == numpy.uint8:
    return t_UCHAR
  elif numeric_type == numpy.uint16:
    return t_USHORT
  elif numeric_type == numpy.int16:
    return t_SHORT
  elif numeric_type == numpy.uint32:
    return t_UINT32
  elif numeric_type == numpy.int32:
    return t_INT32
  elif numeric_type == numpy.uint64:
    return t_UINT64
  elif numeric_type == numpy.int64:
    return t_INT64
  elif numeric_type == numpy.float32:
    return t_FLOAT
  elif numeric_type == numpy.float64:
    return t_DOUBLE
  else:
    raise TypeError("provided numeric_type not compatible with available IMAGE data types")

def ImDataToNumPyTypeCode(ImDataType):
    """Returns the numpy Array data type code matching a given an MIALib
    image data type."""
    if not isinstance(ImDataType, int):
        raise TypeError("Input must be an integer value")

    if ImDataType == t_UCHAR:
        return numpy.uint8
    elif ImDataType == t_USHORT:
        return numpy.uint16
    elif ImDataType == t_SHORT:
        return numpy.int16
    elif ImDataType == t_UINT32:
        return numpy.uint32
    elif ImDataType == t_INT32:
        return numpy.int32
    elif ImDataType == t_UINT64:
        return numpy.uint64
    elif ImDataType == t_INT64:
        return numpy.int64
    elif ImDataType == t_FLOAT:
        return numpy.float32
    elif ImDataType == t_DOUBLE:
        return numpy.float64
    else:
        return None

def JimToNumPyTypeCode(JimDataType):
    """Returns the numpy Array data type code matching a given an JIPLib
    image data type."""
    if JimDataType == GDT_Byte:
        return numpy.uint8
    elif JimDataType == GDT_UInt16:
        return numpy.uint16
    elif JimDataType == GDT_Int16:
        return numpy.int16
    elif JimDataType == GDT_UInt32:
        return numpy.uint32
    elif JimDataType == GDT_Int32:
        return numpy.int32
    elif JimDataType == JDT_UInt64:
        return numpy.uint64
    elif JimDataType == JDT_Int64:
        return numpy.int64
    elif JimDataType == GDT_Float32:
        return numpy.float32
    elif JimDataType == GDT_Float64:
        return numpy.float64
    else:
        return None


def ConvertToNumPyArray( im ):
    """Pure python implementation of converting a MIALib image
    into a numpy array.  Data values are copied!"""

    buf_obj = numpy.empty([im.ny,im.nx], dtype = ImDataToNumPyTypeCode(im.DataType))

    if RasterIOMIALib(im, buf_obj) != NO_ERROR:
       return None

    return buf_obj

def ConvertNumPyArrayToMIALibImage( psArray ):
    """Pure python implementation of converting a numpy array into
    a MIALib image.  Data values are copied!"""

    im=_mialib.create_image(NumPyToImDataTypeCode(psArray.dtype),psArray.shape[0],psArray.shape[1],1)

    if _ConvertNumPyArrayToMIALibIMAGE(psArray, im) != NO_ERROR:
      return None

    return im




def np2jim( psArray):
    """Pure python implementation of converting a numpy array into
    a JIPLib image.  Data values are copied!"""
      print(psArray.dtype)
      print(NumPyToImDataTypeCode(psArray.dtype))
      print(psArray.shape[0])
      print(psArray.shape[1])
    #jim=create_image(NumPyToImDataTypeCode(psArray.dtype),psArray.shape[0],psArray.shape[1],1)
    jim=Jim.createImg({'nrow': psArray.shape[0], 'ncol': psArray.shape[1], 'otype': 'GDT_Float64','mean':10})

    #jim.iminfo()

    #if _ConvertNumPyArrayToJim( psArray, jim) != NO_ERROR:
    #   return None

    return jim

def jim2np( jim ):
#  #"""Pure python implementation of converting a MIALib image into a numpy array.  Data values are copied!"""
  print("entering jim2np")
  print(jim.nrOfRow())
  print(jim.nrOfCol())
  print(jim.getDataType())
  print(JimToNumPyTypeCode(jim.getDataType()))
    #buf_obj = numpy.empty([jim.nrOfRow(),jim.nrOfCol()], dtype = JimToNumPyTypeCode(jim.getDataType()))
    buf_obj = numpy.zeros([jim.nrOfRow(),jim.nrOfCol()], dtype = JimToNumPyTypeCode(jim.getDataType()))
  print("buf_obj created, now calling RasterIOJim()")
    #if(RasterIOJim(jim, buf_obj) != NO_ERROR):
    #RasterIOJim(jim, buf_obj)
    #buf_obj=numpy.add(buf_obj,buf_obj);
    buf_obj
    RasterIOJim(buf_obj)
    return buf_obj


def jim2np():
  buf_obj = numpy.zeros([10,10], dtype = numpy.uint32)
    print("buf_obj created, now calling RasterIOJim()")
  #RasterIOJim(jim, buf_obj)
  #buf_obj=numpy.add(buf_obj,buf_obj);
    print(buf_obj)
  RasterIOJim(buf_obj)
  return buf_obj

%}
