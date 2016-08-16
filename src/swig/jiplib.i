//SWIG interface for jiplib
%include <std_string.i>
%include <std_vector.i>
%include <std_iostream.i>

%module jiplib
%{
  #include "config.h"
  #include "imageclasses/ImgRaster.h"
  #include "jim.h"
  #include "jipl_glue.h"
  %}

//Parse the header file
%include "imageclasses/ImgRaster.h"
%include "jim.h"
%include "jipl_glue.h"

   %ignore Jim::operator=;




%inline %{
extern "C"
{
 void *__dso_handle = 0;
}
 %}

%include "std_vector.i"
%include "std_string.i"
// Instantiate templates for vector
namespace std {
   %template(ByteVector) vector<char>;
   %template(Int16Vector) vector<short>;
   %template(UInt16Vector) vector<unsigned short>;
   %template(Int32Vector) vector<int>;
   %template(UInt32Vector) vector<unsigned int>;
   %template(Float32Vector) vector<float>;
   %template(Float64Vector) vector<double>;
   %template(StringVector) vector<std::string>;
}

// instantiate read and write data from ImgRaster
%template(readDataUInt16) jiplib::Jim::readData<unsigned short>;
%template(writeDataUInt16) jiplib::Jim::writeData<unsigned short>;
/* %template(writeDataUInt16) ImgRaster::writeData<unsigned short> */

/* namespace std { */
/*   %template(ImgVector) vector<jiplib::Jim>; */
/* } */

   /* %rename(__assign__) *::operator=; */
// Instantiate templates used by example
   /* %rename(__assignJim__) jiplib::Jim::operator=; */
   %rename(__isEqual__) jiplib::Jim::operator==;
   %rename(__isNot__) jiplib::Jim::operator!=;

   ///////////// how to build _jiplib.so /////////////////
   // swig -c++ -I.. -I/usr/local/include/mia -I/usr/local/include/pktools -python -o jiplib_wrap.cc jiplib.i
// add following lines to jiplib_wrap.cc
// extern "C"
// {
// void *__dso_handle = 0;
// }
// g++ -fPIC -I.. -I../../build -I/usr/local/include/mia -I/usr/local/include/pktools -c jiplib_wrap.cc $(python-config --cflags) -o jiplib_wrap.o
// g++ -shared -v -nostartfiles -L../../build/src -L/usr/local/lib jiplib_wrap.o -ljip_generic -ljiplib -limageClasses -lalgorithms -lgsl -ldl -lgdal $(python-config --ldflags) -o _jiplib.so

