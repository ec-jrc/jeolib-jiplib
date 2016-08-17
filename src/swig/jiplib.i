//SWIG interface for jiplib
%module jiplib
%{
  #include "config.h"
  #include "imageclasses/ImgRaster.h"
  #include "jim.h"
  #include "jipl_glue.h"
  #include <cpl_error.h>
  %}

//Parse the header file
//%include "imageclasses/ImgRaster.h"
%include "/home/kempepi/pktools/src/swig/pktools.i"
%include "jim.h"
//%include "jipl_glue.h"

//%ignore Jim::operator=;

%inline %{
extern "C"
{
 void *__dso_handle = 0;
}
 %}

%include <std_string.i>
%include <std_vector.i>
%include <std_iostream.i>

// Instantiate templates for vector
%template(ByteVector) std::vector<char>;
%template(Int16Vector) std::vector<short>;
%template(UInt16Vector) std::vector<unsigned short>;
%template(Int32Vector) std::vector<int>;
%template(UInt32Vector) std::vector<unsigned int>;
%template(Float32Vector) std::vector<float>;
%template(Float64Vector) std::vector<double>;
%template(StringVector) std::vector<std::string>;

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

//%catches(std::string)

enum CPLErr {CE_None = 0, CE_Debug = 1, CE_Warning = 2, CE_Failure = 3, CE_Fatal = 4};

%include <typemaps.i>
%apply unsigned short *OUTPUT { unsigned short & theValue, unsigned int, unsigned int, unsigned int };
