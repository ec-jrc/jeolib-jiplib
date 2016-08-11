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

