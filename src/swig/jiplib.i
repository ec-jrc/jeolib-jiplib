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

   ///////////// how to build _jiplib.so /////////////////
/*
cd ~/jiplib/build/
make -j
sudo cp ~/jiplib/build/src/libjiplib.so /usr/local/lib
cd ~/jiplib/src/swig/
swig -c++ -I.. -I/usr/local/include/mia -I/usr/local/include/pktools -python -o jiplib_wrap.cc jiplib.i
g++ -fPIC -I.. -I../../build -I/usr/local/include/mia -I/usr/local/include/pktools -c jiplib_wrap.cc $(python-config --cflags) -o jiplib_wrap.o
g++ -shared -v -nostartfiles -L../../build/src -L/usr/local/lib jiplib_wrap.o -ljip_generic -ljiplib -limageClasses -lalgorithms -lgsl -ldl -lgdal $(python-config --ldflags) -o _jiplib.so
sudo cp _jiplib.so jiplib.py /usr/local/lib/python2.7/site-packages
*/
//////// how to use jiplib module within Python //////////
/*
import sys
sys.path.append("/usr/local/lib/python2.7/site-packages/")
import jiplib
*/
enum CPLErr {CE_None = 0, CE_Debug = 1, CE_Warning = 2, CE_Failure = 3, CE_Fatal = 4};

%include <typemaps.i>
%apply unsigned short *OUTPUT { unsigned short & theValue, unsigned int, unsigned int, unsigned int };

%allowexception;                // turn on globally
