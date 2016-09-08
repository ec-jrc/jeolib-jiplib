//SWIG interface for jiplib
%include <std_string.i>
%include <std_vector.i>
%include <std_list.i>
%include <std_iostream.i>
%include <std_shared_ptr.i>
%shared_ptr(ImgRaster)
%shared_ptr(jiplib::Jim)

%module jiplib
%{
#include <memory>
#include "config.h"
#include "imageclasses/ImgRaster.h"
#include "imageclasses/ImgCollection.h"
#include "apps/AppFactory.h"
#include "algorithms/Filter2d.h"
#include "jim.h"
#include "mialib_swig.h"
#include <cpl_error.h>
  %}

%template(ImgVectorRaster) std::vector< std::shared_ptr< ImgRaster > >;
%template(ImgVectorJim) std::vector< std::shared_ptr< jiplib::Jim > >;

//Parse the header file
//%include "imageclasses/ImgRaster.h"
%include "swig/pktools.i"
%include "imageclasses/ImgCollection.h"
%include "imageclasses/ImgRaster.h"
%include "apps/AppFactory.h"
%include "algorithms/Filter2d.h"
%include "jim.h"

//%ignore Jim::operator=;
/* %rename(ImgCollection) ImgCollection<std::shared_ptr<jiplib::Jim> >; */

/* %inline %{ */
/* extern "C" */
/* { */
/*  void *__dso_handle = 0; */
/* } */
/*  %} */


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
g++ -fPIC -std=c++11 -I.. -I../../build -I/usr/local/include/mia -I/usr/local/include/pktools -c jiplib_wrap.cc $(python-config --cflags) -o jiplib_wrap.o
g++ -shared -v -nostartfiles -L../../build/src -L/usr/local/lib jiplib_wrap.o -lmia_generic -ljiplib -limageClasses -lalgorithms -lgsl -ldl -lgdal $(python-config --ldflags) -o _jiplib.so
sudo cp _jiplib.so jiplib.py /usr/local/lib/python2.7/site-packages
*/
//////// how to use jiplib module within Python //////////
/*
import sys
sys.path.append("/usr/local/lib/python2.7/site-packages/")
import jiplib
*/
enum CPLErr {CE_None = 0, CE_Debug = 1, CE_Warning = 2, CE_Failure = 3, CE_Fatal = 4};

/* %include <typemaps.i> */
/* %apply unsigned short *OUTPUT { unsigned short & theValue, unsigned int, unsigned int, unsigned int }; */

/* %typemap(out) std::shared_ptr<ImgRaster> { */
/*   $result=std::dynamic_pointer_cast<jiplib::Jim> $1; */
/*  } */

/* //todo: does not work yet */
/* %typemap(out) std::shared_ptr<ImgRaster> { */
/*   $result=jiplib::createJim($1); */
/* } */

%typemap(out) std::shared_ptr<ImgRaster> ImgCollection::composite {
  $result=jiplib::createJim($1);
}

/* %typemap(out) std::shared_ptr<ImgRaster> { */
/*   jiplib::Jim *downcast = dynamic_cast<jiplib::Jim *>($1); */
/*     *(jiplib::Jim **)&$result = downcast; */
/*  } */
/* %typemap(out) std::shared_ptr<ImgRaster> ImgCollection::composite { */
/*     /\* const std::string lookup_typename = *arg2 + " *"; *\/ */
/*     /\* swig_type_info * const outtype = SWIG_TypeQuery(lookup_typename.c_str()); *\/ */
/*   std::shared_ptr<jiplib:Jim> pJim = std::dynamic_pointer_cast<jiplib::Jim> (base); */
/*   swig_type_info * const outtype = SWIG_TypeQuery(jiplib::Jimlookup_typename.c_str()); */
/*     $result = SWIG_NewPointerObj(SWIG_as_voidptr($1), outtype, $owner); */
/* } */

%typemap(out) ImgRaster * {
  jiplib::Jim *downcast = dynamic_cast<jiplib::Jim *>($1);
  *(jiplib::Jim **)&$result = downcast;
}

%typemap(out) std::shared_ptr<ImgRaster> {
  std::dynamic_pointer_cast<jiplib::Jim>($1);
}

%newobject createJim;
%inline %{
  std::shared_ptr<jiplib::Jim> createJim(){std::shared_ptr<jiplib::Jim> pJim(new jiplib::Jim()); return pJim;};
  std::shared_ptr<jiplib::Jim> createJim(const std::string& filename){std::shared_ptr<jiplib::Jim> pJim(new jiplib::Jim(filename)); return pJim;};
  std::shared_ptr<jiplib::Jim> createJim(const std::string& filename, const jiplib::Jim& imgSrc){std::shared_ptr<jiplib::Jim> pJim(new jiplib::Jim(filename, imgSrc)); return pJim;};
  std::shared_ptr<jiplib::Jim> createJim(const std::string& filename, const jiplib::Jim& imgSrc, const std::vector<std::string>& options){std::shared_ptr<jiplib::Jim> pJim(new jiplib::Jim(filename, imgSrc, 0, options)); return pJim;};
  std::shared_ptr<jiplib::Jim> createJim(jiplib::Jim& imgSrc, bool copyData){std::shared_ptr<jiplib::Jim> pJim(new jiplib::Jim(imgSrc,copyData)); return pJim;};
  std::shared_ptr<jiplib::Jim> createJim(jiplib::Jim& imgSrc){std::shared_ptr<jiplib::Jim> pJim(new jiplib::Jim(imgSrc,true)); return pJim;};
  //todo: create Jim with proper attributes as derived from imgRaster, perhaps creating new object and delete old?
  std::shared_ptr<jiplib::Jim> createJim(std::shared_ptr<ImgRaster> imgSrc){return(std::dynamic_pointer_cast<jiplib::Jim>(imgSrc));};
 %}

/* %ignore("ImgCollection::pushImage"); */
/* %rename("ImgCollection::pushImage") ImgCollection::pushImageWrap;//(jiplib::Jim); */

%allowexception;                // turn on globally
