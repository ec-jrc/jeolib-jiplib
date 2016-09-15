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

// Instantiate templates for vector
/* %template(ByteVector) std::vector<char>; */
/* %template(Int16Vector) std::vector<short>; */
/* %template(UInt16Vector) std::vector<unsigned short>; */
/* %template(Int32Vector) std::vector<int>; */
/* %template(UInt32Vector) std::vector<unsigned int>; */
/* %template(Float32Vector) std::vector<float>; */
/* %template(Float64Vector) std::vector<double>; */
/* %template(StringVector) std::vector<std::string>; */

enum CPLErr {CE_None = 0, CE_Debug = 1, CE_Warning = 2, CE_Failure = 3, CE_Fatal = 4};
enum GDALDataType {GDT_Unknown = 0, GDT_Byte = 1, GDT_UInt16 = 2, GDT_Int16 = 3, GDT_UInt32 = 4, GDT_Int32 = 5, GDT_Float32 = 6, GDT_Float64 = 7, GDT_CInt16 = 8, GDT_CInt32 = 9, GDT_CFloat32 = 10, GDT_CFloat64 = 11, GDT_TypeCount = 12}; 

/* %newobject createJim; */
/* %inline %{ */
/*   std::shared_ptr<jiplib::Jim> createJim(){std::shared_ptr<jiplib::Jim> pJim(new jiplib::Jim()); return pJim;}; */
/*   std::shared_ptr<jiplib::Jim> createJim(const std::string& filename){std::shared_ptr<jiplib::Jim> pJim(new jiplib::Jim(filename)); return pJim;}; */
/*   std::shared_ptr<jiplib::Jim> createJim(const std::string& filename, const jiplib::Jim& imgSrc){std::shared_ptr<jiplib::Jim> pJim(new jiplib::Jim(filename, imgSrc)); return pJim;}; */
/*   std::shared_ptr<jiplib::Jim> createJim(const std::string& filename, const jiplib::Jim& imgSrc, const std::vector<std::string>& options){std::shared_ptr<jiplib::Jim> pJim(new jiplib::Jim(filename, imgSrc, 0, options)); return pJim;}; */
/*   std::shared_ptr<jiplib::Jim> createJim(jiplib::Jim& imgSrc, bool copyData){std::shared_ptr<jiplib::Jim> pJim(new jiplib::Jim(imgSrc,copyData)); return pJim;}; */
/*   std::shared_ptr<jiplib::Jim> createJim(jiplib::Jim& imgSrc){std::shared_ptr<jiplib::Jim> pJim(new jiplib::Jim(imgSrc,true)); return pJim;}; */
/*   //todo: create Jim with proper attributes as derived from imgRaster, perhaps creating new object and delete old? */
/*   std::shared_ptr<jiplib::Jim> createJim(std::shared_ptr<ImgRaster> imgSrc){return(std::dynamic_pointer_cast<jiplib::Jim>(imgSrc));}; */
/*  %} */

/* %allowexception;                // turn on globally */
/* %catches(std::string,...) jiplib::Jim::createImg(); */

//from :  http://stackoverflow.com/questions/39436632/wrap-a-function-that-takes-a-struct-of-optional-arguments-using-kwargs
/* %pythoncode %{ */
/*   def StructArgs(type_name): */
/*   def wrap(f): */
/*   def _wrapper(*args, **kwargs): */
/*   ty=globals()[type_name] */
/*     arg=(ty(),) if kwargs else tuple() */
/*     for it in kwargs.iteritems(): */
/*       setattr(arg[0], *it) */
/*     return f(*(args+arg)) */
/*     return _wrapper */
/*     return wrap */
/*     %} */

/* %define %StructArgs(func, ret, type) */
/* %pythoncode %{ @StructArgs(#type) %} // *very* position sensitive */
/* %pythonprepend func %{ %} // Hack to workaround problem with #3 */
/* ret func(const type*); */
/* %ignore func; */
/* %enddef */

/* typedef bool _Bool; */

/* %StructArgs(createImg, std::shared_ptr<jiplib::Jim>, const app::AppFactory&) */
