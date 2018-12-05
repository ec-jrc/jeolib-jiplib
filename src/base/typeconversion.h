/**********************************************************************
typeconversion.h: class to handle type conversions
Copyright (c) 2016-2018 European Union (Joint Research Centre)
License EUPLv1.2

This file is part of jiplib
***********************************************************************/
#ifndef _TYPECONVERSION_H_
#define _TYPECONVERSION_H_

/* #ifndef WIN32 */
/* #include <cxxabi.h> */
/* #include <typeinfo> */
#include <iostream>
#include <sstream>
#include <string>
#include <iomanip>
#include <stdexcept>
/* #include <memory> */
/* #include <cstdlib> */
#ifndef WIN32
#include <cxxabi.h>
#define mytypeid(T) abi::__cxa_demangle(typeid(T).name(),0,0,&status)
#else
#define mytypeid(T) typeid(T).name()
#endif
/* #ifndef WIN32 */
/* #include <cxxabi.h> */
/* #include <typeinfo> */
/* #include <iostream> */
/* #include <string> */
/* #include <memory> */
/* #include <cstdlib> */

/* template<typename T> std::string mytypeid(T) */
/* { */
/*   int status; */
/*   return(abi::__cxa_demangle(typeid(T).name(),0,0,&status)); */
/* } */
/* #else */
/* template<typename T> std::string mytypeid(T); */
/* { */
/*   return(typeid(T).name()); */
/* } */
/* #endif */

class BadConversion : public std::runtime_error {
 public:
 BadConversion(std::string const& s)
   : runtime_error(s)
  { }
};

///convert command line option to value of the defined type, throw exception in case of failure
template<typename T> inline T string2type(std::string const& s){
  std::istringstream i(s);
  T x;
  if (!(i >> x) )
    throw BadConversion(s);
  return x;
}

///convert command line option to value of the defined type, throw if entire string could not get converted
template<typename T> inline T string2type(std::string const& s,bool failIfLeftoverChars){
  std::istringstream i(s);
  char c;
  T x;
  if (!(i >> x) || (failIfLeftoverChars && i.get(c)))
    throw BadConversion(s);
  return x;
}

///serialization for help or to dump option values to screen in verbose mode
template<typename T> inline std::string type2string(T const& value){
  std::ostringstream oss;
  oss << value;
  return oss.str();
}

#endif
