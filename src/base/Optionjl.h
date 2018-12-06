/**********************************************************************
Optionjl.h: class to handle command line options (inherits from stl vector class)
Author(s): Pieter.Kempeneers@ec.europa.eu
Copyright (c) 2016-2018 European Union (Joint Research Centre)
License EUPLv1.2

This file is part of jiplib
***********************************************************************/
#ifndef _OPTIONPK_H_
#define _OPTIONPK_H_

#include <vector>
#include <string>
#include <cstdlib>
#include <assert.h>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <typeinfo>
/* #ifndef WIN32 */
/* #include <cxxabi.h> */
/* #include <type.hpp> */
/* #define mytypeid(T) abi::__cxa_demangle(typeid(T).name(),0,0,&status) */
/* #else */
/* #define mytypeid(T) typeid(T).name() */
/* #endif */
#include "ogr_feature.h"

#ifdef HAVE_CONFIG_H
#include <config_jiplib.h>
#endif

#include "typeconversion.h"
#include "apps/AppFactory.h"

/**
Class to implement command line options. With the constructor you can define an option, in both short `-` and long `--` format, of a specific type, help information and a default value.\n
This class inherits from std::vector, so the option variable is a vector, supporting multiple inputs for the same option (e.g., --input file1 [--input file2 ...].
Several command line option formats are supported:
- `-shortOption value`
- `-shortOption=value`
- `--longOption value`
- `--longOption=value`
- `-shortOption` (no value for boolean options, which are automatically set by invoking the option)
- `--longOption` (no value for boolean options, which are automatically set by invoking the option)

Option names should have regular characters and no white space in them. Some names are reserved and can not be used either:
- short option `h` or long option `help`: shows usage
- long option `help` shows long help info
- long option `license`: shows license info
- long option `version`: shows current version of pktools
- long option `doxygen`: shows help info in table format, ready to be included in doxygen
- long option `doxypar`: shows list of parameters for doxygen inline documentation

A call to member function \ref retrieveOption reads the command line arguments and initializes the object (vector). Make sure to call this member function before using the option object in your main program (or a segmentation error due to an un-initialized vector will occur).

All calls to retrieveOption should reside in a try{} block. If one of the reserved options
- `license`
- `version`
is used, an exception of type std::string is thrown. This can be caught with a catch(string predefinedString) right after the try block, where the message can be sent to stdout and the program can be ended.

Similarly, if help is invoked with the short option `-h` or long option `--help`, the main program is informed by the return value `false` of \ref retrieveOption (for any option). An example how to use Optionjl is shown in \ref pktestOption.cc
**/

template<class T> class Optionjl : public std::vector <T>
{
public:
  ///default constructor
  Optionjl();
  ///constructor for option without default value
  Optionjl(const std::string& shortName, const std::string& longName, const std::string& helpInfo);
  ///constructor for option with default value. Option can be hidden for help info
  Optionjl(const std::string& shortName, const std::string& longName, const std::string& helpInfo,const T& defaultValue, short hide=0);
  ///default destructor
  ~Optionjl();
  ///set help information
  void setHelp(const std::string& helpInfo){m_help=helpInfo;};
  ///hide option from short help -h (1) or make invisible to short and long help --help (2)
  void setHide(short hide){m_hide=hide;};
  ///read option from command line (use for all options!), std::string implementation
  bool retrieveOption(int argc, const std::vector<std::string>& argv);
  ///read option from command line (use for all options!)
  bool retrieveOption(int argc, char ** argv);
  ///read option from command line (use for all options!) using AppFactory and register option in the app
  bool retrieveOption(const app::AppFactory& app);
  bool retrieveOption(app::AppFactory& app);
  ///print values for this option
  template<class T1> friend std::ostream& operator<<(std::ostream & os, const Optionjl<T1>& theOption);
  ///set all attributes of the option, except default and hide
  void setAll(const std::string& shortName, const std::string& longName, const std::string& helpInfo);
  ///set all attributes of the option
  void setAll(const std::string& shortName, const std::string& longName, const std::string& helpInfo,const T& defaultValue, short hide);
  ///set a default value for the option
  void setDefault(const T& defaultValue);
  std::string getDefaultValue() const {return m_defaultValue;};
  ///set the short name to be used as -shortName
  void setShortName(const std::string& shortName);
  ///set the long name to be used as --longName
  void setLongName(const std::string& longName);
  ///get the short name to be used as -shortName
  std::string getShortName() const {return m_shortName;};
  ///get the long name to be used as --longName
  std::string getLongName() const {return m_longName;};
  ///get help info stored in m_help
  std::string getHelp() const {return m_help;};
  ///get license info
  static std::string getGPLv3License(){
    return static_cast<std::string>("\n\
    This program is free software: you can redistribute it and/or modify\n\
    it under the terms of the GNU General Public License as published by\n\
    the Free Software Foundation, either version 3 of the License, or\n\
    (at your option) any later version.\n\
    \n\
    This program is distributed in the hope that it will be useful,\n\
    but WITHOUT ANY WARRANTY; without even the implied warranty of\n\
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n\
    GNU General Public License for more details.\n\
                                          \n\
    You should have received a copy of the GNU General Public License\n\
    along with this program.  If not, see <http://www.gnu.org/licenses/>.\n");};

  ///find substring in options of type string (e.g., -co INTERLEAVE=BAND)
  ///this template function only makes sense for T=std::string (implemented via a specialization)
  typename std::vector<T>::const_iterator findSubstring(const T& argument) const {std::string errorString="Error: findSubstring only defined for options of type std::string"; throw(errorString);};
  ///this template function only makes sense for T=std::string (implemented via a specialization)
  int countSubstring(const T& argument) const{std::string errorString="Error: countSubstring only defined for options of type std::string"; throw(errorString);};

 private:
  ///all options except bools should have arguments
  bool hasArgument() const {return m_hasArgument;};
  ///true unless option was defined with \0 as short option
  bool hasShortOption() const {return m_shortName.compare("\0");};
  ///true unless option was defined with \0 as long option
  bool hasLongOption() const {return m_longName.compare("\0");};
  ///report the use of the option (called when utility is run with -h or --help)
  std::string usage() const;
  ///used to generate help in Doxygen style (used for web documentation)
  std::string usageDoxygen() const;
  ///used to generate help in Dictionary style (used for Python binding)
  std::string usageDict() const;
  ///used to generate help with in Doxygen style (used for parameters in inline documentation)
  std::string usageDoxypar() const;
  ///short name to be used as -shortName
  std::string m_shortName;
  ///long name to be used as --longName
  std::string m_longName;
  ///string with help information
  std::string m_help;
  ///all options except bools should have arguments
  bool m_hasArgument;
  ///default value of the option
  T m_defaultValue;
  ///has the default been defined
  bool m_hasDefault;
  ///0: always show; 1: only show with --help; 2: invisible option
  short m_hide;
};

template<class T1> std::ostream& operator<<(std::ostream& os, const Optionjl<T1>& theOption)
{
  os << theOption.getLongName() << ": ";
  for(int index=0;index<theOption.size();++index)
    os << type2string<T1>(theOption[index]) << " ";
  os << std::endl;
  return os;
}

template<class T> inline Optionjl<T>::Optionjl()
: m_hasDefault(false)
{
}

/**
constructor without default value\n
shortName is option invoked with `-`\n
longName is option invoked with `--`\n
helpInfo is the help message that is shown when option -h or --help is invoked\n
**/
template<class T> inline Optionjl<T>::Optionjl(const std::string& shortName, const std::string& longName, const std::string& helpInfo)
: m_hasDefault(false)
{
  setAll(shortName,longName,helpInfo);
}

/**
constructor with default value.\n
shortName is option invoked with `-`\n
longName is option invoked with `--`\n
helpInfo is the help message that is shown when option -h or --help is invoked\n
defaultValue is default value of the option (first value of vector: option[0])\n
hide=0 : option is visible for in both short (`-h`). Typical use: mandatory options\n
hide=1 : option is only visible in long help (`--help`). Typical use: expert options\n
hide=2 : option is hidden for user. Typical use: Easter eggs or options only known to author
**/
template<class T> inline Optionjl<T>::Optionjl(const std::string& shortName, const std::string& longName, const std::string& helpInfo,const T& defaultValue, short hide)
{
  setAll(shortName,longName,helpInfo,defaultValue, hide);
}

template<class T> inline std::string Optionjl<T>::usage() const
{
  std::ostringstream helpss;
  std::string shortOption=m_shortName;
  std::string longOption=m_longName;
  shortOption.insert(0,"-");
  longOption.insert(0,"--");
  if(hasShortOption()){
    helpss << "   " << std::setiosflags(std::ios::left) << std::setw(6) << shortOption;
  }
  else{
    helpss << "   " << std::setiosflags(std::ios::left) << std::setw(6) << " ";
  }
  if(hasLongOption()){
    helpss << "   " << std::setiosflags(std::ios::left) << std::setw(20) << longOption;
  }
  else{
    helpss << "   " << std::setiosflags(std::ios::left) << std::setw(20) << " ";
  }
  helpss << "   " << m_help;
  if(m_hasDefault)
    helpss << " (default: " << type2string<T>(m_defaultValue) << ")";
  return helpss.str();
}

template<class T> inline std::string Optionjl<T>::usageDoxypar() const
{
  std::ostringstream helpss;
  std::string shortOption=m_shortName;
  std::string longOption=m_longName;

  if(hasLongOption())
    helpss << std::setiosflags(std::ios::left) << " * @param " << longOption << " ";
  else if(hasShortOption())
      helpss << " | " << std::setiosflags(std::ios::left) << " * @param "<< shortOption << " ";
  int status;
  helpss << std::setiosflags(std::ios::left) << std::setw(4) << "(type: " << mytypeid(T) << ") ";
  if(m_hasDefault)
    helpss << std::setiosflags(std::ios::left) << std::setw(4) << "(default: " << type2string<T>(m_defaultValue) << ") ";
  helpss << m_help;
  return helpss.str();
}

template<class T> inline std::string Optionjl<T>::usageDict() const
{
  std::ostringstream helpss;
  std::string shortOption=m_shortName;
  std::string longOption=m_longName;

  if(hasLongOption())
    helpss << std::setiosflags(std::ios::left) << longOption << ":<value> ";
  else if(hasShortOption())
    helpss << std::setiosflags(std::ios::left) << shortOption << ":<value> ";
  int status;
  helpss << std::setiosflags(std::ios::left) << std::setw(4) << " (type: " << mytypeid(T) << ") ";
  if(m_hasDefault)
    helpss << std::setiosflags(std::ios::left) << std::setw(4) << " (default: " << type2string<T>(m_defaultValue) << ") : ";
  helpss << m_help;
  return helpss.str();
}

template<class T> inline std::string Optionjl<T>::usageDoxygen() const
{
  std::ostringstream helpss;
  std::string shortOption=m_shortName;
  std::string longOption=m_longName;

  if(hasShortOption())
    helpss << " | " << std::setiosflags(std::ios::left) << std::setw(6) << shortOption << " | ";
  else
    helpss << " | " << std::setiosflags(std::ios::left) << "       | ";
  if(hasLongOption())
    helpss << std::setiosflags(std::ios::left) << std::setw(20) << longOption << " | ";
  else
    helpss << std::setiosflags(std::ios::left) << "                     | ";
  int status;
  helpss << std::setiosflags(std::ios::left) << std::setw(4) << mytypeid(T) << " | ";
  //helpss << std::setiosflags(std::ios::left) << std::setw(4) << abi::__cxa_demangle(typeid(T).name(),0,0,&status) << " | ";
  if(m_hasDefault)
    helpss <<std::setiosflags(std::ios::left) << std::setw(5) << type2string<T>(m_defaultValue) << " |";
  else
    helpss << "      |";
  helpss << m_help << " | ";

  return helpss.str();
}

template<class T> inline void Optionjl<T>::setAll(const std::string& shortName, const std::string& longName, const std::string& helpInfo)
{
  m_shortName=shortName;
  m_longName=longName;
  m_hasArgument=true;
  m_help=helpInfo;
  m_hide=0;
}

template<class T> inline void Optionjl<T>::setAll(const std::string& shortName, const std::string& longName, const std::string& helpInfo,const T& defaultValue, short hide)
{
  m_shortName=shortName;
  m_longName=longName;
  m_hasArgument=true;
  m_help=helpInfo;
  m_defaultValue=defaultValue;
  m_hasDefault=true;
  m_hide=hide;
}


template<class T> inline Optionjl<T>::~Optionjl()
{
}

template<class T> inline bool Optionjl<T>::retrieveOption(const app::AppFactory& app){
  return(retrieveOption(app.getArgc(),app.getArgv()));
}

template<class T> inline bool Optionjl<T>::retrieveOption(app::AppFactory& app){
  if(getShortName()!="\0")
    app.registerShortOption(getShortName());
  if(getLongName()!="\0")
    app.registerLongOption(getLongName());
  return(retrieveOption(app.getArgc(),app.getArgv()));
}

///make sure to call this function first before using the option in main program (or segmentation fault will occur...)
template<class T> inline bool Optionjl<T>::retrieveOption(int argc, const std::vector<std::string>& argv){
  bool noHelp=true;//return value, alert main program that hard coded option (help, version, license, doxygen) was invoked
  std::string helpStringShort="-h";//short option for help (hard coded)
  std::string helpStringLong="--help";//long option for help (hard coded)
  std::string helpStringDict="--dict";//option for help in Dictionary format
  std::string helpStringDoxygen="--doxygen";//option to create table of options ready to use for doxygen
  std::string helpStringDoxypar="--doxypar";//option to create list of doxygen parameters for inline documentation
  std::string versionString="--version";//option to show current version
  std::string licenseString="--license";//option to show current version
  for(int i = 1; i < argc; ++i ){
    std::string currentArgument;
    std::string currentOption=argv[i];
    std::string shortOption=m_shortName;
    std::string longOption=m_longName;
    shortOption.insert(0,"-");
    longOption.insert(0,"--");
    size_t foundEqual=currentOption.rfind("=");
    if(foundEqual!=std::string::npos){
      currentArgument=currentOption.substr(foundEqual+1);
      currentOption=currentOption.substr(0,foundEqual);
    }
    if(!helpStringShort.compare(currentOption)){
      if(m_hide<1)
        std::cout << usage() << std::endl;
      noHelp=false;
    }
    else if(!helpStringLong.compare(currentOption)){
      if(m_hide<2)
        std::cout << usage() << std::endl;
      noHelp=false;
    }
    else if(!helpStringDoxygen.compare(currentOption)){
      if(m_hide<2)
        std::cout << usageDoxygen() << std::endl;
      noHelp=false;
    }
    else if(!helpStringDoxypar.compare(currentOption)){
      if(m_hide<2)
        std::cout << usageDoxypar() << std::endl;
      noHelp=false;
    }
    else if(!helpStringDict.compare(currentOption)){
      if(m_hide<2)
        std::cout << usageDict() << std::endl;
      noHelp=false;
    }
    else if(!versionString.compare(currentOption)){
      std::ostringstream oss;
      oss << "version " << JIPLIB_VERSION;
      std::string theVersion="version ";
      /* theVersion+=VERSION; */
      oss << ", Copyright (C) Pieter Kempeneers.\n\
   This program comes with ABSOLUTELY NO WARRANTY; for details type use option -h.\n \
   This is free software, and you are welcome to redistribute it\n      \
   under certain conditions; use option --license for details.";
      throw(oss.str());//no need to continue registering (break prevents from multiplication of version info)
    }
    else if(!licenseString.compare(currentOption)){
      throw(getGPLv3License());
    }
    if(hasShortOption()&&!(shortOption.compare(currentOption))){//for -option
      if(foundEqual!=std::string::npos)
        this->push_back(string2type<T>(currentArgument));
      else if(m_hasArgument && i < argc-1)
        this->push_back(string2type<T>(argv[++i]));
      else
        this->push_back(string2type<T>("1"));
    }
    else if(hasLongOption()&&!(longOption.compare(currentOption))){//for --option
      if(foundEqual!=std::string::npos)
        this->push_back(string2type<T>(currentArgument));
      else if(m_hasArgument && i < argc-1){
        std::string argValue=argv[++i];
        this->push_back(string2type<T>(argValue));
        //this->push_back(string2type<T>(argv[++i]));
      }
      else
        this->push_back(string2type<T>("1"));
    }
  }
  if(!(this->size())&&m_hasDefault)//only set default value if no options were given
    this->push_back(m_defaultValue);
  return(noHelp);
}

/**
make sure to call this function first before using the option in main program (or segmentation fault will occur...)
**/
template<class T> inline bool Optionjl<T>::retrieveOption(int argc, char **argv){
  bool noHelp=true;//return value, alert main program that hard coded option (help, version, license, doxygen) was invoked
  std::string helpStringShort="-h";//short option for help (hard coded)
  std::string helpStringLong="--help";//long option for help (hard coded)
  std::string helpStringDict="--dict";//option for help in Dictionary format
  std::string helpStringDoxygen="--doxygen";//option to create table of options ready to use for doxygen
  std::string helpStringDoxypar="--doxypar";//option to list of parameters for doxygen inline documentation
  std::string versionString="--version";//option to show current version
  std::string licenseString="--license";//option to show current version
  for(int i = 1; i < argc; ++i ){
    std::string currentArgument;
    std::string currentOption=argv[i];
    std::string shortOption=m_shortName;
    std::string longOption=m_longName;
    shortOption.insert(0,"-");
    longOption.insert(0,"--");
    size_t foundEqual=currentOption.rfind("=");
    if(foundEqual!=std::string::npos){
      currentArgument=currentOption.substr(foundEqual+1);
      currentOption=currentOption.substr(0,foundEqual);
    }
    if(!helpStringShort.compare(currentOption)){
      if(m_hide<1)
        std::cout << usage() << std::endl;
      noHelp=false;
    }
    else if(!helpStringLong.compare(currentOption)){
      if(m_hide<2)
        std::cout << usage() << std::endl;
      noHelp=false;
    }
    else if(!helpStringDoxygen.compare(currentOption)){
      if(m_hide<2)
        std::cout << usageDoxygen() << std::endl;
      noHelp=false;
    }
    else if(!helpStringDict.compare(currentOption)){
      if(m_hide<2)
        std::cout << usageDict() << std::endl;
      noHelp=false;
    }
    else if(!helpStringDoxypar.compare(currentOption)){
      if(m_hide<2)
        std::cout << usageDoxypar() << std::endl;
      noHelp=false;
    }
    else if(!versionString.compare(currentOption)){
      std::ostringstream oss;
      oss << "version " << JIPLIB_VERSION;
      std::string theVersion="version ";
      /* theVersion+=VERSION; */
      oss << ", Copyright (C) Pieter Kempeneers.\n\
   This program comes with ABSOLUTELY NO WARRANTY; for details type use option -h.\n \
   This is free software, and you are welcome to redistribute it\n      \
   under certain conditions; use option --license for details.";
      throw(oss.str());//no need to continue registering (break prevents from multiplication of version info)
    }
    else if(!licenseString.compare(currentOption)){
      throw(getGPLv3License());
    }
    if(hasShortOption()&&!(shortOption.compare(currentOption))){//for -option
      if(foundEqual!=std::string::npos)
        this->push_back(string2type<T>(currentArgument));
      else if(m_hasArgument && i < argc-1)
        this->push_back(string2type<T>(argv[++i]));
      else
        this->push_back(string2type<T>("1"));
    }
    else if(hasLongOption()&&!(longOption.compare(currentOption))){//for --option
      if(foundEqual!=std::string::npos)
        this->push_back(string2type<T>(currentArgument));
      else if(m_hasArgument && i < argc-1)
        this->push_back(string2type<T>(argv[++i]));
      else
        this->push_back(string2type<T>("1"));
    }
  }
  if(!(this->size())&&m_hasDefault)//only set default value if no options were given
    this->push_back(m_defaultValue);
  return(noHelp);
}

//template<class T> typename std::vector<T>::const_iterator Optionjl<T>::findSubstring(const T& argument) const {std::string errorString="Error: findSubstring only defined for options of type std::string"; throw(errorString);}

///specialization for bool
template<> inline void Optionjl<bool>::setAll(const std::string& shortName, const std::string& longName, const std::string& helpInfo)
{
  m_shortName=shortName;
  m_longName=longName;
  m_hasArgument=false;
  m_help=helpInfo;
  m_hide=0;
}

///specialization for bool
template<> inline void Optionjl<bool>::setAll(const std::string& shortName, const std::string& longName, const std::string& helpInfo,const bool& defaultValue, short hide)
{
  m_shortName=shortName;
  m_longName=longName;
  m_hasArgument=false;
  m_help=helpInfo;
  m_defaultValue=defaultValue;
  m_hasDefault=true;
  m_hide=hide;
}

///specialization for bool
template<> inline Optionjl<bool>::Optionjl(const std::string& shortName, const std::string& longName, const std::string& helpInfo)
{
  setAll(shortName,longName,helpInfo);
}

///specialization for bool
template<> inline Optionjl<bool>::Optionjl(const std::string& shortName, const std::string& longName, const std::string& helpInfo,const bool& defaultValue, short hide)
{
  setAll(shortName,longName,helpInfo,defaultValue, hide);
}

//specialization (only makes sense for T=std::string), generic function throws exception
//find a substring in string option (e.g., option is of type -co INTERLEAVE=BAND)
template<> inline std::vector<std::string>::const_iterator Optionjl<std::string>::findSubstring(const std::string& argument) const{
  std::vector<std::string>::const_iterator opit=this->begin();
  while(opit!=this->end()){
    if(opit->find(argument)!=std::string::npos)
      break;
    ++opit;
  }
  return opit;
}

//specialization (only makes sense for T=std::string), generic function throws exception
//find a substring in string option (e.g., option is of type -co INTERLEAVE=BAND)
template<> inline int Optionjl<std::string>::countSubstring(const std::string& argument) const{
  int count=0;
  std::vector<std::string>::const_iterator opit=this->begin();
  while(opit!=this->end()){
    if(opit->find(argument)!=std::string::npos)
      ++count;
    ++opit;
  }
  return count;
}

#endif
