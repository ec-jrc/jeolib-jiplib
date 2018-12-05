/**********************************************************************
AppFactory.h: class for application functions
Copyright (c) 2016-2018 European Union (Joint Research Centre)
License EUPLv1.2

This file is part of jiplib
***********************************************************************/
#ifndef _APPFACTORY_H_
#define _APPFACTORY_H_

#include <string>
#include <vector>
#include <list>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <locale> //std::locale, std::isdigit
#include "gdal_priv.h"
#include "base/typeconversion.h"
//#include "config.h"
#include "json/value.h"
#include "json/json.h"

namespace app
{
#if PKTOOLS_BUILD_WITH_PYTHON==1
#define MyProgressFunc(dfComplete,pszMessage,pProgressArg) {}
#else
#define MyProgressFunc(dfComplete,pszMessage,pProgressArg) pfnProgress(dfComplete,pszMessage,pProgressArg)
#endif
  class AppFactory{

  public:
  AppFactory(void) : m_argc(1), m_argv(std::vector<std::string>(1,"appFactory")){}
  AppFactory(int argc, char* argv[]) : m_argc(1), m_argv(std::vector<std::string>(1,"appFactory")){
      setOptions(argc, argv);
    }
    AppFactory(Json::Value &jsonobj){json2app(jsonobj);};
    AppFactory(const std::string &jsonAppString){
      Json::Value root;
      Json::Reader reader;
      bool parsedSuccess = reader.parse(jsonAppString, root, false);
      if( parsedSuccess )
        {
          json2app(root);
        }
    }
    ///copy constructor
    AppFactory(const AppFactory &theApp){
      m_argc=theApp.getArgc();
      m_argv=theApp.getArgv();
      m_shortRegistered=theApp.getShortRegister();
      m_longRegistered=theApp.getLongRegister();
    }
    ///copy operator
    AppFactory &operator=(const AppFactory &theApp){
      if( &theApp == this ) return *this;
      m_argc=theApp.getArgc();
      m_argv=theApp.getArgv();
      m_longRegistered=theApp.getLongRegister();
      m_shortRegistered=theApp.getShortRegister();
      return *this;
    }
    virtual ~AppFactory(void){};
    bool empty() const {return(m_argv.empty());};
    bool size() const {return(m_argv.size());};
    void setOptions(int argc, const std::vector<std::string> argv){
      m_argc=argc;
      m_argv.clear();
      m_argv=argv;
    }
    ///get all options and store in argv, return number of options const version
    std::vector<std::string> getOptions() const{return m_argv;};
    ///get all options and store in argv, return number of options
    std::vector<std::string> getOptions() {return m_argv;};
    ///clear short register
    void clearShortRegister(){
      m_shortRegistered.clear();
    }
    ///clear long register
    void clearLongRegister(){
      m_longRegistered.clear();
    }
    ///clear register
    void clearRegister(){
      clearRegister();
      clearRegister();
    }
    ///get short register const version
    std::vector<std::string> getShortRegister() const {return(m_shortRegistered);};
    ///get short register
    std::vector<std::string> getShortRegister() {return(m_shortRegistered);};
    ///get long register const version
    std::vector<std::string> getLongRegister() const {return(m_longRegistered);};
    ///get long register
    std::vector<std::string> getLongRegister() {return(m_longRegistered);};
    ///check if app was called with not registered options
    int badKeys(std::vector<std::string>& badKeys){
      badKeys.clear();
      std::locale loc;
      std::string helpStringShort="-h";//short option for help (hard coded)
      std::string helpStringLong="--help";//long option for help (hard coded)
      std::string helpStringDoxygen="--doxygen";//option to create table of options ready to use for doxygen
      std::string helpStringDoxypar="--doxypar";//option to create list of doxygen parameters for inline documentation
      std::string versionString="--version";//option to show current version
      std::string licenseString="--license";//option to show current version
      for(int i = 1; i < m_argc; ++i ){
        std::string currentArgument;
        std::string currentOption=m_argv[i];
        /* if(currentOption.compare(0,1,"-")) */
        /*   continue;//not an option */
        bool isLongOption=(!currentOption.compare(0,2,"--"));
        bool isShortOption=(!currentOption.compare(0,1,"-")&&!std::isdigit(currentOption[1],loc));
        bool currentRegistered=true;
        //bool specialOption=false;//return value, alert main program that hard coded option (help, version, license, doxygen) was invoked
        if(!helpStringShort.compare(currentOption))
          continue;
        if(!helpStringLong.compare(currentOption))
          continue;
        if(!helpStringDoxygen.compare(currentOption))
          continue;
        if(!helpStringDoxypar.compare(currentOption))
          continue;
        if(!versionString.compare(currentOption))
          continue;
        if(!licenseString.compare(currentOption))
          continue;
        if(isLongOption){
          currentRegistered=false;
          for(unsigned int ireg=0;ireg<m_longRegistered.size();++ireg){
            std::string longOption=m_longRegistered[ireg];
            longOption.insert(0,"--");
            size_t foundEqual=currentOption.rfind("=");
            if(foundEqual!=std::string::npos){
              currentArgument=currentOption.substr(foundEqual+1);
              currentOption=currentOption.substr(0,foundEqual);
            }
            if(longOption==currentOption){
              currentRegistered=true;
              break;
            }
          }
          if(!currentRegistered){
            badKeys.push_back(currentOption.substr(2));
          }
        }
        else if(isShortOption){//short option
          currentRegistered=false;
          for(unsigned int ireg=0;ireg<m_shortRegistered.size();++ireg){
            std::string shortOption=m_shortRegistered[ireg];
            shortOption.insert(0,"-");
            size_t foundEqual=currentOption.rfind("=");
            if(foundEqual!=std::string::npos){
              currentArgument=currentOption.substr(foundEqual+1);
              currentOption=currentOption.substr(0,foundEqual);
            }
            if(shortOption==currentOption){
              currentRegistered=true;
              break;
            }
          }
          if(!currentRegistered){
            badKeys.push_back(currentOption.substr(1));
          }
        }
        else//not an option
          continue;
      }
      return(badKeys.size());
    }
    ///register short option
    void registerShortOption(const std::string& shortName){
      m_shortRegistered.push_back(shortName);
    }
    ///register long option
    void registerLongOption(const std::string& longName){
      m_longRegistered.push_back(longName);
    }
    ///set all options from argc and argv
    void setOptions(int argc, char* argv[]){
      m_argc=argc;
      m_argv.clear();
      for(int iarg=0;iarg<argc;++iarg)
        m_argv.push_back(argv[iarg]);
    }
    ///push bool option (used as flag)
    void pushShortOption(const std::string &key)
    {
      std::ostringstream os;
      /* // os << "--" << key;; */
      /* if(key=="help") */
      /*   os << "--" << key; */
      /* else */
      os << "-" << key;
      m_argv.push_back(os.str().c_str());
      ++m_argc;
    };
    ///push bool long option (used as flag)
    void pushLongOption(const std::string &key)
    {
      std::ostringstream os;
      os << "--" << key;
      m_argv.push_back(os.str().c_str());
      ++m_argc;
    };
    ///set short bool option (used as flag)
    void setShortOption(const std::string &key)
    {
      clearOption(key);
      std::ostringstream os;
      /* // os << "--" << key;; */
      /* if(key=="help") */
      /*   os << "--" << key; */
      /* else */
      os << "-" << key;
      m_argv.push_back(os.str().c_str());
      ++m_argc;
    };
    ///set long bool option (used as flag)
    void setLongOption(const std::string &key)
    {
      clearOption(key);
      std::ostringstream os;
      os << "--" << key;;
      m_argv.push_back(os.str().c_str());
      ++m_argc;
    };
    //template function to set short option
    template<typename T> void setShortOption(const std::string &key, const T &value){
      setShortOption(key,type2string<T>(value));
    }
    //template function to set long option
    template<typename T> void setLongOption(const std::string &key, const T &value){
      setLongOption(key,type2string<T>(value));
    }
    ///set key value short option
    void setShortOption(const std::string &key, const std::string &value)
    {
      clearOption(key);
      std::ostringstream os;
      os << "-" << key;
      m_argv.push_back(os.str());
      ++m_argc;
      m_argv.push_back(value);
      ++m_argc;
    };
    ///set key value long option
    void setLongOption(const std::string &key, const std::string &value)
    {
      clearOption(key);
      std::ostringstream os;
      os << "--" << key;
      m_argv.push_back(os.str());
      ++m_argc;
      m_argv.push_back(value);
      ++m_argc;
    };
    ///push key value short option
    void pushShortOption(const std::string &key, const std::string &value)
    {
      std::ostringstream os;
      os << "-" << key;
      m_argv.push_back(os.str());
      ++m_argc;
      m_argv.push_back(value);
      ++m_argc;
    };
    ///push key value short option template version
    template<typename T> void pushShortOption(const std::string &key, T &value){
      pushShortOption(key,type2string<T>(value));
    }
    ///push key value long option
    void pushLongOption(const std::string &key, const std::string &value)
    {
      std::ostringstream os;
      os << "--" << key;
      m_argv.push_back(os.str());
      ++m_argc;
      /* if(value.find(" ")!=std::string::npos){ */
      /*   m_argv.push_back("\"" + value + "\""); */
      /*   //from C++14:  */
      /*   /\* std::stringstream ss; *\/ */
      /*   /\* ss << std::quoted(value); *\/ */
      /*   /\* m_argv.push_back(ss.str()); *\/ */
      /* } */
      /* else */
      m_argv.push_back(value);
      ++m_argc;
    };
    ///push key value long option template version
    template<typename T> void pushLongOption(const std::string &key, T &value){
      pushLongOption(key,type2string<T>(value));
    }
    void getHelp() {setLongOption("help");};
    void clearOptions() {m_argc=1;m_argv.clear();m_argv.push_back("appFactory");};
    void clearOption(const std::string &key)
    {
      std::vector<std::string>::iterator opit=m_argv.begin();
      while(opit!=m_argv.end()){
        if(opit->find("-"+key)!=std::string::npos){
          m_argv.erase(opit);
          --m_argc;
          if(opit!=m_argv.end()){
            if(opit->find("-")==std::string::npos){//not a bool option
              m_argv.erase(opit);
              --m_argc;
            }
          }
        }
        else
          ++opit;
      }
    };
    void showOptions() const
    {
      for(unsigned int iarg=1;iarg<m_argv.size();++iarg)
        std::cout << m_argv[iarg] << " ";
      std::cout << std::endl;
    };
    int getArgc() const {return m_argc;};
    std::string getArgv(unsigned int i) const {
      if((i>0)&&(i<m_argv.size()))
        return m_argv[i];
      else
        throw(std::string("Error: invalid index"));
    }
    std::vector<std::string> getArgv() const {
      return m_argv;
    }
    std::vector<std::string> getArgv() {
      return m_argv;
    }
    Json::Value app2json(){
      Json::Value jsonobj;
      jsonobj["argc"] = getArgc();
      std::vector<std::string> argv = getArgv();
      Json::Value strarray;
      for(size_t i=0; i<argv.size(); i++) strarray.append(argv[i]);
      jsonobj["argv"] = strarray;
      return jsonobj;
    }
    bool json2app(Json::Value &jsonobj){
      /* clearOptions(); */
      m_argc=0;
      m_argv.clear();
      m_argc=jsonobj["argc"].asInt();
      const Json::Value strarray = jsonobj["argv"];
      for(unsigned int index = 0; index<strarray.size(); ++index) m_argv.push_back(strarray[index].asString());
      return true;
    }
  private:
    //todo: create member attribute for pointer to memory buffer?
    int m_argc;
    std::vector<std::string> m_argv;
    std::vector<std::string> m_longRegistered;
    std::vector<std::string> m_shortRegistered;
  };
}

#endif /* _APPFACTORY_H_ */
