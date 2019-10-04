/**********************************************************************
jlfilter1d_lib.h: program to filter raster images: median, min/max, morphological, filtering
Author(s): Pieter.Kempeneers@ec.europa.eu
Copyright (c) 2016-2019 European Union (Joint Research Centre)
License EUPLv1.2

This file is part of jiplib
***********************************************************************/
#ifndef _JLFILTER1D_LIB_H_
#define _JLFILTER1D_LIB_H_

#include "imageclasses/Jim.h"
#include "algorithms/Filter.h"
#include "apps/AppFactory.h"

/* enum FILTER_TYPE { median=100, var=101 , min=102, max=103, sum=104, mean=105, minmax=106, dilate=107, erode=108, close=109, open=110, homog=111, sobelx=112, sobely=113, sobelxy=114, sobelyx=115, smooth=116, density=117, mode=118, mixed=119, threshold=120, ismin=121, ismax=122, heterog=123, order=124, stdev=125, mrf=126, dwt=127, dwti=128, dwt_cut=129, scramble=130, shift=131, linearfeature=132, smoothnodata=133, countid=134, dwt_cut_from=135, savgolay=136, percentile=137, proportion=138, nvalid=139, sauvola=140,first=141,last=142, minindex=143, maxindex=144}; */

/* static FILTER_TYPE getFilterType(const std::string filterType){ */
/*   std::map<std::string, filter::FILTER_TYPE> filterMap; */
/*   filterMap["median"]=filter::median; */
/*   filterMap["var"]=filter::var; */
/*   filterMap["min"]=filter::min; */
/*   filterMap["max"]=filter::max; */
/*   filterMap["sum"]=filter::sum; */
/*   filterMap["mean"]=filter::mean; */
/*   filterMap["threshold"]=filter::threshold; */
/*   filterMap["ismin"]=filter::ismin; */
/*   filterMap["ismax"]=filter::ismax; */
/*   filterMap["order"]=filter::order; */
/*   filterMap["stdev"]=filter::stdev; */
/*   filterMap["scramble"]=filter::scramble; */
/*   filterMap["countid"]=filter::countid; */
/*   filterMap["percentile"]=filter::percentile; */
/*   filterMap["proportion"]=filter::proportion; */
/*   filterMap["nvalid"]=filter::nvalid; */
/*   filterMap["first"]=filter::first; */
/*   filterMap["last"]=filter::last; */
/*   return filterMap[filterType]; */
/* }; */

template<typename T> void Jim::firfilter1d_t(Jim& imgWriter, app::AppFactory& app){
  Optionjl<double> taps_opt("taps", "taps", "taps used for spectral filtering");
  Optionjl<std::string> padding_opt("pad","pad", "Padding method for filtering (how to handle edge effects). Choose between: symmetric, replicate, circular, zero (pad with 0).", "symmetric");
  Optionjl<short> verbose_opt("v", "verbose", "verbose mode if > 0", 0,2);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=taps_opt.retrieveOption(app);
    padding_opt.retrieveOption(app);
    verbose_opt.retrieveOption(app);
    if(!doProcess){
      std::cout << std::endl;
      std::ostringstream helpStream;
      helpStream << "short option -h shows basic options only, use long option --help to show all options" << std::endl;
      throw(helpStream.str());//help was invoked, stop processing
    }

#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
    for(size_t iband=0;iband<nrOfBand();++iband){
      T* pin=static_cast<T*>(getDataPointer(iband));
      double* pout=static_cast<double*>(imgWriter.getDataPointer(iband));

      filter::Filter filter1d;
      if(verbose_opt[0])
        std::cout << "set padding" << std::endl;
      filter1d.setPadding(padding_opt[0]);
      if(taps_opt.size()>nrOfPlane()){
        std::ostringstream errorStream;
        errorStream << "Error: number of taps must be < " << nrOfPlane() << std::endl;
        throw(errorStream.str());//error was invoked, stop processing
      }

      if(verbose_opt[0]){
        std::cout << "taps: " << std::endl;
        std::cout << taps_opt << std::endl;
      }
      for(size_t index=0;index<nrOfCol()*nrOfRow();++index){
        size_t i=0;
        //start: extend pin by padding
        for(i=0;i<taps_opt.size()/2;++i){
          pout[index+i*nrOfCol()*nrOfRow()]=taps_opt[taps_opt.size()/2]*pin[index+i*nrOfCol()*nrOfRow()];
          //todo:introduce nodata?
          //todo: check if t<taps_opt.size() or <= ?
          for(int t=1;t<taps_opt.size()/2;++t){
            pout[index+i*nrOfCol()*nrOfRow()]+=taps_opt[taps_opt.size()/2+t]*pin[index+(i+t)*nrOfCol()*nrOfRow()];
            if(i>=t)
              pout[index+i*nrOfCol()*nrOfRow()]+=taps_opt[taps_opt.size()/2-t]*pin[index+(i-t)*nrOfCol()*nrOfRow()];
            else{
              switch(filter1d.getPadding(padding_opt[0])){
              case(filter::replicate):
                pout[index+i*nrOfCol()*nrOfRow()]+=taps_opt[taps_opt.size()/2-t]*pin[index];
                break;
              case(filter::circular):
                pout[index+i*nrOfCol()*nrOfRow()]+=taps_opt[taps_opt.size()/2-t]*pin[index+(nrOfPlane()+i-t)*nrOfCol()*nrOfRow()];
                break;
              case(filter::zero):
                pout[index+i*nrOfCol()*nrOfRow()]+=taps_opt[taps_opt.size()/2-t]*0;
                break;
              case(filter::symmetric):
              default:
                pout[index+i*nrOfCol()*nrOfRow()]+=taps_opt[taps_opt.size()/2-t]*pin[index+(t-i)*nrOfCol()*nrOfRow()];
                break;
              }
            }
          }
        }
        //main
        for(i=taps_opt.size()/2;i<nrOfPlane()-taps_opt.size()/2;++i){
          //todo:introduce nodata
          pout[index+i*nrOfCol()*nrOfRow()]=0;
          for(int t=0;t<taps_opt.size();++t)
            pout[index+i*nrOfCol()*nrOfRow()]+=pin[index+(i-taps_opt.size()/2+t)*nrOfCol()*nrOfRow()]*taps_opt[t];
        }
        //end: extend pin by padding
        for(i=nrOfPlane()-taps_opt.size()/2;i<nrOfPlane();++i){
          //todo:introduce nodata?
          pout[index+i*nrOfCol()*nrOfRow()]=taps_opt[taps_opt.size()/2]*pin[index+i*nrOfCol()*nrOfRow()];
          //todo:introduce nodata?
          //todo: check if t<taps_opt.size() or <= ?
          for(int t=1;t<taps_opt.size()/2;++t){
            pout[index+i*nrOfCol()*nrOfRow()]+=taps_opt[taps_opt.size()/2-t]*pin[index+(i-t)*nrOfCol()*nrOfRow()];
            if(i+t<nrOfPlane())
              pout[index+i*nrOfCol()*nrOfRow()]+=taps_opt[taps_opt.size()/2+t]*pin[index+(i+t)*nrOfCol()*nrOfRow()];
            else{
              switch(filter1d.getPadding(padding_opt[0])){
              case(filter::replicate):
                pout[index+i*nrOfCol()*nrOfRow()]+=taps_opt[taps_opt.size()/2+t]*pin[index+(nrOfPlane()-1)*nrOfCol()*nrOfRow()];
                break;
              case(filter::circular):
                pout[index+i*nrOfCol()*nrOfRow()]+=taps_opt[taps_opt.size()/2+t]*pin[index+(t-1)*nrOfCol()*nrOfRow()];
                break;
              case(filter::zero):
                pout[index+i*nrOfCol()*nrOfRow()]+=taps_opt[taps_opt.size()/2+t]*0;
                break;
              case(filter::symmetric):
              default:
                pout[index+i*nrOfCol()*nrOfRow()]+=taps_opt[taps_opt.size()/2+t]*pin[index+(i-t)*nrOfCol()*nrOfRow()];
                break;
              }
            }
          }
        }
      }
    }
  }
  catch(std::string predefinedString){
    std::cout << predefinedString << std::endl;
    throw;
  }
}

template<typename T> void Jim::smoothNoData1d_t(Jim& imgWriter, app::AppFactory& app){
  Optionjl<double> nodata_opt("nodata", "nodata", "nodata to interpolate",0);
  Optionjl<std::string> interpolationType_opt("interp", "interp", "type of interpolation for spectral filtering (see http://www.gnu.org/software/gsl/manual/html_node/Interpolation-Types.html)","akima");

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=interpolationType_opt.retrieveOption(app);
    nodata_opt.retrieveOption(app);
    if(!doProcess){
      std::cout << std::endl;
      std::ostringstream helpStream;
      helpStream << "short option -h shows basic options only, use long option --help to show all options" << std::endl;
      throw(helpStream.str());//help was invoked, stop processing
    }
    if(nrOfPlane()<2){
      std::ostringstream errorStream;
      errorStream << "Error: not a 3D object, consider band2plane" << std::endl;
      throw(errorStream.str());
    }
    statfactory::StatFactory stat;
    stat.setNoDataValues(nodata_opt);
    std::vector<double> abscis(nrOfPlane());
    for(int i=0;i<abscis.size();++i)
      abscis[i]=i;
#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
    for(size_t iband=0;iband<nrOfBand();++iband){
      T* pin=static_cast<T*>(getDataPointer(iband));
      T* pout=static_cast<T*>(imgWriter.getDataPointer(iband));
      for(size_t index=0;index<nrOfCol()*nrOfRow();++index){
        std::vector<double> input(nrOfPlane());
        std::vector<double> output(imgWriter.nrOfPlane());
        for(size_t iplane=0;iplane<nrOfPlane();++iplane)
          input[iplane]=static_cast<double>(pin[index+iplane*nrOfCol()*nrOfRow()]);
        stat.interpolateNoData(abscis,input,interpolationType_opt[0],output);
        for(size_t iplane=0;iplane<nrOfPlane();++iplane)
          pout[index+iplane*nrOfCol()*nrOfRow()]=static_cast<T>(output[iplane]);
      }
    }
  }
  catch(std::string predefinedString ){
    std::cout << predefinedString << std::endl;
    throw;
  }
}

template<typename T> void Jim::stats1d_t(Jim& imgWriter, app::AppFactory& app){
  Optionjl<double> nodata_opt("nodata", "nodata", "nodata to interpolate",0);
  Optionjl<std::string> methods_opt("methods", "methods", "statistical method");
  Optionjl<double> threshold_opt("t", "threshold", "threshold value(s) to use for threshold filter (one for each class), or threshold to cut for dwt_cut (use 0 to keep all) or dwt_cut_from", 0);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=methods_opt.retrieveOption(app);
    nodata_opt.retrieveOption(app);
    threshold_opt.retrieveOption(app);
    if(!doProcess){
      std::cout << std::endl;
      std::ostringstream helpStream;
      helpStream << "short option -h shows basic options only, use long option --help to show all options" << std::endl;
      throw(helpStream.str());//help was invoked, stop processing
    }
    if(methods_opt.empty()){
      std::ostringstream errorStream;
      errorStream << "Error: no method provided" << std::endl;
      throw(errorStream.str());
    }
    if(nrOfBand()>1){
      std::ostringstream errorStream;
      errorStream << "Error: multi-band input not supported"<< std::endl;
      throw(errorStream.str());
    }
    if(nrOfPlane()<2){
      std::ostringstream errorStream;
      errorStream << "Error: not a 3D object, consider band2plane" << std::endl;
      throw(errorStream.str());
    }
    imgWriter.open(nrOfCol(),nrOfRow(),methods_opt.size(),1,getGDALDataType());
    imgWriter.setProjection(this->getProjection());
    double gt[6];
    this->getGeoTransform(gt);
    imgWriter.setGeoTransform(gt);
    statfactory::StatFactory stat;
    stat.setNoDataValues(nodata_opt);
    std::vector<double> abscis(nrOfPlane());
    for(int i=0;i<abscis.size();++i)
      abscis[i]=i;
    T* pin=static_cast<T*>(getDataPointer(0));
    std::vector<T*> pout(methods_opt.size());
#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
    for(size_t imethod=0;imethod<methods_opt.size();++imethod){
      pout[imethod]=static_cast<T*>(imgWriter.getDataPointer(imethod));
      for(size_t index=0;index<nrOfCol()*nrOfRow();++index){
        std::vector<double> input(nrOfPlane());
        for(size_t iplane=0;iplane<nrOfPlane();++iplane)
          input[iplane]=static_cast<double>(pin[index+iplane*nrOfCol()*nrOfRow()]);
        for(int imethod=0;imethod<methods_opt.size();++imethod){
          switch(filter::Filter::getFilterType(methods_opt[imethod])){
          case(filter::first):
            pout[imethod][index]=input[0];
            break;
          case(filter::last):
            pout[imethod][index]=input.back();
            break;
          case(filter::nvalid):
            pout[imethod][index]=stat.nvalid(input);
            break;
          case(filter::median):
            pout[imethod][index]=stat.median(input);
            break;
          case(filter::min):
            pout[imethod][index]=stat.mymin(input);
            break;
          case(filter::max):
            pout[imethod][index]=stat.mymax(input);
            break;
          case(filter::sum):
            pout[imethod][index]=stat.sum(input);
            break;
          case(filter::var):
            pout[imethod][index]=stat.var(input);
            break;
          case(filter::stdev):
            pout[imethod][index]=sqrt(stat.var(input));
            break;
          case(filter::mean):
            pout[imethod][index]=stat.mean(input);
            break;
          case(filter::percentile):{
            /* double threshold=(ithreshold<m_threshold.size())? m_threshold[ithreshold] : m_threshold[0]; */
            pout[imethod][index]=stat.percentile(input,input.begin(),input.end(),threshold_opt[0]);
            break;
          }
          default:
            std::string errorString="method not supported";
            throw(errorString);
            break;
          }
        }
      }
    }
  }
  catch(std::string predefinedString ){
    std::cout << predefinedString << std::endl;
    throw;
  }
}

#endif // _JLFILTER1D_LIB_H_
