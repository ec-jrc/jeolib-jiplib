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

template<typename T> void Jim::filter3d_t(Jim& imgWriter, app::AppFactory& app){
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

#endif // _JLFILTER1D_LIB_H_
