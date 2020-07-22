/**********************************************************************
jlfilter2d_lib.h: program to filter raster images in spatial domain
Author(s): Pieter.Kempeneers@ec.europa.eu
Copyright (C) 2016-2020 European Union (Joint Research Centre)

This file is part of jiplib.

jiplib is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

jiplib is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with jiplib.  If not, see <https://www.gnu.org/licenses/>.
***********************************************************************/
#ifndef _JLFILTER2D_LIB_H_
#define _JLFILTER2D_LIB_H_

#include "imageclasses/Jim.h"
#include "apps/AppFactory.h"

//todo: add support for different padding strategies
template<typename T> void Jim::firfilter2d_t(Jim& imgWriter, app::AppFactory& app){
  Optionjl<double> taps_opt("taps", "taps", "taps used for spatial filtering");
  Optionjl<size_t> dimx_opt("dimx", "dimx", "number of taps used for columns");
  Optionjl<size_t> dimy_opt("dimy", "dimy", "number of taps used for rows");
  Optionjl<double> nodata_opt("nodata", "nodata", "list of no data values not taken into account");
  /* Optionjl<std::string> padding_opt("pad","pad", "Padding method for filtering (how to handle edge effects). Choose between: symmetric, replicate, circular, zero (pad with 0).", "symmetric"); */
  /* Optionjl<bool> abs_opt("abs", "abs", "use absolute values when filtering",false); */
  Optionjl<bool> norm_opt("norm", "norm", "normalize tap values values when filtering",false);
  Optionjl<short> verbose_opt("v", "verbose", "verbose mode if > 0", 0,2);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=taps_opt.retrieveOption(app);
    dimx_opt.retrieveOption(app);
    dimy_opt.retrieveOption(app);
    nodata_opt.retrieveOption(app);
    /* abs_opt.retrieveOption(app); */
    norm_opt.retrieveOption(app);
    /* padding_opt.retrieveOption(app); */
    verbose_opt.retrieveOption(app);
    if(!doProcess){
      std::cout << std::endl;
      std::ostringstream helpStream;
      helpStream << "short option -h shows basic options only, use long option --help to show all options" << std::endl;
      throw(helpStream.str());//help was invoked, stop processing
    }

    std::vector<double> taps(taps_opt.begin(),taps_opt.end());

    if(norm_opt[0]&&nodata_opt.empty()){
      if(verbose_opt[0])
        std::cout << "normalizing taps" << std::endl;
      double norm=0;
      for(size_t itap=0;itap<taps.size();++itap){
        norm+=abs(static_cast<double>(taps[itap]));
      }
      if(norm){
        for(size_t itap=0;itap<taps.size();++itap){
          taps[itap]/=norm;
        }
      }
    }
    size_t ncol=nrOfCol();
    size_t nrow=nrOfRow();

    if(dimx_opt.empty()){
      std::ostringstream errorStream;
      errorStream << "Error: could not determine dimx of taps" << std::endl;
      throw(errorStream.str());
    }
    if(dimy_opt.empty()){
      std::ostringstream errorStream;
      errorStream << "Error: could not determine dimy of taps" << std::endl;
      throw(errorStream.str());
    }
    int dimX=dimx_opt[0];//horizontal!!!
    int dimY=dimy_opt[0];//vertical!!!
    if(verbose_opt[0]){
      std::cout << "dimX: " << dimX << std::endl;
      std::cout << "dimY: " << dimY << std::endl;
    }
#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
    for(size_t iband=0;iband<nrOfBand();++iband){
      if(verbose_opt[0])
        std::cout << "processing band " << iband << std::endl;
      T* pin=static_cast<T*>(getDataPointer(iband));
      T* pout=static_cast<T*>(imgWriter.getDataPointer(iband));

      int indexI=0;
      int indexJ=0;
      if(nodata_opt.size()){
        for(int y=0;y<nrow;++y){
          for(int x=0;x<ncol;++x){
            double norm=0;
            for(int j=-(dimY-1)/2;j<=dimY/2;++j){
              for(int i=-(dimX-1)/2;i<=dimX/2;++i){
                indexI=x+i;
                indexJ=y+j;
                //check if out of bounds
                if(x<(dimX-1)/2)
                  indexI=x+abs(i);
                else if(x>=ncol-(dimX-1)/2)
                  indexI=x-abs(i);
                if(y<(dimY-1)/2)
                  indexJ=y+abs(j);
                else if(y>=nrow-(dimY-1)/2)
                  indexJ=y-abs(j);
                //do not take masked values into account
                bool masked=false;
                for(int imask=0;imask<nodata_opt.size();++imask){
                  if(pin[indexJ*ncol+indexI]==nodata_opt[imask]){
                    masked=true;
                    break;
                  }
                }
                if(!masked){
                  norm+=abs(taps[((dimY-1)/2+j)*dimX+(dimX-1)/2+i]);
                  pout[y*ncol+x]+=(taps[((dimY-1)/2+j)*dimX+(dimX-1)/2+i]*pin[indexJ*ncol+indexI]);
                }
              }
            }
            if(norm>0)
              pout[y*ncol+x]/=norm;
          }
        }
      }
      else{
        for(int y=0;y<nrow;++y){
          for(int x=0;x<ncol;++x){
            for(int j=-(dimY-1)/2;j<=dimY/2;++j){
              for(int i=-(dimX-1)/2;i<=dimX/2;++i){
                indexI=x+i;
                indexJ=y+j;
                //check if out of bounds
                if(x<(dimX-1)/2)
                  indexI=x+abs(i);
                else if(x>=ncol-(dimX-1)/2)
                  indexI=x-abs(i);
                if(y<(dimY-1)/2)
                  indexJ=y+abs(j);
                else if(y>=nrow-(dimY-1)/2)
                  indexJ=y-abs(j);
                pout[y*ncol+x]+=(taps[((dimY-1)/2+j)*dimX+(dimX-1)/2+i]*pin[indexJ*ncol+indexI]);
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

#endif // _JLFILTER2D_LIB_H_
