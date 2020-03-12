/**********************************************************************
jlcrop_lib.h: perform raster data operations on image such as crop, extract and stack bands
Author(s): Pieter.Kempeneers@ec.europa.eu
Copyright (c) 2016-2020 European Union (Joint Research Centre)
License EUPLv1.2

This file is part of jiplib
***********************************************************************/
#ifndef _JLCROP_LIB_H_
#define _JLCROP_LIB_H_
#include "imageclasses/Jim.h"

template<typename T1, typename T2> void Jim::convertDataType_t(Jim& imgWriter, const GDALDataType& dataType){
#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
  for(size_t iband=0;iband<nrOfBand();++iband){
    T1* psrc=static_cast<T1*>(getDataPointer(iband));
    T2* ptrt=static_cast<T2*>(imgWriter.getDataPointer(iband));
    for(size_t index=0;index<nrOfCol()*nrOfRow()*nrOfPlane();++index){
      ptrt[index]=psrc[index];
    }
  }
}

#endif // _JLCROP_LIB_H_
