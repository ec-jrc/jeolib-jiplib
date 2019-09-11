/**********************************************************************
jloperators.h: operators for Jim
Author(s): Pieter.Kempeneers@ec.europa.eu
Copyright (c) 2016-2019 European Union (Joint Research Centre)
License EUPLv1.2

This file is part of jiplib
***********************************************************************/
#ifndef _JLOPERATORS_LIB_H_
#define _JLOPERATORS_LIB_H_

#include "Jim.h"

template<typename T> void Jim::lt_t(Jim& other, Jim& imgWriter){
#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
  for(size_t iband=0;iband<nrOfBand();++iband){
    T* p1=static_cast<T*>(getDataPointer(iband));
    T* p2=static_cast<T*>(other.getDataPointer(iband));
    unsigned char* pout=static_cast<unsigned char*>(imgWriter.getDataPointer(static_cast<int>(iband)));
    for(size_t index=0;index<nrOfCol()*nrOfRow()*nrOfPlane();++index){
      if(*p1<*p2)
        *pout=1;
      else
        *pout=0;
      ++p1;
      ++p2;
      ++pout;
    }
  }
}

template<typename T> void Jim::lt_t(double value, Jim& imgWriter){
#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
  for(size_t iband=0;iband<nrOfBand();++iband){
    T* p1=static_cast<T*>(getDataPointer(iband));
    unsigned char* pout=static_cast<unsigned char*>(imgWriter.getDataPointer(static_cast<int>(iband)));
    for(size_t index=0;index<nrOfCol()*nrOfRow()*nrOfPlane();++index){
      if(*p1<value)
        *pout=1;
      else
        *pout=0;
      ++p1;
      ++pout;
    }
  }
}

#endif // _JLOPERATORS_LIB_H_
