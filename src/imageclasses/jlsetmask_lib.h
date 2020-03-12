/**********************************************************************
jlsetmask.h:  program to apply mask image (set invalid values) to raster image
Author(s): Pieter.Kempeneers@ec.europa.eu
Copyright (c) 2016-2019 European Union (Joint Research Centre)
License EUPLv1.2

This file is part of jiplib
***********************************************************************/
#ifndef _JLSETMASK_LIB_H_
#define _JLSETMASK_LIB_H_

#include "Jim.h"

template<typename T1, typename T2> void Jim::d_setMask_t(Jim& mask, Jim& other){
#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
  for(size_t iband=0;iband<nrOfBand();++iband){
    unsigned char* pmask=0;
    T1* pim=static_cast<T1*>(getDataPointer(iband));
    if(mask.nrOfBand()<nrOfBand())
      pmask=static_cast<unsigned char*>(mask.getDataPointer(0));
    else
      pmask=static_cast<unsigned char*>(mask.getDataPointer(iband));
    T2* pother=0;
    if(other.nrOfBand()<nrOfBand())
      pother=static_cast<T2*>(other.getDataPointer(0));
    else
      pother=static_cast<T2*>(other.getDataPointer(iband));

    for(size_t iplane=0;iplane<nrOfPlane();++iplane){
      for(size_t index=nrOfCol()*nrOfRow()*iplane;index<nrOfCol()*nrOfRow()*(iplane+1);++index){
        if(*pmask>0)
          *pim=*pother;
        ++pim;
        ++pother;
        ++pmask;
      }
      if(mask.nrOfPlane()<nrOfPlane()){
        if(mask.nrOfBand()<nrOfBand())
          pmask=static_cast<unsigned char*>(mask.getDataPointer(0));
        else
          pmask=static_cast<unsigned char*>(mask.getDataPointer(iband));
      }
      if(other.nrOfPlane()<nrOfPlane()){
        if(other.nrOfBand()<nrOfBand())
          pother=static_cast<T2*>(other.getDataPointer(0));
        else
          pother=static_cast<T2*>(other.getDataPointer(iband));
      }
    }
  }
}

template<typename T> void Jim::d_setMask_t(Jim& mask, double value){
#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
  for(size_t iband=0;iband<nrOfBand();++iband){
    unsigned char* pmask=0;
    T* pim=static_cast<T*>(getDataPointer(iband));
    if(mask.nrOfBand()<nrOfBand())
      pmask=static_cast<unsigned char*>(mask.getDataPointer(0));
    else
      pmask=static_cast<unsigned char*>(mask.getDataPointer(iband));
    for(size_t iplane=0;iplane<nrOfPlane();++iplane){
      for(size_t index=nrOfCol()*nrOfRow()*iplane;index<nrOfCol()*nrOfRow()*(iplane+1);++index){
        if(*pmask>0)
          *pim=static_cast<T>(value);
        ++pim;
        ++pmask;
      }
      if(mask.nrOfPlane()<nrOfPlane()){
        if(mask.nrOfBand()<nrOfBand())
          pmask=static_cast<unsigned char*>(mask.getDataPointer(0));
        else
          pmask=static_cast<unsigned char*>(mask.getDataPointer(iband));
      }
    }
  }
}

#endif // _JLSETMASK_LIB_H
