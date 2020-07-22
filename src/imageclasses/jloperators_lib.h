/**********************************************************************
jloperators.h: operators for Jim
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
