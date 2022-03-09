/**********************************************************************
jldsm_lib.h: perform digital elevation model data operations on image
Author(s): Pieter.Kempeneers@ec.europa.eu
Copyright (C) 2016-2022 European Union (Joint Research Centre)

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
#ifndef _JLDEM_LIB_H_
#define _JLDEM_LIB_H_
#include "imageclasses/Jim.h"

#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif

#ifndef DEG2RAD
#define DEG2RAD(DEG) (DEG/180.0*PI)
#endif

#ifndef RAD2DEG
#define RAD2DEG(RAD) (RAD/PI*180)
#endif

template<class T1, class T2> void Jim::hillShade_t(Jim& imgWriter, Jim& sza, Jim& saa){
  T1* psrc=static_cast<T1*>(getDataPointer());
  T2* psza=static_cast<T2*>(sza.getDataPointer());
  T2* psaa=static_cast<T2*>(saa.getDataPointer());
  unsigned char* ptrt=static_cast<unsigned char*>(imgWriter.getDataPointer());
  double pixelSize = getDeltaX();
  for(size_t index=0;index<nrOfCol()*nrOfRow();++index){
    double currentValue = psrc[index];
    ptrt[index]=0;
    double tansza = tan(DEG2RAD(psza[index]));
    double product = currentValue*tansza/pixelSize;
    int theDist=static_cast<int>(sqrt(product*product));//in pixels
    // int theDist=static_cast<int>(sqrt((currentValue*tan(DEG2RAD(psza[index]))/pixelSize)*(currentValue*tan(DEG2RAD(psza[index]))/pixelSize)));//in pixels
    double theDir=DEG2RAD(psaa[index])+PI/2.0;
    if(theDir<0)
      theDir+=2*PI;
    std::div_t dv{};
    dv = std::div(index, nrOfCol());
    double x = dv.rem;
    double y = dv.quot;
    for(int d=0;d<theDist;++d){//d in pixels
      size_t indexI=x+d*cos(theDir);//in pixels
      size_t indexJ=y+d*sin(theDir);//in pixels
      size_t pos = (indexJ)*nrOfCol() + indexI;
      if(indexJ<0||indexJ>=nrOfRow())
        continue;
      if(indexI<0||indexI>=nrOfCol())
        continue;
      if(tansza > 0 && (psrc[pos]<currentValue-d*pixelSize/tansza))//in m
        ptrt[pos]=1;
    }
  }
}
#endif // _JLDEM_LIB_H_
