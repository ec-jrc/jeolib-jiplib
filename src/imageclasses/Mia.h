/**********************************************************************
Mia.h: class to test MIA functions in C++
Author(s): Pieter.Kempeneers@ec.europa.eu Pierre.Soille@ec.europa.eu
Copyright (c) 2016-2019 European Union (Joint Research Centre)
License EUPLv1.2

This file is part of jiplib
***********************************************************************/
#ifndef _JMIA_H_
#define _JMIA_H_

#include <vector>
#include <queue>
#include <limits>
#include "Jim.h"

/* #define  FICT_PIX  1 */

void set_seq_shift(long int nx, long int ny, long int nz, long int graph, long int *shift);

template<typename T> void Jim::d_jlframebox_t(std::vector<int> box, T val, std::size_t band){
  if(box.size()!=6){
    std::ostringstream s;
    s << "Error: box should be of size 6";
    throw(s.str());
  }
  long int nx = nrOfCol();
  long int ny = nrOfRow();
  long int nz = nrOfPlane();
  long int l1,l2;
  /* check the validity of input parmaters */
  if (box[0] > nx || box[1] > nx ||
      box[2] > ny || box[3] > ny ||
      box[4] > nz || box[5] > nz){
    std::ostringstream s;
    s << "Error: box parameters out of bounds";
    throw(s.str());
  }

  T *p;
  T* pdata=static_cast<T*>(getDataPointer(static_cast<int>(band)));

  long int x;
  long int y;
  long int z;

  l1 = box[0]; l2 = box[1];	/* left and right borders */
  for (z = 0; z < nz; z++){
    for (y = 0; y < ny; y++){
      p = pdata + z * nx * ny + y * nx;
      for (x = 0; x < l1; x++)
        *p++ = val;
      p = pdata + z * nx * ny + y * nx + nx - l2;
      for (x = 0; x < l2; x++)
        *p++ = val;
    }
  }
  l1 = box[2] * nx; l2 = box[3] * nx;	/* top and bottom borders */
  for (z = 0; z < nz; z++){
    p = pdata + z * nx * ny;
    for (x = 0; x < l1; x++)
      *p++ = val;
    p = pdata + z * nx * ny + nx * (ny - box[3]);
    for (x = 0; x < l2; x++)
      *p++ = val;
  }

  l1 = box[4] * nx * ny; l2 = box[5] * nx * ny;	/* up and down borders */
  p = pdata;
  for (x = 0; x < l1; x++)
    *p++ = val;
  p = pdata + nx * ny * (nz - box[5]);
  for (x = 0; x < l2; x++)
    *p++ = val;
}

template<typename T> void Jim::d_jldistanceGeodesic_t(Jim& reference, std::size_t graph, std::size_t band){
  long int nx = nrOfCol();
  long int ny = nrOfRow();
  long int nz = nrOfPlane();
  long int shft[27];

  T* pm;
  unsigned char* pr;
  long int i, k;
  std::queue<std::intptr_t> stlq;

  std::vector<int> box(6,1);
  if(nrOfPlane()==1){
    box[4]=0;
    box[5]=0;
  }

  /* set shift array */
  set_seq_shift(nx, ny, nz, graph, shft);

  if(reference.getDataType()!=GDT_Byte){
    std::ostringstream s;
    s << "Error: data type of reference should be GDT_Byte";
    throw(s.str());
  }
  pm=static_cast<T*>(getDataPointer(static_cast<int>(band)));
  pr=static_cast<unsigned char*>(reference.getDataPointer(static_cast<int>(band)));

  for (i=0; i < nx*ny*nz; ++i){
    if(pm){
      *pm ^= (T)(*pr);
      if (*pm)
        *pm = std::numeric_limits<T>::max();
    }
    pm++;
    pr++;
  }

  d_jlframebox_t<T>(box,static_cast<T>(0),band);

  pm=static_cast<T*>(getDataPointer(static_cast<int>(band)));
  pr=static_cast<unsigned char*>(reference.getDataPointer(static_cast<int>(band)));

  for (i=0; i < nx*ny*nz; ++i){
    if (*pr){
      for (k = 0; k < graph; k++){
        if(i+shft[k]>=0 && i+shft[k]<nx*ny*nz){
          if (*(pm + shft[k]) == std::numeric_limits<T>::max()){
            *(pm + shft[k]) = std::numeric_limits<T>::max()-1;
            stlq.push(reinterpret_cast<std::intptr_t>(pm+shft[k]));
          }
        }
      }
    }
    pm++;
    pr++;
  }

  T dcrt = 0;
  bool init=true;
  while(!stlq.empty()){
    if(init){
      stlq.push(1L);
      dcrt++;
      init=false;
    }
    if(stlq.front()==1L){
      stlq.pop();
      init=true;
      continue;
    }
    pm=(T*)(stlq.front());
    stlq.pop();
    *pm = dcrt;
    for (k=0; k < graph; ++k){
      if(pm + shft[k]){
        if (*(pm + shft[k]) == std::numeric_limits<T>::max()){
          *(pm + shft[k]) = std::numeric_limits<T>::max()- 1;
          stlq.push(reinterpret_cast<std::intptr_t>(pm + shft[k]));
        }
      }
    }
  }
}

#endif _JMIA_H_
