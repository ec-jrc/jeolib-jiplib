/**********************************************************************
jldem_lib.cc: perform digital elevation model operations on image
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
#include <assert.h>
#include <string>
#include <iostream>
#include <algorithm>
#include <memory>
#include "imageclasses/Jim.h"
#include "jldem_lib.h"

using namespace std;
using namespace app;

shared_ptr<Jim> Jim::hillShade(Jim& sza, Jim& saa){
  shared_ptr<Jim> imgWriter=Jim::createImg();
  hillShade(*imgWriter, sza, saa);
  return(imgWriter);
}

void Jim::hillShade(Jim& imgWriter, Jim& sza, Jim& saa){
  assert(sza.getDataType() == saa.getDataType());
  imgWriter.open(nrOfCol(),nrOfRow(), 1, 1, GDT_Byte);
  imgWriter.copyGeoTransform(*this);
  imgWriter.setProjection(this->getProjection());
  switch(getDataType()){
  case(GDT_Byte):
    switch(sza.getDataType()){
    case(GDT_Byte):
      hillShade_t<unsigned char,unsigned char>(imgWriter, sza, saa);
      break;
    case(GDT_Int16):
      hillShade_t<unsigned char,short>(imgWriter, sza, saa);
      break;
    case(GDT_UInt16):
      hillShade_t<unsigned char,unsigned short>(imgWriter, sza, saa);
      break;
    case(GDT_Int32):
      hillShade_t<unsigned char,int>(imgWriter, sza, saa);
      break;
    case(GDT_UInt32):
      hillShade_t<unsigned char,unsigned int>(imgWriter, sza, saa);
      break;
    case(GDT_Float32):
      hillShade_t<unsigned char,float>(imgWriter, sza, saa);
      break;
    case(GDT_Float64):
      hillShade_t<unsigned char,double>(imgWriter, sza, saa);
      break;
    default:
      std::string errorString="Error: data type not supported";
      throw(errorString);
      break;
    }
    break;
  case(GDT_Int16):
    switch(sza.getDataType()){
    case(GDT_Byte):
      hillShade_t<short,unsigned char>(imgWriter, sza, saa);
      break;
    case(GDT_Int16):
      hillShade_t<short,short>(imgWriter, sza, saa);
      break;
    case(GDT_UInt16):
      hillShade_t<short,unsigned short>(imgWriter, sza, saa);
      break;
    case(GDT_Int32):
      hillShade_t<short,int>(imgWriter, sza, saa);
      break;
    case(GDT_UInt32):
      hillShade_t<short,unsigned int>(imgWriter, sza, saa);
      break;
    case(GDT_Float32):
      hillShade_t<short,float>(imgWriter, sza, saa);
      break;
    case(GDT_Float64):
      hillShade_t<short,double>(imgWriter, sza, saa);
      break;
    default:
      std::string errorString="Error: data type not supported";
      throw(errorString);
      break;
    }
    break;
  case(GDT_UInt16):
    switch(sza.getDataType()){
    case(GDT_Byte):
      hillShade_t<unsigned short,unsigned char>(imgWriter, sza, saa);
      break;
    case(GDT_Int16):
      hillShade_t<unsigned short,short>(imgWriter, sza, saa);
      break;
    case(GDT_UInt16):
      hillShade_t<unsigned short,unsigned short>(imgWriter, sza, saa);
      break;
    case(GDT_Int32):
      hillShade_t<unsigned short,int>(imgWriter, sza, saa);
      break;
    case(GDT_UInt32):
      hillShade_t<unsigned short,unsigned int>(imgWriter, sza, saa);
      break;
    case(GDT_Float32):
      hillShade_t<unsigned short,float>(imgWriter, sza, saa);
      break;
    case(GDT_Float64):
      hillShade_t<unsigned short,double>(imgWriter, sza, saa);
      break;
    default:
      std::string errorString="Error: data type not supported";
      throw(errorString);
      break;
    }
    break;
  case(GDT_Int32):
    switch(sza.getDataType()){
    case(GDT_Byte):
      hillShade_t<int,unsigned char>(imgWriter, sza, saa);
      break;
    case(GDT_Int16):
      hillShade_t<int,short>(imgWriter, sza, saa);
      break;
    case(GDT_UInt16):
      hillShade_t<int,unsigned short>(imgWriter, sza, saa);
      break;
    case(GDT_Int32):
      hillShade_t<int,int>(imgWriter, sza, saa);
      break;
    case(GDT_UInt32):
      hillShade_t<int,unsigned int>(imgWriter, sza, saa);
      break;
    case(GDT_Float32):
      hillShade_t<int,float>(imgWriter, sza, saa);
      break;
    case(GDT_Float64):
      hillShade_t<int,double>(imgWriter, sza, saa);
      break;
    default:
      std::string errorString="Error: data type not supported";
      throw(errorString);
      break;
    }
    break;
  case(GDT_UInt32):
    switch(sza.getDataType()){
    case(GDT_Byte):
      hillShade_t<unsigned int,unsigned char>(imgWriter, sza, saa);
      break;
    case(GDT_Int16):
      hillShade_t<unsigned int,short>(imgWriter, sza, saa);
      break;
    case(GDT_UInt16):
      hillShade_t<unsigned int,unsigned short>(imgWriter, sza, saa);
      break;
    case(GDT_Int32):
      hillShade_t<unsigned int,int>(imgWriter, sza, saa);
      break;
    case(GDT_UInt32):
      hillShade_t<unsigned int,unsigned int>(imgWriter, sza, saa);
      break;
    case(GDT_Float32):
      hillShade_t<unsigned int,float>(imgWriter, sza, saa);
      break;
    case(GDT_Float64):
      hillShade_t<unsigned int,double>(imgWriter, sza, saa);
      break;
    default:
      std::string errorString="Error: data type not supported";
      throw(errorString);
      break;
    }
    break;
  case(GDT_Float32):
    switch(sza.getDataType()){
    case(GDT_Byte):
      hillShade_t<float,unsigned char>(imgWriter, sza, saa);
      break;
    case(GDT_Int16):
      hillShade_t<float,short>(imgWriter, sza, saa);
      break;
    case(GDT_UInt16):
      hillShade_t<float,unsigned short>(imgWriter, sza, saa);
      break;
    case(GDT_Int32):
      hillShade_t<float,int>(imgWriter, sza, saa);
      break;
    case(GDT_UInt32):
      hillShade_t<float,unsigned int>(imgWriter, sza, saa);
      break;
    case(GDT_Float32):
      hillShade_t<float,float>(imgWriter, sza, saa);
      break;
    case(GDT_Float64):
      hillShade_t<float,double>(imgWriter, sza, saa);
      break;
    default:
      std::string errorString="Error: data type not supported";
      throw(errorString);
      break;
    }
    break;
  case(GDT_Float64):
    switch(sza.getDataType()){
    case(GDT_Byte):
      hillShade_t<double,unsigned char>(imgWriter, sza, saa);
      break;
    case(GDT_Int16):
      hillShade_t<double,short>(imgWriter, sza, saa);
      break;
    case(GDT_UInt16):
      hillShade_t<double,unsigned short>(imgWriter, sza, saa);
      break;
    case(GDT_Int32):
      hillShade_t<double,int>(imgWriter, sza, saa);
      break;
    case(GDT_UInt32):
      hillShade_t<double,unsigned int>(imgWriter, sza, saa);
      break;
    case(GDT_Float32):
      hillShade_t<double,float>(imgWriter, sza, saa);
      break;
    case(GDT_Float64):
      hillShade_t<double,double>(imgWriter, sza, saa);
      break;
    default:
      std::string errorString="Error: data type not supported";
      throw(errorString);
      break;
    }
    break;
  default:
    std::string errorString="Error: data type not supported";
    throw(errorString);
    break;
  }
}
