/**********************************************************************
Mia.c: class to test MIA functions in C++
Author(s): Pieter.Kempeneers@ec.europa.eu Pierre.Soille@ec.europa.eu
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
#include "Mia.h"

void set_seq_shift(long int nx, long int ny, long int nz, long int graph, long int *shift){
  if (graph == 2){      /* added 20110310 for shadow ray processing */
    shift[0] = -1;
    shift[1] = 1;
  }
  else if (graph == 4){      /* && nz == 1: suppressed on 2004-11-18 (side effects?) */
    shift[0] = -nx;
    shift[1] = -1;
    shift[2] = nx;
    shift[3] = 1;
  }
  else if (graph == 8){ /* && nz == 1: suppressed on 2004-11-18 (side effects?) */
    shift[0] = -nx;		shift[1] = -1;
    shift[2] = -nx + 1;	shift[3] = -nx - 1;
    shift[4] = 1;		shift[5] = nx;
    shift[6] = nx - 1;		shift[7] = nx + 1;
  }
  else if (graph == 6 && nz > 1){
    shift[0] = -nx;  shift[1] = -1;
    shift[2] = -(nx * ny);
    shift[3] = nx;   shift[4] = 1;
    shift[5] = -shift[2];
   }
  else if (graph == 18 && nz > 1){
    shift[0] = -nx;   shift[1] = -1;
    shift[2] = shift[0] + 1;         shift[3] = shift[0] - 1;
    shift[4] = -(nx * ny);
    shift[5] = shift[4] + shift[0];   shift[6] = shift[4] - 1;
    shift[7] = shift[4] + 1;         shift[8] = shift[4] - shift[0];
    shift[9] = -shift[0];	           shift[10] = 1;
    shift[11] = -shift[2];           shift[12] = -shift[3];
    shift[13] = -shift[4];           shift[14] = -shift[5];
    shift[15] = -shift[6];           shift[16] = -shift[7];
    shift[17] = -shift[8];
  }
  else if (graph == 26 && nz > 1){
    shift[0] = -nx;  shift[1] = -1;
    shift[2] = shift[0] + 1;        shift[3] = shift[0] - 1;
    shift[4] = -(nx * ny);
    shift[5] = shift[4] + shift[0];  shift[6] = shift[4] - 1;
    shift[7] = shift[4] + 1;        shift[8] = shift[4] - shift[0];
    shift[9] = shift[5] - 1;        shift[10] = shift[5] + 1; 
    shift[11] = shift[8] - 1;       shift[12] = shift[8] + 1; 
    shift[13] = -shift[0];          shift[14] = -shift[1];
    shift[15] = -shift[2];          shift[16] = -shift[3];
    shift[17] = -shift[4];          shift[18] = -shift[5];
    shift[19] = -shift[6];          shift[20] = -shift[7];
    shift[21] = -shift[8];          shift[22] = -shift[9];
    shift[23] = -shift[10];         shift[24] = -shift[11];
    shift[25] = -shift[12];
  }
  else{
    std::ostringstream s;
    s << "Error: set_seq_shift(): invalid parameters";
    throw(s.str());
  }
}

void Jim::d_jlframebox(std::vector<int> box, double val, std::size_t band){
  switch(getDataType()){
  case(GDT_Byte):
    d_jlframebox_t<unsigned char>(box,static_cast<unsigned char>(val),band);
    break;
  case(GDT_Int16):
    d_jlframebox_t<short>(box,static_cast<short>(val),band);
    break;
  case(GDT_UInt16):
    d_jlframebox_t<unsigned short>(box,static_cast<unsigned short>(val),band);
    break;
  case(GDT_Int32):
    d_jlframebox_t<int>(box,static_cast<int>(val),band);
    break;
  case(GDT_UInt32):
    d_jlframebox_t<unsigned int>(box,static_cast<unsigned int>(val),band);
    break;
  case(GDT_Float32):
    d_jlframebox_t<float>(box,static_cast<float>(val),band);
    break;
  case(GDT_Float64):
    d_jlframebox_t<double>(box,static_cast<double>(val),band);
    break;
  default:
    std::string errorString="Error: data type not supported";
    throw(errorString);
    break;
  }
}

void Jim::d_jldistanceGeodesic(Jim& reference, std::size_t graph, std::size_t band){
  switch(getDataType()){
  case(GDT_Byte):
    d_jldistanceGeodesic_t<unsigned char>(reference,graph,band);
    break;
  case(GDT_Int16):
    d_jldistanceGeodesic_t<short>(reference,graph,band);
    break;
  case(GDT_UInt16):
    d_jldistanceGeodesic_t<unsigned short>(reference,graph,band);
    break;
  case(GDT_Int32):
    d_jldistanceGeodesic_t<int>(reference,graph,band);
    break;
  case(GDT_UInt32):
    d_jldistanceGeodesic_t<unsigned int>(reference,graph,band);
    break;
  default:
    std::string errorString="Error: data type not supported";
    throw(errorString);
    break;
  }
}

std::shared_ptr<Jim> Jim::jldistanceGeodesic(Jim& reference, std::size_t graph, std::size_t band){
	try{
    std::shared_ptr<Jim> copyImg=this->clone();
    copyImg->d_jldistanceGeodesic(reference,graph,band);
    return(copyImg);
  }
  catch(std::string errorString){
    std::cerr << errorString << std::endl;
    throw;
  }
  catch(...){
    throw;
  }
}
