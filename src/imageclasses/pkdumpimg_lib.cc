/**********************************************************************
pkdumpimg_lib.cc: dump image on screen or ASCII file
Copyright (C) 2008-2016 Pieter Kempeneers

This file is part of pktools

pktools is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

pktools is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with pktools.  If not, see <http://www.gnu.org/licenses/>.
***********************************************************************/
#include <string>
#include <iostream>
#include <memory>
#include "imageclasses/ImgRaster.h"
#include "imageclasses/ImgReaderOgr.h"
#include "base/Optionpk.h"
#include "apps/AppFactory.h"

using namespace std;
using namespace app;

/**
 * @param imgWriter output raster dumpimg dataset
 * @return CE_None if successful, CE_Failure if failed
 **/
CPLErr ImgRaster::dumpImg(app::AppFactory& app){

  Optionpk<string> output_opt("o", "output", "Output ascii file (Default is empty: dump to standard output)");
  Optionpk<string> oformat_opt("of", "oformat", "Output format: matrix or list (x,y,z) form. Default is matrix", "matrix");
  Optionpk<bool>  geo_opt("geo", "geo", "Dump x and y in spatial reference system of raster dataset (for list form only)", false);
  Optionpk<int> band_opt("b", "band", "Band index to crop");
  // Optionpk<short> dstnodata_opt("dstnodata", "dstnodata", "nodata value for output if out of bounds.", 0);
  Optionpk<double> srcnodata_opt("srcnodata", "srcnodata", "Do not dump these no data values if oformat is in list form");
  Optionpk<bool> force_opt("f", "force", "Force full dump even for large images (above 100 rows and cols)", false);
  Optionpk<short> verbose_opt("v", "verbose", "verbose", 0,2);

  srcnodata_opt.setHide(1);
  // dstnodata_opt.setHide(1);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=output_opt.retrieveOption(app);
    oformat_opt.retrieveOption(app);
    geo_opt.retrieveOption(app);
    band_opt.retrieveOption(app);
    srcnodata_opt.retrieveOption(app);
    // dstnodata_opt.retrieveOption(app);
    force_opt.retrieveOption(app);
    verbose_opt.retrieveOption(app);
  }
  catch(string predefinedString){
    std::cout << predefinedString << std::endl;
    return(CE_Failure);
  }
  if(!doProcess){
    cout << endl;
    std::ostringstream helpStream;
    helpStream << "short option -h shows basic options only, use long option --help to show all options" << std::endl;
    throw(helpStream.str());//help was invoked, stop processing
  }
  ofstream outputStream;
  if(output_opt.size())
    outputStream.open(output_opt[0].c_str());

  for(int inodata=0;inodata<srcnodata_opt.size();++inodata)
    pushNoDataValue(srcnodata_opt[inodata]);

  if(band_opt.empty()){
    for(int iband=0;iband<nrOfBand();++iband)
      band_opt.push_back(iband);
  }
  std::vector<double> readBuffer(nrOfCol());
  for(int iband=0;iband<band_opt.size();++iband){
    assert(band_opt[iband]>=0);
    assert(band_opt[iband]<nrOfBand());
    for(int irow=0;irow<nrOfRow();++irow){
      if(!force_opt[0]){
        if(irow>10&&irow<nrOfRow()-10)
          continue;
        else if(irow==10){
          std::cout << "..." << std::endl;
          continue;
        }
      }
      readData(readBuffer,irow,band_opt[iband]);
      for(int icol=0;icol<nrOfCol();++icol){
        if(!force_opt[0]){
          if(icol>10&&icol<nrOfCol()-10)
            continue;
          else if(icol==10){
            std::cout << " ... ";
            continue;
          }
        }
        if(oformat_opt[0]=="matrix"){
          if(output_opt.empty())
            std::cout << readBuffer[icol] << " ";
          else
            outputStream << readBuffer[icol] << " ";
        }
        else if(!isNoData(readBuffer[icol])){
          if(geo_opt[0]){
            double ix,iy;
            image2geo(icol,irow,ix,iy);
            if(output_opt.empty())
              std::cout << ix << " " << iy << " " << readBuffer[icol] << std::endl;
            else
              outputStream << ix << " " << iy << " " << readBuffer[icol] << std::endl;
          }
          else{
            if(output_opt.empty())
              std::cout << icol << " " << irow << " " << readBuffer[icol] << std::endl;
            else
              outputStream << icol << " " << irow << " " << readBuffer[icol] << std::endl;
          }
        }
      }
      std::cout << std::endl;
    }
  }
  if(oformat_opt[0]=="matrix"){
    if(output_opt.empty())
      std::cout << std::endl;
    else
      outputStream << std::endl;
  }
  if(!output_opt.empty())
    outputStream.close();
  return(CE_None);
}
