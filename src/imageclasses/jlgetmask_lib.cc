/**********************************************************************
jlgetmask_lib.cc: program to create mask image based on values in input raster image
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
#include <assert.h>
#include <vector>
#include "imageclasses/Jim.h"
#include "base/Optionjl.h"
#include "apps/AppFactory.h"

using namespace std;
using namespace app;

/**
 * @param app application specific option arguments
 * @return output image
 **/
shared_ptr<Jim> Jim::getMask(app::AppFactory& app){
  shared_ptr<Jim> imgWriter=createImg();
  getMask(*imgWriter, app);
  return(imgWriter);
}

/**
 * @param imgWriter output raster getmask dataset
 * @return CE_None if successful, CE_Failure if failed
 **/
void Jim::getMask(Jim& imgWriter, app::AppFactory& app){
  Optionjl<short> band_opt("b", "band", "band(s) used for mask", 0);
  Optionjl<double> min_opt("min", "min", "Values smaller than min threshold(s) are masked as invalid. Use one threshold for each band");
  Optionjl<double> max_opt("max", "max", "Values greater than max threshold(s) are masked as invalid. Use one threshold for each band");
  Optionjl<string> operator_opt("p", "operator", "Operator: [AND,OR].", "OR");
  Optionjl<unsigned short> data_opt("data", "data", "value(s) for valid pixels: between min and max", 1);
  Optionjl<unsigned short> nodata_opt("nodata", "nodata", "value(s) for invalid pixels: not between min and max", 0);
  Optionjl<string> otype_opt("ot", "otype", "Data type for output image ({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}). Empty string: inherit type from input image", "Byte");
  // Optionjl<string> oformat_opt("of", "oformat", "Output image format (see also gdal_translate).","GTiff");
  Optionjl<string> option_opt("co", "co", "Creation option for output file. Multiple options can be specified.");
  Optionjl<string> colorTable_opt("ct", "ct", "color table (file with 5 columns: id R G B ALFA (0: transparent, 255: solid)");
  Optionjl<unsigned long int>  memory_opt("mem", "mem", "Buffer size (in MB) to read image data blocks in memory",0,1);
  Optionjl<short> verbose_opt("v", "verbose", "verbose", 0,2);

  band_opt.setHide(1);
  operator_opt.setHide(1);
  otype_opt.setHide(1);
  // oformat_opt.setHide(1);
  option_opt.setHide(1);
  colorTable_opt.setHide(1);
  memory_opt.setHide(1);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=min_opt.retrieveOption(app);
    max_opt.retrieveOption(app);
    data_opt.retrieveOption(app);
    nodata_opt.retrieveOption(app);
    band_opt.retrieveOption(app);
    operator_opt.retrieveOption(app);
    otype_opt.retrieveOption(app);
    colorTable_opt.retrieveOption(app);
    verbose_opt.retrieveOption(app);

    if(!doProcess){
      cout << endl;
      std::ostringstream helpStream;
      helpStream << "short option -h shows basic options only, use long option --help to show all options" << std::endl;
      throw(helpStream.str());//help was invoked, stop processing
    }

    std::vector<std::string> badKeys;
    app.badKeys(badKeys);
    if(badKeys.size()){
      std::ostringstream errorStream;
      if(badKeys.size()>1)
        errorStream << "Error: unknown keys: ";
      else
        errorStream << "Error: unknown key: ";
      for(int ikey=0;ikey<badKeys.size();++ikey){
        errorStream << badKeys[ikey] << " ";
      }
      errorStream << std::endl;
      throw(errorStream.str());
    }
    GDALDataType theType=GDT_Unknown;
    if(otype_opt.size()){
      theType=string2GDAL(otype_opt[0]);
      if(theType==GDT_Unknown)
        std::cout << "Warning: unknown output pixel type: " << otype_opt[0] << ", using input type as default" << std::endl;
    }
    //if output type not set, get type from input image
    if(theType==GDT_Unknown){
      theType=getGDALDataType();
      if(verbose_opt[0])
        cout << "Using data type from input image: " << GDALGetDataTypeName(theType) << endl;
    }
    if(verbose_opt[0])
      cout << "Output pixel type:  " << GDALGetDataTypeName(theType) << endl;
    const char* pszMessage;
    void* pProgressArg=NULL;
    GDALProgressFunc pfnProgress=GDALTermProgress;
    double progress=0;
    MyProgressFunc(progress,pszMessage,pProgressArg);

    if(band_opt.empty()){
      std::string errorString="Error: band is empty, use option -b";
      throw(errorString);
    }
    for(int iband=0;iband<band_opt.size();++iband){
      if(band_opt[iband]>=nrOfBand()){
        std::string errorString="Error: bands exceed number of band in input image";
        throw(errorString);
      }
    }

    if(min_opt.size()&&max_opt.size()){
      if(min_opt.size()!=max_opt.size()){
        std::string errorString="Error: number of min and max options must correspond if both min and max options are provided";
        throw(errorString);
      }
    }
    if(min_opt.size()){
      while(band_opt.size()>min_opt.size())
        min_opt.push_back(min_opt[0]);
      while(min_opt.size()>data_opt.size())
        data_opt.push_back(data_opt[0]);
    }
    if(max_opt.size()){
      while(band_opt.size()>max_opt.size())
        max_opt.push_back(max_opt[0]);
      while(max_opt.size()>data_opt.size())
        data_opt.push_back(data_opt[0]);
    }

    vector< vector<float> > lineBuffer(band_opt.size());
    for(unsigned int iband=0;iband<band_opt.size();++iband)
      lineBuffer.resize(nrOfCol());
    //if output type not set, get type from input image
    if(theType==GDT_Unknown){
      theType=getGDALDataType();
      if(verbose_opt[0])
        cout << "Using data type from input image: " << GDALGetDataTypeName(theType) << endl;
    }
    // string imageType;//=getImageType();
    // if(oformat_opt.size())//default
    //   imageType=oformat_opt[0];
    // if(option_opt.findSubstring("INTERLEAVE=")==option_opt.end()){
    //   string theInterleave="INTERLEAVE=";
    //   theInterleave+=getInterleave();
    //   option_opt.push_back(theInterleave);
    // }
    imgWriter.open(nrOfCol(),nrOfRow(),1,theType);
    if(colorTable_opt.size()){
      if(colorTable_opt[0]!="none")
        imgWriter.setColorTable(colorTable_opt[0]);
    }
    else if (getColorTable()!=NULL)//copy colorTable from input image
      imgWriter.setColorTable(getColorTable());

    imgWriter.setProjection(getProjection());
    double gt[6];
    getGeoTransform(gt);
    imgWriter.setGeoTransform(gt);//ulx,uly,getDeltaX(),getDeltaY(),0,0);

    if(nodata_opt.size())
      imgWriter.GDALSetNoDataValue(nodata_opt[0]);

    vector<char> writeBuffer(imgWriter.nrOfCol());
    for(unsigned int irow=0;irow<nrOfRow();++irow){
      for(unsigned int iband=0;iband<band_opt.size();++iband)
        readData(lineBuffer[iband],irow,band_opt[iband]);
      for(unsigned int icol=0;icol<nrOfCol();++icol){
        bool valid=(operator_opt[0]=="OR")?false:true;
        unsigned short validValue=data_opt[0];
        if(min_opt.size()&&max_opt.size()){
          assert(max_opt.size()==min_opt.size());
          for(unsigned int ivalid=0;ivalid<min_opt.size();++ivalid){
            bool validBand=false;
            // for(unsigned int iband=0;iband<band_opt.size();++iband){
            unsigned short theBand=(band_opt.size()==min_opt.size())? ivalid:0;
            if(lineBuffer[theBand][icol]>=min_opt[ivalid]&&lineBuffer[theBand][icol]<=max_opt[ivalid]){
              validValue=data_opt[ivalid];
              validBand=true;
            }
            valid=(operator_opt[0]=="OR")?valid||validBand : valid&&validBand;
          }
        }
        else if(min_opt.size()){
          for(int ivalid=0;ivalid<min_opt.size();++ivalid){
            bool validBand=false;
            // for(int iband=0;iband<band_opt.size();++iband){
            unsigned short theBand=(band_opt.size()==min_opt.size())? ivalid:0;
            if(lineBuffer[theBand][icol]>=min_opt[ivalid]){
              validValue=data_opt[ivalid];
              validBand=true;
            }
            valid=(operator_opt[0]=="OR")?valid||validBand : valid&&validBand;
          }
        }
        else if(max_opt.size()){
          for(int ivalid=0;ivalid<max_opt.size();++ivalid){
            bool validBand=false;
            // for(int iband=0;iband<band_opt.size();++iband){
            unsigned short theBand=(band_opt.size()==max_opt.size())? ivalid:0;
            if(lineBuffer[theBand][icol]<=max_opt[ivalid]){
              validValue=data_opt[ivalid];
              validBand=true;
            }
            valid=(operator_opt[0]=="OR")?valid||validBand : valid&&validBand;
          }
        }
        if(valid)
          writeBuffer[icol]=validValue;
        else
          writeBuffer[icol]=nodata_opt[0];
      }
      imgWriter.writeData(writeBuffer,irow);
      progress=(1.0+irow)/imgWriter.nrOfRow();
      MyProgressFunc(progress,pszMessage,pProgressArg);
    }
  }
  catch(string predefinedString){
    std::cout << predefinedString << std::endl;
    throw;
  }
}
