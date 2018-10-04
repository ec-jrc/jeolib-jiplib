/**********************************************************************
jlcrop_lib.cc: perform raster data operations on image such as crop, extract and stack bands
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
#include <assert.h>
#include <string>
#include <iostream>
#include <algorithm>
#include <memory>
#include "imageclasses/Jim.h"
#include "imageclasses/VectorOgr.h"
#include "imageclasses/JimList.h"
#include "base/Optionjl.h"
#include "algorithms/Egcs.h"
#include "algorithms/StatFactory.h"
#include "apps/AppFactory.h"

using namespace std;
using namespace app;

shared_ptr<Jim> Jim::convert(AppFactory& app){
  shared_ptr<Jim> imgWriter=Jim::createImg();
  convert(*imgWriter, app);
  return(imgWriter);
}

/**
 * @param app application specific option arguments
 * @return output image
 **/
shared_ptr<Jim> Jim::crop(AppFactory& app){
  shared_ptr<Jim> imgWriter=Jim::createImg();
  crop(*imgWriter, app);
  return(imgWriter);
}

/**
 * @param app application specific option arguments
 * @return output image
 **/
shared_ptr<Jim> Jim::crop(VectorOgr& sampleReader, AppFactory& app){
  shared_ptr<Jim> imgWriter=Jim::createImg();
  crop(sampleReader, *imgWriter, app);
  return(imgWriter);
}

/**
 * @param app application specific option arguments
 * @return output image
 **/
shared_ptr<Jim> Jim::crop(double ulx, double uly, double lrx, double lry){
  shared_ptr<Jim> imgWriter=Jim::createImg();
  crop(*imgWriter, ulx, uly, lrx, lry);
  return(imgWriter);
}

CPLErr Jim::crop(Jim& imgWriter, double ulx, double uly, double lrx, double lry){
  app::AppFactory app;
  app.setLongOption("ulx",ulx);
  app.setLongOption("uly",uly);
  app.setLongOption("lrx",lrx);
  app.setLongOption("lry",lry);
  return(crop(imgWriter,app));
}


CPLErr Jim::convert(Jim& imgWriter, AppFactory& app){
  Optionjl<string>  projection_opt("a_srs", "a_srs", "Override the projection for the output file (leave blank to copy from input file, use epsg:3035 to use European projection and force to European grid");
  Optionjl<double> autoscale_opt("as", "autoscale", "scale output to min and max, e.g., --autoscale 0 --autoscale 255");
  Optionjl<double> scale_opt("scale", "scale", "output=scale*input+offset");
  Optionjl<double> offset_opt("offset", "offset", "output=scale*input+offset");
  Optionjl<string>  otype_opt("ot", "otype", "Data type for output image ({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}). Empty string: inherit type from input image");
  Optionjl<double>  nodata_opt("nodata", "nodata", "No data value");
  Optionjl<string>  description_opt("d", "description", "Set image description");
  Optionjl<short>  verbose_opt("v", "verbose", "verbose", 0,2);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=projection_opt.retrieveOption(app);
    autoscale_opt.retrieveOption(app);
    otype_opt.retrieveOption(app);
    scale_opt.retrieveOption(app);
    offset_opt.retrieveOption(app);
    nodata_opt.retrieveOption(app);
    description_opt.retrieveOption(app);
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

    double nodataValue=nodata_opt.size()? nodata_opt[0] : 0;
    const char* pszMessage;
    void* pProgressArg=NULL;
    GDALProgressFunc pfnProgress=GDALTermProgress;
    double progress=0;
    MyProgressFunc(progress,pszMessage,pProgressArg);

    GDALDataType theType=getGDALDataType();
    if(otype_opt.size()){
      theType=string2GDAL(otype_opt[0]);
      if(theType==GDT_Unknown)
        std::cout << "Warning: unknown output pixel type: " << otype_opt[0] << ", using input type as default" << std::endl;
    }
    if(verbose_opt[0])
      cout << "Output pixel type:  " << GDALGetDataTypeName(theType) << endl;

    if(scale_opt.size()){
      while(scale_opt.size()<nrOfBand())
        scale_opt.push_back(scale_opt[0]);
    }
    if(offset_opt.size()){
      while(offset_opt.size()<nrOfBand())
        offset_opt.push_back(offset_opt[0]);
    }
    if(autoscale_opt.size()){
      assert(autoscale_opt.size()%2==0);
    }

    if(theType==GDT_Unknown){
      theType=this->getGDALDataType();
      if(verbose_opt[0])
        cout << "Using data type from input image: " << GDALGetDataTypeName(theType) << endl;
    }
    imgWriter.open(nrOfCol(),nrOfRow(),nrOfBand(),theType);
    imgWriter.setNoData(nodata_opt);
    if(nodata_opt.size())
      imgWriter.GDALSetNoDataValue(nodata_opt[0]);
    imgWriter.copyGeoTransform(*this);
    imgWriter.setProjection(this->getProjection());

    if(description_opt.size())
      imgWriter.setImageDescription(description_opt[0]);

    vector<double> readBuffer(nrOfCol());
    vector<double> writeBuffer(nrOfCol());
    unsigned int nband=this->nrOfBand();
    for(unsigned int iband=0;iband<nband;++iband){
      unsigned int readBand=iband;
      if(verbose_opt[0]){
        cout << "extracting band " << readBand << endl;
        MyProgressFunc(progress,pszMessage,pProgressArg);
      }
      double theMin=0;
      double theMax=0;
      if(autoscale_opt.size()){
        this->getMinMax(0,nrOfCol()-1,0,nrOfRow()-1,readBand,theMin,theMax);
        if(verbose_opt[0])
          cout << "minmax: " << theMin << ", " << theMax << endl;
        double theScale=(autoscale_opt[1]-autoscale_opt[0])/(theMax-theMin);
        double theOffset=autoscale_opt[0]-theScale*theMin;
        this->setScale(theScale,readBand);
        this->setOffset(theOffset,readBand);
      }
      else{
        if(scale_opt.size()){
          if(scale_opt.size()>iband)
            this->setScale(scale_opt[iband],readBand);
          else
            this->setScale(scale_opt[0],readBand);
        }
        if(offset_opt.size()){
          if(offset_opt.size()>iband)
            this->setOffset(offset_opt[iband],readBand);
          else
            this->setOffset(offset_opt[0],readBand);
        }
      }

      double readRow=0;
      double readCol=0;
      double lowerCol=0;
      double upperCol=0;
      for(int irow=0;irow<imgWriter.nrOfRow();++irow){
        readRow=irow;
        readData(readBuffer,readRow,readBand);
        for(int icol=0;icol<imgWriter.nrOfCol();++icol)
          writeBuffer[icol]=readBuffer[icol];
        imgWriter.writeData(writeBuffer,irow,readBand);
        progress=(1.0+irow);
        progress+=(imgWriter.nrOfRow()*readBand);
        progress/=imgWriter.nrOfBand()*imgWriter.nrOfRow();
        assert(progress>=0);
        assert(progress<=1);
        MyProgressFunc(progress,pszMessage,pProgressArg);
      }
    }
    return(CE_None);
  }
  catch(string predefinedString){
    std::cerr << predefinedString << std::endl;
    throw;
  }
}

CPLErr Jim::crop(Jim& imgWriter, AppFactory& app){
  Optionjl<string>  projection_opt("a_srs", "a_srs", "Override the projection for the output file (leave blank to copy from input file, use epsg:3035 to use European projection and force to European grid");
  //todo: support layer names
  Optionjl<string>  extent_opt("e", "extent", "get boundary from extent from polygons in vector file");
  Optionjl<string>  layer_opt("ln", "ln", "layer name of extent to crop");
  Optionjl<bool> cut_to_cutline_opt("crop_to_cutline", "crop_to_cutline", "Crop the extent of the target dataset to the extent of the cutline, setting the outside area to nodata.",false);
  Optionjl<bool> cut_in_cutline_opt("crop_in_cutline", "crop_in_cutline", "Crop the extent of the target dataset to the extent of the cutline, setting the inner area to nodata.",false);
  Optionjl<string> eoption_opt("eo","eo", "special extent options controlling rasterization: ATTRIBUTE|CHUNKYSIZE|ALL_TOUCHED|BURN_VALUE_FROM|MERGE_ALG, e.g., -eo ATTRIBUTE=fieldname");
  Optionjl<string> mask_opt("m", "mask", "Use the the specified file as a validity mask (0 is nodata).");
  Optionjl<double> msknodata_opt("msknodata", "msknodata", "Mask value not to consider for crop.", 0);
  Optionjl<unsigned int> mskband_opt("mskband", "mskband", "Mask band to read (0 indexed)", 0);
  Optionjl<double>  ulx_opt("ulx", "ulx", "Upper left x value bounding box", 0.0);
  Optionjl<double>  uly_opt("uly", "uly", "Upper left y value bounding box", 0.0);
  Optionjl<double>  lrx_opt("lrx", "lrx", "Lower right x value bounding box", 0.0);
  Optionjl<double>  lry_opt("lry", "lry", "Lower right y value bounding box", 0.0);
  Optionjl<double>  dx_opt("dx", "dx", "Output resolution in x (in meter) (empty: keep original resolution)");
  Optionjl<double>  dy_opt("dy", "dy", "Output resolution in y (in meter) (empty: keep original resolution)");
  Optionjl<double> cx_opt("x", "x", "x-coordinate of image center to crop (in meter)");
  Optionjl<double> cy_opt("y", "y", "y-coordinate of image center to crop (in meter)");
  Optionjl<double> nx_opt("nx", "nx", "image size in x to crop (in meter)");
  Optionjl<double> ny_opt("ny", "ny", "image size in y to crop (in meter)");
  Optionjl<unsigned int> ns_opt("ns", "ns", "number of samples  to crop (in pixels)");
  Optionjl<unsigned int> nl_opt("nl", "nl", "number of lines to crop (in pixels)");
  Optionjl<unsigned int>  band_opt("b", "band", "band index to crop (leave empty to retain all bands)");
  Optionjl<unsigned int> bstart_opt("sband", "startband", "Start band sequence number");
  Optionjl<unsigned int> bend_opt("eband", "endband", "End band sequence number");
  Optionjl<double> autoscale_opt("as", "autoscale", "scale output to min and max, e.g., --autoscale 0 --autoscale 255");
  Optionjl<double> scale_opt("scale", "scale", "output=scale*input+offset");
  Optionjl<double> offset_opt("offset", "offset", "output=scale*input+offset");
  Optionjl<string>  otype_opt("ot", "otype", "Data type for output image ({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}). Empty string: inherit type from input image");
  // Optionjl<string>  oformat_opt("of", "oformat", "Output image format (see also gdal_translate).","GTiff");
  // Optionjl<string> option_opt("co", "co", "Creation option for output file. Multiple options can be specified.");
  Optionjl<string>  colorTable_opt("ct", "ct", "color table (file with 5 columns: id R G B ALFA (0: transparent, 255: solid)");
  Optionjl<double>  nodata_opt("nodata", "nodata", "Nodata value to put in image if out of bounds.");
  Optionjl<string>  resample_opt("r", "resampling-method", "Resampling method (near: nearest neighbor, bilinear: bi-linear interpolation).", "near");
  Optionjl<string>  description_opt("d", "description", "Set image description");
  Optionjl<bool>  align_opt("align", "align", "Align output bounding box to input image",false);
  Optionjl<short>  verbose_opt("v", "verbose", "verbose", 0,2);

  extent_opt.setHide(1);
  layer_opt.setHide(1);
  cut_to_cutline_opt.setHide(1);
  cut_in_cutline_opt.setHide(1);
  eoption_opt.setHide(1);
  bstart_opt.setHide(1);
  bend_opt.setHide(1);
  mask_opt.setHide(1);
  msknodata_opt.setHide(1);
  mskband_opt.setHide(1);
  // option_opt.setHide(1);
  cx_opt.setHide(1);
  cy_opt.setHide(1);
  nx_opt.setHide(1);
  ny_opt.setHide(1);
  ns_opt.setHide(1);
  nl_opt.setHide(1);
  scale_opt.setHide(1);
  offset_opt.setHide(1);
  nodata_opt.setHide(1);
  description_opt.setHide(1);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=projection_opt.retrieveOption(app);
    ulx_opt.retrieveOption(app);
    uly_opt.retrieveOption(app);
    lrx_opt.retrieveOption(app);
    lry_opt.retrieveOption(app);
    band_opt.retrieveOption(app);
    bstart_opt.retrieveOption(app);
    bend_opt.retrieveOption(app);
    autoscale_opt.retrieveOption(app);
    otype_opt.retrieveOption(app);
    // oformat_opt.retrieveOption(app);
    colorTable_opt.retrieveOption(app);
    dx_opt.retrieveOption(app);
    dy_opt.retrieveOption(app);
    resample_opt.retrieveOption(app);
    extent_opt.retrieveOption(app);
    layer_opt.retrieveOption(app);
    cut_to_cutline_opt.retrieveOption(app);
    cut_in_cutline_opt.retrieveOption(app);
    eoption_opt.retrieveOption(app);
    mask_opt.retrieveOption(app);
    msknodata_opt.retrieveOption(app);
    mskband_opt.retrieveOption(app);
    // option_opt.retrieveOption(app);
    cx_opt.retrieveOption(app);
    cy_opt.retrieveOption(app);
    nx_opt.retrieveOption(app);
    ny_opt.retrieveOption(app);
    ns_opt.retrieveOption(app);
    nl_opt.retrieveOption(app);
    scale_opt.retrieveOption(app);
    offset_opt.retrieveOption(app);
    nodata_opt.retrieveOption(app);
    description_opt.retrieveOption(app);
    align_opt.retrieveOption(app);
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

    double nodataValue=nodata_opt.size()? nodata_opt[0] : 0;
    RESAMPLE theResample;
    if(resample_opt[0]=="near"){
      theResample=NEAR;
      if(verbose_opt[0])
        cout << "resampling: nearest neighbor" << endl;
    }
    else if(resample_opt[0]=="bilinear"){
      theResample=BILINEAR;
      if(verbose_opt[0])
        cout << "resampling: bilinear interpolation" << endl;
    }
    else{
      std::cout << "Error: resampling method " << resample_opt[0] << " not supported" << std::endl;
      return(CE_Failure);
    }

    const char* pszMessage;
    void* pProgressArg=NULL;
    GDALProgressFunc pfnProgress=GDALTermProgress;
    double progress=0;
    MyProgressFunc(progress,pszMessage,pProgressArg);
    // ImgReaderGdal imgReader;
    // ImgWriterGdal imgWriter;
    //open input images to extract number of bands and spatial resolution
    int ncropband=0;//total number of bands to write
    double dx=0;
    double dy=0;
    if(dx_opt.size())
      dx=dx_opt[0];
    if(dy_opt.size())
      dy=dy_opt[0];

    try{
      //convert start and end band options to vector of band indexes
      if(bstart_opt.size()){
        if(bend_opt.size()!=bstart_opt.size()){
          string errorstring="Error: options for start and end band indexes must be provided as pairs, missing end band";
          throw(errorstring);
        }
        band_opt.clear();
        for(int ipair=0;ipair<bstart_opt.size();++ipair){
          if(bend_opt[ipair]<=bstart_opt[ipair]){
            string errorstring="Error: index for end band must be smaller then start band";
            throw(errorstring);
          }
          for(unsigned int iband=bstart_opt[ipair];iband<=bend_opt[ipair];++iband)
            band_opt.push_back(iband);
        }
      }
    }
    catch(string error){
      cerr << error << std::endl;
      throw;
    }


    bool isGeoRef=false;
    string projectionString;
    // for(int iimg=0;iimg<input_opt.size();++iimg){

    if(!isGeoRef)
      isGeoRef=this->isGeoRef();
    if(this->isGeoRef()&&projection_opt.empty())
      projectionString=this->getProjection();
    if(dx_opt.empty()){
      dx=this->getDeltaX();
    }

    if(dy_opt.empty()){
      dy=this->getDeltaY();
    }
    if(band_opt.size())
      ncropband+=band_opt.size();
    else
      ncropband+=this->nrOfBand();

    GDALDataType theType=getGDALDataType();
    if(otype_opt.size()){
      theType=string2GDAL(otype_opt[0]);
      if(theType==GDT_Unknown)
        std::cout << "Warning: unknown output pixel type: " << otype_opt[0] << ", using input type as default" << std::endl;
    }
    if(verbose_opt[0])
      cout << "Output pixel type:  " << GDALGetDataTypeName(theType) << endl;

    //bounding box of cropped image
    double cropulx=ulx_opt[0];
    double cropuly=uly_opt[0];
    double croplrx=lrx_opt[0];
    double croplry=lry_opt[0];
    //get bounding box from extentReader if defined
    VectorOgr extentReader;

    OGRSpatialReference gdsSpatialRef(getProjectionRef().c_str());
    if(extent_opt.size()){
      //image must be georeferenced
      if(!this->isGeoRef()){
        string errorstring="Warning: input image is not georeferenced using extent";
        std::cerr << errorstring << std::endl;
        throw(errorstring);
      }
      statfactory::StatFactory stat;
      double e_ulx;
      double e_uly;
      double e_lrx;
      double e_lry;
      for(int iextent=0;iextent<extent_opt.size();++iextent){
        extentReader.open(extent_opt[iextent],layer_opt,true);//noread=true

        OGRSpatialReference *vectorSpatialRef=extentReader.getLayer(0)->GetSpatialRef();
        OGRCoordinateTransformation *vector2raster=0;
        vector2raster = OGRCreateCoordinateTransformation(vectorSpatialRef, &gdsSpatialRef);
        if(gdsSpatialRef.IsSame(vectorSpatialRef)){
          vector2raster=0;
        }
        else{
          if(!vector2raster){
            std::ostringstream errorStream;
            errorStream << "Error: cannot create OGRCoordinateTransformation vector to GDAL raster dataset" << std::endl;
            throw(errorStream.str());
          }
        }
        extentReader.getExtent(e_ulx,e_uly,e_lrx,e_lry,vector2raster);
        ulx_opt.push_back(e_ulx);
        uly_opt.push_back(e_uly);
        lrx_opt.push_back(e_lrx);
        lry_opt.push_back(e_lry);
        extentReader.close();
      }
      e_ulx=stat.mymin(ulx_opt);
      e_uly=stat.mymax(uly_opt);
      e_lrx=stat.mymax(lrx_opt);
      e_lry=stat.mymin(lry_opt);
      ulx_opt.clear();
      uly_opt.clear();
      lrx_opt.clear();
      lrx_opt.clear();
      ulx_opt.push_back(e_ulx);
      uly_opt.push_back(e_uly);
      lrx_opt.push_back(e_lrx);
      lry_opt.push_back(e_lry);
      if(cut_to_cutline_opt.size()||cut_in_cutline_opt.size()||eoption_opt.size())
        extentReader.open(extent_opt[0],layer_opt,true);
    }
    else if(cx_opt.size()&&cy_opt.size()&&nx_opt.size()&&ny_opt.size()){
      ulx_opt[0]=cx_opt[0]-nx_opt[0]/2.0;
      uly_opt[0]=(isGeoRef) ? cy_opt[0]+ny_opt[0]/2.0 : cy_opt[0]-ny_opt[0]/2.0;
      lrx_opt[0]=cx_opt[0]+nx_opt[0]/2.0;
      lry_opt[0]=(isGeoRef) ? cy_opt[0]-ny_opt[0]/2.0 : cy_opt[0]+ny_opt[0]/2.0;
    }
    else if(cx_opt.size()&&cy_opt.size()&&ns_opt.size()&&nl_opt.size()){
      ulx_opt[0]=cx_opt[0]-ns_opt[0]*dx/2.0;
      uly_opt[0]=(isGeoRef) ? cy_opt[0]+nl_opt[0]*dy/2.0 : cy_opt[0]-nl_opt[0]*dy/2.0;
      lrx_opt[0]=cx_opt[0]+ns_opt[0]*dx/2.0;
      lry_opt[0]=(isGeoRef) ? cy_opt[0]-nl_opt[0]*dy/2.0 : cy_opt[0]+nl_opt[0]*dy/2.0;
    }

    if(verbose_opt[0])
      cout << "--ulx=" << ulx_opt[0] << " --uly=" << uly_opt[0] << " --lrx=" << lrx_opt[0] << " --lry=" << lry_opt[0] << endl;

    int ncropcol=0;
    int ncroprow=0;

    Jim maskReader;
    //todo: support transform of extent with cutline
    if(extent_opt.size()&&(cut_to_cutline_opt[0]||cut_in_cutline_opt[0]||eoption_opt.size())){
      if(mask_opt.size()){
        string errorString="Error: can only either mask or extent extent with cut_to_cutline / cut_in_cutline, not both";
        throw(errorString);
      }
      try{
        // ncropcol=abs(static_cast<unsigned int>(ceil((lrx_opt[0]-ulx_opt[0])/dx)));
        // ncroprow=abs(static_cast<unsigned int>(ceil((uly_opt[0]-lry_opt[0])/dy)));
        ncropcol=static_cast<unsigned int>(ceil((lrx_opt[0]-ulx_opt[0])/dx));
        ncroprow=static_cast<unsigned int>(ceil((uly_opt[0]-lry_opt[0])/dy));
        maskReader.open(ncropcol,ncroprow,1,GDT_Float64);
        double gt[6];
        gt[0]=ulx_opt[0];
        gt[1]=dx;
        gt[2]=0;
        gt[3]=uly_opt[0];
        gt[4]=0;
        gt[5]=-dy;
        maskReader.setGeoTransform(gt);
        if(projection_opt.size())
          maskReader.setProjectionProj4(projection_opt[0]);
        else if(projectionString.size())
          maskReader.setProjection(projectionString);

        // maskReader.rasterizeBuf(extentReader,msknodata_opt[0],eoption_opt,layer_opt);
        maskReader.rasterizeBuf(extentReader,1,eoption_opt,layer_opt);
      }
      catch(string error){
        cerr << error << std::endl;
        throw;
      }
    }
    else if(mask_opt.size()==1){
      try{
        //there is only a single mask
        maskReader.open(mask_opt[0]);
        if(mskband_opt[0]>=maskReader.nrOfBand()){
          string errorString="Error: illegal mask band";
          throw;
        }
      }
      catch(string error){
        cerr << error << std::endl;
        throw;
      }
    }

    //determine number of output bands
    int writeBand=0;//write band

    if(scale_opt.size()){
      while(scale_opt.size()<band_opt.size())
        scale_opt.push_back(scale_opt[0]);
    }
    if(offset_opt.size()){
      while(offset_opt.size()<band_opt.size())
        offset_opt.push_back(offset_opt[0]);
    }
    if(autoscale_opt.size()){
      assert(autoscale_opt.size()%2==0);
    }

    if(theType==GDT_Unknown){
      theType=this->getGDALDataType();
      if(verbose_opt[0])
        cout << "Using data type from input image: " << GDALGetDataTypeName(theType) << endl;
    }
    // if(option_opt.findSubstring("INTERLEAVE=")==option_opt.end()){
    //   string theInterleave="INTERLEAVE=";
    //   theInterleave+=this->getInterleave();
    //   option_opt.push_back(theInterleave);
    // }
    // if(verbose_opt[0])
    //   cout << "size of " << input_opt[iimg] << ": " << ncol << " cols, "<< nrow << " rows" << endl;
    double uli,ulj,lri,lrj;//image coordinates
    bool forceEUgrid=false;
    if(projection_opt.size())
      forceEUgrid=(!(projection_opt[0].compare("EPSG:3035"))||!(projection_opt[0].compare("EPSG:3035"))||projection_opt[0].find("ETRS-LAEA")!=string::npos);
    if(ulx_opt[0]>=lrx_opt[0]){//default bounding box: no cropping
      uli=0;
      lri=this->nrOfCol()-1;
      ulj=0;
      lrj=this->nrOfRow()-1;
      ncropcol=this->nrOfCol();
      ncroprow=this->nrOfRow();
      this->getBoundingBox(cropulx,cropuly,croplrx,croplry);
      double magicX=1,magicY=1;
      // this->getMagicPixel(magicX,magicY);
      if(forceEUgrid){
        //force to LAEA grid
        Egcs egcs;
        egcs.setLevel(egcs.res2level(dx));
        egcs.force2grid(cropulx,cropuly,croplrx,croplry);
        this->geo2image(cropulx+(magicX-1.0)*this->getDeltaX(),cropuly-(magicY-1.0)*this->getDeltaY(),uli,ulj);
        this->geo2image(croplrx+(magicX-2.0)*this->getDeltaX(),croplry-(magicY-2.0)*this->getDeltaY(),lri,lrj);
      }
      this->geo2image(cropulx+(magicX-1.0)*this->getDeltaX(),cropuly-(magicY-1.0)*this->getDeltaY(),uli,ulj);
      this->geo2image(croplrx+(magicX-2.0)*this->getDeltaX(),croplry-(magicY-2.0)*this->getDeltaY(),lri,lrj);
      // ncropcol=abs(static_cast<unsigned int>(ceil((croplrx-cropulx)/dx)));
      // ncroprow=abs(static_cast<unsigned int>(ceil((cropuly-croplry)/dy)));
      ncropcol=static_cast<unsigned int>(ceil((croplrx-cropulx)/dx));
      ncroprow=static_cast<unsigned int>(ceil((cropuly-croplry)/dy));
    }
    else{
      double magicX=1,magicY=1;
      // this->getMagicPixel(magicX,magicY);
      cropulx=ulx_opt[0];
      cropuly=uly_opt[0];
      croplrx=lrx_opt[0];
      croplry=lry_opt[0];
      if(forceEUgrid){
        //force to LAEA grid
        Egcs egcs;
        egcs.setLevel(egcs.res2level(dx));
        egcs.force2grid(cropulx,cropuly,croplrx,croplry);
      }
      else if(align_opt[0]){
        if(cropulx>this->getUlx())
          cropulx-=fmod(cropulx-this->getUlx(),dx);
        else if(cropulx<this->getUlx())
          cropulx+=fmod(this->getUlx()-cropulx,dx)-dx;
        if(croplrx<this->getLrx())
          croplrx+=fmod(this->getLrx()-croplrx,dx);
        else if(croplrx>this->getLrx())
          croplrx-=fmod(croplrx-this->getLrx(),dx)+dx;
        if(croplry>this->getLry())
          croplry-=fmod(croplry-this->getLry(),dy);
        else if(croplry<this->getLry())
          croplry+=fmod(this->getLry()-croplry,dy)-dy;
        if(cropuly<this->getUly())
          cropuly+=fmod(this->getUly()-cropuly,dy);
        else if(cropuly>this->getUly())
          cropuly-=fmod(cropuly-this->getUly(),dy)+dy;
      }
      this->geo2image(cropulx+(magicX-1.0)*this->getDeltaX(),cropuly-(magicY-1.0)*this->getDeltaY(),uli,ulj);
      this->geo2image(croplrx+(magicX-2.0)*this->getDeltaX(),croplry-(magicY-2.0)*this->getDeltaY(),lri,lrj);

      ncropcol=static_cast<unsigned int>(ceil((croplrx-cropulx)/dx));
      ncroprow=static_cast<unsigned int>(ceil((cropuly-croplry)/dy));
      uli=floor(uli);
      ulj=floor(ulj);
      lri=floor(lri);
      lrj=floor(lrj);
    }

    // double deltaX=this->getDeltaX();
    // double deltaY=this->getDeltaY();
    if(!imgWriter.nrOfBand()){//not opened yet
      if(verbose_opt[0]){
        cout << "cropulx: " << cropulx << endl;
        cout << "cropuly: " << cropuly << endl;
        cout << "croplrx: " << croplrx << endl;
        cout << "croplry: " << croplry << endl;
        cout << "ncropcol: " << ncropcol << endl;
        cout << "ncroprow: " << ncroprow << endl;
        cout << "cropulx+ncropcol*dx: " << cropulx+ncropcol*dx << endl;
        cout << "cropuly-ncroprow*dy: " << cropuly-ncroprow*dy << endl;
        cout << "upper left column of input image: " << uli << endl;
        cout << "upper left row of input image: " << ulj << endl;
        cout << "lower right column of input image: " << lri << endl;
        cout << "lower right row of input image: " << lrj << endl;
        cout << "new number of cols: " << ncropcol << endl;
        cout << "new number of rows: " << ncroprow << endl;
        cout << "new number of bands: " << ncropband << endl;
      }
      // string imageType;//=this->getImageType();
      // if(oformat_opt.size())//default
      //   imageType=oformat_opt[0];
      try{
        imgWriter.open(ncropcol,ncroprow,ncropband,theType);
        imgWriter.setNoData(nodata_opt);
        // if(nodata_opt.size()){
        //   imgWriter.setNoData(nodata_opt);
        // }
      }
      catch(string errorstring){
        cout << errorstring << endl;
        throw;
      }
      if(description_opt.size())
        imgWriter.setImageDescription(description_opt[0]);
      double gt[6];
      gt[0]=cropulx;
      gt[1]=dx;
      gt[2]=0;
      gt[3]=cropuly;
      gt[4]=0;
      gt[5]=(this->isGeoRef())? -dy : dy;
      imgWriter.setGeoTransform(gt);
      if(projection_opt.size()){
        if(verbose_opt[0])
          cout << "projection: " << projection_opt[0] << endl;
        imgWriter.setProjectionProj4(projection_opt[0]);
      }
      else
        imgWriter.setProjection(this->getProjection());
      if(imgWriter.getDataType()==GDT_Byte){
        if(colorTable_opt.size()){
          if(colorTable_opt[0]!="none")
            imgWriter.setColorTable(colorTable_opt[0]);
        }
        else if (this->getColorTable()!=NULL)//copy colorTable from input image
          imgWriter.setColorTable(this->getColorTable());
      }
    }

    double startCol=uli;
    double endCol=lri;
    if(uli<0)
      startCol=0;
    else if(uli>=this->nrOfCol())
      startCol=this->nrOfCol()-1;
    if(lri<0)
      endCol=0;
    else if(lri>=this->nrOfCol())
      endCol=this->nrOfCol()-1;
    double startRow=ulj;
    double endRow=lrj;
    if(ulj<0)
      startRow=0;
    else if(ulj>=this->nrOfRow())
      startRow=this->nrOfRow()-1;
    if(lrj<0)
      endRow=0;
    else if(lrj>=this->nrOfRow())
      endRow=this->nrOfRow()-1;

    vector<double> readBuffer;
    unsigned int nband=(band_opt.size())?band_opt.size() : this->nrOfBand();
    for(unsigned int iband=0;iband<nband;++iband){
      unsigned int readBand=(band_opt.size()>iband)?band_opt[iband]:iband;
      if(verbose_opt[0]){
        cout << "extracting band " << readBand << endl;
        MyProgressFunc(progress,pszMessage,pProgressArg);
      }
      double theMin=0;
      double theMax=0;
      if(autoscale_opt.size()){
        try{
          this->getMinMax(static_cast<unsigned int>(startCol),static_cast<unsigned int>(endCol),static_cast<unsigned int>(startRow),static_cast<unsigned int>(endRow),readBand,theMin,theMax);
        }
        catch(string errorString){
          cout << errorString << endl;
        }
        if(verbose_opt[0])
          cout << "minmax: " << theMin << ", " << theMax << endl;
        double theScale=(autoscale_opt[1]-autoscale_opt[0])/(theMax-theMin);
        double theOffset=autoscale_opt[0]-theScale*theMin;
        this->setScale(theScale,readBand);
        this->setOffset(theOffset,readBand);
      }
      else{
        if(scale_opt.size()){
          if(scale_opt.size()>iband)
            this->setScale(scale_opt[iband],readBand);
          else
            this->setScale(scale_opt[0],readBand);
        }
        if(offset_opt.size()){
          if(offset_opt.size()>iband)
            this->setOffset(offset_opt[iband],readBand);
          else
            this->setOffset(offset_opt[0],readBand);
        }
      }

      double readRow=0;
      double readCol=0;
      double lowerCol=0;
      double upperCol=0;
      for(int irow=0;irow<imgWriter.nrOfRow();++irow){
        vector<double> lineMask;
        double x=0;
        double y=0;
        //convert irow to geo
        imgWriter.image2geo(0,irow,x,y);
        //lookup corresponding row for irow in this file
        this->geo2image(x,y,readCol,readRow);
        vector<double> writeBuffer;
        if(readRow<0||readRow>=this->nrOfRow()){
          for(int icol=0;icol<imgWriter.nrOfCol();++icol)
            writeBuffer.push_back(nodataValue);
        }
        else{
          try{
            if(endCol<this->nrOfCol()-1){
              this->readData(readBuffer,startCol,endCol+1,readRow,readBand,theResample);
            }
            else{
              this->readData(readBuffer,startCol,endCol,readRow,readBand,theResample);
            }
            double oldRowMask=-1;//keep track of row mask to optimize number of line readings
            for(int icol=0;icol<imgWriter.nrOfCol();++icol){
              imgWriter.image2geo(icol,irow,x,y);
              //lookup corresponding row for irow in this file
              this->geo2image(x,y,readCol,readRow);
              if(readCol<0||readCol>=this->nrOfCol()){
                writeBuffer.push_back(nodataValue);
              }
              else{
                bool valid=true;
                double geox=0;
                double geoy=0;
                if(maskReader.isInit()){
                  //read mask
                  double colMask=0;
                  double rowMask=0;

                  imgWriter.image2geo(icol,irow,geox,geoy);
                  maskReader.geo2image(geox,geoy,colMask,rowMask);
                  colMask=static_cast<unsigned int>(colMask);
                  rowMask=static_cast<unsigned int>(rowMask);
                  if(rowMask>=0&&rowMask<maskReader.nrOfRow()&&colMask>=0&&colMask<maskReader.nrOfCol()){
                    if(static_cast<unsigned int>(rowMask)!=static_cast<unsigned int>(oldRowMask)){

                      try{
                        maskReader.readData(lineMask,static_cast<unsigned int>(rowMask),mskband_opt[0]);
                      }
                      catch(string errorstring){
                        cerr << errorstring << endl;
                        throw;
                      }
                      catch(...){
                        cerr << "error caught" << std::endl;
                        throw;
                      }
                      oldRowMask=rowMask;
                    }
                    if(cut_to_cutline_opt[0]){
                      if(lineMask[colMask]!=1){
                        nodataValue=nodata_opt[0];
                        valid=false;
                      }
                    }
                    else if(cut_in_cutline_opt[0]){
                      if(lineMask[colMask]==1){
                        nodataValue=nodata_opt[0];
                        valid=false;
                      }
                    }
                    else{
                      for(int ivalue=0;ivalue<msknodata_opt.size();++ivalue){
                        if(lineMask[colMask]==msknodata_opt[ivalue]){
                          if(nodata_opt.size()>ivalue)
                            nodataValue=nodata_opt[ivalue];
                          valid=false;
                          break;
                        }
                      }
                    }
                  }
                }
                if(!valid)
                  writeBuffer.push_back(nodataValue);
                else{
                  switch(theResample){
                  case(BILINEAR):
                    lowerCol=readCol-0.5;
                    lowerCol=static_cast<unsigned int>(lowerCol);
                    upperCol=readCol+0.5;
                    upperCol=static_cast<unsigned int>(upperCol);
                    if(lowerCol<0)
                      lowerCol=0;
                    if(upperCol>=this->nrOfCol())
                      upperCol=this->nrOfCol()-1;
                    writeBuffer.push_back((readCol-0.5-lowerCol)*readBuffer[upperCol-startCol]+(1-readCol+0.5+lowerCol)*readBuffer[lowerCol-startCol]);
                    break;
                  default:
                    readCol=static_cast<unsigned int>(readCol);
                    readCol-=startCol;//we only start reading from startCol
                    writeBuffer.push_back(readBuffer[readCol]);
                    break;
                  }
                }
              }
            }
          }
          catch(string errorstring){
            cout << errorstring << endl;
            throw;
          }
        }
        if(writeBuffer.size()!=imgWriter.nrOfCol())
          cout << "writeBuffer.size()=" << writeBuffer.size() << ", imgWriter.nrOfCol()=" << imgWriter.nrOfCol() << endl;

        assert(writeBuffer.size()==imgWriter.nrOfCol());
        try{
          imgWriter.writeData(writeBuffer,irow,writeBand);
        }
        catch(string errorstring){
          cout << errorstring << endl;
          throw;
        }
        if(verbose_opt[0]){
          progress=(1.0+irow);
          progress/=imgWriter.nrOfRow();
          MyProgressFunc(progress,pszMessage,pProgressArg);
        }
        else{
          progress=(1.0+irow);
          progress+=(imgWriter.nrOfRow()*writeBand);
          progress/=imgWriter.nrOfBand()*imgWriter.nrOfRow();
          assert(progress>=0);
          assert(progress<=1);
          MyProgressFunc(progress,pszMessage,pProgressArg);
        }
      }
      ++writeBand;
    }
    if(extent_opt.size()&&(cut_to_cutline_opt[0]||cut_in_cutline_opt[0]||eoption_opt.size())){
      extentReader.close();
    }
    if(maskReader.isInit())
      maskReader.close();
    return(CE_None);
  }
  catch(string predefinedString){
    std::cout << predefinedString << std::endl;
    throw;
  }
}

CPLErr Jim::crop(VectorOgr& sampleReader, Jim& imgWriter, AppFactory& app){
  Optionjl<string>  projection_opt("a_srs", "a_srs", "Override the projection for the output file (leave blank to copy from input file, use epsg:3035 to use European projection and force to European grid");
  //todo: support layer names
  Optionjl<string>  layer_opt("ln", "ln", "layer name of extent to crop");
  Optionjl<bool> cut_to_cutline_opt(
"cut_to_cutline", "crop_to_cutline", "Crop the extent of the target dataset to the extent of the cutline, setting the outside area to nodata.",false);
  Optionjl<bool> cut_in_cutline_opt("cut_in_cutline", "crop_in_cutline", "Crop the extent of the target dataset to the extent of the cutline, setting the inner area to nodata.",false);
  Optionjl<string> eoption_opt("eo","eo", "special extent options controlling rasterization: ATTRIBUTE|CHUNKYSIZE|ALL_TOUCHED|BURN_VALUE_FROM|MERGE_ALG, e.g., -eo ATTRIBUTE=fieldname");
  Optionjl<double> msknodata_opt("msknodata", "msknodata", "Mask value not to consider for crop.", 0);
  Optionjl<double>  ulx_opt("ulx", "ulx", "Upper left x value bounding box", 0.0);
  Optionjl<double>  uly_opt("uly", "uly", "Upper left y value bounding box", 0.0);
  Optionjl<double>  lrx_opt("lrx", "lrx", "Lower right x value bounding box", 0.0);
  Optionjl<double>  lry_opt("lry", "lry", "Lower right y value bounding box", 0.0);
  Optionjl<double>  dx_opt("dx", "dx", "Output resolution in x (in meter) (empty: keep original resolution)");
  Optionjl<double>  dy_opt("dy", "dy", "Output resolution in y (in meter) (empty: keep original resolution)");
  Optionjl<unsigned int>  band_opt("b", "band", "band index to crop (leave empty to retain all bands)");
  Optionjl<unsigned int> bstart_opt("sband", "startband", "Start band sequence number");
  Optionjl<unsigned int> bend_opt("eband", "endband", "End band sequence number");
  Optionjl<double> autoscale_opt("as", "autoscale", "scale output to min and max, e.g., --autoscale 0 --autoscale 255");
  Optionjl<double> scale_opt("scale", "scale", "output=scale*input+offset");
  Optionjl<double> offset_opt("offset", "offset", "output=scale*input+offset");
  Optionjl<string>  otype_opt("ot", "otype", "Data type for output image ({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}). Empty string: inherit type from input image");
  Optionjl<string>  colorTable_opt("ct", "ct", "color table (file with 5 columns: id R G B ALFA (0: transparent, 255: solid)");
  Optionjl<double>  nodata_opt("nodata", "nodata", "Nodata value to put in image if out of bounds.");
  Optionjl<string>  resample_opt("r", "resampling-method", "Resampling method (near: nearest neighbor, bilinear: bi-linear interpolation).", "near");
  Optionjl<string>  description_opt("d", "description", "Set image description");
  Optionjl<bool>  align_opt("align", "align", "Align output bounding box to input image",false);
  Optionjl<short>  verbose_opt("v", "verbose", "verbose", 0,2);

  layer_opt.setHide(1);
  cut_to_cutline_opt.setHide(1);
  cut_in_cutline_opt.setHide(1);
  eoption_opt.setHide(1);
  msknodata_opt.setHide(1);
  bstart_opt.setHide(1);
  bend_opt.setHide(1);
  scale_opt.setHide(1);
  offset_opt.setHide(1);
  nodata_opt.setHide(1);
  description_opt.setHide(1);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=projection_opt.retrieveOption(app);
    ulx_opt.retrieveOption(app);
    uly_opt.retrieveOption(app);
    lrx_opt.retrieveOption(app);
    lry_opt.retrieveOption(app);
    band_opt.retrieveOption(app);
    bstart_opt.retrieveOption(app);
    bend_opt.retrieveOption(app);
    autoscale_opt.retrieveOption(app);
    otype_opt.retrieveOption(app);
    colorTable_opt.retrieveOption(app);
    dx_opt.retrieveOption(app);
    dy_opt.retrieveOption(app);
    resample_opt.retrieveOption(app);
    layer_opt.retrieveOption(app);
    cut_to_cutline_opt.retrieveOption(app);
    cut_in_cutline_opt.retrieveOption(app);
    eoption_opt.retrieveOption(app);
    msknodata_opt.retrieveOption(app);
    scale_opt.retrieveOption(app);
    offset_opt.retrieveOption(app);
    nodata_opt.retrieveOption(app);
    description_opt.retrieveOption(app);
    align_opt.retrieveOption(app);
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

    double nodataValue=nodata_opt.size()? nodata_opt[0] : 0;
    RESAMPLE theResample;
    if(resample_opt[0]=="near"){
      theResample=NEAR;
      if(verbose_opt[0])
        cout << "resampling: nearest neighbor" << endl;
    }
    else if(resample_opt[0]=="bilinear"){
      theResample=BILINEAR;
      if(verbose_opt[0])
        cout << "resampling: bilinear interpolation" << endl;
    }
    else{
      std::cout << "Error: resampling method " << resample_opt[0] << " not supported" << std::endl;
      return(CE_Failure);
    }

    const char* pszMessage;
    void* pProgressArg=NULL;
    GDALProgressFunc pfnProgress=GDALTermProgress;
    double progress=0;
    MyProgressFunc(progress,pszMessage,pProgressArg);
    // ImgReaderGdal imgReader;
    // ImgWriterGdal imgWriter;
    //open input images to extract number of bands and spatial resolution
    int ncropband=0;//total number of bands to write
    double dx=0;
    double dy=0;
    if(dx_opt.size())
      dx=dx_opt[0];
    if(dy_opt.size())
      dy=dy_opt[0];

    try{
      //convert start and end band options to vector of band indexes
      if(bstart_opt.size()){
        if(bend_opt.size()!=bstart_opt.size()){
          string errorstring="Error: options for start and end band indexes must be provided as pairs, missing end band";
          throw(errorstring);
        }
        band_opt.clear();
        for(int ipair=0;ipair<bstart_opt.size();++ipair){
          if(bend_opt[ipair]<=bstart_opt[ipair]){
            string errorstring="Error: index for end band must be smaller then start band";
            throw(errorstring);
          }
          for(unsigned int iband=bstart_opt[ipair];iband<=bend_opt[ipair];++iband)
            band_opt.push_back(iband);
        }
      }
      //image must be georeferenced
      if(!this->isGeoRef()){
        string errorstring="Warning: input image is not georeferenced using start and end band options";
        std::cerr << errorstring << std::endl;
        // throw(errorstring);
      }
    }
    catch(string error){
      cerr << error << std::endl;
      throw;
    }


    bool isGeoRef=false;
    string projectionString;
    // for(int iimg=0;iimg<input_opt.size();++iimg){

    if(!isGeoRef)
      isGeoRef=this->isGeoRef();
    if(this->isGeoRef()&&projection_opt.empty())
      projectionString=this->getProjection();
    if(dx_opt.empty()){
      dx=this->getDeltaX();
    }

    if(dy_opt.empty()){
      dy=this->getDeltaY();
    }
    if(band_opt.size())
      ncropband+=band_opt.size();
    else
      ncropband+=this->nrOfBand();

    GDALDataType theType=getGDALDataType();
    if(otype_opt.size()){
      theType=string2GDAL(otype_opt[0]);
      if(theType==GDT_Unknown)
        std::cout << "Warning: unknown output pixel type: " << otype_opt[0] << ", using input type as default" << std::endl;
    }
    if(verbose_opt[0])
      cout << "Output pixel type:  " << GDALGetDataTypeName(theType) << endl;

    //bounding box of cropped image
    double cropulx=ulx_opt[0];
    double cropuly=uly_opt[0];
    double croplrx=lrx_opt[0];
    double croplry=lry_opt[0];
    //get bounding box from extentReader if defined

    bool doInit=true;
    for(int ilayer=0;ilayer<sampleReader.getLayerCount();++ilayer){
      std::string currentLayername=sampleReader.getLayer(ilayer)->GetName();
      if(layer_opt.size())
        if(find(layer_opt.begin(),layer_opt.end(),currentLayername)==layer_opt.end())
          continue;
      if(verbose_opt[0])
        std::cout << "getLayer " << std::endl;
      OGRLayer *readLayer=sampleReader.getLayer(ilayer);
      if(!readLayer){
        ostringstream ess;
        ess << "Error: could not get layer of sampleReader" << endl;
        throw(ess.str());
      }
      OGRSpatialReference thisSpatialRef(getProjectionRef().c_str());
      OGRSpatialReference *sampleSpatialRef=readLayer->GetSpatialRef();
      OGRCoordinateTransformation *sample2img = OGRCreateCoordinateTransformation(sampleSpatialRef, &thisSpatialRef);
      OGRCoordinateTransformation *img2sample = OGRCreateCoordinateTransformation(&thisSpatialRef, sampleSpatialRef);
      if(thisSpatialRef.IsSame(sampleSpatialRef)){
        sample2img=0;
        img2sample=0;
      }
      else{
        if(cut_to_cutline_opt[0]||cut_in_cutline_opt[0]||eoption_opt.size()){
          string errorString="Error: projection of vector and raster should be identical when using cut_to_cutline, cut_in_cutline or eoption";
          throw(errorString);
        }
        if(!sample2img){
          std::ostringstream errorStream;
          errorStream << "Error: cannot create OGRCoordinateTransformation sample to image" << std::endl;
          throw(errorStream.str());
        }
        if(!img2sample){
          std::ostringstream errorStream;
          errorStream << "Error: cannot create OGRCoordinateTransformation image to sample" << std::endl;
          throw(errorStream.str());
        }
      }
      double layer_ulx;
      double layer_uly;
      double layer_lrx;
      double layer_lry;
      if(verbose_opt[0])
        std::cout << "getExtent " << std::endl;
      sampleReader.getExtent(layer_ulx,layer_uly,layer_lrx,layer_lry,ilayer,sample2img);

      if(doInit){
        ulx_opt[0]=layer_ulx;
        uly_opt[0]=layer_uly;
        lrx_opt[0]=layer_lrx;
        lry_opt[0]=layer_lry;
        doInit=false;
      }
      else{
        if(layer_ulx<ulx_opt[0])
          ulx_opt[0]=layer_ulx;
        if(layer_uly>uly_opt[0])
          uly_opt[0]=layer_uly;
        if(layer_lrx>lrx_opt[0])
          ulx_opt[0]=layer_lrx;
        if(layer_lry<lry_opt[0])
          lry_opt[0]=layer_lry;
      }
    }
    //ulx_opt[0],uly_opt[0],lrx_opt[0],lry_opt[0] now is maximum extent over all selected layers
    if(croplrx>cropulx&&cropulx>ulx_opt[0])
      ulx_opt[0]=cropulx;
    if(croplrx>cropulx&&croplrx<lrx_opt[0])
      lrx_opt[0]=croplrx;
    if(cropuly>croplry&&cropuly<uly_opt[0])
      uly_opt[0]=cropuly;
    if(croplry<cropuly&&croplry>lry_opt[0])
      lry_opt[0]=croplry;
    //ulx_opt[0],uly_opt[0],lrx_opt[0],lry_opt[0] now is minimum extent over all selected layers and user defined bounding box
    if(verbose_opt[0])
      cout << "--ulx=" << ulx_opt[0] << " --uly=" << uly_opt[0] << " --lrx=" << lrx_opt[0] << " --lry=" << lry_opt[0] << endl;

    int ncropcol=0;
    int ncroprow=0;

    Jim maskReader;
    if(cut_to_cutline_opt[0]||cut_in_cutline_opt[0]||eoption_opt.size()){
      try{
        if(sampleReader.getLayerCount()>1&&(layer_opt.size()>1||layer_opt.empty())){
          std::ostringstream errorStream;
          errorStream << "Error: multiple layers not supported with cut_to_cutline or cut_to cutline, please specify a single layer" << std::endl;
          throw(errorStream.str());//help was invoked, stop processing
        }

        ncropcol=static_cast<unsigned int>(ceil((lrx_opt[0]-ulx_opt[0])/dx));
        ncroprow=static_cast<unsigned int>(ceil((uly_opt[0]-lry_opt[0])/dy));
        maskReader.open(ncropcol,ncroprow,1,GDT_Float64);
        double gt[6];
        gt[0]=ulx_opt[0];
        gt[1]=dx;
        gt[2]=0;
        gt[3]=uly_opt[0];
        gt[4]=0;
        gt[5]=-dy;
        maskReader.setGeoTransform(gt);
        if(projection_opt.size())
          maskReader.setProjectionProj4(projection_opt[0]);
        else if(projectionString.size())
          maskReader.setProjection(projectionString);

        // maskReader.rasterizeBuf(sampleReader,msknodata_opt[0],eoption_opt,layer_opt);
        maskReader.rasterizeBuf(sampleReader,1,eoption_opt,layer_opt);
      }
      catch(string error){
        cerr << error << std::endl;
        throw;
      }
    }
    //determine number of output bands
    int writeBand=0;//write band

    if(scale_opt.size()){
      while(scale_opt.size()<band_opt.size())
        scale_opt.push_back(scale_opt[0]);
    }
    if(offset_opt.size()){
      while(offset_opt.size()<band_opt.size())
        offset_opt.push_back(offset_opt[0]);
    }
    if(autoscale_opt.size()){
      assert(autoscale_opt.size()%2==0);
    }

    if(theType==GDT_Unknown){
      theType=this->getGDALDataType();
      if(verbose_opt[0])
        cout << "Using data type from input image: " << GDALGetDataTypeName(theType) << endl;
    }
    // if(option_opt.findSubstring("INTERLEAVE=")==option_opt.end()){
    //   string theInterleave="INTERLEAVE=";
    //   theInterleave+=this->getInterleave();
    //   option_opt.push_back(theInterleave);
    // }
    // if(verbose_opt[0])
    //   cout << "size of " << input_opt[iimg] << ": " << ncol << " cols, "<< nrow << " rows" << endl;
    double uli,ulj,lri,lrj;//image coordinates
    bool forceEUgrid=false;
    if(projection_opt.size())
      forceEUgrid=(!(projection_opt[0].compare("EPSG:3035"))||!(projection_opt[0].compare("EPSG:3035"))||projection_opt[0].find("ETRS-LAEA")!=string::npos);
    if(ulx_opt[0]>=lrx_opt[0]){//default bounding box: no cropping
      uli=0;
      lri=this->nrOfCol()-1;
      ulj=0;
      lrj=this->nrOfRow()-1;
      ncropcol=this->nrOfCol();
      ncroprow=this->nrOfRow();
      this->getBoundingBox(cropulx,cropuly,croplrx,croplry);
      double magicX=1,magicY=1;
      // this->getMagicPixel(magicX,magicY);
      if(forceEUgrid){
        //force to LAEA grid
        Egcs egcs;
        egcs.setLevel(egcs.res2level(dx));
        egcs.force2grid(cropulx,cropuly,croplrx,croplry);
        this->geo2image(cropulx+(magicX-1.0)*this->getDeltaX(),cropuly-(magicY-1.0)*this->getDeltaY(),uli,ulj);
        this->geo2image(croplrx+(magicX-2.0)*this->getDeltaX(),croplry-(magicY-2.0)*this->getDeltaY(),lri,lrj);
      }
      this->geo2image(cropulx+(magicX-1.0)*this->getDeltaX(),cropuly-(magicY-1.0)*this->getDeltaY(),uli,ulj);
      this->geo2image(croplrx+(magicX-2.0)*this->getDeltaX(),croplry-(magicY-2.0)*this->getDeltaY(),lri,lrj);
      // ncropcol=abs(static_cast<unsigned int>(ceil((croplrx-cropulx)/dx)));
      // ncroprow=abs(static_cast<unsigned int>(ceil((cropuly-croplry)/dy)));
      ncropcol=static_cast<unsigned int>(ceil((croplrx-cropulx)/dx));
      ncroprow=static_cast<unsigned int>(ceil((cropuly-croplry)/dy));
    }
    else{
      double magicX=1,magicY=1;
      // this->getMagicPixel(magicX,magicY);
      cropulx=ulx_opt[0];
      cropuly=uly_opt[0];
      croplrx=lrx_opt[0];
      croplry=lry_opt[0];
      if(forceEUgrid){
        //force to LAEA grid
        Egcs egcs;
        egcs.setLevel(egcs.res2level(dx));
        egcs.force2grid(cropulx,cropuly,croplrx,croplry);
      }
      else if(align_opt[0]){
        if(cropulx>this->getUlx())
          cropulx-=fmod(cropulx-this->getUlx(),dx);
        else if(cropulx<this->getUlx())
          cropulx+=fmod(this->getUlx()-cropulx,dx)-dx;
        if(croplrx<this->getLrx())
          croplrx+=fmod(this->getLrx()-croplrx,dx);
        else if(croplrx>this->getLrx())
          croplrx-=fmod(croplrx-this->getLrx(),dx)+dx;
        if(croplry>this->getLry())
          croplry-=fmod(croplry-this->getLry(),dy);
        else if(croplry<this->getLry())
          croplry+=fmod(this->getLry()-croplry,dy)-dy;
        if(cropuly<this->getUly())
          cropuly+=fmod(this->getUly()-cropuly,dy);
        else if(cropuly>this->getUly())
          cropuly-=fmod(cropuly-this->getUly(),dy)+dy;
      }
      this->geo2image(cropulx+(magicX-1.0)*this->getDeltaX(),cropuly-(magicY-1.0)*this->getDeltaY(),uli,ulj);
      this->geo2image(croplrx+(magicX-2.0)*this->getDeltaX(),croplry-(magicY-2.0)*this->getDeltaY(),lri,lrj);

      // ncropcol=abs(static_cast<unsigned int>(ceil((croplrx-cropulx)/dx)));
      // ncroprow=abs(static_cast<unsigned int>(ceil((cropuly-croplry)/dy)));
      ncropcol=static_cast<unsigned int>(ceil((croplrx-cropulx)/dx));
      ncroprow=static_cast<unsigned int>(ceil((cropuly-croplry)/dy));
      uli=floor(uli);
      ulj=floor(ulj);
      lri=floor(lri);
      lrj=floor(lrj);
    }

    // double deltaX=this->getDeltaX();
    // double deltaY=this->getDeltaY();
    if(!imgWriter.nrOfBand()){//not opened yet
      if(verbose_opt[0]){
        cout << "cropulx: " << cropulx << endl;
        cout << "cropuly: " << cropuly << endl;
        cout << "croplrx: " << croplrx << endl;
        cout << "croplry: " << croplry << endl;
        cout << "ncropcol: " << ncropcol << endl;
        cout << "ncroprow: " << ncroprow << endl;
        cout << "cropulx+ncropcol*dx: " << cropulx+ncropcol*dx << endl;
        cout << "cropuly-ncroprow*dy: " << cropuly-ncroprow*dy << endl;
        cout << "upper left column of input image: " << uli << endl;
        cout << "upper left row of input image: " << ulj << endl;
        cout << "lower right column of input image: " << lri << endl;
        cout << "lower right row of input image: " << lrj << endl;
        cout << "new number of cols: " << ncropcol << endl;
        cout << "new number of rows: " << ncroprow << endl;
        cout << "new number of bands: " << ncropband << endl;
      }
      // string imageType;//=this->getImageType();
      // if(oformat_opt.size())//default
      //   imageType=oformat_opt[0];
      try{
        imgWriter.open(ncropcol,ncroprow,ncropband,theType);
        imgWriter.setNoData(nodata_opt);
        // if(nodata_opt.size()){
        //   imgWriter.setNoData(nodata_opt);
        // }
      }
      catch(string errorstring){
        cout << errorstring << endl;
        throw;
      }
      if(description_opt.size())
        imgWriter.setImageDescription(description_opt[0]);
      double gt[6];
      gt[0]=cropulx;
      gt[1]=dx;
      gt[2]=0;
      gt[3]=cropuly;
      gt[4]=0;
      gt[5]=(this->isGeoRef())? -dy : dy;
      imgWriter.setGeoTransform(gt);
      if(projection_opt.size()){
        if(verbose_opt[0])
          cout << "projection: " << projection_opt[0] << endl;
        imgWriter.setProjectionProj4(projection_opt[0]);
      }
      else
        imgWriter.setProjection(this->getProjection());
      if(imgWriter.getDataType()==GDT_Byte){
        if(colorTable_opt.size()){
          if(colorTable_opt[0]!="none")
            imgWriter.setColorTable(colorTable_opt[0]);
        }
        else if (this->getColorTable()!=NULL)//copy colorTable from input image
          imgWriter.setColorTable(this->getColorTable());
      }
    }

    double startCol=uli;
    double endCol=lri;
    if(uli<0)
      startCol=0;
    else if(uli>=this->nrOfCol())
      startCol=this->nrOfCol()-1;
    if(lri<0)
      endCol=0;
    else if(lri>=this->nrOfCol())
      endCol=this->nrOfCol()-1;
    double startRow=ulj;
    double endRow=lrj;
    if(ulj<0)
      startRow=0;
    else if(ulj>=this->nrOfRow())
      startRow=this->nrOfRow()-1;
    if(lrj<0)
      endRow=0;
    else if(lrj>=this->nrOfRow())
      endRow=this->nrOfRow()-1;

    vector<double> readBuffer;
    unsigned int nband=(band_opt.size())?band_opt.size() : this->nrOfBand();
    for(unsigned int iband=0;iband<nband;++iband){
      unsigned int readBand=(band_opt.size()>iband)?band_opt[iband]:iband;
      if(verbose_opt[0]){
        cout << "extracting band " << readBand << endl;
        MyProgressFunc(progress,pszMessage,pProgressArg);
      }
      double theMin=0;
      double theMax=0;
      if(autoscale_opt.size()){
        try{
          this->getMinMax(static_cast<unsigned int>(startCol),static_cast<unsigned int>(endCol),static_cast<unsigned int>(startRow),static_cast<unsigned int>(endRow),readBand,theMin,theMax);
        }
        catch(string errorString){
          cout << errorString << endl;
        }
        if(verbose_opt[0])
          cout << "minmax: " << theMin << ", " << theMax << endl;
        double theScale=(autoscale_opt[1]-autoscale_opt[0])/(theMax-theMin);
        double theOffset=autoscale_opt[0]-theScale*theMin;
        this->setScale(theScale,readBand);
        this->setOffset(theOffset,readBand);
      }
      else{
        if(scale_opt.size()){
          if(scale_opt.size()>iband)
            this->setScale(scale_opt[iband],readBand);
          else
            this->setScale(scale_opt[0],readBand);
        }
        if(offset_opt.size()){
          if(offset_opt.size()>iband)
            this->setOffset(offset_opt[iband],readBand);
          else
            this->setOffset(offset_opt[0],readBand);
        }
      }

      double readRow=0;
      double readCol=0;
      double lowerCol=0;
      double upperCol=0;
      for(int irow=0;irow<imgWriter.nrOfRow();++irow){
        vector<double> lineMask;
        double x=0;
        double y=0;
        //convert irow to geo
        imgWriter.image2geo(0,irow,x,y);
        //lookup corresponding row for irow in this file
        this->geo2image(x,y,readCol,readRow);
        vector<double> writeBuffer;
        if(readRow<0||readRow>=this->nrOfRow()){
          for(int icol=0;icol<imgWriter.nrOfCol();++icol)
            writeBuffer.push_back(nodataValue);
        }
        else{
          try{
            if(endCol<this->nrOfCol()-1){
              this->readData(readBuffer,startCol,endCol+1,readRow,readBand,theResample);
            }
            else{
              this->readData(readBuffer,startCol,endCol,readRow,readBand,theResample);
            }
            double oldRowMask=-1;//keep track of row mask to optimize number of line readings
            for(int icol=0;icol<imgWriter.nrOfCol();++icol){
              imgWriter.image2geo(icol,irow,x,y);
              //lookup corresponding row for irow in this file
              this->geo2image(x,y,readCol,readRow);
              if(readCol<0||readCol>=this->nrOfCol()){
                writeBuffer.push_back(nodataValue);
              }
              else{
                bool valid=true;
                double geox=0;
                double geoy=0;
                if(maskReader.isInit()){
                  //read mask
                  double colMask=0;
                  double rowMask=0;

                  imgWriter.image2geo(icol,irow,geox,geoy);
                  maskReader.geo2image(geox,geoy,colMask,rowMask);
                  colMask=static_cast<unsigned int>(colMask);
                  rowMask=static_cast<unsigned int>(rowMask);
                  if(rowMask>=0&&rowMask<maskReader.nrOfRow()&&colMask>=0&&colMask<maskReader.nrOfCol()){
                    if(static_cast<unsigned int>(rowMask)!=static_cast<unsigned int>(oldRowMask)){

                      try{
                        maskReader.readData(lineMask,static_cast<unsigned int>(rowMask),0);
                      }
                      catch(string errorstring){
                        cerr << errorstring << endl;
                        throw;
                      }
                      catch(...){
                        cerr << "error caught" << std::endl;
                        throw;
                      }
                      oldRowMask=rowMask;
                    }
                    if(cut_to_cutline_opt[0]){
                      if(lineMask[colMask]!=1){
                        nodataValue=nodata_opt[0];
                        valid=false;
                      }
                    }
                    else if(cut_in_cutline_opt[0]){
                      if(lineMask[colMask]==1){
                        nodataValue=nodata_opt[0];
                        valid=false;
                      }
                    }
                    else{
                      for(int ivalue=0;ivalue<msknodata_opt.size();++ivalue){
                        if(lineMask[colMask]==msknodata_opt[ivalue]){
                          if(nodata_opt.size()>ivalue)
                            nodataValue=nodata_opt[ivalue];
                          valid=false;
                          break;
                        }
                      }
                    }
                  }
                }
                if(!valid)
                  writeBuffer.push_back(nodataValue);
                else{
                  switch(theResample){
                  case(BILINEAR):
                    lowerCol=readCol-0.5;
                    lowerCol=static_cast<unsigned int>(lowerCol);
                    upperCol=readCol+0.5;
                    upperCol=static_cast<unsigned int>(upperCol);
                    if(lowerCol<0)
                      lowerCol=0;
                    if(upperCol>=this->nrOfCol())
                      upperCol=this->nrOfCol()-1;
                    writeBuffer.push_back((readCol-0.5-lowerCol)*readBuffer[upperCol-startCol]+(1-readCol+0.5+lowerCol)*readBuffer[lowerCol-startCol]);
                    break;
                  default:
                    readCol=static_cast<unsigned int>(readCol);
                    readCol-=startCol;//we only start reading from startCol
                    writeBuffer.push_back(readBuffer[readCol]);
                    break;
                  }
                }
              }
            }
          }
          catch(string errorstring){
            cout << errorstring << endl;
            throw;
          }
        }
        if(writeBuffer.size()!=imgWriter.nrOfCol())
          cout << "writeBuffer.size()=" << writeBuffer.size() << ", imgWriter.nrOfCol()=" << imgWriter.nrOfCol() << endl;

        assert(writeBuffer.size()==imgWriter.nrOfCol());
        try{
          imgWriter.writeData(writeBuffer,irow,writeBand);
        }
        catch(string errorstring){
          cout << errorstring << endl;
          throw;
        }
        if(verbose_opt[0]){
          progress=(1.0+irow);
          progress/=imgWriter.nrOfRow();
          MyProgressFunc(progress,pszMessage,pProgressArg);
        }
        else{
          progress=(1.0+irow);
          progress+=(imgWriter.nrOfRow()*writeBand);
          progress/=imgWriter.nrOfBand()*imgWriter.nrOfRow();
          assert(progress>=0);
          assert(progress<=1);
          MyProgressFunc(progress,pszMessage,pProgressArg);
        }
      }
      ++writeBand;
    }
    if(maskReader.isInit())
      maskReader.close();
    return(CE_None);
  }
  catch(string predefinedString){
    std::cout << predefinedString << std::endl;
    throw;
  }
}

/**
 * read the data of the current raster dataset assuming it has not been read yet (otherwise use crop instead). Typically used when current dataset was opened with argument noRead true.
 * @param app application options
 **/
CPLErr Jim::cropDS(Jim& imgWriter, AppFactory& app){
  Optionjl<std::string> resample_opt("r", "resample", "resample: GRIORA_NearestNeighbour|GRIORA_Bilinear|GRIORA_Cubic|GRIORA_CubicSpline|GRIORA_Lanczos|GRIORA_Average|GRIORA_Average|GRIORA_Gauss (check http://www.gdal.org/gdal_8h.html#a640ada511cbddeefac67c548e009d5a)","GRIORA_NearestNeighbour");
  Optionjl<string>  projection_opt("a_srs", "a_srs", "Override the projection for the output file (leave blank to copy from input file, use epsg:3035 to use European projection and force to European grid");
  //todo: support layer names
  Optionjl<string>  extent_opt("e", "extent", "get boundary from extent from polygons in vector file");
  Optionjl<string>  layer_opt("ln", "ln", "layer name of extent to crop");
  Optionjl<double>  ulx_opt("ulx", "ulx", "Upper left x value bounding box", 0.0);
  Optionjl<double>  uly_opt("uly", "uly", "Upper left y value bounding box", 0.0);
  Optionjl<double>  lrx_opt("lrx", "lrx", "Lower right x value bounding box", 0.0);
  Optionjl<double>  lry_opt("lry", "lry", "Lower right y value bounding box", 0.0);
  Optionjl<double>  dx_opt("dx", "dx", "Output resolution in x (in meter) (empty: keep original resolution)");
  Optionjl<double>  dy_opt("dy", "dy", "Output resolution in y (in meter) (empty: keep original resolution)");
  Optionjl<double> cx_opt("x", "x", "x-coordinate of image center to crop (in meter)");
  Optionjl<double> cy_opt("y", "y", "y-coordinate of image center to crop (in meter)");
  Optionjl<double> nx_opt("nx", "nx", "image size in x to crop (in meter)");
  Optionjl<double> ny_opt("ny", "ny", "image size in y to crop (in meter)");
  Optionjl<unsigned int> ns_opt("ns", "ns", "number of samples  to crop (in pixels)");
  Optionjl<unsigned int> nl_opt("nl", "nl", "number of lines to crop (in pixels)");
  Optionjl<unsigned int>  band_opt("b", "band", "band index to crop (leave empty to retain all bands)");
  Optionjl<unsigned int> bstart_opt("sband", "startband", "Start band sequence number");
  Optionjl<unsigned int> bend_opt("eband", "endband", "End band sequence number");
  Optionjl<string>  otype_opt("ot", "otype", "Data type for output image ({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}). Empty string: inherit type from input image");
  // Optionjl<string>  oformat_opt("of", "oformat", "Output image format (see also gdal_translate).","GTiff");
  // Optionjl<string> option_opt("co", "co", "Creation option for output file. Multiple options can be specified.");
  Optionjl<string>  colorTable_opt("ct", "ct", "color table (file with 5 columns: id R G B ALFA (0: transparent, 255: solid)");
  Optionjl<double>  nodata_opt("nodata", "nodata", "Nodata value to put in image if out of bounds.");
  Optionjl<string>  description_opt("d", "description", "Set image description");
  Optionjl<bool>  align_opt("align", "align", "Align output bounding box to input image",false);
  Optionjl<short>  verbose_opt("v", "verbose", "verbose", 0,2);

  extent_opt.setHide(1);
  layer_opt.setHide(1);
  bstart_opt.setHide(1);
  bend_opt.setHide(1);
  // option_opt.setHide(1);
  cx_opt.setHide(1);
  cy_opt.setHide(1);
  nx_opt.setHide(1);
  ny_opt.setHide(1);
  ns_opt.setHide(1);
  nl_opt.setHide(1);
  nodata_opt.setHide(1);
  description_opt.setHide(1);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=projection_opt.retrieveOption(app);
    ulx_opt.retrieveOption(app);
    uly_opt.retrieveOption(app);
    lrx_opt.retrieveOption(app);
    lry_opt.retrieveOption(app);
    band_opt.retrieveOption(app);
    bstart_opt.retrieveOption(app);
    bend_opt.retrieveOption(app);
    otype_opt.retrieveOption(app);
    // oformat_opt.retrieveOption(app);
    colorTable_opt.retrieveOption(app);
    dx_opt.retrieveOption(app);
    dy_opt.retrieveOption(app);
    resample_opt.retrieveOption(app);
    extent_opt.retrieveOption(app);
    layer_opt.retrieveOption(app);
    // option_opt.retrieveOption(app);
    cx_opt.retrieveOption(app);
    cy_opt.retrieveOption(app);
    nx_opt.retrieveOption(app);
    ny_opt.retrieveOption(app);
    ns_opt.retrieveOption(app);
    nl_opt.retrieveOption(app);
    nodata_opt.retrieveOption(app);
    description_opt.retrieveOption(app);
    align_opt.retrieveOption(app);
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

    double nodataValue=nodata_opt.size()? nodata_opt[0] : 0;
    RESAMPLE theResample;
    if(resample_opt[0]=="near"){
      theResample=NEAR;
      if(verbose_opt[0])
        cout << "resampling: nearest neighbor" << endl;
    }
    else if(resample_opt[0]=="bilinear"){
      theResample=BILINEAR;
      if(verbose_opt[0])
        cout << "resampling: bilinear interpolation" << endl;
    }
    else{
      std::cout << "Error: resampling method " << resample_opt[0] << " not supported" << std::endl;
      return(CE_Failure);
    }

    const char* pszMessage;
    void* pProgressArg=NULL;
    GDALProgressFunc pfnProgress=GDALTermProgress;
    double progress=0;
    MyProgressFunc(progress,pszMessage,pProgressArg);
    // ImgReaderGdal imgReader;
    // ImgWriterGdal imgWriter;
    //open input images to extract number of bands and spatial resolution
    int ncropband=0;//total number of bands to write
    double dx=0;
    double dy=0;
    if(dx_opt.size())
      dx=dx_opt[0];
    if(dy_opt.size())
      dy=dy_opt[0];

    try{
      //convert start and end band options to vector of band indexes
      if(bstart_opt.size()){
        if(bend_opt.size()!=bstart_opt.size()){
          string errorstring="Error: options for start and end band indexes must be provided as pairs, missing end band";
          throw(errorstring);
        }
        band_opt.clear();
        for(int ipair=0;ipair<bstart_opt.size();++ipair){
          if(bend_opt[ipair]<=bstart_opt[ipair]){
            string errorstring="Error: index for end band must be smaller then start band";
            throw(errorstring);
          }
          for(unsigned int iband=bstart_opt[ipair];iband<=bend_opt[ipair];++iband)
            band_opt.push_back(iband);
        }
      }
      //image must be georeferenced
      if(!this->isGeoRef()){
        string errorstring="Warning: input image is not georeferenced in cropDS";
        std::cerr << errorstring << std::endl;
        // throw(errorstring);
      }
    }
    catch(string error){
      cerr << error << std::endl;
      throw;
    }


    bool isGeoRef=false;
    string projectionString;
    // for(int iimg=0;iimg<input_opt.size();++iimg){

    if(!isGeoRef)
      isGeoRef=this->isGeoRef();
    if(this->isGeoRef()&&projection_opt.empty())
      projectionString=this->getProjection();
    if(dx_opt.empty()){
      dx=this->getDeltaX();
    }

    if(dy_opt.empty()){
      dy=this->getDeltaY();
    }
    if(band_opt.size())
      ncropband+=band_opt.size();
    else
      ncropband+=this->nrOfBand();

    GDALDataType theType=getGDALDataType();
    if(otype_opt.size()){
      theType=string2GDAL(otype_opt[0]);
      if(theType==GDT_Unknown)
        std::cout << "Warning: unknown output pixel type: " << otype_opt[0] << ", using input type as default" << std::endl;
    }
    if(verbose_opt[0])
      cout << "Output pixel type:  " << GDALGetDataTypeName(theType) << endl;

    //bounding box of cropped image
    double cropulx=ulx_opt[0];
    double cropuly=uly_opt[0];
    double croplrx=lrx_opt[0];
    double croplry=lry_opt[0];
    //get bounding box from extentReader if defined
    VectorOgr extentReader;

    if(extent_opt.size()){
      double e_ulx;
      double e_uly;
      double e_lrx;
      double e_lry;
      for(int iextent=0;iextent<extent_opt.size();++iextent){
        extentReader.open(extent_opt[iextent],layer_opt,true);//noread=true
        extentReader.getExtent(e_ulx,e_uly,e_lrx,e_lry);
        if(!iextent){
          ulx_opt[0]=e_ulx;
          uly_opt[0]=e_uly;
          lrx_opt[0]=e_lrx;
          lry_opt[0]=e_lry;
        }
        else{
          if(e_ulx<ulx_opt[0])
            ulx_opt[0]=e_ulx;
          if(e_uly>uly_opt[0])
            uly_opt[0]=e_uly;
          if(e_lrx>lrx_opt[0])
            lrx_opt[0]=e_lrx;
          if(e_lry<lry_opt[0])
            lry_opt[0]=e_lry;
        }
        extentReader.close();
      }
      if(croplrx>cropulx&&cropulx>ulx_opt[0])
        ulx_opt[0]=cropulx;
      if(croplrx>cropulx&&croplrx<lrx_opt[0])
        lrx_opt[0]=croplrx;
      if(cropuly>croplry&&cropuly<uly_opt[0])
        uly_opt[0]=cropuly;
      if(croplry<cropuly&&croplry>lry_opt[0])
        lry_opt[0]=croplry;
    }
    else if(cx_opt.size()&&cy_opt.size()&&nx_opt.size()&&ny_opt.size()){
      ulx_opt[0]=cx_opt[0]-nx_opt[0]/2.0;
      uly_opt[0]=(isGeoRef) ? cy_opt[0]+ny_opt[0]/2.0 : cy_opt[0]-ny_opt[0]/2.0;
      lrx_opt[0]=cx_opt[0]+nx_opt[0]/2.0;
      lry_opt[0]=(isGeoRef) ? cy_opt[0]-ny_opt[0]/2.0 : cy_opt[0]+ny_opt[0]/2.0;
    }
    else if(cx_opt.size()&&cy_opt.size()&&ns_opt.size()&&nl_opt.size()){
      ulx_opt[0]=cx_opt[0]-ns_opt[0]*dx/2.0;
      uly_opt[0]=(isGeoRef) ? cy_opt[0]+nl_opt[0]*dy/2.0 : cy_opt[0]-nl_opt[0]*dy/2.0;
      lrx_opt[0]=cx_opt[0]+ns_opt[0]*dx/2.0;
      lry_opt[0]=(isGeoRef) ? cy_opt[0]-nl_opt[0]*dy/2.0 : cy_opt[0]+nl_opt[0]*dy/2.0;
    }

    if(verbose_opt[0])
      cout << "--ulx=" << ulx_opt[0] << " --uly=" << uly_opt[0] << " --lrx=" << lrx_opt[0] << " --lry=" << lry_opt[0] << endl;

    int ncropcol=0;
    int ncroprow=0;

    //determine number of output bands
    int writeBand=0;//write band

    if(theType==GDT_Unknown){
      theType=this->getGDALDataType();
      if(verbose_opt[0])
        cout << "Using data type from input image: " << GDALGetDataTypeName(theType) << endl;
    }
    // if(option_opt.findSubstring("INTERLEAVE=")==option_opt.end()){
    //   string theInterleave="INTERLEAVE=";
    //   theInterleave+=this->getInterleave();
    //   option_opt.push_back(theInterleave);
    // }
    // if(verbose_opt[0])
    //   cout << "size of " << input_opt[iimg] << ": " << ncol << " cols, "<< nrow << " rows" << endl;
    // double uli,ulj,lri,lrj;//image coordinates
    bool forceEUgrid=false;
    if(projection_opt.size())
      forceEUgrid=(!(projection_opt[0].compare("EPSG:3035"))||!(projection_opt[0].compare("EPSG:3035"))||projection_opt[0].find("ETRS-LAEA")!=string::npos);
    if(ulx_opt[0]>=lrx_opt[0]){//default bounding box: no cropping
      // uli=0;
      // lri=this->nrOfCol()-1;
      // ulj=0;
      // lrj=this->nrOfRow()-1;
      ncropcol=this->nrOfCol();
      ncroprow=this->nrOfRow();
      this->getBoundingBox(cropulx,cropuly,croplrx,croplry);
      double magicX=1,magicY=1;
      // this->getMagicPixel(magicX,magicY);
      if(forceEUgrid){
        //force to LAEA grid
        Egcs egcs;
        egcs.setLevel(egcs.res2level(dx));
        egcs.force2grid(cropulx,cropuly,croplrx,croplry);
        // this->geo2image(cropulx+(magicX-1.0)*this->getDeltaX(),cropuly-(magicY-1.0)*this->getDeltaY(),uli,ulj);
        // this->geo2image(croplrx+(magicX-2.0)*this->getDeltaX(),croplry-(magicY-2.0)*this->getDeltaY(),lri,lrj);
      }
      // this->geo2image(cropulx+(magicX-1.0)*this->getDeltaX(),cropuly-(magicY-1.0)*this->getDeltaY(),uli,ulj);
      // this->geo2image(croplrx+(magicX-2.0)*this->getDeltaX(),croplry-(magicY-2.0)*this->getDeltaY(),lri,lrj);
      // ncropcol=abs(static_cast<unsigned int>(ceil((croplrx-cropulx)/dx)));
      // ncroprow=abs(static_cast<unsigned int>(ceil((cropuly-croplry)/dy)));
      ncropcol=static_cast<unsigned int>(ceil((croplrx-cropulx)/dx));
      ncroprow=static_cast<unsigned int>(ceil((cropuly-croplry)/dy));
    }
    else{
      double magicX=1,magicY=1;
      // this->getMagicPixel(magicX,magicY);
      cropulx=ulx_opt[0];
      cropuly=uly_opt[0];
      croplrx=lrx_opt[0];
      croplry=lry_opt[0];
      if(forceEUgrid){
        //force to LAEA grid
        Egcs egcs;
        egcs.setLevel(egcs.res2level(dx));
        egcs.force2grid(cropulx,cropuly,croplrx,croplry);
      }
      else if(align_opt[0]){
        if(cropulx>this->getUlx())
          cropulx-=fmod(cropulx-this->getUlx(),dx);
        else if(cropulx<this->getUlx())
          cropulx+=fmod(this->getUlx()-cropulx,dx)-dx;
        if(croplrx<this->getLrx())
          croplrx+=fmod(this->getLrx()-croplrx,dx);
        else if(croplrx>this->getLrx())
          croplrx-=fmod(croplrx-this->getLrx(),dx)+dx;
        if(croplry>this->getLry())
          croplry-=fmod(croplry-this->getLry(),dy);
        else if(croplry<this->getLry())
          croplry+=fmod(this->getLry()-croplry,dy)-dy;
        if(cropuly<this->getUly())
          cropuly+=fmod(this->getUly()-cropuly,dy);
        else if(cropuly>this->getUly())
          cropuly-=fmod(cropuly-this->getUly(),dy)+dy;
      }
      // this->geo2image(cropulx+(magicX-1.0)*this->getDeltaX(),cropuly-(magicY-1.0)*this->getDeltaY(),uli,ulj);
      // this->geo2image(croplrx+(magicX-2.0)*this->getDeltaX(),croplry-(magicY-2.0)*this->getDeltaY(),lri,lrj);

      // ncropcol=abs(static_cast<unsigned int>(ceil((croplrx-cropulx)/dx)));
      // ncropcol=abs(static_cast<unsigned int>(ceil((croplrx-cropulx)/dx)));
      ncroprow=static_cast<unsigned int>(ceil((cropuly-croplry)/dy));
      ncroprow=static_cast<unsigned int>(ceil((cropuly-croplry)/dy));
      // uli=floor(uli);
      // ulj=floor(ulj);
      // lri=floor(lri);
      // lrj=floor(lrj);
    }

    // double deltaX=this->getDeltaX();
    // double deltaY=this->getDeltaY();
    if(!imgWriter.nrOfBand()){//not opened yet
      if(verbose_opt[0]){
        cout << "cropulx: " << cropulx << endl;
        cout << "cropuly: " << cropuly << endl;
        cout << "croplrx: " << croplrx << endl;
        cout << "croplry: " << croplry << endl;
        cout << "ncropcol: " << ncropcol << endl;
        cout << "ncroprow: " << ncroprow << endl;
        cout << "cropulx+ncropcol*dx: " << cropulx+ncropcol*dx << endl;
        cout << "cropuly-ncroprow*dy: " << cropuly-ncroprow*dy << endl;
        // cout << "upper left column of input image: " << uli << endl;
        // cout << "upper left row of input image: " << ulj << endl;
        // cout << "lower right column of input image: " << lri << endl;
        // cout << "lower right row of input image: " << lrj << endl;
        cout << "new number of cols: " << ncropcol << endl;
        cout << "new number of rows: " << ncroprow << endl;
        cout << "new number of bands: " << ncropband << endl;
      }
      // string imageType;//=this->getImageType();
      // if(oformat_opt.size())//default
      //   imageType=oformat_opt[0];
      try{
        imgWriter.open(ncropcol,ncroprow,ncropband,theType);
        imgWriter.setNoData(nodata_opt);
        // if(nodata_opt.size()){
        //   imgWriter.setNoData(nodata_opt);
        // }
      }
      catch(string errorstring){
        cout << errorstring << endl;
        throw;
      }
      if(description_opt.size())
        imgWriter.setImageDescription(description_opt[0]);
      double gt[6];
      gt[0]=cropulx;
      gt[1]=dx;
      gt[2]=0;
      gt[3]=cropuly;
      gt[4]=0;
      gt[5]=(this->isGeoRef())? -dy : dy;
      imgWriter.setGeoTransform(gt);
      if(projection_opt.size()){
        if(verbose_opt[0])
          cout << "projection: " << projection_opt[0] << endl;
        imgWriter.setProjectionProj4(projection_opt[0]);
      }
      else
        imgWriter.setProjection(this->getProjection());
      if(imgWriter.getDataType()==GDT_Byte){
        if(colorTable_opt.size()){
          if(colorTable_opt[0]!="none")
            imgWriter.setColorTable(colorTable_opt[0]);
        }
        else if (this->getColorTable()!=NULL)//copy colorTable from input image
          imgWriter.setColorTable(this->getColorTable());
      }
    }

    // double startCol=uli;
    // double endCol=lri;
    // if(uli<0)
    //   startCol=0;
    // else if(uli>=this->nrOfCol())
    //   startCol=this->nrOfCol()-1;
    // if(lri<0)
    //   endCol=0;
    // else if(lri>=this->nrOfCol())
    //   endCol=this->nrOfCol()-1;
    // double startRow=ulj;
    // double endRow=lrj;
    // if(ulj<0)
    //   startRow=0;
    // else if(ulj>=this->nrOfRow())
    //   startRow=this->nrOfRow()-1;
    // if(lrj<0)
    //   endRow=0;
    // else if(lrj>=this->nrOfRow())
    //   endRow=this->nrOfRow()-1;

    //todo: readDS here
    CPLErr returnValue=CE_None;
    if(m_gds == NULL){
      std::string errorString="Error in readNewBlock";
      throw(errorString);
    }
    // if(m_end[iband]<m_blockSize)//first time
    //   m_end[iband]=m_blockSize;
    // while(row>=m_end[iband]&&m_begin[iband]<nrOfRow()){
    //   m_begin[iband]+=m_blockSize;
    //   m_end[iband]=m_begin[iband]+m_blockSize;
    // }
    // if(m_end[iband]>nrOfRow())
    //   m_end[iband]=nrOfRow();

    int gds_ncol=m_gds->GetRasterXSize();
    int gds_nrow=m_gds->GetRasterYSize();
    int gds_nband=m_gds->GetRasterCount();
    double gds_gt[6];
    m_gds->GetGeoTransform(gds_gt);
    double gds_ulx=gds_gt[0];
    double gds_uly=gds_gt[3];
    double gds_lrx=gds_gt[0]+gds_ncol*gds_gt[1]+gds_nrow*gds_gt[2];
    double gds_lry=gds_gt[3]+gds_ncol*gds_gt[4]+gds_nrow*gds_gt[5];
    double gds_dx=gds_gt[1];
    double gds_dy=-gds_gt[5];
    double diffXm=getUlx()-gds_ulx;
    // double diffYm=gds_uly-getUly();

    // double dfXSize=diffXm/gds_dx;
    double dfXSize=(getLrx()-getUlx())/gds_dx;//x-size in pixels of region to read in original image
    double dfXOff=diffXm/gds_dx;
    // double dfYSize=diffYm/gds_dy;
    // double dfYSize=(getUly()-getLry())/gds_dy;//y-size in piyels of region to read in original image
    // double dfYOff=diffYm/gds_dy;
    // int nYOff=static_cast<int>(dfYOff);
    // int nXSize=abs(static_cast<unsigned int>(ceil((getLrx()-getUlx())/gds_dx)));//x-size in pixels of region to read in original image
    int nXSize=static_cast<unsigned int>(ceil((getLrx()-getUlx())/gds_dx));//x-size in pixels of region to read in original image
    int nXOff=static_cast<int>(dfXOff);
    if(nXSize>gds_ncol)
      nXSize=gds_ncol;

    double dfYSize=0;
    double dfYOff=0;
    int nYSize=0;
    int nYOff=0;

    GDALRasterIOExtraArg sExtraArg;
    INIT_RASTERIO_EXTRA_ARG(sExtraArg);
    sExtraArg.eResampleAlg = m_resample;
    for(int iband=0;iband<m_nband;++iband){
      //fetch raster band
      GDALRasterBand  *poBand;
      if(nrOfBand()<=iband){
        std::string errorString="Error: band number exceeds available bands in readNewBlock";
        throw(errorString);
      }
      poBand = m_gds->GetRasterBand(iband+1);//GDAL uses 1 based index

      dfYSize=(m_end[iband]-m_begin[iband])*getDeltaY()/gds_dy;//y-size in pixels of region to read in original image
      // nYSize=abs(static_cast<unsigned int>(ceil((m_end[iband]-m_begin[iband])*getDeltaY()/gds_dy)));//y-size in pixels of region to read in original image
      nYSize=static_cast<unsigned int>(ceil((m_end[iband]-m_begin[iband])*getDeltaY()/gds_dy));//y-size in pixels of region to read in original image
      if(nYSize>gds_nrow)
        nYSize=gds_nrow;
      dfYOff=(gds_uly-getUly())/gds_dy+m_begin[iband]*getDeltaY()/gds_dy;
      nYOff=static_cast<int>(dfYOff);
      if(poBand->GetOverviewCount()){
        //calculate number of desired samples in overview
        // int nDesiredSamples=abs(static_cast<unsigned int>(ceil((gds_lrx-gds_ulx)/getDeltaX()))*static_cast<unsigned int>(ceil((gds_uly-gds_lry)/getDeltaY())));
        int nDesiredSamples=static_cast<unsigned int>(ceil((gds_lrx-gds_ulx)/getDeltaX()))*static_cast<unsigned int>(ceil((gds_uly-gds_lry)/getDeltaY()));
        poBand=poBand->GetRasterSampleOverview(nDesiredSamples);
        if(poBand->GetXSize()*poBand->GetYSize()<nDesiredSamples){
          //should never be entered as GetRasterSampleOverview must return best overview or original band in worst case...
          // std::cout << "Warning: not enough samples in best overview, falling back to original band" << std::endl;
          poBand = m_gds->GetRasterBand(iband+1);//GDAL uses 1 based index
        }
        int ods_ncol=poBand->GetXSize();
        int ods_nrow=poBand->GetYSize();
        double ods_dx=gds_dx*gds_ncol/ods_ncol;
        double ods_dy=gds_dy*gds_nrow/ods_nrow;

        // dfXSize=diffXm/ods_dx;
        dfXSize=(getLrx()-getUlx())/ods_dx;
        // nXSize=abs(static_cast<unsigned int>(ceil((getLrx()-getUlx())/ods_dx)));//x-size in pixels of region to read in overview image
        nXSize=static_cast<unsigned int>(ceil((getLrx()-getUlx())/ods_dx));//x-size in pixels of region to read in overview image
        if(nXSize>ods_ncol)
          nXSize=ods_ncol;
        dfXOff=diffXm/ods_dx;
        nXOff=static_cast<int>(dfXOff);
        dfYSize=(m_end[iband]-m_begin[iband])*getDeltaY()/ods_dy;//y-size in pixels of region to read in overview image
        // nYSize=abs(static_cast<unsigned int>(ceil((m_end[iband]-m_begin[iband])*getDeltaY()/ods_dy)));//y-size in pixels of region to read in overview image
        nYSize=static_cast<unsigned int>(ceil((m_end[iband]-m_begin[iband])*getDeltaY()/ods_dy));//y-size in pixels of region to read in overview image
        if(nYSize>ods_nrow)
          nYSize=ods_nrow;
        dfYOff=(gds_uly-getUly())/ods_dy+m_begin[iband]*getDeltaY()/ods_dy;
        nYOff=static_cast<int>(dfYOff);
      }
      if(dfXOff-nXOff>0||dfYOff-nYOff>0||getDeltaX()<gds_dx||getDeltaX()>gds_dx||getDeltaY()<gds_dy||getDeltaY()>gds_dy){
        sExtraArg.bFloatingPointWindowValidity = TRUE;
        sExtraArg.dfXOff = dfXOff;
        sExtraArg.dfYOff = dfYOff;
        sExtraArg.dfXSize = dfXSize;
        sExtraArg.dfYSize = dfYSize;
      }
      else{
        sExtraArg.bFloatingPointWindowValidity = FALSE;
        sExtraArg.dfXOff = 0;
        sExtraArg.dfYOff = 0;
        sExtraArg.dfXSize = dfXSize;
        sExtraArg.dfYSize = dfYSize;
      }
      // //test
      // std::cout << "nXOff: " << nXOff << std::endl;
      // std::cout << "nYOff: " << nYOff << std::endl;
      // std::cout << "dfXOff: " << dfXOff << std::endl;
      // std::cout << "dfYOff: " << dfYOff << std::endl;
      // std::cout << "nXSize: " << nXSize << std::endl;
      // std::cout << "nYSize: " << nYSize << std::endl;
      // std::cout << "nrOfCol(): " << nrOfCol() << std::endl;
      // std::cout << "nrOfRow(): " << nrOfRow() << std::endl;
      // std::cout << "getDeltaX(): " << getDeltaX() << std::endl;
      // std::cout << "getDeltaY(): " << getDeltaY() << std::endl;
      // std::cout << "gds_dx: " << gds_dx << std::endl;
      // std::cout << "gds_dy: " << gds_dy << std::endl;
      // std::cout << "getUlx(): " << getUlx() << std::endl;
      // std::cout << "getUly(): " << getUly() << std::endl;
      // std::cout << "gds_ulx: " << gds_ulx << std::endl;
      // std::cout << "gds_uly: " << gds_uly << std::endl;
      // eRWFlag	Either GF_Read to read a region of data, or GF_Write to write a region of data.
      // nXOff	The pixel offset to the top left corner of the region of the band to be accessed. This would be zero to start from the left side.
      // nYOff	The line offset to the top left corner of the region of the band to be accessed. This would be zero to start from the top.
      // nXSize	The width of the region of the band to be accessed in pixels.
      // nYSize	The height of the region of the band to be accessed in lines.
      // pData	The buffer into which the data should be read, or from which it should be written. This buffer must contain at least nBufXSize * nBufYSize words of type eBufType. It is organized in left to right, top to bottom pixel order. Spacing is controlled by the nPixelSpace, and nLineSpace parameters.
      // nBufXSize	the width of the buffer image into which the desired region is to be read, or from which it is to be written.
      // nBufYSize	the height of the buffer image into which the desired region is to be read, or from which it is to be written.
      // eBufType	the type of the pixel values in the pData data buffer. The pixel values will automatically be translated to/from the GDALRasterBand data type as needed.
      // nPixelSpace	The byte offset from the start of one pixel value in pData to the start of the next pixel value within a scanline. If defaulted (0) the size of the datatype eBufType is used.
      // nLineSpace	The byte offset from the start of one scanline in pData to the start of the next. If defaulted (0) the size of the datatype eBufType * nBufXSize is used.
      // psExtraArg	(new in GDAL 2.0) pointer to a GDALRasterIOExtraArg structure with additional arguments to specify resampling and progress callback, or NULL for default behaviour. The GDAL_RASTERIO_RESAMPLING configuration option can also be defined to override the default resampling to one of BILINEAR, CUBIC, CUBICSPLINE, LANCZOS, AVERAGE or MODE.

      return(poBand->RasterIO(GF_Read,nXOff,nYOff+m_begin[iband],nXSize,nYSize,imgWriter.getDataPointer(iband),imgWriter.nrOfCol(),imgWriter.nrOfRow(),imgWriter.getGDALDataType(),0,0,&sExtraArg));
    }
  }
  catch(string predefinedString){
    std::cout << predefinedString << std::endl;
    throw;
  }
}

shared_ptr<Jim> Jim::createct(app::AppFactory& app){
  shared_ptr<Jim> imgWriter=Jim::createImg();
  createct(*imgWriter, app);
  return(imgWriter);
}

CPLErr Jim::createct(Jim& imgWriter, app::AppFactory& app){
  Optionjl<double> min_opt("min", "min", "minimum value", 0);
  Optionjl<double> max_opt("max", "max", "maximum value", 100);
  Optionjl<bool> grey_opt("g", "grey", "grey scale", false);
  Optionjl<string> colorTable_opt("ct", "ct", "color table (file with 5 columns: id R G B ALFA (0: transparent, 255: solid)");
  Optionjl<bool> verbose_opt("v", "verbose", "verbose", false,2);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  doProcess=min_opt.retrieveOption(app);
  max_opt.retrieveOption(app);
  grey_opt.retrieveOption(app);
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

  GDALColorTable colorTable;
  GDALColorEntry sEntry;
  if(colorTable_opt.empty()){
    sEntry.c4=255;
    for(int i=min_opt[0];i<=max_opt[0];++i){
      if(grey_opt[0]){
        sEntry.c1=255*(i-min_opt[0])/(max_opt[0]-min_opt[0]);
        sEntry.c2=255*(i-min_opt[0])/(max_opt[0]-min_opt[0]);
        sEntry.c3=255*(i-min_opt[0])/(max_opt[0]-min_opt[0]);
      }
      else{//hot to cold colour ramp
        sEntry.c1=255;
        sEntry.c2=255;
        sEntry.c3=255;
        double delta=max_opt[0]-min_opt[0];
        if(i<(min_opt[0]+0.25*delta)){
          sEntry.c1=0;
          sEntry.c2=255*4*(i-min_opt[0])/delta;
        }
        else if(i<(min_opt[0]+0.5*delta)){
          sEntry.c1=0;
          sEntry.c3=255*(1+4*(min_opt[0]+0.25*delta-i)/delta);
        }
        else if(i<(min_opt[0]+0.75*delta)){
          sEntry.c1=255*4*(i-min_opt[0]-0.5*delta)/delta;
          sEntry.c3=0;
        }
        else{
          sEntry.c2=255*(1+4*(min_opt[0]+0.75*delta-i)/delta);
          sEntry.c3=0;
        }
      }
      colorTable.SetColorEntry(i,&sEntry);
      // if(output_opt.empty())
      //   cout << i << " " << sEntry.c1 << " " << sEntry.c2 << " " << sEntry.c3 << " " << sEntry.c4 << endl;
    }
  }
  imgWriter.open(nrOfCol(),nrOfRow(),1,GDT_Byte);
  std::vector<double> gt;
  getGeoTransform(gt);
  imgWriter.setGeoTransform(gt);
  imgWriter.setProjection(getProjection());
  if(colorTable_opt.size()){
    if(colorTable_opt[0]!="none")
      imgWriter.setColorTable(colorTable_opt[0]);
  }
  else
    imgWriter.setColorTable(&colorTable);
  switch(getDataType()){
  case(GDT_Byte):{
    vector<char> buffer;
    for(unsigned int irow=0;irow<nrOfRow();++irow){
      readData(buffer,irow);
      imgWriter.writeData(buffer,irow);
    }
    break;
  }
  case(GDT_Int16):{
    vector<short> buffer;
    cout << "Warning: copying short to unsigned short without conversion, use convert with -scale if needed..." << endl;
    for(unsigned int irow=0;irow<nrOfRow();++irow){
      readData(buffer,irow);
      imgWriter.writeData(buffer,irow);
    }
    break;
  }
  case(GDT_UInt16):{
    vector<unsigned short> buffer;
    for(unsigned int irow=0;irow<nrOfRow();++irow){
      readData(buffer,irow);
      imgWriter.writeData(buffer,irow);
    }
    break;
  }
  default:
    cerr << "data type " << getDataType() << " not supported for adding a colortable" << endl;
    break;
  }
  return(CE_None);
}
/**
 * @param app application specific option arguments
 * @return output image
 **/
shared_ptr<Jim> JimList::crop(AppFactory& app){
  shared_ptr<Jim> imgWriter=Jim::createImg();
  crop(*imgWriter, app);
  return(imgWriter);
}

/**
 * @param imgWriter output raster crop dataset
 * @return CE_None if successful, CE_Failure if failed
 **/
//todo: support extent a VectorOgr argument instead of option in app
 JimList& JimList::crop(Jim& imgWriter, AppFactory& app){
   Optionjl<string>  projection_opt("a_srs", "a_srs", "Override the projection for the output file (leave blank to copy from input file, use epsg:3035 to use European projection and force to European grid");
   //todo: support layer names
   Optionjl<string>  extent_opt("e", "extent", "get boundary from extent from polygons in vector file");
   Optionjl<string>  layer_opt("ln", "ln", "layer name of extent to crop");
   Optionjl<bool> cut_to_cutline_opt("cut_to_cutline", "crop_to_cutline", "Crop the extent of the target dataset to the extent of the cutline, setting the outside area to nodata.",false);
   Optionjl<bool> cut_in_cutline_opt("cut_in_cutline", "crop_in_cutline", "Crop the extent of the target dataset to the extent of the cutline, setting the inner area to nodata.",false);
   Optionjl<string> eoption_opt("eo","eo", "special extent options controlling rasterization: ATTRIBUTE|CHUNKYSIZE|ALL_TOUCHED|BURN_VALUE_FROM|MERGE_ALG, e.g., -eo ATTRIBUTE=fieldname");
   Optionjl<string> mask_opt("m", "mask", "Use the the specified file as a validity mask (0 is nodata).");
   Optionjl<double> msknodata_opt("msknodata", "msknodata", "Mask value not to consider for crop.", 0);
   Optionjl<unsigned int> mskband_opt("mskband", "mskband", "Mask band to read (0 indexed)", 0);
   Optionjl<double>  ulx_opt("ulx", "ulx", "Upper left x value bounding box", 0.0);
   Optionjl<double>  uly_opt("uly", "uly", "Upper left y value bounding box", 0.0);
   Optionjl<double>  lrx_opt("lrx", "lrx", "Lower right x value bounding box", 0.0);
   Optionjl<double>  lry_opt("lry", "lry", "Lower right y value bounding box", 0.0);
   Optionjl<double>  dx_opt("dx", "dx", "Output resolution in x (in meter) (empty: keep original resolution)");
   Optionjl<double>  dy_opt("dy", "dy", "Output resolution in y (in meter) (empty: keep original resolution)");
   Optionjl<double> cx_opt("x", "x", "x-coordinate of image center to crop (in meter)");
   Optionjl<double> cy_opt("y", "y", "y-coordinate of image center to crop (in meter)");
   Optionjl<double> nx_opt("nx", "nx", "image size in x to crop (in meter)");
   Optionjl<double> ny_opt("ny", "ny", "image size in y to crop (in meter)");
   Optionjl<unsigned int> ns_opt("ns", "ns", "number of samples  to crop (in pixels)");
   Optionjl<unsigned int> nl_opt("nl", "nl", "number of lines to crop (in pixels)");
   Optionjl<unsigned int>  band_opt("b", "band", "band index to crop (leave empty to retain all bands)");
   Optionjl<unsigned int> bstart_opt("sband", "startband", "Start band sequence number");
   Optionjl<unsigned int> bend_opt("eband", "endband", "End band sequence number");
   Optionjl<double> autoscale_opt("as", "autoscale", "scale output to min and max, e.g., --autoscale 0 --autoscale 255");
   Optionjl<double> scale_opt("scale", "scale", "output=scale*input+offset");
   Optionjl<double> offset_opt("offset", "offset", "output=scale*input+offset");
   Optionjl<string>  otype_opt("ot", "otype", "Data type for output image ({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}). Empty string: inherit type from input image");
   // Optionjl<string>  oformat_opt("of", "oformat", "Output image format (see also gdal_translate).","GTiff");
   // Optionjl<string> option_opt("co", "co", "Creation option for output file. Multiple options can be specified.");
   Optionjl<string>  colorTable_opt("ct", "ct", "color table (file with 5 columns: id R G B ALFA (0: transparent, 255: solid)");
   Optionjl<double>  nodata_opt("nodata", "nodata", "Nodata value to put in image if out of bounds.");
   Optionjl<string>  resample_opt("r", "resample", "Resampling method (near: nearest neighbor, bilinear: bi-linear interpolation).", "near");
   Optionjl<string>  description_opt("d", "description", "Set image description");
   Optionjl<bool>  align_opt("align", "align", "Align output bounding box to input image",false);
   Optionjl<short>  verbose_opt("v", "verbose", "verbose", 0,2);

   extent_opt.setHide(1);
   layer_opt.setHide(1);
   cut_to_cutline_opt.setHide(1);
   cut_in_cutline_opt.setHide(1);
   eoption_opt.setHide(1);
   bstart_opt.setHide(1);
   bend_opt.setHide(1);
   mask_opt.setHide(1);
   msknodata_opt.setHide(1);
   mskband_opt.setHide(1);
   // option_opt.setHide(1);
   cx_opt.setHide(1);
   cy_opt.setHide(1);
   nx_opt.setHide(1);
   ny_opt.setHide(1);
   ns_opt.setHide(1);
   nl_opt.setHide(1);
   scale_opt.setHide(1);
   offset_opt.setHide(1);
   nodata_opt.setHide(1);
   description_opt.setHide(1);

   bool doProcess;//stop process when program was invoked with help option (-h --help)
   try{
     doProcess=projection_opt.retrieveOption(app);
     ulx_opt.retrieveOption(app);
     uly_opt.retrieveOption(app);
     lrx_opt.retrieveOption(app);
     lry_opt.retrieveOption(app);
     band_opt.retrieveOption(app);
     bstart_opt.retrieveOption(app);
     bend_opt.retrieveOption(app);
     autoscale_opt.retrieveOption(app);
     otype_opt.retrieveOption(app);
     // oformat_opt.retrieveOption(app);
     colorTable_opt.retrieveOption(app);
     dx_opt.retrieveOption(app);
     dy_opt.retrieveOption(app);
     resample_opt.retrieveOption(app);
     extent_opt.retrieveOption(app);
     layer_opt.retrieveOption(app);
     cut_to_cutline_opt.retrieveOption(app);
     cut_in_cutline_opt.retrieveOption(app);
     eoption_opt.retrieveOption(app);
     mask_opt.retrieveOption(app);
     msknodata_opt.retrieveOption(app);
     mskband_opt.retrieveOption(app);
     // option_opt.retrieveOption(app);
     cx_opt.retrieveOption(app);
     cy_opt.retrieveOption(app);
     nx_opt.retrieveOption(app);
     ny_opt.retrieveOption(app);
     ns_opt.retrieveOption(app);
     nl_opt.retrieveOption(app);
     scale_opt.retrieveOption(app);
     offset_opt.retrieveOption(app);
     nodata_opt.retrieveOption(app);
     description_opt.retrieveOption(app);
     align_opt.retrieveOption(app);
     verbose_opt.retrieveOption(app);

     if(!doProcess){
       cout << endl;
       std::ostringstream helpStream;
       helpStream << "short option -h shows basic options only, use long option --help to show all options" << std::endl;
       throw(helpStream.str());//help was invoked, stop processing
     }
     if(empty()){
       std::ostringstream errorStream;
       errorStream << "Input collection is empty. Use --help for more help information" << std::endl;
       throw(errorStream.str());
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

     double nodataValue=nodata_opt.size()? nodata_opt[0] : 0;
     RESAMPLE theResample;
     if(resample_opt[0]=="near"){
       theResample=NEAR;
       if(verbose_opt[0])
         cout << "resampling: nearest neighbor" << endl;
     }
     else if(resample_opt[0]=="bilinear"){
       theResample=BILINEAR;
       if(verbose_opt[0])
         cout << "resampling: bilinear interpolation" << endl;
     }
     else{
       std::ostringstream errorStream;
       errorStream << "Error: resampling method " << resample_opt[0] << " not supported" << std::endl;
       throw(errorStream.str());
       // return(CE_Failure);
     }

     const char* pszMessage;
     void* pProgressArg=NULL;
     GDALProgressFunc pfnProgress=GDALTermProgress;
     double progress=0;
     MyProgressFunc(progress,pszMessage,pProgressArg);
     // ImgReaderGdal imgReader;
     // ImgWriterGdal imgWriter;
     //open input images to extract number of bands and spatial resolution
     int ncropband=0;//total number of bands to write
     double dx=0;
     double dy=0;
     if(dx_opt.size())
       dx=dx_opt[0];
     if(dy_opt.size())
       dy=dy_opt[0];

     try{
       //convert start and end band options to vector of band indexes
       if(bstart_opt.size()){
         if(bend_opt.size()!=bstart_opt.size()){
           string errorstring="Error: options for start and end band indexes must be provided as pairs, missing end band";
           throw(errorstring);
         }
         band_opt.clear();
         for(int ipair=0;ipair<bstart_opt.size();++ipair){
           if(bend_opt[ipair]<=bstart_opt[ipair]){
             string errorstring="Error: index for end band must be smaller then start band";
             throw(errorstring);
           }
           for(unsigned int iband=bstart_opt[ipair];iband<=bend_opt[ipair];++iband)
             band_opt.push_back(iband);
         }
       }
     }
     catch(string error){
       throw;
       // return(CE_Failure);
     }

     bool isGeoRef=false;
     string projectionString;
     // for(int iimg=0;iimg<input_opt.size();++iimg){

     // std::vector<std::shared_ptr<Jim> >::const_iterator imit=begin();
     std::list<std::shared_ptr<Jim> >::const_iterator imit=begin();

     for(imit=begin();imit!=end();++imit){
       //image must be georeferenced
       if(!((*imit)->isGeoRef())){
         string errorstring="Warning: input image is not georeferenced in JimList";
         std::cerr << errorstring << std::endl;
         // throw(errorstring);
       }
       // while((imgReader=getNextImage())){
       // for(int iimg=0;iimg<imgReader.size();++iimg){
       // try{
       // }
       // catch(string error){
       //   cerr << "Error: could not open file " << input_opt[iimg] << ": " << error << std::endl;
       //   exit(1);
       // }
       if(!isGeoRef)
         isGeoRef=(*imit)->isGeoRef();
       if((*imit)->isGeoRef()&&projection_opt.empty())
         projectionString=(*imit)->getProjection();
       if(dx_opt.empty()){
         if(imit==begin()||(*imit)->getDeltaX()<dx)
           dx=(*imit)->getDeltaX();
         if(dx<=0){
           string errorstring="Warning: pixel size in x has not been defined in input image";
           std::cerr << errorstring << std::endl;
           dx=1;
           // throw(errorstring);
         }
       }

       if(dy_opt.empty()){
         if(imit==begin()||(*imit)->getDeltaY()<dy)
           dy=(*imit)->getDeltaY();
         if(dy<=0){
           string errorstring="Warning: pixel size in y has not been defined in input image";
           std::cerr << errorstring << std::endl;
           dy=1;
           // throw(errorstring);
         }
       }
       if(band_opt.size())
         ncropband+=band_opt.size();
       else
         ncropband+=(*imit)->nrOfBand();
       // (*imit)->close();
     }

     GDALDataType theType=GDT_Unknown;
     if(otype_opt.size()){
       theType=string2GDAL(otype_opt[0]);
       if(theType==GDT_Unknown)
         std::cout << "Warning: unknown output pixel type: " << otype_opt[0] << ", using input type as default" << std::endl;
     }
     if(verbose_opt[0])
       cout << "Output pixel type:  " << GDALGetDataTypeName(theType) << endl;

     //bounding box of cropped image
     double cropulx=ulx_opt[0];
     double cropuly=uly_opt[0];
     double croplrx=lrx_opt[0];
     double croplry=lry_opt[0];
     //get bounding box from extentReader if defined
     VectorOgr extentReader;

     if(extent_opt.size()){
       double e_ulx;
       double e_uly;
       double e_lrx;
       double e_lry;
       for(int iextent=0;iextent<extent_opt.size();++iextent){
         extentReader.open(extent_opt[iextent],layer_opt,true);//noread=true
         extentReader.getExtent(e_ulx,e_uly,e_lrx,e_lry);
         if(!iextent){
           ulx_opt[0]=e_ulx;
           uly_opt[0]=e_uly;
           lrx_opt[0]=e_lrx;
           lry_opt[0]=e_lry;
         }
         else{
           if(e_ulx<ulx_opt[0])
             ulx_opt[0]=e_ulx;
           if(e_uly>uly_opt[0])
             uly_opt[0]=e_uly;
           if(e_lrx>lrx_opt[0])
             lrx_opt[0]=e_lrx;
           if(e_lry<lry_opt[0])
             lry_opt[0]=e_lry;
         }
         extentReader.close();
       }
       if(croplrx>cropulx&&cropulx>ulx_opt[0])
         ulx_opt[0]=cropulx;
       if(croplrx>cropulx&&croplrx<lrx_opt[0])
         lrx_opt[0]=croplrx;
       if(cropuly>croplry&&cropuly<uly_opt[0])
         uly_opt[0]=cropuly;
       if(croplry<cropuly&&croplry>lry_opt[0])
         lry_opt[0]=croplry;
       if(cut_to_cutline_opt[0]||cut_in_cutline_opt[0]||eoption_opt.size())
         extentReader.open(extent_opt[0],layer_opt,true);
     }
     else if(cx_opt.size()&&cy_opt.size()&&nx_opt.size()&&ny_opt.size()){
       ulx_opt[0]=cx_opt[0]-nx_opt[0]/2.0;
       uly_opt[0]=(isGeoRef) ? cy_opt[0]+ny_opt[0]/2.0 : cy_opt[0]-ny_opt[0]/2.0;
       lrx_opt[0]=cx_opt[0]+nx_opt[0]/2.0;
       lry_opt[0]=(isGeoRef) ? cy_opt[0]-ny_opt[0]/2.0 : cy_opt[0]+ny_opt[0]/2.0;
     }
     else if(cx_opt.size()&&cy_opt.size()&&ns_opt.size()&&nl_opt.size()){
       ulx_opt[0]=cx_opt[0]-ns_opt[0]*dx/2.0;
       uly_opt[0]=(isGeoRef) ? cy_opt[0]+nl_opt[0]*dy/2.0 : cy_opt[0]-nl_opt[0]*dy/2.0;
       lrx_opt[0]=cx_opt[0]+ns_opt[0]*dx/2.0;
       lry_opt[0]=(isGeoRef) ? cy_opt[0]-nl_opt[0]*dy/2.0 : cy_opt[0]+nl_opt[0]*dy/2.0;
     }

     if(verbose_opt[0])
       cout << "--ulx=" << ulx_opt[0] << " --uly=" << uly_opt[0] << " --lrx=" << lrx_opt[0] << " --lry=" << lry_opt[0] << endl;

     int ncropcol=0;
     int ncroprow=0;

     Jim maskReader;
     if(extent_opt.size()&&(cut_to_cutline_opt[0]||cut_in_cutline_opt[0]||eoption_opt.size())){
       if(mask_opt.size()){
         string errorString="Error: can only either mask or extent extent with cut_to_cutline / cut_in_cutline, not both";
         throw(errorString);
       }
       try{
         // ncropcol=abs(static_cast<unsigned int>(ceil((lrx_opt[0]-ulx_opt[0])/dx)));
         // ncroprow=abs(static_cast<unsigned int>(ceil((uly_opt[0]-lry_opt[0])/dy)));
         ncropcol=static_cast<unsigned int>(ceil((lrx_opt[0]-ulx_opt[0])/dx));
         ncroprow=static_cast<unsigned int>(ceil((uly_opt[0]-lry_opt[0])/dy));
         maskReader.open(ncropcol,ncroprow,1,GDT_Float64);
         double gt[6];
         gt[0]=ulx_opt[0];
         gt[1]=dx;
         gt[2]=0;
         gt[3]=uly_opt[0];
         gt[4]=0;
         gt[5]=-dy;
         maskReader.setGeoTransform(gt);
         if(projection_opt.size())
           maskReader.setProjectionProj4(projection_opt[0]);
         else if(projectionString.size())
           maskReader.setProjection(projectionString);

         // vector<double> burnValues(1,1);//burn value is 1 (single band)
         // maskReader.rasterizeBuf(extentReader,msknodata_opt[0],eoption_opt,layer_opt);
         maskReader.rasterizeBuf(extentReader,1,eoption_opt,layer_opt);

         // if(eoption_opt.size())
         //   maskReader.rasterizeBuf(extentReader,eoption_opt);
         // else
         //   maskReader.rasterizeBuf(extentReader);
       }
       catch(string error){
         throw;
         // return(CE_Failure);
       }
     }
     else if(mask_opt.size()==1){
       try{
         //there is only a single mask
         maskReader.open(mask_opt[0]);
         if(mskband_opt[0]>=maskReader.nrOfBand()){
           string errorString="Error: illegal mask band";
           throw(errorString);
         }
       }
       catch(string error){
         throw;
         // return(CE_Failure);
       }
     }

     //determine number of output bands
     int writeBand=0;//write band

     if(scale_opt.size()){
       while(scale_opt.size()<band_opt.size())
         scale_opt.push_back(scale_opt[0]);
     }
     if(offset_opt.size()){
       while(offset_opt.size()<band_opt.size())
         offset_opt.push_back(offset_opt[0]);
     }
     if(autoscale_opt.size()){
       assert(autoscale_opt.size()%2==0);
     }

     // for(int iimg=0;iimg<input_opt.size();++iimg){
     for(imit=begin();imit!=end();++imit){
       // for(int iimg=0;iimg<imgReader.size();++iimg){
       // if(verbose_opt[0])
       //   cout << "opening image " << input_opt[iimg] << endl;
       // try{
       // }
       // catch(string error){
       //   cerr << error << std::endl;
       //   exit(2);
       // }
       //if output type not set, get type from input image
       if(theType==GDT_Unknown){
         theType=(*imit)->getGDALDataType();
         if(verbose_opt[0])
           cout << "Using data type from input image: " << GDALGetDataTypeName(theType) << endl;
       }
       // if(option_opt.findSubstring("INTERLEAVE=")==option_opt.end()){
       //   string theInterleave="INTERLEAVE=";
       //   theInterleave+=(*imit)->getInterleave();
       //   option_opt.push_back(theInterleave);
       // }
       // if(verbose_opt[0])
       //   cout << "size of " << input_opt[iimg] << ": " << ncol << " cols, "<< nrow << " rows" << endl;
       double uli,ulj,lri,lrj;//image coordinates
       bool forceEUgrid=false;
       if(projection_opt.size())
         forceEUgrid=(!(projection_opt[0].compare("EPSG:3035"))||!(projection_opt[0].compare("EPSG:3035"))||projection_opt[0].find("ETRS-LAEA")!=string::npos);
       if(ulx_opt[0]>=lrx_opt[0]){//default bounding box: no cropping
         uli=0;
         lri=(*imit)->nrOfCol()-1;
         ulj=0;
         lrj=(*imit)->nrOfRow()-1;
         ncropcol=(*imit)->nrOfCol();
         ncroprow=(*imit)->nrOfRow();
         (*imit)->getBoundingBox(cropulx,cropuly,croplrx,croplry);
         double magicX=1,magicY=1;
         // (*imit)->getMagicPixel(magicX,magicY);
         if(forceEUgrid){
           //force to LAEA grid
           Egcs egcs;
           egcs.setLevel(egcs.res2level(dx));
           egcs.force2grid(cropulx,cropuly,croplrx,croplry);
           (*imit)->geo2image(cropulx+(magicX-1.0)*(*imit)->getDeltaX(),cropuly-(magicY-1.0)*(*imit)->getDeltaY(),uli,ulj);
           (*imit)->geo2image(croplrx+(magicX-2.0)*(*imit)->getDeltaX(),croplry-(magicY-2.0)*(*imit)->getDeltaY(),lri,lrj);
         }
         (*imit)->geo2image(cropulx+(magicX-1.0)*(*imit)->getDeltaX(),cropuly-(magicY-1.0)*(*imit)->getDeltaY(),uli,ulj);
         (*imit)->geo2image(croplrx+(magicX-2.0)*(*imit)->getDeltaX(),croplry-(magicY-2.0)*(*imit)->getDeltaY(),lri,lrj);
         // ncropcol=abs(static_cast<unsigned int>(ceil((croplrx-cropulx)/dx)));
         // ncroprow=abs(static_cast<unsigned int>(ceil((cropuly-croplry)/dy)));
         ncropcol=static_cast<unsigned int>(ceil((croplrx-cropulx)/dx));
         ncroprow=static_cast<unsigned int>(ceil((cropuly-croplry)/dy));
         if(verbose_opt[0]){
           cout << "default bounding box" << endl;
           cout << "ulx_opt[0]: " << ulx_opt[0]<< endl;
           cout << "uly_opt[0]: " << uly_opt[0]<< endl;
           cout << "lrx_opt[0]: " << lrx_opt[0]<< endl;
           cout << "lry_opt[0]: " << lry_opt[0]<< endl;
           cout << "croplrx,cropulx: " << croplrx << "," << cropulx << endl;
           cout << "dx: " << dx << endl;
           cout << "cropuly,croplry: " << cropuly << "," << croplry << endl;
           cout << "dy: " << dy << endl;
           cout << "filename: " << (*imit)->getFileName() << endl;
         }
       }
       else{
         double magicX=1,magicY=1;
         // (*imit)->getMagicPixel(magicX,magicY);
         cropulx=ulx_opt[0];
         cropuly=uly_opt[0];
         croplrx=lrx_opt[0];
         croplry=lry_opt[0];
         if(forceEUgrid){
           //force to LAEA grid
           Egcs egcs;
           egcs.setLevel(egcs.res2level(dx));
           egcs.force2grid(cropulx,cropuly,croplrx,croplry);
         }
         else if(align_opt[0]){
           if(cropulx>(*imit)->getUlx())
             cropulx-=fmod(cropulx-(*imit)->getUlx(),dx);
           else if(cropulx<(*imit)->getUlx())
             cropulx+=fmod((*imit)->getUlx()-cropulx,dx)-dx;
           if(croplrx<(*imit)->getLrx())
             croplrx+=fmod((*imit)->getLrx()-croplrx,dx);
           else if(croplrx>(*imit)->getLrx())
             croplrx-=fmod(croplrx-(*imit)->getLrx(),dx)+dx;
           if(croplry>(*imit)->getLry())
             croplry-=fmod(croplry-(*imit)->getLry(),dy);
           else if(croplry<(*imit)->getLry())
             croplry+=fmod((*imit)->getLry()-croplry,dy)-dy;
           if(cropuly<(*imit)->getUly())
             cropuly+=fmod((*imit)->getUly()-cropuly,dy);
           else if(cropuly>(*imit)->getUly())
             cropuly-=fmod(cropuly-(*imit)->getUly(),dy)+dy;
         }
         (*imit)->geo2image(cropulx+(magicX-1.0)*(*imit)->getDeltaX(),cropuly-(magicY-1.0)*(*imit)->getDeltaY(),uli,ulj);
         (*imit)->geo2image(croplrx+(magicX-2.0)*(*imit)->getDeltaX(),croplry-(magicY-2.0)*(*imit)->getDeltaY(),lri,lrj);

         // ncropcol=abs(static_cast<unsigned int>(ceil((croplrx-cropulx)/dx)));
         // ncroprow=abs(static_cast<unsigned int>(ceil((cropuly-croplry)/dy)));
         ncropcol=static_cast<unsigned int>(ceil((croplrx-cropulx)/dx));
         ncroprow=static_cast<unsigned int>(ceil((cropuly-croplry)/dy));
         uli=floor(uli);
         ulj=floor(ulj);
         lri=floor(lri);
         lrj=floor(lrj);
       }

       // double deltaX=(*imit)->getDeltaX();
       // double deltaY=(*imit)->getDeltaY();
       if(!imgWriter.nrOfBand()){//not opened yet
         if(verbose_opt[0]){
           cout << "cropulx: " << cropulx << endl;
           cout << "cropuly: " << cropuly << endl;
           cout << "croplrx: " << croplrx << endl;
           cout << "croplry: " << croplry << endl;
           cout << "ncropcol: " << ncropcol << endl;
           cout << "ncroprow: " << ncroprow << endl;
           cout << "cropulx+ncropcol*dx: " << cropulx+ncropcol*dx << endl;
           cout << "cropuly-ncroprow*dy: " << cropuly-ncroprow*dy << endl;
           cout << "upper left column of input image: " << uli << endl;
           cout << "upper left row of input image: " << ulj << endl;
           cout << "lower right column of input image: " << lri << endl;
           cout << "lower right row of input image: " << lrj << endl;
           cout << "new number of cols: " << ncropcol << endl;
           cout << "new number of rows: " << ncroprow << endl;
           cout << "new number of bands: " << ncropband << endl;
         }
         // string imageType;//=(*imit)->getImageType();
         // if(oformat_opt.size())//default
         //   imageType=oformat_opt[0];
         try{
           imgWriter.open(ncropcol,ncroprow,ncropband,theType);
           imgWriter.setNoData(nodata_opt);
           // if(nodata_opt.size()){
           //   imgWriter.setNoData(nodata_opt);
           // }
         }
         catch(string errorstring){
           throw;
           // cout << errorstring << endl;
           // return(CE_Failure);
         }
         if(description_opt.size())
           imgWriter.setImageDescription(description_opt[0]);
         double gt[6];
         gt[0]=cropulx;
         gt[1]=dx;
         gt[2]=0;
         gt[3]=cropuly;
         gt[4]=0;
         gt[5]=((*imit)->isGeoRef())? -dy : dy;
         imgWriter.setGeoTransform(gt);
         if(projection_opt.size()){
           if(verbose_opt[0])
             cout << "projection: " << projection_opt[0] << endl;
           imgWriter.setProjectionProj4(projection_opt[0]);
         }
         else
           imgWriter.setProjection((*imit)->getProjection());
         if(imgWriter.getDataType()==GDT_Byte){
           if(colorTable_opt.size()){
             if(colorTable_opt[0]!="none")
               imgWriter.setColorTable(colorTable_opt[0]);
           }
           else if ((*imit)->getColorTable()!=NULL)//copy colorTable from input image
             imgWriter.setColorTable((*imit)->getColorTable());
         }
       }

       double startCol=uli;
       double endCol=lri;
       if(uli<0)
         startCol=0;
       else if(uli>=(*imit)->nrOfCol())
         startCol=(*imit)->nrOfCol()-1;
       if(lri<0)
         endCol=0;
       else if(lri>=(*imit)->nrOfCol())
         endCol=(*imit)->nrOfCol()-1;
       double startRow=ulj;
       double endRow=lrj;
       if(ulj<0)
         startRow=0;
       else if(ulj>=(*imit)->nrOfRow())
         startRow=(*imit)->nrOfRow()-1;
       if(lrj<0)
         endRow=0;
       else if(lrj>=(*imit)->nrOfRow())
         endRow=(*imit)->nrOfRow()-1;

       vector<double> readBuffer;
       unsigned int nband=(band_opt.size())?band_opt.size() : (*imit)->nrOfBand();
       for(unsigned int iband=0;iband<nband;++iband){
         unsigned int readBand=(band_opt.size()>iband)?band_opt[iband]:iband;
         if(verbose_opt[0]){
           cout << "extracting band " << readBand << endl;
           MyProgressFunc(progress,pszMessage,pProgressArg);
         }
         double theMin=0;
         double theMax=0;
         if(autoscale_opt.size()){
           try{
             (*imit)->getMinMax(static_cast<unsigned int>(startCol),static_cast<unsigned int>(endCol),static_cast<unsigned int>(startRow),static_cast<unsigned int>(endRow),readBand,theMin,theMax);
           }
           catch(string errorString){
             cout << errorString << endl;
           }
           if(verbose_opt[0])
             cout << "minmax: " << theMin << ", " << theMax << endl;
           double theScale=(autoscale_opt[1]-autoscale_opt[0])/(theMax-theMin);
           double theOffset=autoscale_opt[0]-theScale*theMin;
           (*imit)->setScale(theScale,readBand);
           (*imit)->setOffset(theOffset,readBand);
         }
         else{
           if(scale_opt.size()){
             if(scale_opt.size()>iband)
               (*imit)->setScale(scale_opt[iband],readBand);
             else
               (*imit)->setScale(scale_opt[0],readBand);
           }
           if(offset_opt.size()){
             if(offset_opt.size()>iband)
               (*imit)->setOffset(offset_opt[iband],readBand);
             else
               (*imit)->setOffset(offset_opt[0],readBand);
           }
         }

         double readRow=0;
         double readCol=0;
         double lowerCol=0;
         double upperCol=0;
         for(int irow=0;irow<imgWriter.nrOfRow();++irow){
           vector<double> lineMask;
           double x=0;
           double y=0;
           //convert irow to geo
           imgWriter.image2geo(0,irow,x,y);
           //lookup corresponding row for irow in this file
           (*imit)->geo2image(x,y,readCol,readRow);
           vector<double> writeBuffer;
           if(readRow<0||readRow>=(*imit)->nrOfRow()){
             for(int icol=0;icol<imgWriter.nrOfCol();++icol)
               writeBuffer.push_back(nodataValue);
           }
           else{
             try{
               if(endCol<(*imit)->nrOfCol()-1){
                 (*imit)->readData(readBuffer,startCol,endCol+1,readRow,readBand,theResample);
               }
               else{
                 (*imit)->readData(readBuffer,startCol,endCol,readRow,readBand,theResample);
               }
               double oldRowMask=-1;//keep track of row mask to optimize number of line readings
               for(int icol=0;icol<imgWriter.nrOfCol();++icol){
                 imgWriter.image2geo(icol,irow,x,y);
                 //lookup corresponding row for irow in this file
                 (*imit)->geo2image(x,y,readCol,readRow);
                 if(readCol<0||readCol>=(*imit)->nrOfCol()){
                   writeBuffer.push_back(nodataValue);
                 }
                 else{
                   bool valid=true;
                   double geox=0;
                   double geoy=0;
                   if(maskReader.isInit()){
                     //read mask
                     double colMask=0;
                     double rowMask=0;

                     imgWriter.image2geo(icol,irow,geox,geoy);
                     maskReader.geo2image(geox,geoy,colMask,rowMask);
                     colMask=static_cast<unsigned int>(colMask);
                     rowMask=static_cast<unsigned int>(rowMask);
                     if(rowMask>=0&&rowMask<maskReader.nrOfRow()&&colMask>=0&&colMask<maskReader.nrOfCol()){
                       if(static_cast<unsigned int>(rowMask)!=static_cast<unsigned int>(oldRowMask)){

                         try{
                           maskReader.readData(lineMask,static_cast<unsigned int>(rowMask),mskband_opt[0]);
                         }
                         catch(string errorstring){
                           throw;
                           // cerr << errorstring << endl;
                           // return(CE_Failure);
                         }
                         catch(...){
                           std::string errorString="error caught";
                           throw;
                           // cerr << "error caught" << std::endl;
                           // return(CE_Failure);
                         }
                         oldRowMask=rowMask;
                       }
                       if(cut_to_cutline_opt[0]){
                         if(lineMask[colMask]!=1){
                           nodataValue=nodata_opt[0];
                           valid=false;
                         }
                       }
                       else if(cut_in_cutline_opt[0]){
                         if(lineMask[colMask]==1){
                           nodataValue=nodata_opt[0];
                           valid=false;
                         }
                       }
                       else{
                         for(int ivalue=0;ivalue<msknodata_opt.size();++ivalue){
                           if(lineMask[colMask]==msknodata_opt[ivalue]){
                             if(nodata_opt.size()>ivalue)
                               nodataValue=nodata_opt[ivalue];
                             valid=false;
                             break;
                           }
                         }
                       }
                     }
                   }
                   if(!valid)
                     writeBuffer.push_back(nodataValue);
                   else{
                     switch(theResample){
                     case(BILINEAR):
                       lowerCol=readCol-0.5;
                       lowerCol=static_cast<unsigned int>(lowerCol);
                       upperCol=readCol+0.5;
                       upperCol=static_cast<unsigned int>(upperCol);
                       if(lowerCol<0)
                         lowerCol=0;
                       if(upperCol>=(*imit)->nrOfCol())
                         upperCol=(*imit)->nrOfCol()-1;
                       writeBuffer.push_back((readCol-0.5-lowerCol)*readBuffer[upperCol-startCol]+(1-readCol+0.5+lowerCol)*readBuffer[lowerCol-startCol]);
                       break;
                     default:
                       readCol=static_cast<unsigned int>(readCol);
                       readCol-=startCol;//we only start reading from startCol
                       writeBuffer.push_back(readBuffer[readCol]);
                       break;
                     }
                   }
                 }
               }
             }
             catch(string errorstring){
               throw;
               // cout << errorstring << endl;
               // return(CE_Failure);
             }
           }
           if(writeBuffer.size()!=imgWriter.nrOfCol())
             cout << "writeBuffer.size()=" << writeBuffer.size() << ", imgWriter.nrOfCol()=" << imgWriter.nrOfCol() << endl;

           assert(writeBuffer.size()==imgWriter.nrOfCol());
           try{
             imgWriter.writeData(writeBuffer,irow,writeBand);
           }
           catch(string errorstring){
             throw;
             // cout << errorstring << endl;
             // return(CE_Failure);
           }
           if(verbose_opt[0]){
             progress=(1.0+irow);
             progress/=imgWriter.nrOfRow();
             MyProgressFunc(progress,pszMessage,pProgressArg);
           }
           else{
             progress=(1.0+irow);
             progress+=(imgWriter.nrOfRow()*writeBand);
             progress/=imgWriter.nrOfBand()*imgWriter.nrOfRow();
             assert(progress>=0);
             assert(progress<=1);
             MyProgressFunc(progress,pszMessage,pProgressArg);
           }
         }
         ++writeBand;
       }
       // (*imit)->close();
     }
     if(extent_opt.size()&&(cut_to_cutline_opt[0]||cut_in_cutline_opt[0]||eoption_opt.size())){
       extentReader.close();
     }
     if(maskReader.isInit())
       maskReader.close();
     // return(CE_None);
   }
   catch(string predefinedString){
     std::cout << predefinedString << std::endl;
     throw;
   }
   return(*this);
 }
