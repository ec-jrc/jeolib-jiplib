/**********************************************************************
jlcrop_lib.cc: perform raster data operations on image such as crop, extract and stack bands
Author(s): Pieter.Kempeneers@ec.europa.eu
Copyright (c) 2016-2018 European Union (Joint Research Centre)
License EUPLv1.2

This file is part of jiplib
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

shared_ptr<Jim> Jim::crop(AppFactory& app){
  shared_ptr<Jim> imgWriter=Jim::createImg();
  crop(*imgWriter, app);
  return(imgWriter);
}

shared_ptr<Jim> Jim::cropBand(AppFactory& app){
  shared_ptr<Jim> imgWriter=Jim::createImg();
  cropBand(*imgWriter, app);
  return(imgWriter);
}

shared_ptr<Jim> Jim::cropPlane(AppFactory& app){
  shared_ptr<Jim> imgWriter=Jim::createImg();
  cropPlane(*imgWriter, app);
  return(imgWriter);
}

shared_ptr<Jim> Jim::cropOgr(VectorOgr& sampleReader, AppFactory& app){
  shared_ptr<Jim> imgWriter=Jim::createImg();
  cropOgr(sampleReader, *imgWriter, app);
  return(imgWriter);
}

// shared_ptr<Jim> Jim::crop(double ulx, double uly, double lrx, double lry, double dx, double dy, bool geo){
//   shared_ptr<Jim> imgWriter=Jim::createImg();
//   crop(*imgWriter, ulx, uly, lrx, lry, dx, dy, geo);
//   return(imgWriter);
// }

// CPLErr Jim::crop(Jim& imgWriter, double ulx, double uly, double lrx, double lry){
//   app::AppFactory app;
//   app.setLongOption("ulx",ulx);
//   app.setLongOption("uly",uly);
//   app.setLongOption("lrx",lrx);
//   app.setLongOption("lry",lry);
//   return(crop(imgWriter,app));
// }

shared_ptr<Jim> Jim::stackBand(Jim& srcImg, AppFactory& app){
  shared_ptr<Jim> imgWriter=Jim::createImg();
  stackBand(srcImg, *imgWriter, app);
  return(imgWriter);
}


// void Jim::d_convertType(AppFactory& app){
//   Optionjl<string>  otype_opt("ot", "otype", "Data type for output image ({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}). Empty string: inherit type from input image");
//   Optionjl<short>  verbose_opt("v", "verbose", "verbose", 0,2);

//   bool doProcess;//stop process when program was invoked with help option (-h --help)
//   try{
//     doProcess=otype_opt.retrieveOption(app);
//     verbose_opt.retrieveOption(app);

//     if(!doProcess){
//       cout << endl;
//       std::ostringstream helpStream;
//       helpStream << "short option -h shows basic options only, use long option --help to show all options" << std::endl;
//       throw(helpStream.str());//help was invoked, stop processing
//     }

//     std::vector<std::string> badKeys;
//     app.badKeys(badKeys);
//     if(badKeys.size()){
//       std::ostringstream errorStream;
//       if(badKeys.size()>1)
//         errorStream << "Error: unknown keys: ";
//       else
//         errorStream << "Error: unknown key: ";
//       for(int ikey=0;ikey<badKeys.size();++ikey){
//         errorStream << badKeys[ikey] << " ";
//       }
//       errorStream << std::endl;
//       throw(errorStream.str());
//     }
//     GDALDataType theType=getGDALDataType();
//     if(otype_opt.size()){
//       theType=string2GDAL(otype_opt[0]);
//       if(theType==GDT_Unknown)
//         std::cout << "Warning: unknown output pixel type: " << otype_opt[0] << ", using input type as default" << std::endl;
//     }
//     if(verbose_opt[0]>1)
//       cout << "Output pixel type:  " << GDALGetDataTypeName(theType) << endl;

//     if(scale_opt.size()){
//       while(scale_opt.size()<nrOfBand())
//         scale_opt.push_back(scale_opt[0]);
//     }
//     if(offset_opt.size()){
//       while(offset_opt.size()<nrOfBand())
//         offset_opt.push_back(offset_opt[0]);
//     }
//     if(autoscale_opt.size()){
//       assert(autoscale_opt.size()%2==0);
//     }

//     if(theType==GDT_Unknown){
//       theType=this->getGDALDataType();
//       if(verbose_opt[0]>1)
//         cout << "Using data type from input image: " << GDALGetDataTypeName(theType) << endl;
//     }
//     Jim imgWriter(nrOfCol(),nrOfRow(),1,nrOfPlane(),getDataType());
//     memcpy(imgWriter.getDataPointer(),m_data[0],getDataTypeSizeBytes()*nrOfCol()*nrOfRow*nrOfPlane());
//     free(m_data[0]);
//     m_data[0]=(void *) calloc(static_cast<size_t>(nrOfPlane()*nrOfCol()*nrOfRow()),getDataTypeSizeBytes(otype_opt[0]));
    
//     for(size_t iband=0;iband<nrOfBand();++iband){
//       memcpy(m_data[nrOfBand()],m_data[iband],getDataTypeSizeBytes()*nrOfCol()*m_blockSize*oldnplane);
//       //allocate memory
//       free(m_data[iband]);
//       m_data[iband]=(void *) calloc(static_cast<size_t>(nrOfCol()*nrOfRow()*nrOfPlane()),getDataTypeSizeBytes());
//       memcpy(m_data[iband],m_data[nrOfBand()],getDataTypeSizeBytes()*nrOfCol()*nrOfRow()*oldnplane);
//       memcpy(m_data[iband]+getDataTypeSizeBytes()*nrOfCol()*nrOfRow()*oldnplane,imgSrc.getDataPointer(iband),imgSrc.getDataTypeSizeBytes()*imgSrc.nrOfCol()*imgSrc.nrOfRow()*imgSrc.nrOfPlane());
//     }
//   }
//   catch(string predefinedString){
//     std::cerr << predefinedString << std::endl;
//     throw;
//   }
// }

void Jim::convert(Jim& imgWriter, AppFactory& app){
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
    GDALDataType theType=getGDALDataType();
    if(otype_opt.size()){
      theType=string2GDAL(otype_opt[0]);
      if(theType==GDT_Unknown)
        std::cout << "Warning: unknown output pixel type: " << otype_opt[0] << ", using input type as default" << std::endl;
    }
    if(verbose_opt[0]>1)
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
      if(verbose_opt[0]>1)
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

    unsigned int nband=this->nrOfBand();

    // const char* pszMessage;
    // void* pProgressArg=NULL;
    // GDALProgressFunc pfnProgress=GDALTermProgress;
    // double progress=0;
    // MyProgressFunc(progress,pszMessage,pProgressArg);

    for(unsigned int iband=0;iband<nband;++iband){
      unsigned int readBand=iband;
      if(verbose_opt[0]>1){
        cout << "extracting band " << readBand << endl;
        // MyProgressFunc(progress,pszMessage,pProgressArg);
      }
      double theMin=0;
      double theMax=0;
      if(autoscale_opt.size()){
        this->getMinMax(0,nrOfCol()-1,0,nrOfRow()-1,readBand,theMin,theMax);
        if(verbose_opt[0]>1)
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
      #if JIPLIB_PROCESS_IN_PARALLEL == 1
      #pragma omp parallel for
      #else
      #endif
      for(int irow=0;irow<imgWriter.nrOfRow();++irow){
        vector<double> readBuffer(nrOfCol());
        vector<double> writeBuffer(nrOfCol());
        // readRow=irow;
        readData(readBuffer,irow,readBand);
        for(int icol=0;icol<imgWriter.nrOfCol();++icol)
          writeBuffer[icol]=readBuffer[icol];
        imgWriter.writeData(writeBuffer,irow,readBand);
        // progress=(1.0+irow);
        // progress+=(imgWriter.nrOfRow()*readBand);
        // progress/=imgWriter.nrOfBand()*imgWriter.nrOfRow();
        // assert(progress>=0);
        // assert(progress<=1);
        // MyProgressFunc(progress,pszMessage,pProgressArg);
      }
    }
  }
  catch(string predefinedString){
    std::cerr << predefinedString << std::endl;
    throw;
  }
}

// CPLErr Jim::crop(Jim& imgWriter, double ulx, double uly, double lrx, double lry, double dx, double dy, bool geo){
void Jim::crop(Jim& imgWriter, AppFactory& app){
  Optionjl<double> ulx_opt("ulx", "ulx", "Upper left x value bounding box", 0.0);
  Optionjl<double> uly_opt("uly", "uly", "Upper left y value bounding box", 0.0);
  Optionjl<double> lrx_opt("lrx", "lrx", "Lower right x value bounding box", 0.0);
  Optionjl<double> lry_opt("lry", "lry", "Lower right y value bounding box", 0.0);
  Optionjl<double> dx_opt("dx", "dx", "spatial resolution in x (in spatial reference system or pixels if nogeo is set)",0);
  Optionjl<double> dy_opt("dy", "dy", "spatial resolution in y (in spatial reference system or piyels if nogeo is set)",0);
  Optionjl<double> nodata_opt("nodata", "nodata", "Nodata value to put in image if out of bounds.", 0);
  Optionjl<bool> align_opt("align", "align", "Align output bounding box to input image",false);
  Optionjl<bool> nogeo_opt("nogeo", "nogeo", "use image coordinates instead of spatial reference system",false);
  Optionjl<short> verbose_opt("v", "verbose", "verbose", 0,2);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=ulx_opt.retrieveOption(app);
    uly_opt.retrieveOption(app);
    lrx_opt.retrieveOption(app);
    lry_opt.retrieveOption(app);
    dx_opt.retrieveOption(app);
    dy_opt.retrieveOption(app);
    nodata_opt.retrieveOption(app);
    align_opt.retrieveOption(app);
    nogeo_opt.retrieveOption(app);
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
    if(m_data.empty()){
      std::ostringstream s;
      s << "Error: Jim not initialized, m_data is empty";
      std::cerr << s.str() << std::endl;
      throw(s.str());
    }
    double cropuli=0;
    double cropulj=0;
    double croplri=0;
    double croplrj=0;
    double stridei=1;
    double stridej=1;
    double cropulx=ulx_opt[0];
    double cropuly=uly_opt[0];
    double croplrx=lrx_opt[0];
    double croplry=lry_opt[0];
    double dx=dx_opt[0];
    double dy=dy_opt[0];
    if(nogeo_opt[0]){
      if(verbose_opt[0]>1){
        std::cout << "crop in nogeo mode" << std::endl;
        std::cout << "cropulx: " << cropulx << std::endl;
        std::cout << "cropuly: " << cropuly << std::endl;
        std::cout << "croplrx: " << croplrx << std::endl;
        std::cout << "croplry: " << croplry << std::endl;
        std::cout << "dx: " << dx << std::endl;
        std::cout << "dy: " << dy << std::endl;
      }
      if(cropulx>=croplrx){
        cropulx=0;
        croplrx=nrOfCol();
      }
      if(cropuly>=croplry){
        cropuly=0;
        croplry=nrOfRow();
      }
      cropuli=cropulx;
      cropulj=cropuly;
      croplri=croplrx;
      croplrj=croplry;
      if(dx>0)
        stridei=dx;
      else
        stridei=1;
      dx=stridei*getDeltaX();
      if(dy>0)
        stridej=dy;
      else
        stridej=1;
      dy=stridej*getDeltaY();

      this->image2geo(cropuli,cropulj,cropulx,cropuly);
      this->image2geo(croplri,croplrj,croplrx,croplry);

      cropulx-=getDeltaX()/2;
      cropuly+=getDeltaY()/2;

      croplrx-=this->getDeltaX()/2.0;
      croplry+=this->getDeltaY()/2.0;
    }
    else{
      if(verbose_opt[0]>1){
        std::cout << "crop in geo mode" << std::endl;
        std::cout << "cropulx: " << cropulx << std::endl;
        std::cout << "cropuly: " << cropuly << std::endl;
        std::cout << "croplrx: " << croplrx << std::endl;
        std::cout << "croplry: " << croplry << std::endl;
        std::cout << "dx: " << dx << std::endl;
        std::cout << "dy: " << dy << std::endl;
      }
      if(cropulx>=croplrx){
        cropulx=getUlx();
        croplrx=getLrx();
      }
      if(cropuly<=croplry){
        cropuly=getUly();
        croplry=getLry();
      }
      if(align_opt[0]){
        if(verbose_opt[0]>1)
          std::cout << "using align option" << std::endl;
        //do align
        if(dx>0){
          if(cropulx>this->getUlx())
            cropulx-=fmod(cropulx-this->getUlx(),dx);
          else if(cropulx<this->getUlx())
            cropulx+=fmod(this->getUlx()-cropulx,dx)-dx;
          if(croplrx<this->getLrx())
            croplrx+=fmod(this->getLrx()-croplrx,dx);
          else if(croplrx>this->getLrx())
            croplrx-=fmod(croplrx-this->getLrx(),dx)+dx;
        }
        if(dy>0){
          if(croplry>this->getLry())
            croplry-=fmod(croplry-this->getLry(),dy);
          else if(croplry<this->getLry())
            croplry+=fmod(this->getLry()-croplry,dy)-dy;
          if(cropuly<this->getUly())
            cropuly+=fmod(this->getUly()-cropuly,dy);
          else if(cropuly>this->getUly())
            cropuly-=fmod(cropuly-this->getUly(),dy)+dy;
        }
      }
      else{
        if(dx>0){
          stridei=dx/getDeltaX();
          // stridei=(croplrx-cropulx)/dx;
        }
        else{
          stridei=1;
          dx=getDeltaX();
        }
        if(dy>0){
          stridej=dy/getDeltaY();
          // stridej=(cropuly-croplry)/dy;
        }
        else{
          stridej=1;
          dy=getDeltaY();
        }
      }
      this->geo2image(cropulx,cropuly,cropuli,cropulj);
      this->geo2image(croplrx-this->getDeltaX(),croplry+this->getDeltaY(),croplri,croplrj);

      cropuli=floor(cropuli);
      cropulj=floor(cropulj);
      croplri=floor(croplri)+1;
      croplrj=floor(croplrj)+1;
      // croplri=ceil(croplri);
      // croplrj=ceil(croplrj);
    }
    // size_t ncropcol=(croplri-cropuli)/stridei;
    // size_t ncroprow=(croplrj-cropulj)/stridej;
    size_t ncropcol=(croplrx-cropulx)/dx;
    size_t ncroprow=(cropuly-croplry)/dy;
    if(verbose_opt[0]>1){
      std::cout << "ncropcol: " << ncropcol << std::endl;
      std::cout << "ncroprow: " << ncroprow << std::endl;
      std::cout << "dx: " << dx << std::endl;
      std::cout << "dy: " << dy << std::endl;
      std::cout << "stridei: " << stridei << std::endl;
      std::cout << "stridej: " << stridej << std::endl;
    }
    try{
      imgWriter.open(ncropcol,ncroprow,nrOfBand(),this->getGDALDataType());
    }
    catch(string errorstring){
      cerr << errorstring << endl;
      throw;
    }
    double gt[6];
    gt[0]=cropulx;
    gt[1]=dx;
    gt[2]=0;
    gt[3]=cropuly;
    gt[4]=0;
    gt[5]=-dy;
    imgWriter.setGeoTransform(gt);
    imgWriter.setProjection(this->getProjection());
    // double nodataValue=m_noDataValues.size() ? m_noDataValues[0] : 0;
    std::ostringstream errorStream;

    if(verbose_opt[0]>1){
      std::cout << "cropuli: " << cropuli << std::endl;
      std::cout << "cropulj: " << cropulj << std::endl;
      std::cout << "stridei: " << stridei << std::endl;
      std::cout << "stridej: " << stridej << std::endl;
    }
    if(cropuli+stridei/2 < 0 || cropuli+(ncropcol-1)*stridei+stridei/2 >= nrOfCol()){
      errorStream << "Warning: columns requested out of bounding box" << std::endl;
      errorStream << "cropuli+stridei/2: " << cropuli+stridei/2 << std::endl;
      errorStream << "cropuli+(ncropcol-1)*stridei+stridei/2: " << cropuli+(ncropcol-1)*stridei+stridei/2 << std::endl;
      std::cerr << errorStream.str() << std::endl;
      imgWriter.setNoDataValue(nodata_opt[0]);
    }
    if(cropulj+stridej/2<0 || cropulj+(ncroprow-1)*stridej+stridej/2 >= nrOfRow()){
      errorStream << "Warning: rows requested out of bounding box" << std::endl;
      errorStream << "cropulj+stridej/2: " << cropulj+stridej/2 << std::endl;
      errorStream << "cropulj+(ncroprow-1)*stridej+stridej/2: " << cropulj+(ncroprow-1)*stridej+stridej/2 << std::endl;
      std::cerr << errorStream.str() << std::endl;
      imgWriter.setNoDataValue(nodata_opt[0]);
    }

    for(size_t iband=0;iband<nrOfBand();++iband){
      for(int irow=0;irow<imgWriter.nrOfRow();++irow){
        double readRow=cropulj+irow*stridej+stridej/2;
        if(readRow<0||readRow>=this->nrOfRow()){
          for(int icol=0;icol<imgWriter.nrOfCol();++icol)
            imgWriter.writeData(nodata_opt[0],icol,irow,iband);
        }
        else{
          for(int icol=0;icol<imgWriter.nrOfCol();++icol){
            double readCol=cropuli+icol*stridei+stridei/2;
            if(readCol<0||readCol>=this->nrOfCol()){
              imgWriter.writeData(nodata_opt[0],icol,irow,iband);
            }
            else{
              double readValue=0;
              this->readData(readValue,readCol,readRow,iband);
              imgWriter.writeData(readValue,icol,irow,iband);
            }
          }
        }
      }
    }
  }
  catch(string errorstring){
    std::cerr << errorstring << std::endl;
    throw;
  }
  catch(BadConversion conversion){
    std::string errorString="Bad conversion in arguments";
    std::cerr << errorString << std::endl;
    app.showOptions();
    throw(errorString);
  }
  catch(...){
    std::string errorString="Unknown error";
    throw(errorString);
  }
}

// CPLErr Jim::crop(Jim& imgWriter, AppFactory& app){
//   Optionjl<string>  projection_opt("a_srs", "a_srs", "Override the projection for the output file (leave blank to copy from input file, use epsg:3035 to use European projection and force to European grid");
//   //todo: support layer names
//   Optionjl<double>  ulx_opt("ulx", "ulx", "Upper left x value bounding box", 0.0);
//   Optionjl<double>  uly_opt("uly", "uly", "Upper left y value bounding box", 0.0);
//   Optionjl<double>  lrx_opt("lrx", "lrx", "Lower right x value bounding box", 0.0);
//   Optionjl<double>  lry_opt("lry", "lry", "Lower right y value bounding box", 0.0);
//   Optionjl<double> cx_opt("x", "x", "x-coordinate of image center to crop (in meter)");
//   Optionjl<double> cy_opt("y", "y", "y-coordinate of image center to crop (in meter)");
//   Optionjl<double> nx_opt("nx", "nx", "image size in x to crop (in meter)");
//   Optionjl<double> ny_opt("ny", "ny", "image size in y to crop (in meter)");
//   Optionjl<unsigned int> ns_opt("ns", "ns", "number of samples  to crop (in pixels)");
//   Optionjl<unsigned int> nl_opt("nl", "nl", "number of lines to crop (in pixels)");
//   Optionjl<double>  nodata_opt("nodata", "nodata", "Nodata value to put in image if out of bounds.");
//   Optionjl<bool>  align_opt("align", "align", "Align output bounding box to input image",false);
//   Optionjl<short>  verbose_opt("v", "verbose", "verbose", 0,2);

//   bool doProcess;//stop process when program was invoked with help option (-h --help)
//   try{
//     doProcess=projection_opt.retrieveOption(app);
//     ulx_opt.retrieveOption(app);
//     uly_opt.retrieveOption(app);
//     lrx_opt.retrieveOption(app);
//     lry_opt.retrieveOption(app);
//     cx_opt.retrieveOption(app);
//     cy_opt.retrieveOption(app);
//     nx_opt.retrieveOption(app);
//     ny_opt.retrieveOption(app);
//     ns_opt.retrieveOption(app);
//     nl_opt.retrieveOption(app);
//     nodata_opt.retrieveOption(app);
//     align_opt.retrieveOption(app);
//     verbose_opt.retrieveOption(app);

//     if(!doProcess){
//       cout << endl;
//       std::ostringstream helpStream;
//       helpStream << "short option -h shows basic options only, use long option --help to show all options" << std::endl;
//       throw(helpStream.str());//help was invoked, stop processing
//     }

//     std::vector<std::string> badKeys;
//     app.badKeys(badKeys);
//     if(badKeys.size()){
//       std::ostringstream errorStream;
//       if(badKeys.size()>1)
//         errorStream << "Error: unknown keys: ";
//       else
//         errorStream << "Error: unknown key: ";
//       for(int ikey=0;ikey<badKeys.size();++ikey){
//         errorStream << badKeys[ikey] << " ";
//       }
//       errorStream << std::endl;
//       throw(errorStream.str());
//     }

//     double nodataValue=nodata_opt.size()? nodata_opt[0] : 0;
//     bool isGeoRef=false;
//     string projectionString;
//     // for(int iimg=0;iimg<input_opt.size();++iimg){

//     if(!isGeoRef)
//       isGeoRef=this->isGeoRef();
//     GDALDataType theType=getGDALDataType();
//     if(verbose_opt[0])
//       cout << "Output pixel type:  " << GDALGetDataTypeName(theType) << endl;

//     //bounding box of cropped image
//     double cropulx=ulx_opt[0];
//     double cropuly=uly_opt[0];
//     double croplrx=lrx_opt[0];
//     double croplry=lry_opt[0];
//     double dx=getDeltaX();
//     double dy=getDeltaY();
//     // if(cx_opt.size()&&cy_opt.size()&&nx_opt.size()&&ny_opt.size()){
//     if(nx_opt.size()&&ny_opt.size()){
//       if(cx_opt.size()&&cy_opt.size()){
//           ulx_opt[0]=cx_opt[0]-nx_opt[0]/2.0;
//           uly_opt[0]=(isGeoRef) ? cy_opt[0]+ny_opt[0]/2.0 : cy_opt[0]-ny_opt[0]/2.0;
//           lrx_opt[0]=cx_opt[0]+nx_opt[0]/2.0;
//           lry_opt[0]=(isGeoRef) ? cy_opt[0]-ny_opt[0]/2.0 : cy_opt[0]+ny_opt[0]/2.0;
//       }
//       else if(ulx_opt.size()&&uly_opt.size()){
//         lrx_opt[0]=ulx_opt[0]+nx_opt[0];
//         lry_opt[0]=lry_opt[0]-ny_opt[0];
//       }
//     }
//     else if(ns_opt.size()&&nl_opt.size()){
//       if(cx_opt.size()&&cy_opt.size()){
//         ulx_opt[0]=cx_opt[0]-ns_opt[0]*dx/2.0;
//         uly_opt[0]=(isGeoRef) ? cy_opt[0]+nl_opt[0]*dy/2.0 : cy_opt[0]-nl_opt[0]*dy/2.0;
//         lrx_opt[0]=cx_opt[0]+ns_opt[0]*dx/2.0;
//         lry_opt[0]=(isGeoRef) ? cy_opt[0]-nl_opt[0]*dy/2.0 : cy_opt[0]+nl_opt[0]*dy/2.0;
//       }
//       else if(ulx_opt.size()&&uly_opt.size()){
//         lrx_opt[0]=ulx_opt[0]+ns_opt[0]*dx;
//         lry_opt[0]=uly_opt[0]-nl_opt[0]*dy;
//       }
//     }

//     if(verbose_opt[0])
//       cout << "--ulx=" << ulx_opt[0] << " --uly=" << uly_opt[0] << " --lrx=" << lrx_opt[0] << " --lry=" << lry_opt[0] << endl;

//     int ncropcol=0;
//     int ncroprow=0;

//     double uli,ulj,lri,lrj;//image coordinates
//     bool forceEUgrid=false;
//     if(projection_opt.size())
//       forceEUgrid=(!(projection_opt[0].compare("EPSG:3035"))||!(projection_opt[0].compare("EPSG:3035"))||projection_opt[0].find("ETRS-LAEA")!=string::npos);
//     if(ulx_opt[0]>=lrx_opt[0]){//default bounding box: no cropping
//       uli=0;
//       lri=this->nrOfCol()-1;
//       ulj=0;
//       lrj=this->nrOfRow()-1;
//       ncropcol=this->nrOfCol();
//       ncroprow=this->nrOfRow();
//       this->getBoundingBox(cropulx,cropuly,croplrx,croplry);
//       double magicX=1,magicY=1;
//       // this->getMagicPixel(magicX,magicY);
//       if(forceEUgrid){
//         //force to LAEA grid
//         Egcs egcs;
//         egcs.setLevel(egcs.res2level(dx));
//         egcs.force2grid(cropulx,cropuly,croplrx,croplry);
//         this->geo2image(cropulx+(magicX-1.0)*this->getDeltaX(),cropuly-(magicY-1.0)*this->getDeltaY(),uli,ulj);
//         this->geo2image(croplrx+(magicX-2.0)*this->getDeltaX(),croplry-(magicY-2.0)*this->getDeltaY(),lri,lrj);
//       }
//       this->geo2image(cropulx+(magicX-1.0)*this->getDeltaX(),cropuly-(magicY-1.0)*this->getDeltaY(),uli,ulj);
//       this->geo2image(croplrx+(magicX-2.0)*this->getDeltaX(),croplry-(magicY-2.0)*this->getDeltaY(),lri,lrj);
//       // ncropcol=abs(static_cast<unsigned int>(ceil((croplrx-cropulx)/dx)));
//       // ncroprow=abs(static_cast<unsigned int>(ceil((cropuly-croplry)/dy)));
//       ncropcol=static_cast<unsigned int>(ceil((croplrx-cropulx)/dx));
//       ncroprow=static_cast<unsigned int>(ceil((cropuly-croplry)/dy));
//       std::cerr << "Warning: unexpected bounding box, using defaults "<< "--ulx=" << cropulx << " --uly=" << cropuly << " --lrx=" << croplrx << " --lry=" << croplry << std::endl;
//     }
//     else{
//       double magicX=1,magicY=1;
//       // this->getMagicPixel(magicX,magicY);
//       cropulx=ulx_opt[0];
//       cropuly=uly_opt[0];
//       croplrx=lrx_opt[0];
//       croplry=lry_opt[0];
//       if(forceEUgrid){
//         //force to LAEA grid
//         Egcs egcs;
//         egcs.setLevel(egcs.res2level(dx));
//         egcs.force2grid(cropulx,cropuly,croplrx,croplry);
//       }
//       else if(align_opt[0]){
//         if(cropulx>this->getUlx())
//           cropulx-=fmod(cropulx-this->getUlx(),dx);
//         else if(cropulx<this->getUlx())
//           cropulx+=fmod(this->getUlx()-cropulx,dx)-dx;
//         if(croplrx<this->getLrx())
//           croplrx+=fmod(this->getLrx()-croplrx,dx);
//         else if(croplrx>this->getLrx())
//           croplrx-=fmod(croplrx-this->getLrx(),dx)+dx;
//         if(croplry>this->getLry())
//           croplry-=fmod(croplry-this->getLry(),dy);
//         else if(croplry<this->getLry())
//           croplry+=fmod(this->getLry()-croplry,dy)-dy;
//         if(cropuly<this->getUly())
//           cropuly+=fmod(this->getUly()-cropuly,dy);
//         else if(cropuly>this->getUly())
//           cropuly-=fmod(cropuly-this->getUly(),dy)+dy;
//       }
//       this->geo2image(cropulx+(magicX-1.0)*this->getDeltaX(),cropuly-(magicY-1.0)*this->getDeltaY(),uli,ulj);
//       this->geo2image(croplrx+(magicX-2.0)*this->getDeltaX(),croplry-(magicY-2.0)*this->getDeltaY(),lri,lrj);

//       ncropcol=static_cast<unsigned int>(ceil((croplrx-cropulx)/dx));
//       ncroprow=static_cast<unsigned int>(ceil((cropuly-croplry)/dy));
//       uli=floor(uli);
//       ulj=floor(ulj);
//       lri=floor(lri);
//       lrj=floor(lrj);

//       if(cropulx<getUlx() || cropuly>getUly() || croplrx>getLrx() || croplry<getLry()){
//         std::cerr << "Warning: requested bounding box not within original bounding box, using "<< "--ulx=" << cropulx << " --uly=" << cropuly << " --lrx=" << croplrx << " --lry=" << croplry << std::endl;
//       }
//     }

//     if(!imgWriter.nrOfBand()){//not opened yet
//       if(verbose_opt[0]){
//         cout << "cropulx: " << cropulx << endl;
//         cout << "cropuly: " << cropuly << endl;
//         cout << "croplrx: " << croplrx << endl;
//         cout << "croplry: " << croplry << endl;
//         cout << "ncropcol: " << ncropcol << endl;
//         cout << "ncroprow: " << ncroprow << endl;
//         cout << "cropulx+ncropcol*dx: " << cropulx+ncropcol*dx << endl;
//         cout << "cropuly-ncroprow*dy: " << cropuly-ncroprow*dy << endl;
//         cout << "upper left column of input image: " << uli << endl;
//         cout << "upper left row of input image: " << ulj << endl;
//         cout << "lower right column of input image: " << lri << endl;
//         cout << "lower right row of input image: " << lrj << endl;
//         cout << "new number of cols: " << ncropcol << endl;
//         cout << "new number of rows: " << ncroprow << endl;
//         cout << "new number of bands: " << nrOfBand()<< endl;
//       }
//       try{
//         imgWriter.open(ncropcol,ncroprow,nrOfBand(),theType);
//         imgWriter.setNoData(nodata_opt);
//       }
//       catch(string errorstring){
//         cout << errorstring << endl;
//         throw;
//       }
//       double gt[6];
//       gt[0]=cropulx;
//       gt[1]=getDeltaX();
//       gt[2]=0;
//       gt[3]=cropuly;
//       gt[4]=0;
//       gt[5]=-getDeltaY();
//       imgWriter.setGeoTransform(gt);
//       if(projection_opt.size()){
//         if(verbose_opt[0])
//           cout << "projection: " << projection_opt[0] << endl;
//         imgWriter.setProjectionProj4(projection_opt[0]);
//       }
//       else
//         imgWriter.setProjection(this->getProjection());
//     }

//     // if(!covers(cropulx,cropuly,croplrx,croplry,true)){
//     //   //todo: extend image in case of no full coverage
//     //   std::cerr << "Error: no full coverage" << std::endl;
//     //   throw;
//     // }
//     double startCol=uli;
//     double endCol=lri;
//     if(uli<0){
//       std::cerr << "Warning: upper left corner out of image boundaries, clipping to 0" << std::endl;
//       startCol=0;
//     }
//     else if(uli>=this->nrOfCol()){
//       std::cerr << "Warning: upper left corner out of image boundaries, clipping to " << this->nrOfCol()-1 << std::endl;
//       startCol=this->nrOfCol()-1;
//     }
//     if(lri<0){
//       std::cerr << "Warning: lower right corner out of image boundaries, clipping to " << 0 << std::endl;
//       endCol=0;
//     }
//     else if(lri>=this->nrOfCol()){
//       std::cerr << "Warning: lower right corner out of image boundaries, clipping to " << this->nrOfCol()-1 << std::endl;
//       endCol=this->nrOfCol()-1;
//     }
//     double startRow=ulj;
//     double endRow=lrj;
//     if(ulj<0){
//       std::cerr << "Warning: upper left corner out of image boundaries, clipping to 0" << std::endl;
//       startRow=0;
//     }
//     else if(ulj>=this->nrOfRow()){
//       std::cerr << "Warning: upper left corner out of image boundaries, clipping to " << this->nrOfRow()-1 << std::endl;
//       startRow=this->nrOfRow()-1;
//     }
//     if(lrj<0){
//       std::cerr << "Warning: lower right corner out of image boundaries, clipping to " << 0 << std::endl;
//       endRow=0;
//     }
//     else if(lrj>=this->nrOfRow()){
//       std::cerr << "Warning: lower right corner out of image boundaries, clipping to " << this->nrOfCol()-1 << std::endl;
//       endRow=this->nrOfRow()-1;
//     }

//     vector<double> readBuffer;
//     double readValue=nodataValue;
//     unsigned int nband=this->nrOfBand();
//     const char* pszMessage;
//     void* pProgressArg=NULL;
//     GDALProgressFunc pfnProgress=GDALTermProgress;
//     double progress=0;
//     MyProgressFunc(progress,pszMessage,pProgressArg);
//     for(size_t iband=0;iband<nband;++iband){
//       if(verbose_opt[0]){
//         cout << "extracting band " << iband << endl;
//         MyProgressFunc(progress,pszMessage,pProgressArg);
//       }
//       for(int irow=0;irow<imgWriter.nrOfRow();++irow){
//         double readRow=ulj+irow;
//         if(readRow<0||readRow>=this->nrOfRow()){
//           if(verbose_opt[0])
//             std::cout << "Warning: readRow is " << readRow << std::endl;
//           for(int icol=0;icol<imgWriter.nrOfCol();++icol)
//             imgWriter.writeData(nodataValue,icol,irow,iband);
//         }
//         else{
//           for(int icol=0;icol<imgWriter.nrOfCol();++icol){
//             double readCol=uli+icol;
//             if(readCol<0||readCol>=this->nrOfCol()){
//               if(verbose_opt[0])
//                 std::cout << "Warning: readCol is " << readCol << std::endl;
//               imgWriter.writeData(nodataValue,icol,irow,iband);
//             }
//             else{
//               this->readData(readValue,readCol,readRow,iband);
//               imgWriter.writeData(readValue,icol,irow,iband);
//             }
//           }
//         }
//         if(verbose_opt[0]){
//           progress=(1.0+irow);
//           progress/=imgWriter.nrOfRow();
//           MyProgressFunc(progress,pszMessage,pProgressArg);
//         }
//         else{
//           progress=(1.0+irow);
//           progress+=(imgWriter.nrOfRow()*iband);
//           progress/=imgWriter.nrOfBand()*imgWriter.nrOfRow();
//           assert(progress>=0);
//           assert(progress<=1);
//           MyProgressFunc(progress,pszMessage,pProgressArg);
//         }
//       }
//     }
//     return(CE_None);
//   }
//   catch(string predefinedString){
//     std::cout << predefinedString << std::endl;
//     throw;
//   }
// }

// CPLErr Jim::crop(Jim& imgWriter, AppFactory& app){
//   Optionjl<string>  projection_opt("a_srs", "a_srs", "Override the projection for the output file (leave blank to copy from input file, use epsg:3035 to use European projection and force to European grid");
//   //todo: support layer names
//   Optionjl<string>  extent_opt("e", "extent", "get boundary from extent from polygons in vector file");
//   Optionjl<string>  layer_opt("ln", "ln", "layer name of extent to crop");
//   Optionjl<bool> cut_to_cutline_opt("crop_to_cutline", "crop_to_cutline", "Crop the extent of the target dataset to the extent of the cutline, setting the outside area to nodata.",false);
//   Optionjl<bool> cut_in_cutline_opt("crop_in_cutline", "crop_in_cutline", "Crop the extent of the target dataset to the extent of the cutline, setting the inner area to nodata.",false);
//   Optionjl<string> eoption_opt("eo","eo", "special extent options controlling rasterization: ATTRIBUTE|CHUNKYSIZE|ALL_TOUCHED|BURN_VALUE_FROM|MERGE_ALG, e.g., -eo ATTRIBUTE=fieldname");
//   Optionjl<string> mask_opt("m", "mask", "Use the the specified file as a validity mask (0 is nodata).");
//   Optionjl<double> msknodata_opt("msknodata", "msknodata", "Mask value not to consider for crop.", 0);
//   Optionjl<unsigned int> mskband_opt("mskband", "mskband", "Mask band to read (0 indexed)", 0);
//   Optionjl<double>  ulx_opt("ulx", "ulx", "Upper left x value bounding box", 0.0);
//   Optionjl<double>  uly_opt("uly", "uly", "Upper left y value bounding box", 0.0);
//   Optionjl<double>  lrx_opt("lrx", "lrx", "Lower right x value bounding box", 0.0);
//   Optionjl<double>  lry_opt("lry", "lry", "Lower right y value bounding box", 0.0);
//   Optionjl<double>  dx_opt("dx", "dx", "Output resolution in x (in meter) (empty: keep original resolution)");
//   Optionjl<double>  dy_opt("dy", "dy", "Output resolution in y (in meter) (empty: keep original resolution)");
//   Optionjl<double> cx_opt("x", "x", "x-coordinate of image center to crop (in meter)");
//   Optionjl<double> cy_opt("y", "y", "y-coordinate of image center to crop (in meter)");
//   Optionjl<double> nx_opt("nx", "nx", "image size in x to crop (in meter)");
//   Optionjl<double> ny_opt("ny", "ny", "image size in y to crop (in meter)");
//   Optionjl<unsigned int> ns_opt("ns", "ns", "number of samples  to crop (in pixels)");
//   Optionjl<unsigned int> nl_opt("nl", "nl", "number of lines to crop (in pixels)");
//   Optionjl<unsigned int>  band_opt("b", "band", "band index to crop (leave empty to retain all bands)");
//   Optionjl<unsigned int> bstart_opt("sband", "startband", "Start band sequence number");
//   Optionjl<unsigned int> bend_opt("eband", "endband", "End band sequence number");
//   Optionjl<double> autoscale_opt("as", "autoscale", "scale output to min and max, e.g., --autoscale 0 --autoscale 255");
//   Optionjl<double> scale_opt("scale", "scale", "output=scale*input+offset");
//   Optionjl<double> offset_opt("offset", "offset", "output=scale*input+offset");
//   Optionjl<string>  otype_opt("ot", "otype", "Data type for output image ({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}). Empty string: inherit type from input image");
//   // Optionjl<string>  oformat_opt("of", "oformat", "Output image format (see also gdal_translate).","GTiff");
//   // Optionjl<string> option_opt("co", "co", "Creation option for output file. Multiple options can be specified.");
//   Optionjl<string>  colorTable_opt("ct", "ct", "color table (file with 5 columns: id R G B ALFA (0: transparent, 255: solid)");
//   Optionjl<double>  nodata_opt("nodata", "nodata", "Nodata value to put in image if out of bounds.");
//   Optionjl<string>  resample_opt("r", "resampling-method", "Resampling method (near: nearest neighbor, bilinear: bi-linear interpolation).", "near");
//   Optionjl<string>  description_opt("d", "description", "Set image description");
//   Optionjl<bool>  align_opt("align", "align", "Align output bounding box to input image",false);
//   Optionjl<short>  verbose_opt("v", "verbose", "verbose", 0,2);

//   extent_opt.setHide(1);
//   layer_opt.setHide(1);
//   cut_to_cutline_opt.setHide(1);
//   cut_in_cutline_opt.setHide(1);
//   eoption_opt.setHide(1);
//   bstart_opt.setHide(1);
//   bend_opt.setHide(1);
//   mask_opt.setHide(1);
//   msknodata_opt.setHide(1);
//   mskband_opt.setHide(1);
//   // option_opt.setHide(1);
//   cx_opt.setHide(1);
//   cy_opt.setHide(1);
//   nx_opt.setHide(1);
//   ny_opt.setHide(1);
//   ns_opt.setHide(1);
//   nl_opt.setHide(1);
//   scale_opt.setHide(1);
//   offset_opt.setHide(1);
//   nodata_opt.setHide(1);
//   description_opt.setHide(1);

//   bool doProcess;//stop process when program was invoked with help option (-h --help)
//   try{
//     doProcess=projection_opt.retrieveOption(app);
//     ulx_opt.retrieveOption(app);
//     uly_opt.retrieveOption(app);
//     lrx_opt.retrieveOption(app);
//     lry_opt.retrieveOption(app);
//     band_opt.retrieveOption(app);
//     bstart_opt.retrieveOption(app);
//     bend_opt.retrieveOption(app);
//     autoscale_opt.retrieveOption(app);
//     otype_opt.retrieveOption(app);
//     // oformat_opt.retrieveOption(app);
//     colorTable_opt.retrieveOption(app);
//     dx_opt.retrieveOption(app);
//     dy_opt.retrieveOption(app);
//     resample_opt.retrieveOption(app);
//     extent_opt.retrieveOption(app);
//     layer_opt.retrieveOption(app);
//     cut_to_cutline_opt.retrieveOption(app);
//     cut_in_cutline_opt.retrieveOption(app);
//     eoption_opt.retrieveOption(app);
//     mask_opt.retrieveOption(app);
//     msknodata_opt.retrieveOption(app);
//     mskband_opt.retrieveOption(app);
//     // option_opt.retrieveOption(app);
//     cx_opt.retrieveOption(app);
//     cy_opt.retrieveOption(app);
//     nx_opt.retrieveOption(app);
//     ny_opt.retrieveOption(app);
//     ns_opt.retrieveOption(app);
//     nl_opt.retrieveOption(app);
//     scale_opt.retrieveOption(app);
//     offset_opt.retrieveOption(app);
//     nodata_opt.retrieveOption(app);
//     description_opt.retrieveOption(app);
//     align_opt.retrieveOption(app);
//     verbose_opt.retrieveOption(app);

//     if(!doProcess){
//       cout << endl;
//       std::ostringstream helpStream;
//       helpStream << "short option -h shows basic options only, use long option --help to show all options" << std::endl;
//       throw(helpStream.str());//help was invoked, stop processing
//     }

//     std::vector<std::string> badKeys;
//     app.badKeys(badKeys);
//     if(badKeys.size()){
//       std::ostringstream errorStream;
//       if(badKeys.size()>1)
//         errorStream << "Error: unknown keys: ";
//       else
//         errorStream << "Error: unknown key: ";
//       for(int ikey=0;ikey<badKeys.size();++ikey){
//         errorStream << badKeys[ikey] << " ";
//       }
//       errorStream << std::endl;
//       throw(errorStream.str());
//     }

//     double nodataValue=nodata_opt.size()? nodata_opt[0] : 0;
//     RESAMPLE theResample;
//     if(resample_opt[0]=="near"){
//       theResample=NEAR;
//       if(verbose_opt[0])
//         cout << "resampling: nearest neighbor" << endl;
//     }
//     else if(resample_opt[0]=="bilinear"){
//       theResample=BILINEAR;
//       if(verbose_opt[0])
//         cout << "resampling: bilinear interpolation" << endl;
//     }
//     else{
//       std::cout << "Error: resampling method " << resample_opt[0] << " not supported" << std::endl;
//       return(CE_Failure);
//     }

//     // ImgReaderGdal imgReader;
//     // ImgWriterGdal imgWriter;
//     //open input images to extract number of bands and spatial resolution
//     int ncropband=0;//total number of bands to write
//     double dx=0;
//     double dy=0;
//     if(dx_opt.size())
//       dx=dx_opt[0];
//     if(dy_opt.size())
//       dy=dy_opt[0];

//     try{
//       //convert start and end band options to vector of band indexes
//       if(bstart_opt.size()){
//         if(bend_opt.size()!=bstart_opt.size()){
//           string errorstring="Error: options for start and end band indexes must be provided as pairs, missing end band";
//           throw(errorstring);
//         }
//         band_opt.clear();
//         for(int ipair=0;ipair<bstart_opt.size();++ipair){
//           if(bend_opt[ipair]<=bstart_opt[ipair]){
//             string errorstring="Error: index for end band must be smaller then start band";
//             throw(errorstring);
//           }
//           for(unsigned int iband=bstart_opt[ipair];iband<=bend_opt[ipair];++iband)
//             band_opt.push_back(iband);
//         }
//       }
//     }
//     catch(string error){
//       cerr << error << std::endl;
//       throw;
//     }


//     bool isGeoRef=false;
//     string projectionString;
//     // for(int iimg=0;iimg<input_opt.size();++iimg){

//     if(!isGeoRef)
//       isGeoRef=this->isGeoRef();
//     if(this->isGeoRef()&&projection_opt.empty())
//       projectionString=this->getProjection();
//     if(dx_opt.empty()){
//       dx=this->getDeltaX();
//     }

//     if(dy_opt.empty()){
//       dy=this->getDeltaY();
//     }
//     if(band_opt.size())
//       ncropband+=band_opt.size();
//     else
//       ncropband+=this->nrOfBand();

//     GDALDataType theType=getGDALDataType();
//     if(otype_opt.size()){
//       theType=string2GDAL(otype_opt[0]);
//       if(theType==GDT_Unknown)
//         std::cout << "Warning: unknown output pixel type: " << otype_opt[0] << ", using input type as default" << std::endl;
//     }
//     if(verbose_opt[0])
//       cout << "Output pixel type:  " << GDALGetDataTypeName(theType) << endl;

//     //bounding box of cropped image
//     double cropulx=ulx_opt[0];
//     double cropuly=uly_opt[0];
//     double croplrx=lrx_opt[0];
//     double croplry=lry_opt[0];
//     //get bounding box from extentReader if defined
//     VectorOgr extentReader;

//     OGRSpatialReference gdsSpatialRef(getProjectionRef().c_str());
//     if(extent_opt.size()){
//       //image must be georeferenced
//       if(!this->isGeoRef()){
//         string errorstring="Warning: input image is not georeferenced using extent";
//         std::cerr << errorstring << std::endl;
//         throw(errorstring);
//       }
//       statfactory::StatFactory stat;
//       double e_ulx;
//       double e_uly;
//       double e_lrx;
//       double e_lry;
//       for(int iextent=0;iextent<extent_opt.size();++iextent){
//         extentReader.open(extent_opt[iextent],layer_opt,true);//noread=true

//         OGRSpatialReference *vectorSpatialRef=extentReader.getLayer(0)->GetSpatialRef();
//         OGRCoordinateTransformation *vector2raster=0;
//         vector2raster = OGRCreateCoordinateTransformation(vectorSpatialRef, &gdsSpatialRef);
//         if(gdsSpatialRef.IsSame(vectorSpatialRef)){
//           vector2raster=0;
//         }
//         else{
//           if(!vector2raster){
//             std::ostringstream errorStream;
//             errorStream << "Error: cannot create OGRCoordinateTransformation vector to GDAL raster dataset" << std::endl;
//             throw(errorStream.str());
//           }
//         }
//         extentReader.getExtent(e_ulx,e_uly,e_lrx,e_lry,vector2raster);
//         ulx_opt.push_back(e_ulx);
//         uly_opt.push_back(e_uly);
//         lrx_opt.push_back(e_lrx);
//         lry_opt.push_back(e_lry);
//         extentReader.close();
//       }
//       e_ulx=stat.mymin(ulx_opt);
//       e_uly=stat.mymax(uly_opt);
//       e_lrx=stat.mymax(lrx_opt);
//       e_lry=stat.mymin(lry_opt);
//       ulx_opt.clear();
//       uly_opt.clear();
//       lrx_opt.clear();
//       lrx_opt.clear();
//       ulx_opt.push_back(e_ulx);
//       uly_opt.push_back(e_uly);
//       lrx_opt.push_back(e_lrx);
//       lry_opt.push_back(e_lry);
//       if(cut_to_cutline_opt.size()||cut_in_cutline_opt.size()||eoption_opt.size())
//         extentReader.open(extent_opt[0],layer_opt,true);
//     }
//     else if(cx_opt.size()&&cy_opt.size()&&nx_opt.size()&&ny_opt.size()){
//       ulx_opt[0]=cx_opt[0]-nx_opt[0]/2.0;
//       uly_opt[0]=(isGeoRef) ? cy_opt[0]+ny_opt[0]/2.0 : cy_opt[0]-ny_opt[0]/2.0;
//       lrx_opt[0]=cx_opt[0]+nx_opt[0]/2.0;
//       lry_opt[0]=(isGeoRef) ? cy_opt[0]-ny_opt[0]/2.0 : cy_opt[0]+ny_opt[0]/2.0;
//     }
//     else if(cx_opt.size()&&cy_opt.size()&&ns_opt.size()&&nl_opt.size()){
//       ulx_opt[0]=cx_opt[0]-ns_opt[0]*dx/2.0;
//       uly_opt[0]=(isGeoRef) ? cy_opt[0]+nl_opt[0]*dy/2.0 : cy_opt[0]-nl_opt[0]*dy/2.0;
//       lrx_opt[0]=cx_opt[0]+ns_opt[0]*dx/2.0;
//       lry_opt[0]=(isGeoRef) ? cy_opt[0]-nl_opt[0]*dy/2.0 : cy_opt[0]+nl_opt[0]*dy/2.0;
//     }

//     if(verbose_opt[0])
//       cout << "--ulx=" << ulx_opt[0] << " --uly=" << uly_opt[0] << " --lrx=" << lrx_opt[0] << " --lry=" << lry_opt[0] << endl;

//     int ncropcol=0;
//     int ncroprow=0;

//     Jim maskReader;
//     //todo: support transform of extent with cutline
//     if(extent_opt.size()&&(cut_to_cutline_opt[0]||cut_in_cutline_opt[0]||eoption_opt.size())){
//       if(mask_opt.size()){
//         string errorString="Error: can only either mask or extent extent with cut_to_cutline / cut_in_cutline, not both";
//         throw(errorString);
//       }
//       try{
//         // ncropcol=abs(static_cast<unsigned int>(ceil((lrx_opt[0]-ulx_opt[0])/dx)));
//         // ncroprow=abs(static_cast<unsigned int>(ceil((uly_opt[0]-lry_opt[0])/dy)));
//         ncropcol=static_cast<unsigned int>(ceil((lrx_opt[0]-ulx_opt[0])/dx));
//         ncroprow=static_cast<unsigned int>(ceil((uly_opt[0]-lry_opt[0])/dy));
//         maskReader.open(ncropcol,ncroprow,1,GDT_Float64);
//         double gt[6];
//         gt[0]=ulx_opt[0];
//         gt[1]=dx;
//         gt[2]=0;
//         gt[3]=uly_opt[0];
//         gt[4]=0;
//         gt[5]=-dy;
//         maskReader.setGeoTransform(gt);
//         if(projection_opt.size())
//           maskReader.setProjectionProj4(projection_opt[0]);
//         else if(projectionString.size())
//           maskReader.setProjection(projectionString);

//         // maskReader.rasterizeBuf(extentReader,msknodata_opt[0],eoption_opt,layer_opt);
//         maskReader.rasterizeBuf(extentReader,1,eoption_opt,layer_opt);
//       }
//       catch(string error){
//         cerr << error << std::endl;
//         throw;
//       }
//     }
//     else if(mask_opt.size()==1){
//       try{
//         //there is only a single mask
//         maskReader.open(mask_opt[0]);
//         if(mskband_opt[0]>=maskReader.nrOfBand()){
//           string errorString="Error: illegal mask band";
//           throw;
//         }
//       }
//       catch(string error){
//         cerr << error << std::endl;
//         throw;
//       }
//     }

//     //determine number of output bands
//     int writeBand=0;//write band

//     if(scale_opt.size()){
//       while(scale_opt.size()<band_opt.size())
//         scale_opt.push_back(scale_opt[0]);
//     }
//     if(offset_opt.size()){
//       while(offset_opt.size()<band_opt.size())
//         offset_opt.push_back(offset_opt[0]);
//     }
//     if(autoscale_opt.size()){
//       assert(autoscale_opt.size()%2==0);
//     }

//     if(theType==GDT_Unknown){
//       theType=this->getGDALDataType();
//       if(verbose_opt[0])
//         cout << "Using data type from input image: " << GDALGetDataTypeName(theType) << endl;
//     }
//     // if(option_opt.findSubstring("INTERLEAVE=")==option_opt.end()){
//     //   string theInterleave="INTERLEAVE=";
//     //   theInterleave+=this->getInterleave();
//     //   option_opt.push_back(theInterleave);
//     // }
//     // if(verbose_opt[0])
//     //   cout << "size of " << input_opt[iimg] << ": " << ncol << " cols, "<< nrow << " rows" << endl;
//     double uli,ulj,lri,lrj;//image coordinates
//     bool forceEUgrid=false;
//     if(projection_opt.size())
//       forceEUgrid=(!(projection_opt[0].compare("EPSG:3035"))||!(projection_opt[0].compare("EPSG:3035"))||projection_opt[0].find("ETRS-LAEA")!=string::npos);
//     if(ulx_opt[0]>=lrx_opt[0]){//default bounding box: no cropping
//       uli=0;
//       lri=this->nrOfCol()-1;
//       ulj=0;
//       lrj=this->nrOfRow()-1;
//       ncropcol=this->nrOfCol();
//       ncroprow=this->nrOfRow();
//       this->getBoundingBox(cropulx,cropuly,croplrx,croplry);
//       double magicX=1,magicY=1;
//       // this->getMagicPixel(magicX,magicY);
//       if(forceEUgrid){
//         //force to LAEA grid
//         Egcs egcs;
//         egcs.setLevel(egcs.res2level(dx));
//         egcs.force2grid(cropulx,cropuly,croplrx,croplry);
//         this->geo2image(cropulx+(magicX-1.0)*this->getDeltaX(),cropuly-(magicY-1.0)*this->getDeltaY(),uli,ulj);
//         this->geo2image(croplrx+(magicX-2.0)*this->getDeltaX(),croplry-(magicY-2.0)*this->getDeltaY(),lri,lrj);
//       }
//       this->geo2image(cropulx+(magicX-1.0)*this->getDeltaX(),cropuly-(magicY-1.0)*this->getDeltaY(),uli,ulj);
//       this->geo2image(croplrx+(magicX-2.0)*this->getDeltaX(),croplry-(magicY-2.0)*this->getDeltaY(),lri,lrj);
//       // ncropcol=abs(static_cast<unsigned int>(ceil((croplrx-cropulx)/dx)));
//       // ncroprow=abs(static_cast<unsigned int>(ceil((cropuly-croplry)/dy)));
//       ncropcol=static_cast<unsigned int>(ceil((croplrx-cropulx)/dx));
//       ncroprow=static_cast<unsigned int>(ceil((cropuly-croplry)/dy));
//       std::cerr << "Warning: unexpected bounding box, using defaults "<< "--ulx=" << cropulx << " --uly=" << cropuly << " --lrx=" << croplrx << " --lry=" << croplry << std::endl;
//     }
//     else{
//       double magicX=1,magicY=1;
//       // this->getMagicPixel(magicX,magicY);
//       cropulx=ulx_opt[0];
//       cropuly=uly_opt[0];
//       croplrx=lrx_opt[0];
//       croplry=lry_opt[0];
//       if(forceEUgrid){
//         //force to LAEA grid
//         Egcs egcs;
//         egcs.setLevel(egcs.res2level(dx));
//         egcs.force2grid(cropulx,cropuly,croplrx,croplry);
//       }
//       else if(align_opt[0]){
//         if(cropulx>this->getUlx())
//           cropulx-=fmod(cropulx-this->getUlx(),dx);
//         else if(cropulx<this->getUlx())
//           cropulx+=fmod(this->getUlx()-cropulx,dx)-dx;
//         if(croplrx<this->getLrx())
//           croplrx+=fmod(this->getLrx()-croplrx,dx);
//         else if(croplrx>this->getLrx())
//           croplrx-=fmod(croplrx-this->getLrx(),dx)+dx;
//         if(croplry>this->getLry())
//           croplry-=fmod(croplry-this->getLry(),dy);
//         else if(croplry<this->getLry())
//           croplry+=fmod(this->getLry()-croplry,dy)-dy;
//         if(cropuly<this->getUly())
//           cropuly+=fmod(this->getUly()-cropuly,dy);
//         else if(cropuly>this->getUly())
//           cropuly-=fmod(cropuly-this->getUly(),dy)+dy;
//       }
//       this->geo2image(cropulx+(magicX-1.0)*this->getDeltaX(),cropuly-(magicY-1.0)*this->getDeltaY(),uli,ulj);
//       this->geo2image(croplrx+(magicX-2.0)*this->getDeltaX(),croplry-(magicY-2.0)*this->getDeltaY(),lri,lrj);

//       ncropcol=static_cast<unsigned int>(ceil((croplrx-cropulx)/dx));
//       ncroprow=static_cast<unsigned int>(ceil((cropuly-croplry)/dy));
//       uli=floor(uli);
//       ulj=floor(ulj);
//       lri=floor(lri);
//       lrj=floor(lrj);

//       if(cropulx<getUlx() || cropuly>getUly() || croplrx>getLrx() || croplry<getLry())
//         std::cerr << "Warning: requested bounding box not within original bounding box, using "<< "--ulx=" << cropulx << " --uly=" << cropuly << " --lrx=" << croplrx << " --lry=" << croplry << std::endl;
//     }

//     // double deltaX=this->getDeltaX();
//     // double deltaY=this->getDeltaY();
//     if(!imgWriter.nrOfBand()){//not opened yet
//       if(verbose_opt[0]){
//         cout << "cropulx: " << cropulx << endl;
//         cout << "cropuly: " << cropuly << endl;
//         cout << "croplrx: " << croplrx << endl;
//         cout << "croplry: " << croplry << endl;
//         cout << "ncropcol: " << ncropcol << endl;
//         cout << "ncroprow: " << ncroprow << endl;
//         cout << "cropulx+ncropcol*dx: " << cropulx+ncropcol*dx << endl;
//         cout << "cropuly-ncroprow*dy: " << cropuly-ncroprow*dy << endl;
//         cout << "upper left column of input image: " << uli << endl;
//         cout << "upper left row of input image: " << ulj << endl;
//         cout << "lower right column of input image: " << lri << endl;
//         cout << "lower right row of input image: " << lrj << endl;
//         cout << "new number of cols: " << ncropcol << endl;
//         cout << "new number of rows: " << ncroprow << endl;
//         cout << "new number of bands: " << ncropband << endl;
//       }
//       // string imageType;//=this->getImageType();
//       // if(oformat_opt.size())//default
//       //   imageType=oformat_opt[0];
//       try{
//         imgWriter.open(ncropcol,ncroprow,ncropband,theType);
//         imgWriter.setNoData(nodata_opt);
//         // if(nodata_opt.size()){
//         //   imgWriter.setNoData(nodata_opt);
//         // }
//       }
//       catch(string errorstring){
//         cout << errorstring << endl;
//         throw;
//       }
//       if(description_opt.size())
//         imgWriter.setImageDescription(description_opt[0]);
//       double gt[6];
//       gt[0]=cropulx;
//       gt[1]=dx;
//       gt[2]=0;
//       gt[3]=cropuly;
//       gt[4]=0;
//       gt[5]=(this->isGeoRef())? -dy : dy;
//       imgWriter.setGeoTransform(gt);
//       if(projection_opt.size()){
//         if(verbose_opt[0])
//           cout << "projection: " << projection_opt[0] << endl;
//         imgWriter.setProjectionProj4(projection_opt[0]);
//       }
//       else
//         imgWriter.setProjection(this->getProjection());
//       if(imgWriter.getDataType()==GDT_Byte){
//         if(colorTable_opt.size()){
//           if(colorTable_opt[0]!="none")
//             imgWriter.setColorTable(colorTable_opt[0]);
//         }
//         else if (this->getColorTable()!=NULL)//copy colorTable from input image
//           imgWriter.setColorTable(this->getColorTable());
//       }
//     }

//     double startCol=uli;
//     double endCol=lri;
//     if(uli<0)
//       startCol=0;
//     else if(uli>=this->nrOfCol())
//       startCol=this->nrOfCol()-1;
//     if(lri<0)
//       endCol=0;
//     else if(lri>=this->nrOfCol())
//       endCol=this->nrOfCol()-1;
//     double startRow=ulj;
//     double endRow=lrj;
//     if(ulj<0)
//       startRow=0;
//     else if(ulj>=this->nrOfRow())
//       startRow=this->nrOfRow()-1;
//     if(lrj<0)
//       endRow=0;
//     else if(lrj>=this->nrOfRow())
//       endRow=this->nrOfRow()-1;

//     vector<double> readBuffer;
//     unsigned int nband=(band_opt.size())?band_opt.size() : this->nrOfBand();
//     const char* pszMessage;
//     void* pProgressArg=NULL;
//     GDALProgressFunc pfnProgress=GDALTermProgress;
//     double progress=0;
//     MyProgressFunc(progress,pszMessage,pProgressArg);
//     for(unsigned int iband=0;iband<nband;++iband){
//       unsigned int readBand=(band_opt.size()>iband)?band_opt[iband]:iband;
//       if(verbose_opt[0]){
//         cout << "extracting band " << readBand << endl;
//         MyProgressFunc(progress,pszMessage,pProgressArg);
//       }
//       double theMin=0;
//       double theMax=0;
//       if(autoscale_opt.size()){
//         try{
//           this->getMinMax(static_cast<unsigned int>(startCol),static_cast<unsigned int>(endCol),static_cast<unsigned int>(startRow),static_cast<unsigned int>(endRow),readBand,theMin,theMax);
//         }
//         catch(string errorString){
//           cout << errorString << endl;
//         }
//         if(verbose_opt[0])
//           cout << "minmax: " << theMin << ", " << theMax << endl;
//         double theScale=(autoscale_opt[1]-autoscale_opt[0])/(theMax-theMin);
//         double theOffset=autoscale_opt[0]-theScale*theMin;
//         this->setScale(theScale,readBand);
//         this->setOffset(theOffset,readBand);
//       }
//       else{
//         if(scale_opt.size()){
//           if(scale_opt.size()>iband)
//             this->setScale(scale_opt[iband],readBand);
//           else
//             this->setScale(scale_opt[0],readBand);
//         }
//         if(offset_opt.size()){
//           if(offset_opt.size()>iband)
//             this->setOffset(offset_opt[iband],readBand);
//           else
//             this->setOffset(offset_opt[0],readBand);
//         }
//       }

//       double readRow=0;
//       double readCol=0;
//       double lowerCol=0;
//       double upperCol=0;
//       for(int irow=0;irow<imgWriter.nrOfRow();++irow){
//         vector<double> lineMask;
//         double x=0;
//         double y=0;
//         //convert irow to geo
//         imgWriter.image2geo(0,irow,x,y);
//         //lookup corresponding row for irow in this file
//         this->geo2image(x,y,readCol,readRow);
//         vector<double> writeBuffer;
//         if(readRow<0||readRow>=this->nrOfRow()){
//           for(int icol=0;icol<imgWriter.nrOfCol();++icol)
//             writeBuffer.push_back(nodataValue);
//         }
//         else{
//           try{
//             if(endCol<this->nrOfCol()-1){
//               this->readData(readBuffer,startCol,endCol+1,readRow,readBand,theResample);
//             }
//             else{
//               this->readData(readBuffer,startCol,endCol,readRow,readBand,theResample);
//             }
//             double oldRowMask=-1;//keep track of row mask to optimize number of line readings
//             for(int icol=0;icol<imgWriter.nrOfCol();++icol){
//               imgWriter.image2geo(icol,irow,x,y);
//               //lookup corresponding row for irow in this file
//               this->geo2image(x,y,readCol,readRow);
//               if(readCol<0||readCol>=this->nrOfCol()){
//                 writeBuffer.push_back(nodataValue);
//               }
//               else{
//                 bool valid=true;
//                 double geox=0;
//                 double geoy=0;
//                 if(maskReader.isInit()){
//                   //read mask
//                   double colMask=0;
//                   double rowMask=0;

//                   imgWriter.image2geo(icol,irow,geox,geoy);
//                   maskReader.geo2image(geox,geoy,colMask,rowMask);
//                   colMask=static_cast<unsigned int>(colMask);
//                   rowMask=static_cast<unsigned int>(rowMask);
//                   if(rowMask>=0&&rowMask<maskReader.nrOfRow()&&colMask>=0&&colMask<maskReader.nrOfCol()){
//                     if(static_cast<unsigned int>(rowMask)!=static_cast<unsigned int>(oldRowMask)){

//                       try{
//                         maskReader.readData(lineMask,static_cast<unsigned int>(rowMask),mskband_opt[0]);
//                       }
//                       catch(string errorstring){
//                         cerr << errorstring << endl;
//                         throw;
//                       }
//                       catch(...){
//                         cerr << "error caught" << std::endl;
//                         throw;
//                       }
//                       oldRowMask=rowMask;
//                     }
//                     if(cut_to_cutline_opt[0]){
//                       if(lineMask[colMask]!=1){
//                         nodataValue=nodata_opt[0];
//                         valid=false;
//                       }
//                     }
//                     else if(cut_in_cutline_opt[0]){
//                       if(lineMask[colMask]==1){
//                         nodataValue=nodata_opt[0];
//                         valid=false;
//                       }
//                     }
//                     else{
//                       for(int ivalue=0;ivalue<msknodata_opt.size();++ivalue){
//                         if(lineMask[colMask]==msknodata_opt[ivalue]){
//                           if(nodata_opt.size()>ivalue)
//                             nodataValue=nodata_opt[ivalue];
//                           valid=false;
//                           break;
//                         }
//                       }
//                     }
//                   }
//                 }
//                 if(!valid)
//                   writeBuffer.push_back(nodataValue);
//                 else{
//                   switch(theResample){
//                   case(BILINEAR):
//                     lowerCol=readCol-0.5;
//                     lowerCol=static_cast<unsigned int>(lowerCol);
//                     upperCol=readCol+0.5;
//                     upperCol=static_cast<unsigned int>(upperCol);
//                     if(lowerCol<0)
//                       lowerCol=0;
//                     if(upperCol>=this->nrOfCol())
//                       upperCol=this->nrOfCol()-1;
//                     writeBuffer.push_back((readCol-0.5-lowerCol)*readBuffer[upperCol-startCol]+(1-readCol+0.5+lowerCol)*readBuffer[lowerCol-startCol]);
//                     break;
//                   default:
//                     readCol=static_cast<unsigned int>(readCol);
//                     readCol-=startCol;//we only start reading from startCol
//                     writeBuffer.push_back(readBuffer[readCol]);
//                     break;
//                   }
//                 }
//               }
//             }
//           }
//           catch(string errorstring){
//             cout << errorstring << endl;
//             throw;
//           }
//         }
//         if(writeBuffer.size()!=imgWriter.nrOfCol())
//           cout << "writeBuffer.size()=" << writeBuffer.size() << ", imgWriter.nrOfCol()=" << imgWriter.nrOfCol() << endl;

//         assert(writeBuffer.size()==imgWriter.nrOfCol());
//         try{
//           imgWriter.writeData(writeBuffer,irow,writeBand);
//         }
//         catch(string errorstring){
//           cout << errorstring << endl;
//           throw;
//         }
//         if(verbose_opt[0]){
//           progress=(1.0+irow);
//           progress/=imgWriter.nrOfRow();
//           MyProgressFunc(progress,pszMessage,pProgressArg);
//         }
//         else{
//           progress=(1.0+irow);
//           progress+=(imgWriter.nrOfRow()*writeBand);
//           progress/=imgWriter.nrOfBand()*imgWriter.nrOfRow();
//           assert(progress>=0);
//           assert(progress<=1);
//           MyProgressFunc(progress,pszMessage,pProgressArg);
//         }
//       }
//       ++writeBand;
//     }
//     if(extent_opt.size()&&(cut_to_cutline_opt[0]||cut_in_cutline_opt[0]||eoption_opt.size())){
//       extentReader.close();
//     }
//     if(maskReader.isInit())
//       maskReader.close();
//     return(CE_None);
//   }
//   catch(string predefinedString){
//     std::cout << predefinedString << std::endl;
//     throw;
//   }
// }

void Jim::cropBand(Jim& imgWriter, AppFactory& app){
  Optionjl<unsigned int> band_opt("b", "band", "band index to crop (leave empty to retain all bands)");
  Optionjl<unsigned int> bstart_opt("sband", "startband", "Start band sequence number");
  Optionjl<unsigned int> bend_opt("eband", "endband", "End band sequence number");
  Optionjl<short> verbose_opt("v", "verbose", "verbose", 0,2);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=band_opt.retrieveOption(app);
    bstart_opt.retrieveOption(app);
    bend_opt.retrieveOption(app);
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
  }
  catch(string predefinedString){
    std::cerr << predefinedString << std::endl;
    throw;
  }
  std::vector<unsigned int> vband=band_opt;
  if(bstart_opt.size()!=bend_opt.size()){
    std::cerr << "Error: size of start band is not equal to size of end band" << std::endl;
    throw;
  }
  for(size_t ipair=0;ipair<bstart_opt.size();++ipair){
    if(bend_opt[ipair]<bstart_opt[ipair]){
      string errorstring="Error: index for start band must be smaller then end band";
      throw(errorstring);
    }
    for(size_t iband=bstart_opt[ipair];iband<=bend_opt[ipair];++iband)
      vband.push_back(iband);
  }
  if(vband.empty()){
    for(size_t iband=0;iband<nrOfBand();++iband)
      vband.push_back(iband);
  }
  if(vband.empty()){
    std::cerr << "Error: no bands selected" << std::endl;
    throw;
  }
  for(size_t iband=0;iband<vband.size();++iband){
    if(vband[iband]>=nrOfBand()){
      std::ostringstream errorStream;
      errorStream << "Error: selected band " << vband[iband] << " is out of range";
      std::cerr << errorStream.str() << std::endl;
      throw(errorStream.str());
    }
  }
  imgWriter.open(nrOfCol(),nrOfRow(),vband.size(),nrOfPlane(),getGDALDataType());
  imgWriter.copyGeoTransform(*this);
  imgWriter.setProjection(this->getProjection());
  for(size_t iband=0;iband<vband.size();++iband){
    copyData(imgWriter.getDataPointer(iband),vband[iband]);
  }
}

///destructive version of cropBand
void Jim::d_cropBand(AppFactory& app){
  Optionjl<unsigned int> band_opt("b", "band", "band index to crop (leave empty to retain all bands)");
  Optionjl<unsigned int> bstart_opt("sband", "startband", "Start band sequence number");
  Optionjl<unsigned int> bend_opt("eband", "endband", "End band sequence number");
  Optionjl<short> verbose_opt("v", "verbose", "verbose", 0,2);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=band_opt.retrieveOption(app);
    bstart_opt.retrieveOption(app);
    bend_opt.retrieveOption(app);
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
  }
  catch(string predefinedString){
    std::cerr << predefinedString << std::endl;
    throw;
  }
  std::vector<unsigned int> vband=band_opt;
  if(bstart_opt.size()!=bend_opt.size()){
    std::cerr << "Error: size of start band is not equal to size of end band" << std::endl;
    throw;
  }
  for(size_t ipair=0;ipair<bstart_opt.size();++ipair){
    if(bend_opt[ipair]<bstart_opt[ipair]){
      string errorstring="Error: index for start band must be smaller then end band";
      throw(errorstring);
    }
    for(size_t iband=bstart_opt[ipair];iband<=bend_opt[ipair];++iband)
      vband.push_back(iband);
  }
  if(vband.empty()){
    for(size_t iband=0;iband<nrOfBand();++iband)
      vband.push_back(iband);
  }
  if(vband.empty()){
    std::cerr << "Error: no bands selected" << std::endl;
    throw;
  }
  for(size_t iband=0;iband<vband.size();++iband){
    if(vband[iband]>=nrOfBand()){
      std::ostringstream errorStream;
      errorStream << "Error: selected band " << vband[iband] << " is out of range";
      std::cerr << errorStream.str() << std::endl;
      throw(errorStream.str());
    }
  }
  std::vector<double>::iterator scale_it=m_scale.begin();
  std::vector<double>::iterator offset_it=m_offset.begin();
  std::vector<int>::iterator begin_it=m_begin.begin();
  std::vector<int>::iterator end_it=m_end.begin();
  std::vector<void*>::iterator data_it=m_data.begin();
  size_t iband=0;
  while(scale_it!=m_scale.end()){
    if(find(vband.begin(),vband.end(),iband)==vband.end()){
      if(verbose_opt[0]>1)
        std::cout << "removing scale for band " << iband << std::endl;
      if(m_scale.size()>1&&m_scale.size()>iband)
        m_scale.erase(scale_it);
      else
        ++scale_it;
    }
    else
      ++scale_it;
    ++iband;
  }
  iband=0;
  while(offset_it!=m_offset.end()){
    if(find(vband.begin(),vband.end(),iband)==vband.end()){
      if(verbose_opt[0]>1)
        std::cout << "removing offset for band " << iband << std::endl;
      if(m_offset.size()>1&&m_offset.size()>iband)
        m_offset.erase(offset_it);
      else
        ++offset_it;
    }
    else
      ++offset_it;
    ++iband;
  }
  iband=0;
  while(begin_it!=m_begin.end()){
    if(find(vband.begin(),vband.end(),iband)==vband.end()){
      if(verbose_opt[0]>1)
        std::cout << "removing begin for band " << iband << std::endl;
      if(m_begin.size()>1&&m_begin.size()>iband)
        m_begin.erase(begin_it);
      else
        ++begin_it;
    }
    else
      ++begin_it;
    ++iband;
  }
  iband=0;
  while(end_it!=m_end.end()){
    if(find(vband.begin(),vband.end(),iband)==vband.end()){
      if(verbose_opt[0]>1)
        std::cout << "removing end for band " << iband << std::endl;
      if(m_end.size()>1&&m_end.size()>iband)
        m_end.erase(end_it);
      else
        ++end_it;
    }
    else
      ++end_it;
    ++iband;
  }
  iband=0;
  while(data_it!=m_data.end()){
    if(find(vband.begin(),vband.end(),iband)==vband.end()){
        if(verbose_opt[0]>1)
          std::cout << "removing data for band " << iband << std::endl;
        m_data.erase(data_it);
    }
    else{
      if(verbose_opt[0]>1)
        std::cout << "keeping data for band " << iband << std::endl;
      ++data_it;
    }
    ++iband;
   }
  m_nband=m_data.size();
}

void Jim::cropPlane(Jim& imgWriter, AppFactory& app){
  Optionjl<unsigned int> plane_opt("p", "plane", "plane index to crop (leave empty to retain all planes)");
  Optionjl<unsigned int> bstart_opt("splane", "startplane", "Start plane sequence number");
  Optionjl<unsigned int> bend_opt("eplane", "endplane", "End plane sequence number");
  Optionjl<short> verbose_opt("v", "verbose", "verbose", 0,2);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=plane_opt.retrieveOption(app);
    bstart_opt.retrieveOption(app);
    bend_opt.retrieveOption(app);
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
  }
  catch(string predefinedString){
    std::cerr << predefinedString << std::endl;
    throw;
  }
  std::vector<unsigned int> vplane=plane_opt;
  if(bstart_opt.size()!=bend_opt.size()){
    std::cerr << "Error: size of start plane is not equal to size of end plane" << std::endl;
    throw;
  }
  for(size_t ipair=0;ipair<bstart_opt.size();++ipair){
    if(bend_opt[ipair]<bstart_opt[ipair]){
      string errorstring="Error: index for start plane must be smaller then end plane";
      throw(errorstring);
    }
    for(size_t iplane=bstart_opt[ipair];iplane<=bend_opt[ipair];++iplane)
      vplane.push_back(iplane);
  }
  if(vplane.empty()){
    for(size_t iplane=0;iplane<nrOfPlane();++iplane)
      vplane.push_back(iplane);
  }
  if(vplane.empty()){
    std::cerr << "Error: no planes selected" << std::endl;
    throw;
  }
  for(size_t iplane=0;iplane<vplane.size();++iplane){
    if(vplane[iplane]>=nrOfPlane()){
      std::ostringstream errorStream;
      errorStream << "Error: selected plane " << vplane[iplane] << " is out of range";
      std::cerr << errorStream.str() << std::endl;
      throw(errorStream.str());
    }
    if(vplane[iplane]<0){
      std::ostringstream errorStream;
      errorStream << "Error: selected plane " << vplane[iplane] << " is out of range";
      std::cerr << errorStream.str() << std::endl;
      throw(errorStream.str());
    }
  }
  // m_data[nrOfBand()]=(void *) calloc(static_cast<size_t>(nrOfCol()*nrOfRow()*vplane.size()),getDataTypeSizeBytes());
  // for(size_t iband=0;iband<nrOfBand();++iband){
  //   for(size_t iplane=0;iplane<vplane.size();++iplane){
  //     memcpy(m_data[nrOfBand()]+getDataTypeSizeBytes()*nrOfCol()*nrOfRow()*iplane,m_data[iband]+getDataTypeSizeBytes()*nrOfCol()*nrOfRow()*vplane[iplane],getDataTypeSizeBytes()*nrOfCol()*nrOfRow());
  //   }
  //   memcpy(m_data[iband],m_data[nrOfBand()],getDataTypeSizeBytes()*nrOfCol()*nrOfRow()*vplane.size());
  // }

  imgWriter.open(nrOfCol(),nrOfRow(),nrOfBand(),vplane.size(),getGDALDataType());
  imgWriter.copyGeoTransform(*this);
  imgWriter.setProjection(this->getProjection());
  for(size_t iband=0;iband<nrOfBand();++iband){
    for(size_t iplane=0;iplane<vplane.size();++iplane){
      memcpy(imgWriter.getDataPointer(iband)+getDataTypeSizeBytes()*nrOfCol()*nrOfRow()*iplane,m_data[iband]+getDataTypeSizeBytes()*nrOfCol()*nrOfRow()*vplane[iplane],getDataTypeSizeBytes()*nrOfCol()*nrOfRow());
    }
  }
}

///destructive version of cropPlane
void Jim::d_cropPlane(AppFactory& app){
  Optionjl<unsigned int> plane_opt("p", "plane", "plane index to crop (leave empty to retain all planes)");
  Optionjl<unsigned int> bstart_opt("splane", "startplane", "Start plane sequence number");
  Optionjl<unsigned int> bend_opt("eplane", "endplane", "End plane sequence number");
  Optionjl<short> verbose_opt("v", "verbose", "verbose", 0,2);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=plane_opt.retrieveOption(app);
    bstart_opt.retrieveOption(app);
    bend_opt.retrieveOption(app);
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
  }
  catch(string predefinedString){
    std::cerr << predefinedString << std::endl;
    throw;
  }
  std::vector<unsigned int> vplane=plane_opt;
  if(bstart_opt.size()!=bend_opt.size()){
    std::cerr << "Error: size of start plane is not equal to size of end plane" << std::endl;
    throw;
  }
  for(size_t ipair=0;ipair<bstart_opt.size();++ipair){
    if(bend_opt[ipair]<bstart_opt[ipair]){
      string errorstring="Error: index for start plane must be smaller then end plane";
      throw(errorstring);
    }
    for(size_t iplane=bstart_opt[ipair];iplane<=bend_opt[ipair];++iplane)
      vplane.push_back(iplane);
  }
  if(vplane.empty()){
    for(size_t iplane=0;iplane<nrOfPlane();++iplane)
      vplane.push_back(iplane);
  }
  if(vplane.empty()){
    std::cerr << "Error: no planes selected" << std::endl;
    throw;
  }
  for(size_t iplane=0;iplane<vplane.size();++iplane){
    if(vplane[iplane]>=nrOfPlane()||vplane[iplane]<0){
      std::ostringstream errorStream;
      errorStream << "Error: selected plane " << vplane[iplane] << " is out of range";
      std::cerr << errorStream.str() << std::endl;
      throw(errorStream.str());
    }
  }
  if(nrOfPlane() > 1){
    m_data.resize(nrOfBand()+1);
    m_data[nrOfBand()]=(void *) calloc(static_cast<size_t>(nrOfCol()*nrOfRow()*vplane.size()),getDataTypeSizeBytes());
    for(size_t iband=0;iband<nrOfBand();++iband){
      for(size_t iplane=0;iplane<vplane.size();++iplane){
        memcpy(m_data[nrOfBand()]+getDataTypeSizeBytes()*nrOfCol()*nrOfRow()*iplane,m_data[iband]+getDataTypeSizeBytes()*nrOfCol()*nrOfRow()*vplane[iplane],getDataTypeSizeBytes()*nrOfCol()*nrOfRow());
      }
      memcpy(m_data[iband],m_data[nrOfBand()],getDataTypeSizeBytes()*nrOfCol()*nrOfRow()*vplane.size());
    }
    free(m_data[nrOfBand()]);
    m_data.resize(nrOfBand());
    m_nplane=vplane.size();
  }
}

void Jim::cropOgr(VectorOgr& sampleReader, Jim& imgWriter, AppFactory& app){
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
      if(verbose_opt[0]>1)
        cout << "resampling: nearest neighbor" << endl;
    }
    else if(resample_opt[0]=="bilinear"){
      theResample=BILINEAR;
      if(verbose_opt[0]>1)
        cout << "resampling: bilinear interpolation" << endl;
    }
    else{
      std::ostringstream errorStream;
      errorStream << "Error: resampling method " << resample_opt[0] << " not supported";
      throw(errorStream.str());
    }

    const char* pszMessage;
    void* pProgressArg=NULL;
    GDALProgressFunc pfnProgress=GDALTermProgress;
    double progress=0;
    if(verbose_opt[0])
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

    std::vector<unsigned int> vband=band_opt;
    try{
      //convert start and end band options to vector of band indexes
      if(bstart_opt.size()){
        if(bend_opt.size()!=bstart_opt.size()){
          string errorstring="Error: options for start and end band indexes must be provided as pairs, missing end band";
          throw(errorstring);
        }
        vband.clear();
        for(int ipair=0;ipair<bstart_opt.size();++ipair){
          if(bend_opt[ipair]<bstart_opt[ipair]){
            string errorstring="Error: index for start band must be smaller then end band";
            throw(errorstring);
          }
          for(unsigned int iband=bstart_opt[ipair];iband<=bend_opt[ipair];++iband)
            vband.push_back(iband);
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
    if(vband.size())
      ncropband+=vband.size();
    else
      ncropband+=this->nrOfBand();

    GDALDataType theType=getGDALDataType();
    if(otype_opt.size()){
      theType=string2GDAL(otype_opt[0]);
      if(theType==GDT_Unknown)
        std::cout << "Warning: unknown output pixel type: " << otype_opt[0] << ", using input type as default" << std::endl;
    }
    if(verbose_opt[0]>1)
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
      if(verbose_opt[0]>1)
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
      if(!sampleSpatialRef){
        sample2img=0;
        img2sample=0;
      }
      else if(thisSpatialRef.IsSame(sampleSpatialRef)){
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
      sampleReader.getExtent(layer_ulx,layer_uly,layer_lrx,layer_lry,ilayer,sample2img);
      if(verbose_opt[0]>1)
        std::cout << "getExtent: " << layer_ulx << ", " << layer_uly << ", " << layer_lrx << ", " << layer_lry << std::endl;

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
          lrx_opt[0]=layer_lrx;
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
    if(verbose_opt[0]>1)
      cout << "--ulx=" << ulx_opt[0] << " --uly=" << uly_opt[0] << " --lrx=" << lrx_opt[0] << " --lry=" << lry_opt[0] << endl;

    if(doInit){//should have been set to false
      std::ostringstream errorStream;
      if(layer_opt.size())
        errorStream << "Error: no layer found with specified name" << std::endl;
      else
        errorStream << "Error: no layer found" << std::endl;
      throw(errorStream.str());//help was invoked, stop processing
    }
    if(ulx_opt[0]>=lrx_opt[0] || uly_opt[0] <= lry_opt[0]){
      std::ostringstream errorStream;
      errorStream << "Error: bounding box not properly defined" << std::endl;
      throw(errorStream.str());//help was invoked, stop processing
    }

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
      while(scale_opt.size()<vband.size())
        scale_opt.push_back(scale_opt[0]);
    }
    if(offset_opt.size()){
      while(offset_opt.size()<vband.size())
        offset_opt.push_back(offset_opt[0]);
    }
    if(autoscale_opt.size()){
      assert(autoscale_opt.size()%2==0);
    }

    if(theType==GDT_Unknown){
      theType=this->getGDALDataType();
      if(verbose_opt[0]>1)
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
      if(verbose_opt[0]>1){
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
        if(verbose_opt[0]>1)
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
    unsigned int nband=(vband.size())?vband.size() : this->nrOfBand();
    for(unsigned int iband=0;iband<nband;++iband){
      unsigned int readBand=(vband.size()>iband)?vband[iband]:iband;
      if(verbose_opt[0]>1){
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
        if(verbose_opt[0]>1)
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
          if(verbose_opt[0])
            MyProgressFunc(progress,pszMessage,pProgressArg);
        }
      }
      ++writeBand;
    }
    if(maskReader.isInit())
      maskReader.close();
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
void Jim::cropDS(Jim& imgWriter, AppFactory& app){
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
      std::ostringstream errorStream;
      errorStream << "Error: resampling method " << resample_opt[0] << " not supported";
      throw(errorStream.str());
    }

    const char* pszMessage;
    void* pProgressArg=NULL;
    GDALProgressFunc pfnProgress=GDALTermProgress;
    double progress=0;
    if(verbose_opt[0])
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

    std::vector<unsigned int> vband=band_opt;
    try{
      //convert start and end band options to vector of band indexes
      if(bstart_opt.size()){
        if(bend_opt.size()!=bstart_opt.size()){
          string errorstring="Error: options for start and end band indexes must be provided as pairs, missing end band";
          throw(errorstring);
        }
        vband.clear();
        for(int ipair=0;ipair<bstart_opt.size();++ipair){
          if(bend_opt[ipair]<bstart_opt[ipair]){
            string errorstring="Error: index for start band must be smaller then end band";
            throw(errorstring);
          }
          for(unsigned int iband=bstart_opt[ipair];iband<=bend_opt[ipair];++iband)
            vband.push_back(iband);
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
    if(vband.size())
      ncropband+=vband.size();
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

      if((poBand->RasterIO(GF_Read,nXOff,nYOff+m_begin[iband],nXSize,nYSize,imgWriter.getDataPointer(iband),imgWriter.nrOfCol(),imgWriter.nrOfRow(),imgWriter.getGDALDataType(),0,0,&sExtraArg) != CE_None)){
        std::ostringstream errorStream;
        errorStream << "Error: could not read raster band using RasterIO";
        throw(errorStream.str());
      }
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

void Jim::createct(Jim& imgWriter, app::AppFactory& app){
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
#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
    for(unsigned int irow=0;irow<nrOfRow();++irow){
      vector<char> buffer;
      readData(buffer,irow);
      imgWriter.writeData(buffer,irow);
    }
    break;
  }
  case(GDT_Int16):{
    cout << "Warning: copying short to unsigned short without conversion, use convert with -scale if needed..." << endl;
#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
    for(unsigned int irow=0;irow<nrOfRow();++irow){
      vector<short> buffer;
      readData(buffer,irow);
      imgWriter.writeData(buffer,irow);
    }
    break;
  }
  case(GDT_UInt16):{
#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
    for(unsigned int irow=0;irow<nrOfRow();++irow){
      vector<unsigned short> buffer;
      readData(buffer,irow);
      imgWriter.writeData(buffer,irow);
    }
    break;
  }
  default:
    cerr << "data type " << getDataType() << " not supported for adding a colortable" << endl;
    break;
  }
}

//stack image to current image
void Jim::stackBand(Jim& imgSrc, Jim& imgWriter, AppFactory& app){
  Optionjl<unsigned int> band_opt("b", "band", "band index to crop (leave empty to retain all bands)");
  Optionjl<unsigned int> bstart_opt("sband", "startband", "Start band sequence number");
  Optionjl<unsigned int> bend_opt("eband", "endband", "End band sequence number");
  Optionjl<short> verbose_opt("v", "verbose", "verbose", 0,2);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=band_opt.retrieveOption(app);
    bstart_opt.retrieveOption(app);
    bend_opt.retrieveOption(app);
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
  }
  catch(string predefinedString){
    std::cerr << predefinedString << std::endl;
    throw;
  }
  if(bstart_opt.size()!=bend_opt.size()){
    std::cerr << "Error: size of start band is not equal to size of end band" << std::endl;
    throw;
  }
  std::vector<unsigned int> vband=band_opt;
  for(size_t ipair=0;ipair<bstart_opt.size();++ipair){
    if(bend_opt[ipair]<bstart_opt[ipair]){
      string errorstring="Error: index for start band must be smaller then end band";
      throw(errorstring);
    }
    for(size_t iband=bstart_opt[ipair];iband<=bend_opt[ipair];++iband)
      vband.push_back(iband);
  }
  if(vband.empty()){
    for(size_t iband=0;iband<imgSrc.nrOfBand();++iband)
      vband.push_back(iband);
  }
  if(vband.empty()){
    std::cerr << "Error: no bands selected" << std::endl;
    throw;
  }
  if(m_ncol!=imgSrc.nrOfCol()){
    std::string errorString="Error: number of columns do not match";
    throw(errorString);
  }
  if(m_nrow!=imgSrc.nrOfRow()){
    std::string errorString="Error: number of rows do not match";
    throw(errorString);
  }
  if(m_nplane!=imgSrc.nrOfPlane()){
    std::string errorString="Error: number of planes do not match";
    throw(errorString);
  }
  if(m_dataType!=imgSrc.getDataType()){
    std::string errorString="Error: data types do not match";
    throw(errorString);
  }
  imgWriter.open(nrOfCol(),nrOfRow(),nrOfBand()+vband.size(),nrOfPlane(),getGDALDataType());
  imgWriter.copyGeoTransform(*this);
  imgWriter.setProjection(this->getProjection());
  for(size_t iband=0;iband<nrOfBand();++iband){
    copyData(imgWriter.getDataPointer(iband),iband);
  }
  for(size_t iband=0;iband<vband.size();++iband){
    imgSrc.copyData(imgWriter.getDataPointer(nrOfBand()+iband),vband[iband]);
  }
}

//destructive version of stack image to current image
void Jim::d_stackBand(Jim& imgSrc, AppFactory& app){
  Optionjl<unsigned int> band_opt("b", "band", "band index to crop (leave empty to retain all bands)");
  Optionjl<unsigned int> bstart_opt("sband", "startband", "Start band sequence number");
  Optionjl<unsigned int> bend_opt("eband", "endband", "End band sequence number");
  Optionjl<short> verbose_opt("v", "verbose", "verbose", 0,2);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=band_opt.retrieveOption(app);
    bstart_opt.retrieveOption(app);
    bend_opt.retrieveOption(app);
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
  }
  catch(string predefinedString){
    std::cerr << predefinedString << std::endl;
    throw;
  }
  if(bstart_opt.size()!=bend_opt.size()){
    std::cerr << "Error: size of start band is not equal to size of end band" << std::endl;
    throw;
  }
  std::vector<unsigned int> vband=band_opt;
  for(size_t ipair=0;ipair<bstart_opt.size();++ipair){
    if(bend_opt[ipair]<bstart_opt[ipair]){
      string errorstring="Error: index for start band must be smaller then end band";
      throw(errorstring);
    }
    for(size_t iband=bstart_opt[ipair];iband<=bend_opt[ipair];++iband)
      vband.push_back(iband);
  }
  if(vband.empty()){
    for(size_t iband=0;iband<imgSrc.nrOfBand();++iband)
      vband.push_back(iband);
  }
  if(vband.empty()){
    std::cerr << "Error: no bands selected" << std::endl;
    throw;
  }
  if(m_ncol!=imgSrc.nrOfCol()){
    std::string errorString="Error: number of columns do not match";
    throw(errorString);
  }
  if(m_nrow!=imgSrc.nrOfRow()){
    std::string errorString="Error: number of rows do not match";
    throw(errorString);
  }
  if(m_nplane!=imgSrc.nrOfPlane()){
    std::string errorString="Error: number of planes do not match";
    throw(errorString);
  }
  if(m_dataType!=imgSrc.getDataType()){
    std::string errorString="Error: data types do not match";
    throw(errorString);
  }
  size_t oldnband=nrOfBand();
  m_data.resize(oldnband+vband.size());
  m_nband+=vband.size();
  m_begin.resize(oldnband+vband.size());
  m_end.resize(oldnband+vband.size());
  for(size_t iband=0;iband<vband.size();++iband){
    m_data[oldnband+iband]=(void *) calloc(static_cast<size_t>(imgSrc.nrOfPlane()*imgSrc.nrOfCol()*imgSrc.getBlockSize()),imgSrc.getDataTypeSizeBytes());
    imgSrc.copyData(getDataPointer(oldnband+iband),vband[iband]);
  }
}

shared_ptr<Jim> JimList::stackBand(AppFactory& app){
  shared_ptr<Jim> imgWriter=Jim::createImg();
  stackBand(*imgWriter, app);
  return(imgWriter);
}

void JimList::stackBand(Jim& imgWriter, AppFactory& app){
  Optionjl<unsigned int> band_opt("b", "band", "band index to crop (leave empty to retain all bands)");
  Optionjl<unsigned int> bstart_opt("sband", "startband", "Start band sequence number");
  Optionjl<unsigned int> bend_opt("eband", "endband", "End band sequence number");
  Optionjl<short> verbose_opt("v", "verbose", "verbose", 0,2);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=band_opt.retrieveOption(app);
    bstart_opt.retrieveOption(app);
    bend_opt.retrieveOption(app);
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
    std::list<std::shared_ptr<Jim> >::const_iterator imit=begin();
    bool initWriter=false;

    std::vector<unsigned int> vband=band_opt;
    if(bstart_opt.size()!=bend_opt.size()){
      std::cerr << "Error: size of start band is not equal to size of end band" << std::endl;
      throw;
    }
    for(size_t ipair=0;ipair<bstart_opt.size();++ipair){
      if(bend_opt[ipair]<bstart_opt[ipair]){
        string errorstring="Error: index for start band must be smaller then end band";
        throw(errorstring);
      }
      for(size_t iband=bstart_opt[ipair];iband<=bend_opt[ipair];++iband)
        vband.push_back(iband);
    }
    size_t nband=vband.size()*getSize();
    if(!nband){
      for(imit=begin();imit!=end();++imit)
        nband+=(*imit)->nrOfBand();
    }
    imgWriter.open((*begin())->nrOfCol(),(*begin())->nrOfRow(),nband,(*begin())->nrOfPlane(),(*begin())->getGDALDataType());
    imgWriter.copyGeoTransform(*(*begin()));
    imgWriter.setProjection((*begin())->getProjection());
    size_t currentBand=0;
    for(imit=begin();imit!=end();++imit){
      if((*begin())->nrOfCol()!=imgWriter.nrOfCol()){
        std::string errorString="Error: number of columns do not match";
        throw(errorString);
      }
      if((*begin())->nrOfRow()!=imgWriter.nrOfRow()){
        std::string errorString="Error: number of rows do not match";
        throw(errorString);
      }
      if((*begin())->nrOfPlane()!=imgWriter.nrOfPlane()){
        std::string errorString="Error: number of planes do not match";
        throw(errorString);
      }
      if((*begin())->getDataType()!=imgWriter.getDataType()){
        std::string errorString="Error: data types do not match";
        throw(errorString);
      }
      if(!(*imit)){
        std::ostringstream errorStream;
        errorStream << "Error: image in list is empty"<< std::endl;
        throw(errorStream.str());
      }
      if(vband.size()){
        for(size_t iband=0;iband<vband.size();++iband){
          if(iband>=(*imit)->nrOfBand()){
            std::string errorString="Error: band number out of range";
            throw(errorString);
          }

          (*imit)->copyData(imgWriter.getDataPointer(currentBand),vband[iband]);
          ++currentBand;
        }
      }
      else{
        for(size_t iband=0;iband<(*imit)->nrOfBand();++iband){
          (*imit)->copyData(imgWriter.getDataPointer(currentBand),iband);
          ++currentBand;
        }
      }
    }
  }
  catch(string predefinedString){
    std::cout << predefinedString << std::endl;
    throw;
  }
}

//destructive version of stack image to current image
void Jim::d_stackPlane(Jim& imgSrc, AppFactory& app){
  Optionjl<short> verbose_opt("v", "verbose", "verbose", 0,2);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=verbose_opt.retrieveOption(app);

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
  }
  catch(string predefinedString){
    std::cerr << predefinedString << std::endl;
    throw;
  }
  if(m_ncol!=imgSrc.nrOfCol()){
    std::string errorString="Error: number of columns do not match";
    throw(errorString);
  }
  if(m_nrow!=imgSrc.nrOfRow()){
    std::string errorString="Error: number of rows do not match";
    throw(errorString);
  }
  if(m_nband!=imgSrc.nrOfBand()){
    std::string errorString="Error: number of bands do not match";
    throw(errorString);
  }
  if(m_dataType!=imgSrc.getDataType()){
    std::string errorString="Error: data types do not match";
    throw(errorString);
  }
  size_t oldnplane=nrOfPlane();
  m_nplane+=imgSrc.nrOfPlane();
  m_data.resize(nrOfBand()+1);
  m_data[nrOfBand()]=(void *) calloc(static_cast<size_t>(nrOfCol()*nrOfRow()*oldnplane),getDataTypeSizeBytes());
  for(size_t iband=0;iband<nrOfBand();++iband){
    memcpy(m_data[nrOfBand()],m_data[iband],getDataTypeSizeBytes()*nrOfCol()*m_blockSize*oldnplane);
    //allocate memory
    free(m_data[iband]);
    m_data[iband]=(void *) calloc(static_cast<size_t>(nrOfCol()*nrOfRow()*nrOfPlane()),getDataTypeSizeBytes());
    memcpy(m_data[iband],m_data[nrOfBand()],getDataTypeSizeBytes()*nrOfCol()*nrOfRow()*oldnplane);
    memcpy(m_data[iband]+getDataTypeSizeBytes()*nrOfCol()*nrOfRow()*oldnplane,imgSrc.getDataPointer(iband),imgSrc.getDataTypeSizeBytes()*imgSrc.nrOfCol()*imgSrc.nrOfRow()*imgSrc.nrOfPlane());
  }
  free(m_data[nrOfBand()]);
  m_data.resize(nrOfBand());
}


shared_ptr<Jim> JimList::stackPlane(AppFactory& app){
  shared_ptr<Jim> imgWriter=Jim::createImg();
  stackPlane(*imgWriter, app);
  return(imgWriter);
}

void JimList::stackPlane(Jim& imgWriter, AppFactory& app){
  Optionjl<short> verbose_opt("v", "verbose", "verbose", 0,2);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=verbose_opt.retrieveOption(app);

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
    std::list<std::shared_ptr<Jim> >::const_iterator imit=begin();
    bool initWriter=false;

    size_t nplane=0;
    for(imit=begin();imit!=end();++imit)
      nplane+=(*imit)->nrOfPlane();

    imgWriter.open((*begin())->nrOfCol(),(*begin())->nrOfRow(),(*begin())->nrOfBand(),nplane,(*begin())->getGDALDataType());
    imgWriter.copyGeoTransform(*(*begin()));
    imgWriter.setProjection((*begin())->getProjection());
    size_t currentBand=0;
    size_t iplane=0;
    for(imit=begin();imit!=end();++imit){
      if((*begin())->nrOfCol()!=imgWriter.nrOfCol()){
        std::string errorString="Error: number of columns do not match";
        throw(errorString);
      }
      if((*begin())->nrOfRow()!=imgWriter.nrOfRow()){
        std::string errorString="Error: number of rows do not match";
        throw(errorString);
      }
      if((*begin())->nrOfBand()!=imgWriter.nrOfBand()){
        std::string errorString="Error: number of bands do not match";
        throw(errorString);
      }
      if((*begin())->getDataType()!=imgWriter.getDataType()){
        std::string errorString="Error: data types do not match";
        throw(errorString);
      }
      if(!(*imit)){
        std::ostringstream errorStream;
        errorStream << "Error: image in list is empty"<< std::endl;
        throw(errorStream.str());
      }
      for(size_t iband=0;iband<imgWriter.nrOfBand();++iband){
          if(iband>=(*imit)->nrOfBand()){
            std::string errorString="Error: band number out of range";
            throw(errorString);
          }
          (*imit)->copyData(imgWriter.getDataPointer(iband)+imgWriter.nrOfCol()*imgWriter.nrOfRow()*iplane*imgWriter.getDataTypeSizeBytes(),iband);
      }
      iplane+=(*imit)->nrOfPlane();
    }
  }
  catch(string predefinedString){
    std::cout << predefinedString << std::endl;
    throw;
  }
}

// shared_ptr<Jim> JimList::crop(AppFactory& app){
//   shared_ptr<Jim> imgWriter=Jim::createImg();
//   crop(*imgWriter, app);
//   return(imgWriter);
// }

//todo: support extent a VectorOgr argument instead of option in app
 // JimList& JimList::crop(Jim& imgWriter, AppFactory& app){
 //   Optionjl<string>  projection_opt("a_srs", "a_srs", "Override the projection for the output file (leave blank to copy from input file, use epsg:3035 to use European projection and force to European grid");
 //   //todo: support layer names
 //   Optionjl<string>  extent_opt("e", "extent", "get boundary from extent from polygons in vector file");
 //   Optionjl<string>  layer_opt("ln", "ln", "layer name of extent to crop");
 //   Optionjl<bool> cut_to_cutline_opt("cut_to_cutline", "crop_to_cutline", "Crop the extent of the target dataset to the extent of the cutline, setting the outside area to nodata.",false);
 //   Optionjl<bool> cut_in_cutline_opt("cut_in_cutline", "crop_in_cutline", "Crop the extent of the target dataset to the extent of the cutline, setting the inner area to nodata.",false);
 //   Optionjl<string> eoption_opt("eo","eo", "special extent options controlling rasterization: ATTRIBUTE|CHUNKYSIZE|ALL_TOUCHED|BURN_VALUE_FROM|MERGE_ALG, e.g., -eo ATTRIBUTE=fieldname");
 //   Optionjl<string> mask_opt("m", "mask", "Use the the specified file as a validity mask (0 is nodata).");
 //   Optionjl<double> msknodata_opt("msknodata", "msknodata", "Mask value not to consider for crop.", 0);
 //   Optionjl<unsigned int> mskband_opt("mskband", "mskband", "Mask band to read (0 indexed)", 0);
 //   Optionjl<double>  ulx_opt("ulx", "ulx", "Upper left x value bounding box", 0.0);
 //   Optionjl<double>  uly_opt("uly", "uly", "Upper left y value bounding box", 0.0);
 //   Optionjl<double>  lrx_opt("lrx", "lrx", "Lower right x value bounding box", 0.0);
 //   Optionjl<double>  lry_opt("lry", "lry", "Lower right y value bounding box", 0.0);
 //   Optionjl<double>  dx_opt("dx", "dx", "Output resolution in x (in meter) (empty: keep original resolution)");
 //   Optionjl<double>  dy_opt("dy", "dy", "Output resolution in y (in meter) (empty: keep original resolution)");
 //   Optionjl<double> cx_opt("x", "x", "x-coordinate of image center to crop (in meter)");
 //   Optionjl<double> cy_opt("y", "y", "y-coordinate of image center to crop (in meter)");
 //   Optionjl<double> nx_opt("nx", "nx", "image size in x to crop (in meter)");
 //   Optionjl<double> ny_opt("ny", "ny", "image size in y to crop (in meter)");
 //   Optionjl<unsigned int> ns_opt("ns", "ns", "number of samples  to crop (in pixels)");
 //   Optionjl<unsigned int> nl_opt("nl", "nl", "number of lines to crop (in pixels)");
 //   Optionjl<unsigned int>  band_opt("b", "band", "band index to crop (leave empty to retain all bands)");
 //   Optionjl<unsigned int> bstart_opt("sband", "startband", "Start band sequence number");
 //   Optionjl<unsigned int> bend_opt("eband", "endband", "End band sequence number");
 //   Optionjl<double> autoscale_opt("as", "autoscale", "scale output to min and max, e.g., --autoscale 0 --autoscale 255");
 //   Optionjl<double> scale_opt("scale", "scale", "output=scale*input+offset");
 //   Optionjl<double> offset_opt("offset", "offset", "output=scale*input+offset");
 //   Optionjl<string>  otype_opt("ot", "otype", "Data type for output image ({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}). Empty string: inherit type from input image");
 //   // Optionjl<string>  oformat_opt("of", "oformat", "Output image format (see also gdal_translate).","GTiff");
 //   // Optionjl<string> option_opt("co", "co", "Creation option for output file. Multiple options can be specified.");
 //   Optionjl<string>  colorTable_opt("ct", "ct", "color table (file with 5 columns: id R G B ALFA (0: transparent, 255: solid)");
 //   Optionjl<double>  nodata_opt("nodata", "nodata", "Nodata value to put in image if out of bounds.");
 //   Optionjl<string>  resample_opt("r", "resample", "Resampling method (near: nearest neighbor, bilinear: bi-linear interpolation).", "near");
 //   Optionjl<string>  description_opt("d", "description", "Set image description");
 //   Optionjl<bool>  align_opt("align", "align", "Align output bounding box to input image",false);
 //   Optionjl<short>  verbose_opt("v", "verbose", "verbose", 0,2);

 //   extent_opt.setHide(1);
 //   layer_opt.setHide(1);
 //   cut_to_cutline_opt.setHide(1);
 //   cut_in_cutline_opt.setHide(1);
 //   eoption_opt.setHide(1);
 //   bstart_opt.setHide(1);
 //   bend_opt.setHide(1);
 //   mask_opt.setHide(1);
 //   msknodata_opt.setHide(1);
 //   mskband_opt.setHide(1);
 //   // option_opt.setHide(1);
 //   cx_opt.setHide(1);
 //   cy_opt.setHide(1);
 //   nx_opt.setHide(1);
 //   ny_opt.setHide(1);
 //   ns_opt.setHide(1);
 //   nl_opt.setHide(1);
 //   scale_opt.setHide(1);
 //   offset_opt.setHide(1);
 //   nodata_opt.setHide(1);
 //   description_opt.setHide(1);

 //   bool doProcess;//stop process when program was invoked with help option (-h --help)
 //   try{
 //     doProcess=projection_opt.retrieveOption(app);
 //     ulx_opt.retrieveOption(app);
 //     uly_opt.retrieveOption(app);
 //     lrx_opt.retrieveOption(app);
 //     lry_opt.retrieveOption(app);
 //     band_opt.retrieveOption(app);
 //     bstart_opt.retrieveOption(app);
 //     bend_opt.retrieveOption(app);
 //     autoscale_opt.retrieveOption(app);
 //     otype_opt.retrieveOption(app);
 //     // oformat_opt.retrieveOption(app);
 //     colorTable_opt.retrieveOption(app);
 //     dx_opt.retrieveOption(app);
 //     dy_opt.retrieveOption(app);
 //     resample_opt.retrieveOption(app);
 //     extent_opt.retrieveOption(app);
 //     layer_opt.retrieveOption(app);
 //     cut_to_cutline_opt.retrieveOption(app);
 //     cut_in_cutline_opt.retrieveOption(app);
 //     eoption_opt.retrieveOption(app);
 //     mask_opt.retrieveOption(app);
 //     msknodata_opt.retrieveOption(app);
 //     mskband_opt.retrieveOption(app);
 //     // option_opt.retrieveOption(app);
 //     cx_opt.retrieveOption(app);
 //     cy_opt.retrieveOption(app);
 //     nx_opt.retrieveOption(app);
 //     ny_opt.retrieveOption(app);
 //     ns_opt.retrieveOption(app);
 //     nl_opt.retrieveOption(app);
 //     scale_opt.retrieveOption(app);
 //     offset_opt.retrieveOption(app);
 //     nodata_opt.retrieveOption(app);
 //     description_opt.retrieveOption(app);
 //     align_opt.retrieveOption(app);
 //     verbose_opt.retrieveOption(app);

 //     if(!doProcess){
 //       cout << endl;
 //       std::ostringstream helpStream;
 //       helpStream << "short option -h shows basic options only, use long option --help to show all options" << std::endl;
 //       throw(helpStream.str());//help was invoked, stop processing
 //     }
 //     if(empty()){
 //       std::ostringstream errorStream;
 //       errorStream << "Input collection is empty. Use --help for more help information" << std::endl;
 //       throw(errorStream.str());
 //     }


 //     std::vector<std::string> badKeys;
 //     app.badKeys(badKeys);
 //     if(badKeys.size()){
 //       std::ostringstream errorStream;
 //       if(badKeys.size()>1)
 //         errorStream << "Error: unknown keys: ";
 //       else
 //         errorStream << "Error: unknown key: ";
 //       for(int ikey=0;ikey<badKeys.size();++ikey){
 //         errorStream << badKeys[ikey] << " ";
 //       }
 //       errorStream << std::endl;
 //       throw(errorStream.str());
 //     }

 //     double nodataValue=nodata_opt.size()? nodata_opt[0] : 0;
 //     RESAMPLE theResample;
 //     if(resample_opt[0]=="near"){
 //       theResample=NEAR;
 //       if(verbose_opt[0])
 //         cout << "resampling: nearest neighbor" << endl;
 //     }
 //     else if(resample_opt[0]=="bilinear"){
 //       theResample=BILINEAR;
 //       if(verbose_opt[0])
 //         cout << "resampling: bilinear interpolation" << endl;
 //     }
 //     else{
 //       std::ostringstream errorStream;
 //       errorStream << "Error: resampling method " << resample_opt[0] << " not supported" << std::endl;
 //       throw(errorStream.str());
 //       // return(CE_Failure);
 //     }

 //     const char* pszMessage;
 //     void* pProgressArg=NULL;
 //     GDALProgressFunc pfnProgress=GDALTermProgress;
 //     double progress=0;
 //     MyProgressFunc(progress,pszMessage,pProgressArg);
 //     // ImgReaderGdal imgReader;
 //     // ImgWriterGdal imgWriter;
 //     //open input images to extract number of bands and spatial resolution
 //     int ncropband=0;//total number of bands to write
 //     double dx=0;
 //     double dy=0;
 //     if(dx_opt.size())
 //       dx=dx_opt[0];
 //     if(dy_opt.size())
 //       dy=dy_opt[0];

 //     try{
 //       //convert start and end band options to vector of band indexes
 //       if(bstart_opt.size()){
 //         if(bend_opt.size()!=bstart_opt.size()){
 //           string errorstring="Error: options for start and end band indexes must be provided as pairs, missing end band";
 //           throw(errorstring);
 //         }
 //         band_opt.clear();
 //         for(int ipair=0;ipair<bstart_opt.size();++ipair){
 //           if(bend_opt[ipair]<=bstart_opt[ipair]){
 //             string errorstring="Error: index for end band must be smaller then start band";
 //             throw(errorstring);
 //           }
 //           for(unsigned int iband=bstart_opt[ipair];iband<=bend_opt[ipair];++iband)
 //             band_opt.push_back(iband);
 //         }
 //       }
 //     }
 //     catch(string error){
 //       throw;
 //       // return(CE_Failure);
 //     }

 //     bool isGeoRef=false;
 //     string projectionString;
 //     // for(int iimg=0;iimg<input_opt.size();++iimg){

 //     // std::vector<std::shared_ptr<Jim> >::const_iterator imit=begin();
 //     std::list<std::shared_ptr<Jim> >::const_iterator imit=begin();

 //     for(imit=begin();imit!=end();++imit){
 //       //image must be georeferenced
 //       if(!((*imit)->isGeoRef())){
 //         string errorstring="Warning: input image is not georeferenced in JimList";
 //         std::cerr << errorstring << std::endl;
 //         // throw(errorstring);
 //       }
 //       // while((imgReader=getNextImage())){
 //       // for(int iimg=0;iimg<imgReader.size();++iimg){
 //       // try{
 //       // }
 //       // catch(string error){
 //       //   cerr << "Error: could not open file " << input_opt[iimg] << ": " << error << std::endl;
 //       //   exit(1);
 //       // }
 //       if(!isGeoRef)
 //         isGeoRef=(*imit)->isGeoRef();
 //       if((*imit)->isGeoRef()&&projection_opt.empty())
 //         projectionString=(*imit)->getProjection();
 //       if(dx_opt.empty()){
 //         if(imit==begin()||(*imit)->getDeltaX()<dx)
 //           dx=(*imit)->getDeltaX();
 //         if(dx<=0){
 //           string errorstring="Warning: pixel size in x has not been defined in input image";
 //           std::cerr << errorstring << std::endl;
 //           dx=1;
 //           // throw(errorstring);
 //         }
 //       }

 //       if(dy_opt.empty()){
 //         if(imit==begin()||(*imit)->getDeltaY()<dy)
 //           dy=(*imit)->getDeltaY();
 //         if(dy<=0){
 //           string errorstring="Warning: pixel size in y has not been defined in input image";
 //           std::cerr << errorstring << std::endl;
 //           dy=1;
 //           // throw(errorstring);
 //         }
 //       }
 //       if(band_opt.size())
 //         ncropband+=band_opt.size();
 //       else
 //         ncropband+=(*imit)->nrOfBand();
 //       // (*imit)->close();
 //     }

 //     GDALDataType theType=GDT_Unknown;
 //     if(otype_opt.size()){
 //       theType=string2GDAL(otype_opt[0]);
 //       if(theType==GDT_Unknown)
 //         std::cout << "Warning: unknown output pixel type: " << otype_opt[0] << ", using input type as default" << std::endl;
 //     }
 //     if(verbose_opt[0])
 //       cout << "Output pixel type:  " << GDALGetDataTypeName(theType) << endl;

 //     //bounding box of cropped image
 //     double cropulx=ulx_opt[0];
 //     double cropuly=uly_opt[0];
 //     double croplrx=lrx_opt[0];
 //     double croplry=lry_opt[0];
 //     //get bounding box from extentReader if defined
 //     VectorOgr extentReader;

 //     if(extent_opt.size()){
 //       double e_ulx;
 //       double e_uly;
 //       double e_lrx;
 //       double e_lry;
 //       for(int iextent=0;iextent<extent_opt.size();++iextent){
 //         extentReader.open(extent_opt[iextent],layer_opt,true);//noread=true
 //         extentReader.getExtent(e_ulx,e_uly,e_lrx,e_lry);
 //         if(!iextent){
 //           ulx_opt[0]=e_ulx;
 //           uly_opt[0]=e_uly;
 //           lrx_opt[0]=e_lrx;
 //           lry_opt[0]=e_lry;
 //         }
 //         else{
 //           if(e_ulx<ulx_opt[0])
 //             ulx_opt[0]=e_ulx;
 //           if(e_uly>uly_opt[0])
 //             uly_opt[0]=e_uly;
 //           if(e_lrx>lrx_opt[0])
 //             lrx_opt[0]=e_lrx;
 //           if(e_lry<lry_opt[0])
 //             lry_opt[0]=e_lry;
 //         }
 //         extentReader.close();
 //       }
 //       if(croplrx>cropulx&&cropulx>ulx_opt[0])
 //         ulx_opt[0]=cropulx;
 //       if(croplrx>cropulx&&croplrx<lrx_opt[0])
 //         lrx_opt[0]=croplrx;
 //       if(cropuly>croplry&&cropuly<uly_opt[0])
 //         uly_opt[0]=cropuly;
 //       if(croplry<cropuly&&croplry>lry_opt[0])
 //         lry_opt[0]=croplry;
 //       if(cut_to_cutline_opt[0]||cut_in_cutline_opt[0]||eoption_opt.size())
 //         extentReader.open(extent_opt[0],layer_opt,true);
 //     }
 //     else if(cx_opt.size()&&cy_opt.size()&&nx_opt.size()&&ny_opt.size()){
 //       ulx_opt[0]=cx_opt[0]-nx_opt[0]/2.0;
 //       uly_opt[0]=(isGeoRef) ? cy_opt[0]+ny_opt[0]/2.0 : cy_opt[0]-ny_opt[0]/2.0;
 //       lrx_opt[0]=cx_opt[0]+nx_opt[0]/2.0;
 //       lry_opt[0]=(isGeoRef) ? cy_opt[0]-ny_opt[0]/2.0 : cy_opt[0]+ny_opt[0]/2.0;
 //     }
 //     else if(cx_opt.size()&&cy_opt.size()&&ns_opt.size()&&nl_opt.size()){
 //       ulx_opt[0]=cx_opt[0]-ns_opt[0]*dx/2.0;
 //       uly_opt[0]=(isGeoRef) ? cy_opt[0]+nl_opt[0]*dy/2.0 : cy_opt[0]-nl_opt[0]*dy/2.0;
 //       lrx_opt[0]=cx_opt[0]+ns_opt[0]*dx/2.0;
 //       lry_opt[0]=(isGeoRef) ? cy_opt[0]-nl_opt[0]*dy/2.0 : cy_opt[0]+nl_opt[0]*dy/2.0;
 //     }

 //     if(verbose_opt[0])
 //       cout << "--ulx=" << ulx_opt[0] << " --uly=" << uly_opt[0] << " --lrx=" << lrx_opt[0] << " --lry=" << lry_opt[0] << endl;

 //     int ncropcol=0;
 //     int ncroprow=0;

 //     Jim maskReader;
 //     if(extent_opt.size()&&(cut_to_cutline_opt[0]||cut_in_cutline_opt[0]||eoption_opt.size())){
 //       if(mask_opt.size()){
 //         string errorString="Error: can only either mask or extent extent with cut_to_cutline / cut_in_cutline, not both";
 //         throw(errorString);
 //       }
 //       try{
 //         // ncropcol=abs(static_cast<unsigned int>(ceil((lrx_opt[0]-ulx_opt[0])/dx)));
 //         // ncroprow=abs(static_cast<unsigned int>(ceil((uly_opt[0]-lry_opt[0])/dy)));
 //         ncropcol=static_cast<unsigned int>(ceil((lrx_opt[0]-ulx_opt[0])/dx));
 //         ncroprow=static_cast<unsigned int>(ceil((uly_opt[0]-lry_opt[0])/dy));
 //         maskReader.open(ncropcol,ncroprow,1,GDT_Float64);
 //         double gt[6];
 //         gt[0]=ulx_opt[0];
 //         gt[1]=dx;
 //         gt[2]=0;
 //         gt[3]=uly_opt[0];
 //         gt[4]=0;
 //         gt[5]=-dy;
 //         maskReader.setGeoTransform(gt);
 //         if(projection_opt.size())
 //           maskReader.setProjectionProj4(projection_opt[0]);
 //         else if(projectionString.size())
 //           maskReader.setProjection(projectionString);

 //         // vector<double> burnValues(1,1);//burn value is 1 (single band)
 //         // maskReader.rasterizeBuf(extentReader,msknodata_opt[0],eoption_opt,layer_opt);
 //         maskReader.rasterizeBuf(extentReader,1,eoption_opt,layer_opt);

 //         // if(eoption_opt.size())
 //         //   maskReader.rasterizeBuf(extentReader,eoption_opt);
 //         // else
 //         //   maskReader.rasterizeBuf(extentReader);
 //       }
 //       catch(string error){
 //         throw;
 //         // return(CE_Failure);
 //       }
 //     }
 //     else if(mask_opt.size()==1){
 //       try{
 //         //there is only a single mask
 //         maskReader.open(mask_opt[0]);
 //         if(mskband_opt[0]>=maskReader.nrOfBand()){
 //           string errorString="Error: illegal mask band";
 //           throw(errorString);
 //         }
 //       }
 //       catch(string error){
 //         throw;
 //         // return(CE_Failure);
 //       }
 //     }

 //     //determine number of output bands
 //     int writeBand=0;//write band

 //     if(scale_opt.size()){
 //       while(scale_opt.size()<band_opt.size())
 //         scale_opt.push_back(scale_opt[0]);
 //     }
 //     if(offset_opt.size()){
 //       while(offset_opt.size()<band_opt.size())
 //         offset_opt.push_back(offset_opt[0]);
 //     }
 //     if(autoscale_opt.size()){
 //       assert(autoscale_opt.size()%2==0);
 //     }

 //     // for(int iimg=0;iimg<input_opt.size();++iimg){
 //     for(imit=begin();imit!=end();++imit){
 //       // for(int iimg=0;iimg<imgReader.size();++iimg){
 //       // if(verbose_opt[0])
 //       //   cout << "opening image " << input_opt[iimg] << endl;
 //       // try{
 //       // }
 //       // catch(string error){
 //       //   cerr << error << std::endl;
 //       //   exit(2);
 //       // }
 //       //if output type not set, get type from input image
 //       if(theType==GDT_Unknown){
 //         theType=(*imit)->getGDALDataType();
 //         if(verbose_opt[0])
 //           cout << "Using data type from input image: " << GDALGetDataTypeName(theType) << endl;
 //       }
 //       // if(option_opt.findSubstring("INTERLEAVE=")==option_opt.end()){
 //       //   string theInterleave="INTERLEAVE=";
 //       //   theInterleave+=(*imit)->getInterleave();
 //       //   option_opt.push_back(theInterleave);
 //       // }
 //       // if(verbose_opt[0])
 //       //   cout << "size of " << input_opt[iimg] << ": " << ncol << " cols, "<< nrow << " rows" << endl;
 //       double uli,ulj,lri,lrj;//image coordinates
 //       bool forceEUgrid=false;
 //       if(projection_opt.size())
 //         forceEUgrid=(!(projection_opt[0].compare("EPSG:3035"))||!(projection_opt[0].compare("EPSG:3035"))||projection_opt[0].find("ETRS-LAEA")!=string::npos);
 //       if(ulx_opt[0]>=lrx_opt[0]){//default bounding box: no cropping
 //         uli=0;
 //         lri=(*imit)->nrOfCol()-1;
 //         ulj=0;
 //         lrj=(*imit)->nrOfRow()-1;
 //         ncropcol=(*imit)->nrOfCol();
 //         ncroprow=(*imit)->nrOfRow();
 //         (*imit)->getBoundingBox(cropulx,cropuly,croplrx,croplry);
 //         double magicX=1,magicY=1;
 //         // (*imit)->getMagicPixel(magicX,magicY);
 //         if(forceEUgrid){
 //           //force to LAEA grid
 //           Egcs egcs;
 //           egcs.setLevel(egcs.res2level(dx));
 //           egcs.force2grid(cropulx,cropuly,croplrx,croplry);
 //           (*imit)->geo2image(cropulx+(magicX-1.0)*(*imit)->getDeltaX(),cropuly-(magicY-1.0)*(*imit)->getDeltaY(),uli,ulj);
 //           (*imit)->geo2image(croplrx+(magicX-2.0)*(*imit)->getDeltaX(),croplry-(magicY-2.0)*(*imit)->getDeltaY(),lri,lrj);
 //         }
 //         (*imit)->geo2image(cropulx+(magicX-1.0)*(*imit)->getDeltaX(),cropuly-(magicY-1.0)*(*imit)->getDeltaY(),uli,ulj);
 //         (*imit)->geo2image(croplrx+(magicX-2.0)*(*imit)->getDeltaX(),croplry-(magicY-2.0)*(*imit)->getDeltaY(),lri,lrj);
 //         // ncropcol=abs(static_cast<unsigned int>(ceil((croplrx-cropulx)/dx)));
 //         // ncroprow=abs(static_cast<unsigned int>(ceil((cropuly-croplry)/dy)));
 //         ncropcol=static_cast<unsigned int>(ceil((croplrx-cropulx)/dx));
 //         ncroprow=static_cast<unsigned int>(ceil((cropuly-croplry)/dy));
 //         if(verbose_opt[0]){
 //           cout << "default bounding box" << endl;
 //           cout << "ulx_opt[0]: " << ulx_opt[0]<< endl;
 //           cout << "uly_opt[0]: " << uly_opt[0]<< endl;
 //           cout << "lrx_opt[0]: " << lrx_opt[0]<< endl;
 //           cout << "lry_opt[0]: " << lry_opt[0]<< endl;
 //           cout << "croplrx,cropulx: " << croplrx << "," << cropulx << endl;
 //           cout << "dx: " << dx << endl;
 //           cout << "cropuly,croplry: " << cropuly << "," << croplry << endl;
 //           cout << "dy: " << dy << endl;
 //           cout << "filename: " << (*imit)->getFileName() << endl;
 //         }
 //       }
 //       else{
 //         double magicX=1,magicY=1;
 //         // (*imit)->getMagicPixel(magicX,magicY);
 //         cropulx=ulx_opt[0];
 //         cropuly=uly_opt[0];
 //         croplrx=lrx_opt[0];
 //         croplry=lry_opt[0];
 //         if(forceEUgrid){
 //           //force to LAEA grid
 //           Egcs egcs;
 //           egcs.setLevel(egcs.res2level(dx));
 //           egcs.force2grid(cropulx,cropuly,croplrx,croplry);
 //         }
 //         else if(align_opt[0]){
 //           if(cropulx>(*imit)->getUlx())
 //             cropulx-=fmod(cropulx-(*imit)->getUlx(),dx);
 //           else if(cropulx<(*imit)->getUlx())
 //             cropulx+=fmod((*imit)->getUlx()-cropulx,dx)-dx;
 //           if(croplrx<(*imit)->getLrx())
 //             croplrx+=fmod((*imit)->getLrx()-croplrx,dx);
 //           else if(croplrx>(*imit)->getLrx())
 //             croplrx-=fmod(croplrx-(*imit)->getLrx(),dx)+dx;
 //           if(croplry>(*imit)->getLry())
 //             croplry-=fmod(croplry-(*imit)->getLry(),dy);
 //           else if(croplry<(*imit)->getLry())
 //             croplry+=fmod((*imit)->getLry()-croplry,dy)-dy;
 //           if(cropuly<(*imit)->getUly())
 //             cropuly+=fmod((*imit)->getUly()-cropuly,dy);
 //           else if(cropuly>(*imit)->getUly())
 //             cropuly-=fmod(cropuly-(*imit)->getUly(),dy)+dy;
 //         }
 //         (*imit)->geo2image(cropulx+(magicX-1.0)*(*imit)->getDeltaX(),cropuly-(magicY-1.0)*(*imit)->getDeltaY(),uli,ulj);
 //         (*imit)->geo2image(croplrx+(magicX-2.0)*(*imit)->getDeltaX(),croplry-(magicY-2.0)*(*imit)->getDeltaY(),lri,lrj);

 //         // ncropcol=abs(static_cast<unsigned int>(ceil((croplrx-cropulx)/dx)));
 //         // ncroprow=abs(static_cast<unsigned int>(ceil((cropuly-croplry)/dy)));
 //         ncropcol=static_cast<unsigned int>(ceil((croplrx-cropulx)/dx));
 //         ncroprow=static_cast<unsigned int>(ceil((cropuly-croplry)/dy));
 //         uli=floor(uli);
 //         ulj=floor(ulj);
 //         lri=floor(lri);
 //         lrj=floor(lrj);
 //       }

 //       // double deltaX=(*imit)->getDeltaX();
 //       // double deltaY=(*imit)->getDeltaY();
 //       if(!imgWriter.nrOfBand()){//not opened yet
 //         if(verbose_opt[0]){
 //           cout << "cropulx: " << cropulx << endl;
 //           cout << "cropuly: " << cropuly << endl;
 //           cout << "croplrx: " << croplrx << endl;
 //           cout << "croplry: " << croplry << endl;
 //           cout << "ncropcol: " << ncropcol << endl;
 //           cout << "ncroprow: " << ncroprow << endl;
 //           cout << "cropulx+ncropcol*dx: " << cropulx+ncropcol*dx << endl;
 //           cout << "cropuly-ncroprow*dy: " << cropuly-ncroprow*dy << endl;
 //           cout << "upper left column of input image: " << uli << endl;
 //           cout << "upper left row of input image: " << ulj << endl;
 //           cout << "lower right column of input image: " << lri << endl;
 //           cout << "lower right row of input image: " << lrj << endl;
 //           cout << "new number of cols: " << ncropcol << endl;
 //           cout << "new number of rows: " << ncroprow << endl;
 //           cout << "new number of bands: " << ncropband << endl;
 //         }
 //         // string imageType;//=(*imit)->getImageType();
 //         // if(oformat_opt.size())//default
 //         //   imageType=oformat_opt[0];
 //         try{
 //           imgWriter.open(ncropcol,ncroprow,ncropband,theType);
 //           imgWriter.setNoData(nodata_opt);
 //           // if(nodata_opt.size()){
 //           //   imgWriter.setNoData(nodata_opt);
 //           // }
 //         }
 //         catch(string errorstring){
 //           throw;
 //           // cout << errorstring << endl;
 //           // return(CE_Failure);
 //         }
 //         if(description_opt.size())
 //           imgWriter.setImageDescription(description_opt[0]);
 //         double gt[6];
 //         gt[0]=cropulx;
 //         gt[1]=dx;
 //         gt[2]=0;
 //         gt[3]=cropuly;
 //         gt[4]=0;
 //         gt[5]=((*imit)->isGeoRef())? -dy : dy;
 //         imgWriter.setGeoTransform(gt);
 //         if(projection_opt.size()){
 //           if(verbose_opt[0])
 //             cout << "projection: " << projection_opt[0] << endl;
 //           imgWriter.setProjectionProj4(projection_opt[0]);
 //         }
 //         else
 //           imgWriter.setProjection((*imit)->getProjection());
 //         if(imgWriter.getDataType()==GDT_Byte){
 //           if(colorTable_opt.size()){
 //             if(colorTable_opt[0]!="none")
 //               imgWriter.setColorTable(colorTable_opt[0]);
 //           }
 //           else if ((*imit)->getColorTable()!=NULL)//copy colorTable from input image
 //             imgWriter.setColorTable((*imit)->getColorTable());
 //         }
 //       }

 //       double startCol=uli;
 //       double endCol=lri;
 //       if(uli<0)
 //         startCol=0;
 //       else if(uli>=(*imit)->nrOfCol())
 //         startCol=(*imit)->nrOfCol()-1;
 //       if(lri<0)
 //         endCol=0;
 //       else if(lri>=(*imit)->nrOfCol())
 //         endCol=(*imit)->nrOfCol()-1;
 //       double startRow=ulj;
 //       double endRow=lrj;
 //       if(ulj<0)
 //         startRow=0;
 //       else if(ulj>=(*imit)->nrOfRow())
 //         startRow=(*imit)->nrOfRow()-1;
 //       if(lrj<0)
 //         endRow=0;
 //       else if(lrj>=(*imit)->nrOfRow())
 //         endRow=(*imit)->nrOfRow()-1;

 //       vector<double> readBuffer;
 //       unsigned int nband=(band_opt.size())?band_opt.size() : (*imit)->nrOfBand();
 //       for(unsigned int iband=0;iband<nband;++iband){
 //         unsigned int readBand=(band_opt.size()>iband)?band_opt[iband]:iband;
 //         if(verbose_opt[0]){
 //           cout << "extracting band " << readBand << endl;
 //           MyProgressFunc(progress,pszMessage,pProgressArg);
 //         }
 //         double theMin=0;
 //         double theMax=0;
 //         if(autoscale_opt.size()){
 //           try{
 //             (*imit)->getMinMax(static_cast<unsigned int>(startCol),static_cast<unsigned int>(endCol),static_cast<unsigned int>(startRow),static_cast<unsigned int>(endRow),readBand,theMin,theMax);
 //           }
 //           catch(string errorString){
 //             cout << errorString << endl;
 //           }
 //           if(verbose_opt[0])
 //             cout << "minmax: " << theMin << ", " << theMax << endl;
 //           double theScale=(autoscale_opt[1]-autoscale_opt[0])/(theMax-theMin);
 //           double theOffset=autoscale_opt[0]-theScale*theMin;
 //           (*imit)->setScale(theScale,readBand);
 //           (*imit)->setOffset(theOffset,readBand);
 //         }
 //         else{
 //           if(scale_opt.size()){
 //             if(scale_opt.size()>iband)
 //               (*imit)->setScale(scale_opt[iband],readBand);
 //             else
 //               (*imit)->setScale(scale_opt[0],readBand);
 //           }
 //           if(offset_opt.size()){
 //             if(offset_opt.size()>iband)
 //               (*imit)->setOffset(offset_opt[iband],readBand);
 //             else
 //               (*imit)->setOffset(offset_opt[0],readBand);
 //           }
 //         }

 //         double readRow=0;
 //         double readCol=0;
 //         double lowerCol=0;
 //         double upperCol=0;
 //         for(int irow=0;irow<imgWriter.nrOfRow();++irow){
 //           vector<double> lineMask;
 //           double x=0;
 //           double y=0;
 //           //convert irow to geo
 //           imgWriter.image2geo(0,irow,x,y);
 //           //lookup corresponding row for irow in this file
 //           (*imit)->geo2image(x,y,readCol,readRow);
 //           vector<double> writeBuffer;
 //           if(readRow<0||readRow>=(*imit)->nrOfRow()){
 //             for(int icol=0;icol<imgWriter.nrOfCol();++icol)
 //               writeBuffer.push_back(nodataValue);
 //           }
 //           else{
 //             try{
 //               if(endCol<(*imit)->nrOfCol()-1){
 //                 (*imit)->readData(readBuffer,startCol,endCol+1,readRow,readBand,theResample);
 //               }
 //               else{
 //                 (*imit)->readData(readBuffer,startCol,endCol,readRow,readBand,theResample);
 //               }
 //               double oldRowMask=-1;//keep track of row mask to optimize number of line readings
 //               for(int icol=0;icol<imgWriter.nrOfCol();++icol){
 //                 imgWriter.image2geo(icol,irow,x,y);
 //                 //lookup corresponding row for irow in this file
 //                 (*imit)->geo2image(x,y,readCol,readRow);
 //                 if(readCol<0||readCol>=(*imit)->nrOfCol()){
 //                   writeBuffer.push_back(nodataValue);
 //                 }
 //                 else{
 //                   bool valid=true;
 //                   double geox=0;
 //                   double geoy=0;
 //                   if(maskReader.isInit()){
 //                     //read mask
 //                     double colMask=0;
 //                     double rowMask=0;

 //                     imgWriter.image2geo(icol,irow,geox,geoy);
 //                     maskReader.geo2image(geox,geoy,colMask,rowMask);
 //                     colMask=static_cast<unsigned int>(colMask);
 //                     rowMask=static_cast<unsigned int>(rowMask);
 //                     if(rowMask>=0&&rowMask<maskReader.nrOfRow()&&colMask>=0&&colMask<maskReader.nrOfCol()){
 //                       if(static_cast<unsigned int>(rowMask)!=static_cast<unsigned int>(oldRowMask)){

 //                         try{
 //                           maskReader.readData(lineMask,static_cast<unsigned int>(rowMask),mskband_opt[0]);
 //                         }
 //                         catch(string errorstring){
 //                           throw;
 //                           // cerr << errorstring << endl;
 //                           // return(CE_Failure);
 //                         }
 //                         catch(...){
 //                           std::string errorString="error caught";
 //                           throw;
 //                           // cerr << "error caught" << std::endl;
 //                           // return(CE_Failure);
 //                         }
 //                         oldRowMask=rowMask;
 //                       }
 //                       if(cut_to_cutline_opt[0]){
 //                         if(lineMask[colMask]!=1){
 //                           nodataValue=nodata_opt[0];
 //                           valid=false;
 //                         }
 //                       }
 //                       else if(cut_in_cutline_opt[0]){
 //                         if(lineMask[colMask]==1){
 //                           nodataValue=nodata_opt[0];
 //                           valid=false;
 //                         }
 //                       }
 //                       else{
 //                         for(int ivalue=0;ivalue<msknodata_opt.size();++ivalue){
 //                           if(lineMask[colMask]==msknodata_opt[ivalue]){
 //                             if(nodata_opt.size()>ivalue)
 //                               nodataValue=nodata_opt[ivalue];
 //                             valid=false;
 //                             break;
 //                           }
 //                         }
 //                       }
 //                     }
 //                   }
 //                   if(!valid)
 //                     writeBuffer.push_back(nodataValue);
 //                   else{
 //                     switch(theResample){
 //                     case(BILINEAR):
 //                       lowerCol=readCol-0.5;
 //                       lowerCol=static_cast<unsigned int>(lowerCol);
 //                       upperCol=readCol+0.5;
 //                       upperCol=static_cast<unsigned int>(upperCol);
 //                       if(lowerCol<0)
 //                         lowerCol=0;
 //                       if(upperCol>=(*imit)->nrOfCol())
 //                         upperCol=(*imit)->nrOfCol()-1;
 //                       writeBuffer.push_back((readCol-0.5-lowerCol)*readBuffer[upperCol-startCol]+(1-readCol+0.5+lowerCol)*readBuffer[lowerCol-startCol]);
 //                       break;
 //                     default:
 //                       readCol=static_cast<unsigned int>(readCol);
 //                       readCol-=startCol;//we only start reading from startCol
 //                       writeBuffer.push_back(readBuffer[readCol]);
 //                       break;
 //                     }
 //                   }
 //                 }
 //               }
 //             }
 //             catch(string errorstring){
 //               throw;
 //               // cout << errorstring << endl;
 //               // return(CE_Failure);
 //             }
 //           }
 //           if(writeBuffer.size()!=imgWriter.nrOfCol())
 //             cout << "writeBuffer.size()=" << writeBuffer.size() << ", imgWriter.nrOfCol()=" << imgWriter.nrOfCol() << endl;

 //           assert(writeBuffer.size()==imgWriter.nrOfCol());
 //           try{
 //             imgWriter.writeData(writeBuffer,irow,writeBand);
 //           }
 //           catch(string errorstring){
 //             throw;
 //             // cout << errorstring << endl;
 //             // return(CE_Failure);
 //           }
 //           if(verbose_opt[0]){
 //             progress=(1.0+irow);
 //             progress/=imgWriter.nrOfRow();
 //             MyProgressFunc(progress,pszMessage,pProgressArg);
 //           }
 //           else{
 //             progress=(1.0+irow);
 //             progress+=(imgWriter.nrOfRow()*writeBand);
 //             progress/=imgWriter.nrOfBand()*imgWriter.nrOfRow();
 //             assert(progress>=0);
 //             assert(progress<=1);
 //             MyProgressFunc(progress,pszMessage,pProgressArg);
 //           }
 //         }
 //         ++writeBand;
 //       }
 //       // (*imit)->close();
 //     }
 //     if(extent_opt.size()&&(cut_to_cutline_opt[0]||cut_in_cutline_opt[0]||eoption_opt.size())){
 //       extentReader.close();
 //     }
 //     if(maskReader.isInit())
 //       maskReader.close();
 //     // return(CE_None);
 //   }
 //   catch(string predefinedString){
 //     std::cout << predefinedString << std::endl;
 //     throw;
 //   }
 //   return(*this);
 // }
