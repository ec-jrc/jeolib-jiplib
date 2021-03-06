/**********************************************************************
jlsml_lib.h: classify raster image using SML
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
#pragma once
//#include <sstream>
//#includ  <iostream>
#include <boost/serialization/map.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>

#include <vector>
#include <map>
#include <memory>
#include "Jim.h"
#include "VectorOgr.h"
#include "base/Optionjl.h"
#include "apps/AppFactory.h"

///train and classify SML
template<typename T> void Jim::classifySML_t(Jim& imgWriter, JimList& referenceReader, app::AppFactory& app){
  Optionjl<unsigned short> class_opt("c", "class", "Class(es) to extract from reference");
  Optionjl<double> srcnodata_opt("srcnodata", "srcnodata", "Nodata value in source");
  Optionjl<double> dstnodata_opt("dstnodata", "dstnodata", "Nodata value to put where image is masked as nodata",0);
  Optionjl<short> verbose_opt("v", "verbose", "Verbose level",0,2);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=class_opt.retrieveOption(app);
    srcnodata_opt.retrieveOption(app);
    dstnodata_opt.retrieveOption(app);
    verbose_opt.retrieveOption(app);

    if(!doProcess){
      std::cout << std::endl;
      std::ostringstream helpStream;
      helpStream << "short option -h shows basic options only, use long option --help to show all options" << std::endl;
      throw(helpStream.str());//help was invoked, stop processing
    }

    if(class_opt.empty()){
      std::string errorString="Error: no classes provided, use option class to provide a list of classes in reference to train";
      throw(errorString);
    }

    if(nrOfBand()>1){
      std::string errorString="Error: only single band (multi-plane) datasets are supported, consider band2plane";
      throw(errorString);
    }

    for(size_t iref=0;iref<referenceReader.size();++iref){
      if(referenceReader.getImage(iref)->getDataType()!=GDT_Byte){
        std::string errorstring="Error: data type of reference must be GDT_Byte";
        throw(errorstring);
      }
    }

    T* pin=static_cast<T*>(getDataPointer(0));

    if(verbose_opt[0]>=1)
      std::cout << "start SML training" << std::endl;

    int nrow=nrOfRow();
    int ncol=nrOfCol();

    //umap: [unique band information]->[index,class1,class2,...] (occurrence is always updated in last node!!!)
    std::map<std::vector<T>,std::vector<std::vector<size_t> > > umap;
    std::vector<T> pixel(nrOfPlane());//pixel with plane information
    std::vector<unsigned char> refpixel(referenceReader.size());//pixel with reference information

    OGRSpatialReference thisSRS(getProjectionRef().c_str());
#if GDAL_VERSION_MAJOR > 2
    thisSRS.SetAxisMappingStrategy(OSRAxisMappingStrategy::OAMS_TRADITIONAL_GIS_ORDER);
#endif
    OGRSpatialReference *thisSpatialRef=&thisSRS;
#if GDAL_VERSION_MAJOR > 2
    thisSpatialRef->SetAxisMappingStrategy(OSRAxisMappingStrategy::OAMS_TRADITIONAL_GIS_ORDER);
#endif
    //currently only a single SRS for all images in reference collection supported
    OGRSpatialReference referenceSRS(referenceReader.getImage(0)->getProjectionRef().c_str());
#if GDAL_VERSION_MAJOR > 2
    referenceSRS.SetAxisMappingStrategy(OSRAxisMappingStrategy::OAMS_TRADITIONAL_GIS_ORDER);
#endif

    OGRSpatialReference *referenceSpatialRef=&referenceSRS;
#if GDAL_VERSION_MAJOR > 2
    referenceSpatialRef->SetAxisMappingStrategy(OSRAxisMappingStrategy::OAMS_TRADITIONAL_GIS_ORDER);
#endif
    OGRCoordinateTransformation *img2ref=OGRCreateCoordinateTransformation(thisSpatialRef, referenceSpatialRef);

    std::vector<unsigned char*> pref(referenceReader.size());
    unsigned short theMax=0;
    for(size_t iref=0;iref<referenceReader.size();++iref)
      pref[iref]=static_cast<unsigned char*>(referenceReader.getImage(iref)->getDataPointer(0));
    unsigned short nclass=class_opt.size();
    if(nclass<2){
      std::ostringstream errorStream;
      errorStream << "Error: number of classes should be at least 2" << std::endl;
      throw(errorStream.str());
    }

    if(verbose_opt[0]>=1){
      std::cout << "number of classes: " << nclass << std::endl;
    }

    for(size_t y=0;y<nrow;++y){
      for(size_t x=0;x<ncol;++x){
        std::vector<size_t> posclass(1+nclass);
        size_t index=y*ncol+x;
        posclass[0]=static_cast<size_t>(index);
        //read reference
        bool validref=false;
        for(size_t iref=0;iref<referenceReader.size();++iref){
          size_t ncolref=referenceReader.getImage(iref)->nrOfCol();
          size_t nrowref=referenceReader.getImage(iref)->nrOfRow();
          double colReference=0;
          double rowReference=0;
          double geox=0;
          double geoy=0;
          image2geo(x,y,geox,geoy);
          referenceReader.getImage(iref)->geo2image(geox,geoy,colReference,rowReference,img2ref);
          if(rowReference>=0&&rowReference<referenceReader.getImage(iref)->nrOfRow()&&colReference>=0&&colReference<referenceReader.getImage(iref)->nrOfCol()){
            size_t indexref=static_cast<size_t>(rowReference)*ncolref+static_cast<size_t>(colReference);
            refpixel[iref]=static_cast<unsigned char>((pref[iref])[indexref]);
            validref=true;
          }
          else{
            std::cout  << "Warning: could not find valid reference pixel for col, row " << x << ", " << y << std::endl;
            std::cout  << "colReference, rowReference: " << colReference << ", " << rowReference << std::endl;
            std::cout  << "geox, geoy: " << geox << ", " << geoy << std::endl;
            continue;
          }
        }

        if(!validref){
          std::ostringstream errorStream;
          errorStream << "Error: could not find valid reference";
          throw(errorStream.str());
        }

        bool valid=true;
        if(srcnodata_opt.size())
          valid=false;
        for(size_t z=0;z<nrOfPlane();++z){
          pixel[z]=pin[index+z*ncol*nrow];
          if(srcnodata_opt.size()){
            //invalid iff pixel has no data in all planes
            if(pixel[z]!=srcnodata_opt[0])
              valid=true;
          }
        }
        auto pit = umap.find(pixel);
        for(size_t iclass=0;iclass<class_opt.size();++iclass){
          size_t increment=0;
          for(size_t iref=0;iref<referenceReader.size();++iref){
            if(refpixel[iref]==class_opt[iclass])
              ++increment;
          }
          if(pit!=umap.end()){
            posclass[1+iclass]=((pit->second).back())[1+iclass]+increment;
          }
          else
            posclass[1+iclass]=increment;
        }
        umap[pixel].push_back(posclass);
      }
    }

    if(verbose_opt[0]>=1)
      std::cout << "end of training" << std::endl;

    if(verbose_opt[0]){
      std::cout << "umap.size(): " << umap.size() << std::endl;
      if(verbose_opt[0]>1){
        for(auto mapit = umap.begin(); mapit != umap.end(); ++mapit){
          for(size_t iclass=0;iclass<nclass;++iclass){
            std::cout << static_cast<size_t>(((mapit->second.back())[0]))/ncol << " " << static_cast<size_t>(((mapit->second.back())[0]))%ncol << ": ";
            double value=(mapit->second.back())[1+iclass];
            std::cout << value << " " << "( " << class_opt[iclass] << ") ";
          }
          std::cout << std::endl;
        }
      }
    }

    imgWriter.open(ncol,nrow,nclass,GDT_Byte);
    imgWriter.GDALSetNoDataValue(dstnodata_opt[0]);
    imgWriter.setNoData(dstnodata_opt);
    imgWriter.copyGeoTransform(*this);
    imgWriter.setProjection(this->getProjection());
    //initialize imgWriter with dstnodata_opt[0]
    imgWriter.setData(dstnodata_opt[0]);

    std::vector<unsigned char*> pout(nclass);
    for(size_t iclass=0;iclass<nclass;++iclass){
      pout[iclass]=static_cast<unsigned char*>(imgWriter.getDataPointer(iclass));
    }

    //loop through umap and assign pixel values based on occurrence
    for(auto mapit = umap.begin(); mapit != umap.end(); ++mapit){
      std::vector<double> fclass(nclass);
      double maxValue=0;
      for(size_t iclass=0;iclass<nclass;++iclass){
        double value=((mapit->second).back())[1+iclass];
        fclass[iclass]=value;
        if(fclass[iclass]>maxValue)
          maxValue=fclass[iclass];
      }
      double scale=(maxValue>0) ? 100.0/maxValue : 0;
      for(size_t iclass=0;iclass<nclass;++iclass)
        fclass[iclass]*=scale;
      for(auto tupit = mapit->second.begin(); tupit != mapit->second.end(); ++tupit){
        if(verbose_opt[0]>1)
          std::cout << (*tupit)[0] << " = " << (*tupit)[0]/ncol << " " << (*tupit)[0]%ncol << ": ";
        for(unsigned int iclass=0;iclass<nclass;++iclass){
          if(static_cast<size_t>(((*tupit)[0]))/ncol<0){
            std::ostringstream errorStream;
            errorStream << "Error: not within rows: " << static_cast<size_t>(((*tupit)[0]))/ncol << std::endl;
            throw(errorStream.str());
          }
          if(static_cast<size_t>(((*tupit)[0]))/ncol>nrow){
            std::ostringstream errorStream;
            errorStream << "Error: not within rows: " << static_cast<size_t>(((*tupit)[0]))/ncol << std::endl;
            throw(errorStream.str());
          }
          if(static_cast<size_t>(((*tupit)[0]))%ncol<0){
            std::ostringstream errorStream;
            errorStream << "Error: not within cols: " << static_cast<size_t>(((*tupit)[0]))%ncol << std::endl;
            throw(errorStream.str());
          }
          if(verbose_opt[0]>1)
            std::cout << static_cast<unsigned short>(fclass[iclass]) << " ";
          pout[iclass][(*tupit)[0]]=static_cast<unsigned char>(fclass[iclass]);
        }
        if(verbose_opt[0]>1)
          std::cout << std::endl;
      }
    }
  }
  catch(std::string errorString){
    std::cerr << errorString << std::endl;
    throw;
  }
}

///train SML with information in 3D
//template<typename T> std::string Jim::trainSML_t(JimList& referenceReader, app::AppFactory& app){
template<typename T> void Jim::trainSML_t(JimList& referenceReader, app::AppFactory& app){
  Optionjl<std::string> model_opt("model", "model", "Model filename to save trained classifier.");
  Optionjl<unsigned short> class_opt("c", "class", "Class(es) to extract from reference. Leave empty to extract two classes only: 1 against rest",1);
  Optionjl<double> srcnodata_opt("srcnodata", "srcnodata", "Nodata value in source",0);
  Optionjl<short> verbose_opt("v", "verbose", "Verbose level",0,2);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=class_opt.retrieveOption(app);
    model_opt.retrieveOption(app);
    srcnodata_opt.retrieveOption(app);
    verbose_opt.retrieveOption(app);

    if(!doProcess){
      std::cout << std::endl;
      std::ostringstream helpStream;
      helpStream << "short option -h shows basic options only, use long option --help to show all options" << std::endl;
      throw(helpStream.str());//help was invoked, stop processing
    }

    if(nrOfBand()>1){
      std::string errorString="Error: only single band (multi-plane) datasets are supported, consider band2plane";
      throw(errorString);
    }

    for(size_t iref=0;iref<referenceReader.size();++iref){
      if(referenceReader.getImage(iref)->getDataType()!=GDT_Byte){
        std::string errorstring="Error: data type of reference must be GDT_Byte";
        throw(errorstring);
      }
    }

    T* pin=static_cast<T*>(getDataPointer(0));

    if(verbose_opt[0]>=1)
      std::cout << "start SML training" << std::endl;

    int nrow=nrOfRow();
    int ncol=nrOfCol();

    //umap: [unique band information]->[index,class1,class2,...] (occurrence is always updated in last node!!!)
    std::map<std::vector<T>,std::vector<std::vector<size_t> > > umap;
    std::vector<T> pixel(nrOfPlane());//pixel with plane information
    std::vector<unsigned char> refpixel(referenceReader.size());//pixel with reference information

    OGRSpatialReference thisSRS(getProjectionRef().c_str());
#if GDAL_VERSION_MAJOR > 2
    thisSRS.SetAxisMappingStrategy(OSRAxisMappingStrategy::OAMS_TRADITIONAL_GIS_ORDER);
#endif
    OGRSpatialReference *thisSpatialRef=&thisSRS;
#if GDAL_VERSION_MAJOR > 2
    thisSpatialRef->SetAxisMappingStrategy(OSRAxisMappingStrategy::OAMS_TRADITIONAL_GIS_ORDER);
#endif
    //currently only a single SRS for all images in reference collection supported
    OGRSpatialReference referenceSRS(referenceReader.getImage(0)->getProjectionRef().c_str());
#if GDAL_VERSION_MAJOR > 2
    referenceSRS.SetAxisMappingStrategy(OSRAxisMappingStrategy::OAMS_TRADITIONAL_GIS_ORDER);
#endif

    OGRSpatialReference *referenceSpatialRef=&referenceSRS;
#if GDAL_VERSION_MAJOR > 2
    referenceSpatialRef->SetAxisMappingStrategy(OSRAxisMappingStrategy::OAMS_TRADITIONAL_GIS_ORDER);
#endif
    OGRCoordinateTransformation *img2ref=OGRCreateCoordinateTransformation(thisSpatialRef, referenceSpatialRef);

    std::vector<unsigned char*> pref(referenceReader.size());
    unsigned short theMax=0;
    for(size_t iref=0;iref<referenceReader.size();++iref)
      pref[iref]=static_cast<unsigned char*>(referenceReader.getImage(iref)->getDataPointer(0));
    unsigned short nclass=class_opt.size()>1?class_opt.size():2;
    if(nclass<2){
      std::ostringstream errorStream;
      errorStream << "Error: number of classes should be at least 2" << std::endl;
      throw(errorStream.str());
    }

    if(verbose_opt[0]>=1){
      std::cout << "number of classes: " << nclass << std::endl;
      std::cout << "create vectors for training" << std::endl;
    }

    for(size_t y=0;y<nrow;++y){
      for(size_t x=0;x<ncol;++x){
        std::vector<size_t> posclass(1+nclass);
        size_t index=y*ncol+x;
        posclass[0]=static_cast<size_t>(index);
        //read reference
        bool validref=false;
        for(size_t iref=0;iref<referenceReader.size();++iref){
          size_t ncolref=referenceReader.getImage(iref)->nrOfCol();
          size_t nrowref=referenceReader.getImage(iref)->nrOfRow();
          double colReference=0;
          double rowReference=0;
          double geox=0;
          double geoy=0;
          image2geo(x,y,geox,geoy);
          referenceReader.getImage(iref)->geo2image(geox,geoy,colReference,rowReference,img2ref);
          if(rowReference>=0&&rowReference<referenceReader.getImage(iref)->nrOfRow()&&colReference>=0&&colReference<referenceReader.getImage(iref)->nrOfCol()){
            size_t indexref=static_cast<size_t>(rowReference)*ncolref+static_cast<size_t>(colReference);
            refpixel[iref]=static_cast<unsigned char>((pref[iref])[indexref]);
            validref=true;
          }
          else{
            std::cout  << "Warning: could not find valid reference pixel for col, row " << x << ", " << y << std::endl;
            std::cout  << "colReference, rowReference: " << colReference << ", " << rowReference << std::endl;
            std::cout  << "geox, geoy: " << geox << ", " << geoy << std::endl;
            continue;
          }
        }
        if(!validref){
          std::ostringstream errorStream;
          errorStream << "Error: could not find valid reference";
          throw(errorStream.str());
        }

        bool valid=true;
        if(srcnodata_opt.size())
          valid=false;
        for(size_t z=0;z<nrOfPlane();++z){
          pixel[z]=pin[index+z*ncol*nrow];
          if(srcnodata_opt.size()){
            //invalid iff pixel has no data in all planes
            if(pixel[z]!=srcnodata_opt[0])
              valid=true;
          }
        }
        auto pit = umap.find(pixel);
        for(size_t iclass=0;iclass<class_opt.size();++iclass){
          size_t increment=0;
          for(size_t iref=0;iref<referenceReader.size();++iref){
            if(refpixel[iref]==class_opt[iclass])
              ++increment;
          }
          if(pit!=umap.end()){
            posclass[1+iclass]=((pit->second).back())[1+iclass]+increment;
          }
          else
            posclass[1+iclass]=increment;
        }
        umap[pixel].push_back(posclass);
      }
    }
    //todo: scale from 0 to 100
    for(auto mapit = umap.begin(); mapit != umap.end(); ++mapit){
      std::vector<double> fclass(nclass);
      double maxValue=0;
      for(size_t iclass=0;iclass<nclass;++iclass){
        double value=((mapit->second).back())[1+iclass];
        fclass[iclass]=value;
        if(fclass[iclass]>maxValue)
          maxValue=fclass[iclass];
      }
      double scale=(maxValue>0) ? 100.0/maxValue : 0;
      for(size_t iclass=0;iclass<nclass;++iclass)
        ((mapit->second).back())[1+iclass]*=scale;
    }
    if(verbose_opt[0]>=1)
      std::cout << "end of training, write to output" << std::endl;
    /* std::ostringstream outputStream; */
    /* boost::archive::text_oarchive oarch(outputStream); */
    std::ofstream ofs(model_opt[0], std::ios::binary);
    boost::archive::binary_oarchive oarch(ofs);
    try{
      oarch & umap;
    }
    catch (const boost::archive::archive_exception &e) {
      if (e.code != boost::archive::archive_exception::output_stream_error) {
        std::cerr << "Error: could not write model 0" << std::endl;
        throw;
      }
      else{
        std::cerr << "Error: could not write model 1" << std::endl;
        throw;
      }
    }
    /* oarch << umap; */

    if(verbose_opt[0]){
      std::cout << "umap.size(): " << umap.size() << std::endl;
      if(verbose_opt[0]>1){
        for(auto mapit = umap.begin(); mapit != umap.end(); ++mapit){
          for(size_t iclass=0;iclass<nclass;++iclass){
            std::cout << static_cast<size_t>(((mapit->second.back())[0]))/ncol << " " << static_cast<size_t>(((mapit->second.back())[0]))%ncol << ": ";
            double value=(mapit->second.back())[1+iclass];
            std::cout << value << " ";
          }
          std::cout << std::endl;
        }
      }
    }
    //return(outputStream.str());
  }
  catch(std::string errorString){
    std::cerr << errorString << std::endl;
    throw;
  }
}

///classify 3D raster dataset with SML
template<typename T> void Jim::classifySML_t(Jim& imgWriter, app::AppFactory& app){
  Optionjl<std::string> model_opt("model", "model", "Model filename to save trained classifier.");
  Optionjl<double> srcnodata_opt("srcnodata", "srcnodata", "Nodata value in source",0);
  Optionjl<double> dstnodata_opt("dstnodata", "dstnodata", "Nodata value to put where image is masked as nodata", 0);
  Optionjl<short> verbose_opt("v", "verbose", "Verbose level",0,2);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=model_opt.retrieveOption(app);
    srcnodata_opt.retrieveOption(app);
    dstnodata_opt.retrieveOption(app);
    verbose_opt.retrieveOption(app);

    if(!doProcess){
      std::cout << std::endl;
      std::ostringstream helpStream;
      helpStream << "short option -h shows basic options only, use long option --help to show all options" << std::endl;
      throw(helpStream.str());//help was invoked, stop processing
    }

    if(model_opt.empty()){
      std::string errorString="Error: model option to write model not set";
      throw(errorString);
    }

    if(nrOfBand()>1){
      std::string errorString="Error: only single band (multi-plane) datasets are supported, consider band2plane";
      throw(errorString);
    }

    T* pin=static_cast<T*>(getDataPointer(0));

    if(verbose_opt[0]>=1)
      std::cout << "start SML classification" << std::endl;

    //umap: [unique band information]->[index,class1,class2,...] (occurrence is always updated in last node!!!)
    std::map<std::vector<T>,std::vector<std::vector<size_t> > > umap;
    std::vector<T> pixel(nrOfPlane());//pixel with plane information

    std::ifstream ifs(model_opt[0],std::ios::binary);
    umap.clear();
    boost::archive::binary_iarchive iarch(ifs);
    /* boost::archive::text_iarchive iarch(ifs); */
    /* iarch >> umap; */
    try{
      iarch & umap;
    }
    catch (const boost::archive::archive_exception &e) {
      if (e.code != boost::archive::archive_exception::input_stream_error) {
        std::cerr << "Error: could not read model 0" << std::endl;
        throw;
      }
      else{
        std::cerr << "Error: could not read model 1" << std::endl;
        throw;
      }
    }

    int nclass=(umap.begin()->second).back().size()-1;

    int nrow=nrOfRow();
    int ncol=nrOfCol();
    if(this->isInit()){
      if(verbose_opt[0]>=1)
        std::cout << "We are in initialize" << std::endl;
      imgWriter.open(ncol,nrow,nclass,GDT_Byte);
      imgWriter.GDALSetNoDataValue(dstnodata_opt[0]);
      imgWriter.setNoData(dstnodata_opt);
      imgWriter.copyGeoTransform(*this);
      imgWriter.setProjection(this->getProjection());
      //initialize imgWriter with dstnodata_opt[0]
      imgWriter.setData(dstnodata_opt[0]);
    }
    std::vector<unsigned char*> pout(nclass);
    for(size_t iclass=0;iclass<nclass;++iclass)
      pout[iclass]=static_cast<unsigned char*>(imgWriter.getDataPointer(iclass));

    bool valid=true;
    if(srcnodata_opt.size())
      valid=false;
    for(size_t index=0;index<nrow*ncol;++index){
      for(size_t z=0;z<nrOfPlane();++z){
        pixel[z]=pin[index+z*ncol*nrow];
        if(srcnodata_opt.size()){
          //invalid iff pixel has no data in all planes
          if(pixel[z]!=srcnodata_opt[0])
            valid=true;
        }
      }
      if(!valid){
        for(size_t iclass=0;iclass<nclass;++iclass)
          pout[iclass][index]=static_cast<unsigned char>(dstnodata_opt[0]);
        continue;
      }
      auto pit = umap.find(pixel);
      if(pit==umap.end()){
        for(size_t iclass=0;iclass<nclass;++iclass)
          pout[iclass][index]=static_cast<unsigned char>(dstnodata_opt[0]);
        continue;
      }
      for(size_t iclass=0;iclass<nclass;++iclass)
        pout[iclass][index]=(pit->second.back())[1+iclass];
    }
  }
  catch(std::string errorString){
    std::cerr << errorString << std::endl;
    throw;
  }
}

///classify 3D raster dataset with SML
/* template<typename T> void Jim::classifySML_t(Jim& imgWriter, app::AppFactory& app){ */
/*   Optionjl<std::string> model_opt("model", "model", "Model filename to save trained classifier."); */
/*   Optionjl<unsigned short> class_opt("c", "class", "Class(es) to extract from reference. Leave empty to extract two classes only: 1 against rest",1); */
/*   Optionjl<double> srcnodata_opt("srcnodata", "srcnodata", "Nodata value in source",0); */
/*   Optionjl<double> dstnodata_opt("dstnodata", "dstnodata", "Nodata value to put where image is masked as nodata", 0); */
/*   Optionjl<short> verbose_opt("v", "verbose", "Verbose level",0,2); */

/*   bool doProcess;//stop process when program was invoked with help option (-h --help) */
/*   try{ */
/*     doProcess=model_opt.retrieveOption(app); */
/*     class_opt.retrieveOption(app); */
/*     srcnodata_opt.retrieveOption(app); */
/*     dstnodata_opt.retrieveOption(app); */
/*     verbose_opt.retrieveOption(app); */

/*     if(!doProcess){ */
/*       std::cout << std::endl; */
/*       std::ostringstream helpStream; */
/*       helpStream << "short option -h shows basic options only, use long option --help to show all options" << std::endl; */
/*       throw(helpStream.str());//help was invoked, stop processing */
/*     } */

/*     if(model_opt.empty()){ */
/*       std::string errorString="Error: model option to write model not set"; */
/*       throw(errorString); */
/*     } */

/*     if(nrOfBand()>1){ */
/*       std::string errorString="Error: only single band (multi-plane) datasets are supported, consider band2plane"; */
/*       throw(errorString); */
/*     } */

/*     if(verbose_opt[0]>=1) */
/*       std::cout << "start SML classification" << std::endl; */

/*     //umap: [unique band information]->[index,class1,class2,...] (occurrence is always updated in last node!!!) */
/*     std::map<std::vector<T>,std::vector<std::vector<size_t> > > umap; */

/*     std::ifstream ifs(model_opt[0]); */
/*     umap.clear(); */
/*     boost::archive::text_iarchive iarch(ifs); */
/*     iarch >> umap; */

/*     int nclass=(umap.begin()->second).back().size()-1; */

/*     int nrow=nrOfRow(); */
/*     int ncol=nrOfCol(); */
/*     if(this->isInit()){ */
/*       if(verbose_opt[0]>=1) */
/*         std::cout << "We are in initialize" << std::endl; */
/*       imgWriter.open(ncol,nrow,nclass,GDT_Byte); */
/*       imgWriter.GDALSetNoDataValue(dstnodata_opt[0]); */
/*       imgWriter.setNoData(dstnodata_opt); */
/*       imgWriter.copyGeoTransform(*this); */
/*       imgWriter.setProjection(this->getProjection()); */
/*       //initialize imgWriter with dstnodata_opt[0] */
/*       imgWriter.setData(dstnodata_opt[0]); */
/*     } */
/*     std::vector<unsigned char*> pout(nclass); */
/*     for(size_t iclass=0;iclass<nclass;++iclass){ */
/*       pout[iclass]=static_cast<unsigned char*>(imgWriter.getDataPointer(iclass)); */
/*     } */

/*     //loop through umap and assign pixel values based on occurrence */
/*     for(auto mapit = umap.begin(); mapit != umap.end(); ++mapit){ */
/*       std::vector<double> fclass(nclass); */
/*       double maxValue=0; */
/*       for(size_t iclass=0;iclass<nclass;++iclass){ */
/*         double value=((mapit->second).back())[1+iclass]; */
/*         fclass[iclass]=value; */
/*         if(fclass[iclass]>maxValue) */
/*           maxValue=fclass[iclass]; */
/*       } */
/*       double scale=(maxValue>0) ? 100.0/maxValue : 0; */
/*       for(size_t iclass=0;iclass<nclass;++iclass) */
/*         fclass[iclass]*=scale; */
/*       for(auto tupit = mapit->second.begin(); tupit != mapit->second.end(); ++tupit){ */
/*         if(verbose_opt[0]>1) */
/*           std::cout << (*tupit)[0] << " = " << (*tupit)[0]/ncol << " " << (*tupit)[0]%ncol << ": "; */
/*         for(unsigned int iclass=0;iclass<nclass;++iclass){ */
/*           if(static_cast<size_t>(((*tupit)[0]))/ncol<0){ */
/*             std::ostringstream errorStream; */
/*             errorStream << "Error: not within rows: " << static_cast<size_t>(((*tupit)[0]))/ncol << std::endl; */
/*             throw(errorStream.str()); */
/*           } */
/*           if(static_cast<size_t>(((*tupit)[0]))/ncol>nrow){ */
/*             std::ostringstream errorStream; */
/*             errorStream << "Error: not within rows: " << static_cast<size_t>(((*tupit)[0]))/ncol << std::endl; */
/*             throw(errorStream.str()); */
/*           } */
/*           if(static_cast<size_t>(((*tupit)[0]))%ncol<0){ */
/*             std::ostringstream errorStream; */
/*             errorStream << "Error: not within cols: " << static_cast<size_t>(((*tupit)[0]))%ncol << std::endl; */
/*             throw(errorStream.str()); */
/*           } */
/*           if(verbose_opt[0]>1) */
/*             std::cout << static_cast<unsigned short>(fclass[iclass]) << " "; */
/*           pout[iclass][(*tupit)[0]]=static_cast<unsigned char>(fclass[iclass]); */
/*         } */
/*         if(verbose_opt[0]>1) */
/*           std::cout << std::endl; */
/*       } */
/*     } */
/*   } */
/*   catch(std::string errorString){ */
/*     std::cerr << errorString << std::endl; */
/*     throw; */
/*   } */
/* } */

///train SML in 2D with information in bands
template<typename T> std::string Jim::trainSML2d_t(JimList& referenceReader, app::AppFactory& app){
  /* Optionjl<std::string> model_opt("model", "model", "Model filename to save trained classifier."); */
  Optionjl<unsigned int> band_opt("b", "band", "Band index (starting from 0, either use band option or use start to end)");
  Optionjl<unsigned int> bstart_opt("sband", "startband", "Start band sequence number");
  Optionjl<unsigned int> bend_opt("eband", "endband", "End band sequence number");
  Optionjl<unsigned short> class_opt("c", "class", "Class(es) to extract from reference. Leave empty to extract two classes only: 1 against rest",1);
  Optionjl<double> srcnodata_opt("srcnodata", "srcnodata", "Nodata value in source",0);
  Optionjl<short> verbose_opt("v", "verbose", "Verbose level",0,2);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    /* doProcess=model_opt.retrieveOption(app); */
    doProcess=band_opt.retrieveOption(app);
    bstart_opt.retrieveOption(app);
    bend_opt.retrieveOption(app);
    class_opt.retrieveOption(app);
    srcnodata_opt.retrieveOption(app);
    verbose_opt.retrieveOption(app);
    // memory_opt.retrieveOption(app);

    if(!doProcess){
      std::cout << std::endl;
      std::ostringstream helpStream;
      helpStream << "short option -h shows basic options only, use long option --help to show all options" << std::endl;
      throw(helpStream.str());//help was invoked, stop processing
    }

    /* if(model_opt.empty()){ */
    /*   std::string errorString="Error: model option to write model not set"; */
    /*   throw(errorString); */
    /* } */

    if(verbose_opt[0]>=1)
      std::cout << "start SML" << std::endl;

    if(verbose_opt[0]>=1)
      std::cout << "configure band options" << std::endl;

    //convert start and end band options to vector of band indexes
    if(bstart_opt.size()){
      if(bend_opt.size()!=bstart_opt.size()){
        std::string errorstring="Error: options for start and end band indexes must be provided as pairs, missing end band";
        throw(errorstring);
      }
      band_opt.clear();
      for(unsigned int ipair=0;ipair<bstart_opt.size();++ipair){
        if(bend_opt[ipair]<=bstart_opt[ipair]){
          std::string errorstring="Error: index for end band must be smaller then start band";
          throw(errorstring);
        }
        for(unsigned int iband=bstart_opt[ipair];iband<=bend_opt[ipair];++iband)
          band_opt.push_back(iband);
      }
    }

    if(verbose_opt[0]>=1)
      std::cout << "sort bands" << std::endl;

    //sort bands
    if(band_opt.size())
      std::sort(band_opt.begin(),band_opt.end());
    else{
      unsigned int iband=0;
      while(band_opt.size()<nrOfBand())
        band_opt.push_back(iband++);
    }

    int nrow=nrOfRow();
    int ncol=nrOfCol();
    int nclass=class_opt.size()>1?class_opt.size():2;

    if(verbose_opt[0]>=1)
      std::cout << "create vectors for training" << std::endl;

    //umap: [unique band information]->[x,y,class1,class2,...]
    std::map<std::vector<T>,std::vector<std::vector<unsigned int> > > umap;
    std::vector<T> pixel(band_opt.size());//pixel with band information
    std::vector<T> refpixel(referenceReader.size());//pixel with reference information

    OGRSpatialReference thisSRS(getProjectionRef().c_str());
#if GDAL_VERSION_MAJOR > 2
    thisSRS.SetAxisMappingStrategy(OSRAxisMappingStrategy::OAMS_TRADITIONAL_GIS_ORDER);
#endif
    OGRSpatialReference *thisSpatialRef=&thisSRS;
#if GDAL_VERSION_MAJOR > 2
    thisSpatialRef->SetAxisMappingStrategy(OSRAxisMappingStrategy::OAMS_TRADITIONAL_GIS_ORDER);
#endif
    //currently only a single SRS for all images in reference collection supported
    OGRSpatialReference referenceSRS(referenceReader.getImage(0)->getProjectionRef().c_str());
#if GDAL_VERSION_MAJOR > 2
    referenceSRS.SetAxisMappingStrategy(OSRAxisMappingStrategy::OAMS_TRADITIONAL_GIS_ORDER);
#endif

    OGRSpatialReference *referenceSpatialRef=&referenceSRS;
#if GDAL_VERSION_MAJOR > 2
    referenceSpatialRef->SetAxisMappingStrategy(OSRAxisMappingStrategy::OAMS_TRADITIONAL_GIS_ORDER);
#endif
    OGRCoordinateTransformation *img2ref=OGRCreateCoordinateTransformation(thisSpatialRef, referenceSpatialRef);

    std::vector<double> oldRowReference(referenceReader.size());
    for(int ireference=0;ireference<referenceReader.size();++ireference)
      oldRowReference[ireference]=-1;
    if(verbose_opt[0]>=1)
      std::cout << "here we go" << std::endl;
    Vector2d<T> classInput(referenceReader.size());
    for(int y=0;y<nrOfRow();++y){
      Vector2d<T> lineInput(band_opt.size(),nrOfCol());
      for(unsigned int iband=0;iband<band_opt.size();++iband)
        readData(lineInput[iband],y,band_opt[iband]);
      for(int x=0;x<this->nrOfCol();++x){
        std::vector<unsigned int> posclass(2+nclass);
        posclass[0]=static_cast<unsigned int>(x);
        posclass[1]=static_cast<unsigned int>(y);
        //read reference
        for(size_t iref=0;iref<referenceReader.size();++iref){
          double colReference=0;
          double rowReference=0;
          double geox=0;
          double geoy=0;
          image2geo(x,y,geox,geoy);
          referenceReader.getImage(iref)->geo2image(geox,geoy,colReference,rowReference,img2ref);
          rowReference=static_cast<int>(rowReference);
          colReference=static_cast<int>(colReference);
          if(rowReference>=0&&rowReference<referenceReader.getImage(iref)->nrOfRow()&&colReference>=0&&colReference<referenceReader.getImage(iref)->nrOfCol()){
            if(static_cast<int>(rowReference)!=static_cast<int>(oldRowReference[iref])){
              referenceReader.getImage(iref)->readData(classInput[iref],static_cast<unsigned int>(rowReference));
              oldRowReference[iref]=rowReference;
            }
            refpixel[iref]=classInput[iref][colReference];
          }
          else
            continue;
        }

        pixel=lineInput.selectCol(x);
        bool valid=false;
        for(size_t iband=0;iband<pixel.size();++iband){
          if(pixel[iband]!=srcnodata_opt[0]){
            valid=true;
            break;
          }
        }
        if(!valid)
          continue;
        for(size_t refband=0;refband<referenceReader.size();++refband){
          bool notFound=true;
          for(size_t iclass=0;iclass<class_opt.size();++iclass){
            if(refpixel[refband]==class_opt[iclass]){
              ++(posclass[2+iclass]);
              notFound=false;
              break;
            }
          }
          if(notFound){
            if(class_opt.size()<2)
              ++(posclass[2+1]);
          }
        }
        umap[pixel].push_back(posclass);
      }
    }
    if(verbose_opt[0]>=1)
      std::cout << "write to output" << std::endl;
    std::ostringstream outputStream;
    boost::archive::text_oarchive oarch(outputStream);
    /* std::ofstream ofs(model_opt[0], std::ios::binary); */
    /* boost::archive::binary_oarchive oarch(ofs); */
    oarch << umap;

    if(verbose_opt[0]){
      std::cout << "umap.size(): " << umap.size() << std::endl;
      auto umapit = umap.begin();
      for(auto tupit = umapit->second.begin(); tupit != umapit->second.end(); ++tupit){
        for(size_t iclass=0;iclass<nclass;++iclass){
          double value=(*tupit)[2+iclass];
          std::cout << value << " ";
        }
        std::cout << std::endl;
      }
    }
    return(outputStream.str());
  }
  catch(std::string errorString){
    std::cerr << errorString << std::endl;
    throw;
  }
}
///classify raster dataset with SML
/* template<typename T> void Jim::classifySML(Jim& imgWriter, app::AppFactory& app){ */
/*   Optionjl<std::string> model_opt("model", "model", "Model filename to save trained classifier."); */
/*   Optionjl<unsigned int> band_opt("b", "band", "Band index (starting from 0, either use band option or use start to end)"); */
/*   Optionjl<unsigned int> bstart_opt("sband", "startband", "Start band sequence number"); */
/*   Optionjl<unsigned int> bend_opt("eband", "endband", "End band sequence number"); */
/*   Optionjl<unsigned short> class_opt("c", "class", "Class(es) to extract from reference. Leave empty to extract two classes only: 1 against rest",1); */
/*   Optionjl<std::string> extent_opt("e", "extent", "Only classify within extent from polygons in vector file"); */
/*   Optionjl<std::string> eoption_opt("eo","eo", "special extent options controlling rasterization: ATTRIBUTE|CHUNKYSIZE|ALL_TOUCHED|BURN_VALUE_FROM|MERGE_ALG, e.g., -eo ATTRIBUTE=fieldname"); */
/*   Optionjl<std::string> mask_opt("m", "mask", "Only classify within specified mask. For raster mask, set nodata values with the option msknodata."); */
/*   Optionjl<short> msknodata_opt("msknodata", "msknodata", "Mask value(s) not to consider for classification. Values will be taken over in classification image.", 0); */
/*   Optionjl<double> srcnodata_opt("srcnodata", "srcnodata", "Nodata value in source",0); */
/*   Optionjl<double> dstnodata_opt("dstnodata", "dstnodata", "Nodata value to put where image is masked as nodata", 0); */
/*   Optionjl<short> verbose_opt("v", "verbose", "Verbose level",0,2); */

/*   extent_opt.setHide(1); */
/*   eoption_opt.setHide(1); */

/*   bool doProcess;//stop process when program was invoked with help option (-h --help) */
/*   try{ */
/*     doProcess=model_opt.retrieveOption(app); */
/*     band_opt.retrieveOption(app); */
/*     bstart_opt.retrieveOption(app); */
/*     bend_opt.retrieveOption(app); */
/*     class_opt.retrieveOption(app); */
/*     model_opt.retrieveOption(app); */
/*     extent_opt.retrieveOption(app); */
/*     eoption_opt.retrieveOption(app); */
/*     mask_opt.retrieveOption(app); */
/*     msknodata_opt.retrieveOption(app); */
/*     srcnodata_opt.retrieveOption(app); */
/*     dstnodata_opt.retrieveOption(app); */
/*     verbose_opt.retrieveOption(app); */

/*     if(!doProcess){ */
/*       std::cout << std::endl; */
/*       std::ostringstream helpStream; */
/*       helpStream << "short option -h shows basic options only, use long option --help to show all options" << std::endl; */
/*       throw(helpStream.str());//help was invoked, stop processing */
/*     } */

/*     if(model_opt.empty()){ */
/*       std::string errorString="Error: model option to write model not set"; */
/*       throw(errorString); */
/*     } */

/*     VectorOgr extentReader; */
/*     Jim maskReader; */

/*     double ulx=0; */
/*     double uly=0; */
/*     double lrx=0; */
/*     double lry=0; */


/*     if(verbose_opt[0]>=1) */
/*       std::cout << "start SML" << std::endl; */

/*     if(extent_opt.size()){ */
/*       if(verbose_opt[0]>=1) */
/*         std::cout << "We are in extent" << std::endl; */
/*       if(mask_opt.size()){ */
/*         std::string errorString="Error: can only either mask or extent, not both"; */
/*         throw(errorString); */
/*       } */
/*       extentReader.open(extent_opt[0]); */
/*       extentReader.getExtent(ulx,uly,lrx,lry); */
/*       maskReader.open(this->nrOfCol(),this->nrOfRow(),1,GDT_Float64); */
/*       double gt[6]; */
/*       this->getGeoTransform(gt); */
/*       maskReader.setGeoTransform(gt); */
/*       maskReader.setProjection(this->getProjection()); */
/*       maskReader.rasterizeBuf(extentReader,dstnodata_opt[0],eoption_opt); */
/*       maskReader.GDALSetNoDataValue(dstnodata_opt[0]); */
/*       extentReader.close(); */
/*     } */

/*     if(verbose_opt[0]>=1) */
/*       std::cout << "configure band options" << std::endl; */

/*     //convert start and end band options to vector of band indexes */
/*     if(bstart_opt.size()){ */
/*       if(bend_opt.size()!=bstart_opt.size()){ */
/*         std::string errorstring="Error: options for start and end band indexes must be provided as pairs, missing end band"; */
/*         throw(errorstring); */
/*       } */
/*       band_opt.clear(); */
/*       for(unsigned int ipair=0;ipair<bstart_opt.size();++ipair){ */
/*         if(bend_opt[ipair]<=bstart_opt[ipair]){ */
/*           std::string errorstring="Error: index for end band must be smaller then start band"; */
/*           throw(errorstring); */
/*         } */
/*         for(unsigned int iband=bstart_opt[ipair];iband<=bend_opt[ipair];++iband) */
/*           band_opt.push_back(iband); */
/*       } */
/*     } */


/*     //sort bands */
/*     if(band_opt.size()) */
/*       std::sort(band_opt.begin(),band_opt.end()); */
/*     else{ */
/*       unsigned int iband=0; */
/*       while(band_opt.size()<nrOfBand()) */
/*         band_opt.push_back(iband++); */
/*     } */

/*     //umap: [unique band information]->[x,y,class1,class2,...] */
/*     std::map<std::vector<T>,std::vector<std::vector<unsigned int> > > umap; */
/*     std::vector<short> lineMask; */

/*     std::ifstream ifs(model_opt[0]); */
/*     umap.clear(); */
/*     boost::archive::text_iarchive iarch(ifs); */
/*     iarch >> umap; */

/*     int nclass=0; */
/*     auto umapit = umap.begin(); */
/*     for(auto tupit = umapit->second.begin(); tupit != umapit->second.end(); ++tupit){ */
/*       nclass=tupit->size()-2; */
/*       if(verbose_opt[0]) */
/*         std::cout << "number of classes: " << nclass; */
/*       for(size_t iclass=0;iclass<nclass;++iclass){ */
/*         double value=(*tupit)[2+iclass]; */
/*         if(verbose_opt[0]) */
/*           std::cout << value << " "; */
/*       } */
/*       if(verbose_opt[0]) */
/*         std::cout << std::endl; */
/*     } */

/*     int nrow=nrOfRow(); */
/*     int ncol=nrOfCol(); */
/*     if(this->isInit()){ */
/*       if(verbose_opt[0]>=1) */
/*         std::cout << "We are in initialize" << std::endl; */
/*       imgWriter.open(ncol,nrow,nclass,GDT_Byte); */
/*       imgWriter.GDALSetNoDataValue(dstnodata_opt[0]); */
/*       imgWriter.setNoData(dstnodata_opt); */
/*       imgWriter.copyGeoTransform(*this); */
/*       imgWriter.setProjection(this->getProjection()); */
/*     } */
/*     //initialize imgWriter with dstnodata_opt[0] */
/*     std::vector<unsigned short> vnodata(imgWriter.nrOfCol()); */
/*     for(size_t icol=0;icol<imgWriter.nrOfCol();++icol) */
/*       vnodata[icol]=static_cast<unsigned short>(dstnodata_opt[0]); */
/*     for(size_t iband=0;iband<imgWriter.nrOfBand();++iband){ */
/*       for(size_t irow=0;irow<imgWriter.nrOfRow();++irow) */
/*         imgWriter.writeData(vnodata,irow,iband); */
/*     } */

/*     if(mask_opt.size()){ */
/*       if(verbose_opt[0]>=1) */
/*         std::cout << "opening mask image file " << mask_opt[0] << std::endl; */
/*       maskReader.open(mask_opt[0]); */
/*     } */

/*     double oldRowMask=-1;//keep track of row mask to optimize number of line readings */
/*     for(int y=0;y<nrOfRow();++y){ */
/*       Vector2d<T> lineInput(band_opt.size(),nrOfCol()); */
/*       for(unsigned int iband=0;iband<band_opt.size();++iband) */
/*         readData(lineInput[iband],y,band_opt[iband]); */
/*       for(int x=0;x<this->nrOfCol();++x){ */
/*         bool doClassify=true; */
/*         double geox=0; */
/*         double geoy=0; */
/*         imgWriter.image2geo(x,y,geox,geoy); */
/*         if(maskReader.isInit()){ */
/*           bool masked=false; */
/*           //read mask */
/*           double colMask=0; */
/*           double rowMask=0; */
/*           maskReader.geo2image(geox,geoy,colMask,rowMask); */
/*           colMask=static_cast<int>(colMask); */
/*           rowMask=static_cast<int>(rowMask); */
/*           if(rowMask>=0&&rowMask<maskReader.nrOfRow()&&colMask>=0&&colMask<maskReader.nrOfCol()){ */
/*             if(static_cast<int>(rowMask)!=static_cast<int>(oldRowMask)){ */
/*               maskReader.readData(lineMask,static_cast<unsigned int>(rowMask)); */
/*               oldRowMask=rowMask; */
/*             } */
/*             short theMask=0; */
/*             for(short ivalue=0;ivalue<msknodata_opt.size();++ivalue){ */
/*               if(lineMask[colMask]==msknodata_opt[ivalue]){ */
/*                 theMask=lineMask[colMask]; */
/*                 masked=true; */
/*                 break; */
/*               } */
/*             } */
/*             if(masked){ */
/*               double prob=dstnodata_opt[0]; */
/*               for(unsigned int iclass=0;iclass<nclass;++iclass) */
/*                 imgWriter.writeData(static_cast<unsigned short>(prob),x,y,iclass); */
/*               continue; */
/*             } */
/*           } */
/*         } */
/*       } */
/*     } */
/*     //loop through umap and assign pixel values based on occurrence */
/*     for(auto mapit = umap.begin(); mapit != umap.end(); ++mapit){ */
/*       std::vector<double> fclass(nclass); */
/*       double maxValue=0; */
/*       for(auto tupit = mapit->second.begin(); tupit != mapit->second.end(); ++tupit){ */
/*         for(size_t iclass=0;iclass<nclass;++iclass){ */
/*           double value=(*tupit)[2+iclass]; */
/*           fclass[iclass]+=value; */
/*           if(fclass[iclass]>maxValue) */
/*             maxValue=fclass[iclass]; */
/*         } */
/*       } */
/*       double scale=255.0/maxValue; */
/*       for(size_t iclass=0;iclass<nclass;++iclass) */
/*         fclass[iclass]*=scale; */
/*       for(auto tupit = mapit->second.begin(); tupit != mapit->second.end(); ++tupit){ */
/*         for(unsigned int iclass=0;iclass<nclass;++iclass) */
/*           imgWriter.writeData(static_cast<unsigned short>(fclass[iclass]),(*tupit)[0],(*tupit)[1],iclass); */
/*       } */
/*     } */
/*     if(mask_opt.size()) */
/*       maskReader.close(); */
/*   } */
/*   catch(std::string errorString){ */
/*     std::cerr << errorString << std::endl; */
/*     throw; */
/*   } */
/* } */

/* ///classify raster dataset with SML */
/* template<typename T> CPLErr Jim::classifySML(JimList& referenceReader, Jim& imgWriter, app::AppFactory& app){ */
/*   Optionjl<unsigned int> band_opt("b", "band", "Band index (starting from 0, either use band option or use start to end)"); */
/*   Optionjl<unsigned int> bstart_opt("sband", "startband", "Start band sequence number"); */
/*   Optionjl<unsigned int> bend_opt("eband", "endband", "End band sequence number"); */
/*   Optionjl<std::string> model_opt("model", "model", "Model filename to save trained classifier."); */
/*   Optionjl<unsigned short> class_opt("c", "class", "Class(es) to extract from reference. Leave empty to extract two classes only: 1 against rest",1); */
/*   Optionjl<std::string> extent_opt("e", "extent", "Only classify within extent from polygons in vector file"); */
/*   Optionjl<std::string> eoption_opt("eo","eo", "special extent options controlling rasterization: ATTRIBUTE|CHUNKYSIZE|ALL_TOUCHED|BURN_VALUE_FROM|MERGE_ALG, e.g., -eo ATTRIBUTE=fieldname"); */
/*   Optionjl<std::string> mask_opt("m", "mask", "Only classify within specified mask. For raster mask, set nodata values with the option msknodata."); */
/*   Optionjl<short> msknodata_opt("msknodata", "msknodata", "Mask value(s) not to consider for classification. Values will be taken over in classification image.", 0); */
/*   Optionjl<double> srcnodata_opt("srcnodata", "srcnodata", "Nodata value in source",0); */
/*   Optionjl<double> dstnodata_opt("dstnodata", "dstnodata", "Nodata value to put where image is masked as nodata", 0); */
/*   /\* Optionjl<double> nodata_opt("nodata", "nodata", "Nodata value to put where image is masked as nodata", 0); *\/ */
/*   Optionjl<std::string> otype_opt("ot", "otype", "Data type for output image ({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}). Empty string: inherit type from input image","GDT_Byte"); */
/*   Optionjl<short> verbose_opt("v", "verbose", "Verbose level",0,2); */

/*   extent_opt.setHide(1); */
/*   eoption_opt.setHide(1); */

/*   bool doProcess;//stop process when program was invoked with help option (-h --help) */
/*   try{ */
/*     doProcess=band_opt.retrieveOption(app); */
/*     bstart_opt.retrieveOption(app); */
/*     bend_opt.retrieveOption(app); */
/*     class_opt.retrieveOption(app); */
/*     model_opt.retrieveOption(app); */
/*     extent_opt.retrieveOption(app); */
/*     eoption_opt.retrieveOption(app); */
/*     mask_opt.retrieveOption(app); */
/*     msknodata_opt.retrieveOption(app); */
/*     srcnodata_opt.retrieveOption(app); */
/*     dstnodata_opt.retrieveOption(app); */
/*     otype_opt.retrieveOption(app); */
/*     verbose_opt.retrieveOption(app); */
/*     // memory_opt.retrieveOption(app); */

/*     if(!doProcess){ */
/*       std::cout << std::endl; */
/*       std::ostringstream helpStream; */
/*       helpStream << "short option -h shows basic options only, use long option --help to show all options" << std::endl; */
/*       throw(helpStream.str());//help was invoked, stop processing */
/*     } */

/*     VectorOgr extentReader; */
/*     Jim maskReader; */

/*     double ulx=0; */
/*     double uly=0; */
/*     double lrx=0; */
/*     double lry=0; */


/*     if(verbose_opt[0]>=1) */
/*       std::cout << "start SML" << std::endl; */

/*     if(extent_opt.size()){ */
/*       if(verbose_opt[0]>=1) */
/*         std::cout << "We are in extent" << std::endl; */
/*       if(mask_opt.size()){ */
/*         std::string errorString="Error: can only either mask or extent, not both"; */
/*         throw(errorString); */
/*       } */
/*       extentReader.open(extent_opt[0]); */
/*       // readLayer = extentReader.getDataSource()->GetLayer(0); */
/*       extentReader.getExtent(ulx,uly,lrx,lry); */
/*       maskReader.open(this->nrOfCol(),this->nrOfRow(),1,GDT_Float64); */
/*       double gt[6]; */
/*       this->getGeoTransform(gt); */
/*       maskReader.setGeoTransform(gt); */
/*       maskReader.setProjection(this->getProjection()); */
/*       // vector<double> burnValues(1,1);//burn value is 1 (single band) */
/*       // maskReader.rasterizeBuf(extentReader); */
/*       maskReader.rasterizeBuf(extentReader,dstnodata_opt[0],eoption_opt); */
/*       maskReader.GDALSetNoDataValue(dstnodata_opt[0]); */
/*       extentReader.close(); */
/*     } */

/*     if(verbose_opt[0]>=1) */
/*       std::cout << "configure band options" << std::endl; */

/*     //convert start and end band options to vector of band indexes */
/*     if(bstart_opt.size()){ */
/*       if(bend_opt.size()!=bstart_opt.size()){ */
/*         std::string errorstring="Error: options for start and end band indexes must be provided as pairs, missing end band"; */
/*         throw(errorstring); */
/*       } */
/*       band_opt.clear(); */
/*       for(unsigned int ipair=0;ipair<bstart_opt.size();++ipair){ */
/*         if(bend_opt[ipair]<=bstart_opt[ipair]){ */
/*           std::string errorstring="Error: index for end band must be smaller then start band"; */
/*           throw(errorstring); */
/*         } */
/*         for(unsigned int iband=bstart_opt[ipair];iband<=bend_opt[ipair];++iband) */
/*           band_opt.push_back(iband); */
/*       } */
/*     } */


/*     //sort bands */
/*     if(band_opt.size()) */
/*       std::sort(band_opt.begin(),band_opt.end()); */
/*     else{ */
/*       unsigned int iband=0; */
/*       while(band_opt.size()<nrOfBand()) */
/*         band_opt.push_back(iband++); */
/*     } */

/*     int nrow=nrOfRow(); */
/*     int ncol=nrOfCol(); */
/*     int nclass=class_opt.size()>1?class_opt.size():2; */
/*     if(this->isInit()){ */
/*       if(verbose_opt[0]>=1) */
/*         std::cout << "We are in initialize" << std::endl; */
/*       GDALDataType theType=string2GDAL(otype_opt[0]); */
/*       if(theType==GDT_Unknown) */
/*         std::cout << "Warning: unknown output pixel type: " << otype_opt[0] << ", using input type as default" << std::endl; */
/*       imgWriter.open(ncol,nrow,nclass,GDT_Byte); */
/*       imgWriter.GDALSetNoDataValue(dstnodata_opt[0]); */
/*       imgWriter.setNoData(dstnodata_opt); */
/*       imgWriter.copyGeoTransform(*this); */
/*       imgWriter.setProjection(this->getProjection()); */
/*       // if(colorTable_opt.size()) */
/*       //   imgWriter.setColorTable(colorTable_opt[0],0); */
/*     } */
/*     //initialize imgWriter with dstnodata_opt[0] */
/*     std::vector<unsigned short> vnodata(imgWriter.nrOfCol()); */
/*     for(size_t icol=0;icol<imgWriter.nrOfCol();++icol) */
/*       vnodata[icol]=static_cast<unsigned short>(dstnodata_opt[0]); */
/*     for(size_t iband=0;iband<imgWriter.nrOfBand();++iband){ */
/*       for(size_t irow=0;irow<imgWriter.nrOfRow();++irow) */
/*         imgWriter.writeData(vnodata,irow,iband); */
/*     } */

/*     if(mask_opt.size()){ */
/*       if(verbose_opt[0]>=1) */
/*         std::cout << "opening mask image file " << mask_opt[0] << std::endl; */
/*       maskReader.open(mask_opt[0]); */
/*     } */

/*     //umap: [unique band information]->[x,y,class1,class2,...] */
/*     std::map<std::vector<T>,std::vector<std::vector<unsigned int> > > umap; */
/*     std::vector<short> lineMask; */
/*     std::vector<T> pixel(band_opt.size());//pixel with band information */
/*     std::vector<T> refpixel(referenceReader.size());//pixel with reference information */

/*     OGRSpatialReference thisSRS(getProjectionRef().c_str()); */
/*     OGRSpatialReference *thisSpatialRef=&thisSRS; */
/*     //currently only a single SRS for all images in reference collection supported */
/*     OGRSpatialReference referenceSRS(referenceReader.getImage(0)->getProjectionRef().c_str()); */

/*     OGRSpatialReference *referenceSpatialRef=&referenceSRS; */
/*     OGRCoordinateTransformation *img2ref=OGRCreateCoordinateTransformation(thisSpatialRef, referenceSpatialRef); */

/*     double oldRowMask=-1;//keep track of row mask to optimize number of line readings */
/*     std::vector<double> oldRowReference(referenceReader.size()); */
/*     for(int ireference=0;ireference<referenceReader.size();++ireference) */
/*       oldRowReference[ireference]=-1; */
/*     for(int y=0;y<nrOfRow();++y){ */
/*       Vector2d<T> lineInput(band_opt.size(),nrOfCol()); */
/*       for(unsigned int iband=0;iband<band_opt.size();++iband) */
/*         readData(lineInput[iband],y,band_opt[iband]); */
/*       /\* Vector2d<T> classInput(referenceReader.size(),nrOfCol()); *\/ */
/*       Vector2d<T> classInput(referenceReader.size()); */
/*       /\* for(unsigned int iband=0;iband<cband_opt.size();++iband) *\/ */
/*       /\*   readData(classInput[iband],y,cband_opt[iband]); *\/ */
/*       for(int x=0;x<this->nrOfCol();++x){ */
/*         std::vector<unsigned int> posclass(2+nclass); */
/*         posclass[0]=static_cast<unsigned int>(x); */
/*         posclass[1]=static_cast<unsigned int>(y); */
/*         bool doClassify=true; */
/*         double geox=0; */
/*         double geoy=0; */
/*         imgWriter.image2geo(x,y,geox,geoy); */
/*         if(maskReader.isInit()){ */
/*           bool masked=false; */
/*           //read mask */
/*           double colMask=0; */
/*           double rowMask=0; */
/*           maskReader.geo2image(geox,geoy,colMask,rowMask); */
/*           colMask=static_cast<int>(colMask); */
/*           rowMask=static_cast<int>(rowMask); */
/*           if(rowMask>=0&&rowMask<maskReader.nrOfRow()&&colMask>=0&&colMask<maskReader.nrOfCol()){ */
/*             if(static_cast<int>(rowMask)!=static_cast<int>(oldRowMask)){ */
/*               maskReader.readData(lineMask,static_cast<unsigned int>(rowMask)); */
/*               oldRowMask=rowMask; */
/*             } */
/*             short theMask=0; */
/*             for(short ivalue=0;ivalue<msknodata_opt.size();++ivalue){ */
/*               if(lineMask[colMask]==msknodata_opt[ivalue]){ */
/*                 theMask=lineMask[colMask]; */
/*                 masked=true; */
/*                 break; */
/*               } */
/*             } */
/*             if(masked){ */
/*               double prob=dstnodata_opt[0]; */
/*               for(unsigned int iclass=0;iclass<nclass;++iclass) */
/*                 imgWriter.writeData(static_cast<unsigned short>(prob),x,y,iclass); */
/*               continue; */
/*             } */
/*           } */
/*         } */

/*         //read reference */
/*         for(size_t iref=0;iref<referenceReader.size();++iref){ */
/*           double colReference=0; */
/*           double rowReference=0; */
/*           referenceReader.getImage(iref)->geo2image(geox,geoy,colReference,rowReference,img2ref); */
/*           rowReference=static_cast<int>(rowReference); */
/*           colReference=static_cast<int>(colReference); */
/*           if(rowReference>=0&&rowReference<referenceReader.getImage(iref)->nrOfRow()&&colReference>=0&&colReference<referenceReader.getImage(iref)->nrOfCol()){ */
/*             if(static_cast<int>(rowReference)!=static_cast<int>(oldRowReference[iref])){ */
/*               referenceReader.getImage(iref)->readData(classInput[iref],static_cast<unsigned int>(rowReference)); */
/*               oldRowReference[iref]=rowReference; */
/*             } */
/*             refpixel[iref]=classInput[iref][colReference]; */
/*           } */
/*           else */
/*             continue; */
/*         } */

/*         pixel=lineInput.selectCol(x); */
/*         bool valid=false; */
/*         for(size_t iband=0;iband<pixel.size();++iband){ */
/*           if(pixel[iband]!=srcnodata_opt[0]){ */
/*             valid=true; */
/*             break; */
/*           } */
/*         } */
/*         if(!valid) */
/*           continue; */
/*         for(size_t refband=0;refband<referenceReader.size();++refband){ */
/*           bool notFound=true; */
/*           for(size_t iclass=0;iclass<class_opt.size();++iclass){ */
/*             if(refpixel[refband]==class_opt[iclass]){ */
/*               ++(posclass[2+iclass]); */
/*               notFound=false; */
/*               break; */
/*             } */
/*           } */
/*           if(notFound){ */
/*             if(class_opt.size()<2) */
/*               ++(posclass[2+1]); */
/*           } */
/*         } */
/*         umap[pixel].push_back(posclass); */
/*       } */
/*     } */
/*     double absMaxValue=0; */

/*     /\*todo: not tested yet... *\/ */
/*     if(model_opt.size()){ */
/*       std::ofstream ofs(model_opt[0], std::ios::binary); */
/*       /\* std::ofstream ofs(model_opt[0]); *\/ */
/*       boost::archive::binary_oarchive oarch(ofs); */
/*       /\* boost::archive::text_oarchive oarch(ofs); *\/ */
/*       oarch << umap; */

/*       if(verbose_opt[0]){ */
/*         std::cout << "umap.size(): " << umap.size() << std::endl; */
/*         auto umapit = umap.begin(); */
/*         for(auto tupit = umapit->second.begin(); tupit != umapit->second.end(); ++tupit){ */
/*           for(size_t iclass=0;iclass<nclass;++iclass){ */
/*             double value=(*tupit)[2+iclass]; */
/*             std::cout << value << " "; */
/*           } */
/*           std::cout << std::endl; */
/*         } */
/*       } */
/*     } */
/*     if(model_opt.size()){ */
/*       //input stream must be in new scope in order output stream to be closed correctly */
/*       /\* std::map<std::vector<T>,std::vector<std::vector<unsigned int> > > newmap; *\/ */
/*       std::ifstream ifs(model_opt[0], std::ios::binary); */
/*       /\* std::ifstream ifs(model_opt[0]); *\/ */
/*       umap.clear(); */
/*       boost::archive::binary_iarchive iarch(ifs); */
/*       /\* boost::archive::text_iarchive iarch(ifs); *\/ */
/*       iarch >> umap; */
/*       /\* iarch >> newmap; *\/ */

/*       if(verbose_opt[0]){ */
/*         /\* std::cout << "newmap.size(): " << newmap.size() << std::endl; *\/ */
/*         auto umapit = umap.begin(); */
/*         /\* for(auto tupit = newmapit->second.begin(); tupit != newmapit->second.end(); ++tupit){ *\/ */
/*         for(auto tupit = umapit->second.begin(); tupit != umapit->second.end(); ++tupit){ */
/*           for(size_t iclass=0;iclass<nclass;++iclass){ */
/*             double value=(*tupit)[2+iclass]; */
/*             std::cout << value << " "; */
/*           } */
/*           std::cout << std::endl; */
/*         } */
/*       } */
/*     } */
/*     //loop through umap and assign pixel values based on occurrence */
/*     for(auto mapit = umap.begin(); mapit != umap.end(); ++mapit){ */
/*       std::vector<double> fclass(nclass); */
/*       double maxValue=0; */
/*       for(auto tupit = mapit->second.begin(); tupit != mapit->second.end(); ++tupit){ */
/*         for(size_t iclass=0;iclass<nclass;++iclass){ */
/*           double value=(*tupit)[2+iclass]; */
/*           fclass[iclass]+=value; */
/*           if(fclass[iclass]>maxValue) */
/*             maxValue=fclass[iclass]; */
/*         } */
/*       } */
/*       double scale=255.0/maxValue; */
/*       for(size_t iclass=0;iclass<nclass;++iclass) */
/*         fclass[iclass]*=scale; */
/*       for(auto tupit = mapit->second.begin(); tupit != mapit->second.end(); ++tupit){ */
/*         for(unsigned int iclass=0;iclass<nclass;++iclass) */
/*           imgWriter.writeData(static_cast<unsigned short>(fclass[iclass]),(*tupit)[0],(*tupit)[1],iclass); */
/*         if(mask_opt.size()) */
/*           maskReader.close(); */
/*       } */
/*     } */
/*   } */
/*   catch(std::string errorString){ */
/*     std::cerr << errorString << std::endl; */
/*     return(CE_Failure); */
/*   } */
/* } */
///classify raster dataset with SML
///this Jim contains input bands and reference with binary class information)
/**
 * @param imgWriter output classified raster dataset
 * @param app application specific option arguments
 * @return CE_None if successful, CE_Failure if failed
 **/
/* template<typename T> CPLErr Jim::classifySML(Jim& imgWriter, app::AppFactory& app){ */
/*   Optionjl<unsigned int> band_opt("b", "band", "Band index (starting from 0, either use band option or use start to end)"); */
/*   Optionjl<unsigned int> bstart_opt("sband", "startband", "Start band sequence number"); */
/*   Optionjl<unsigned int> bend_opt("eband", "endband", "End band sequence number"); */
/*   Optionjl<unsigned int> cband_opt("cb", "cband", "band index(es) used for the reference class information", 0); */
/*   Optionjl<unsigned short> class_opt("c", "class", "Class(es) to extract from reference. Leave empty to extract two classes only: 1 against rest",1); */
/*   Optionjl<std::string> extent_opt("e", "extent", "Only classify within extent from polygons in vector file"); */
/*   Optionjl<std::string> eoption_opt("eo","eo", "special extent options controlling rasterization: ATTRIBUTE|CHUNKYSIZE|ALL_TOUCHED|BURN_VALUE_FROM|MERGE_ALG, e.g., -eo ATTRIBUTE=fieldname"); */
/*   Optionjl<std::string> mask_opt("m", "mask", "Only classify within specified mask. For raster mask, set nodata values with the option msknodata."); */
/*   Optionjl<short> msknodata_opt("msknodata", "msknodata", "Mask value(s) not to consider for classification. Values will be taken over in classification image.", 0); */
/*   Optionjl<double> nodata_opt("nodata", "nodata", "Nodata value to put where image is masked as nodata", 0); */
/*   Optionjl<std::string> otype_opt("ot", "otype", "Data type for output image ({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}). Empty string: inherit type from input image","GDT_Byte"); */
/*   Optionjl<short> verbose_opt("v", "verbose", "Verbose level",0,2); */

/*   extent_opt.setHide(1); */
/*   eoption_opt.setHide(1); */

/*   bool doProcess;//stop process when program was invoked with help option (-h --help) */
/*   try{ */
/*     doProcess=band_opt.retrieveOption(app); */
/*     bstart_opt.retrieveOption(app); */
/*     bend_opt.retrieveOption(app); */
/*     cband_opt.retrieveOption(app); */
/*     class_opt.retrieveOption(app); */
/*     extent_opt.retrieveOption(app); */
/*     eoption_opt.retrieveOption(app); */
/*     mask_opt.retrieveOption(app); */
/*     msknodata_opt.retrieveOption(app); */
/*     nodata_opt.retrieveOption(app); */
/*     otype_opt.retrieveOption(app); */
/*     verbose_opt.retrieveOption(app); */
/*     // memory_opt.retrieveOption(app); */

/*     if(!doProcess){ */
/*       std::cout << std::endl; */
/*       std::ostringstream helpStream; */
/*       helpStream << "short option -h shows basic options only, use long option --help to show all options" << std::endl; */
/*       throw(helpStream.str());//help was invoked, stop processing */
/*     } */

/*     VectorOgr extentReader; */
/*     Jim maskReader; */

/*     double ulx=0; */
/*     double uly=0; */
/*     double lrx=0; */
/*     double lry=0; */

/*     if(extent_opt.size()){ */
/*       if(mask_opt.size()){ */
/*         std::string errorString="Error: can only either mask or extent, not both"; */
/*         throw(errorString); */
/*       } */
/*       extentReader.open(extent_opt[0]); */
/*       // readLayer = extentReader.getDataSource()->GetLayer(0); */
/*       extentReader.getExtent(ulx,uly,lrx,lry); */
/*       maskReader.open(this->nrOfCol(),this->nrOfRow(),1,GDT_Float64); */
/*       double gt[6]; */
/*       this->getGeoTransform(gt); */
/*       maskReader.setGeoTransform(gt); */
/*       maskReader.setProjection(this->getProjection()); */
/*       // vector<double> burnValues(1,1);//burn value is 1 (single band) */
/*       // maskReader.rasterizeBuf(extentReader); */
/*       maskReader.rasterizeBuf(extentReader,nodata_opt[0],eoption_opt); */
/*       maskReader.GDALSetNoDataValue(nodata_opt[0]); */
/*       extentReader.close(); */
/*     } */

/*     //convert start and end band options to vector of band indexes */
/*     if(bstart_opt.size()){ */
/*       if(bend_opt.size()!=bstart_opt.size()){ */
/*         std::string errorstring="Error: options for start and end band indexes must be provided as pairs, missing end band"; */
/*         throw(errorstring); */
/*       } */
/*       band_opt.clear(); */
/*       for(unsigned int ipair=0;ipair<bstart_opt.size();++ipair){ */
/*         if(bend_opt[ipair]<=bstart_opt[ipair]){ */
/*           std::string errorstring="Error: index for end band must be smaller then start band"; */
/*           throw(errorstring); */
/*         } */
/*         for(unsigned int iband=bstart_opt[ipair];iband<=bend_opt[ipair];++iband) */
/*           band_opt.push_back(iband); */
/*       } */
/*     } */

/*     //sort bands */
/*     if(band_opt.size()) */
/*       std::sort(band_opt.begin(),band_opt.end()); */
/*     if(cband_opt.size()) */
/*       std::sort(cband_opt.begin(),cband_opt.end()); */

/*     int nrow=nrOfRow(); */
/*     int ncol=nrOfCol(); */
/*     int nclass=class_opt.size()>1?class_opt.size():2; */
/*     if(this->isInit()){ */
/*       GDALDataType theType=string2GDAL(otype_opt[0]); */
/*       if(theType==GDT_Unknown) */
/*         std::cout << "Warning: unknown output pixel type: " << otype_opt[0] << ", using input type as default" << std::endl; */
/*       imgWriter.open(ncol,nrow,nclass,GDT_Byte); */
/*       imgWriter.GDALSetNoDataValue(nodata_opt[0]); */
/*       imgWriter.setNoData(nodata_opt); */
/*       imgWriter.copyGeoTransform(*this); */
/*       imgWriter.setProjection(this->getProjection()); */
/*       // if(colorTable_opt.size()) */
/*       //   imgWriter.setColorTable(colorTable_opt[0],0); */
/*     } */

/*     if(mask_opt.size()){ */
/*       if(verbose_opt[0]>=1) */
/*         std::cout << "opening mask image file " << mask_opt[0] << std::endl; */
/*       maskReader.open(mask_opt[0]); */
/*     } */

/*     //umap: [unique band information]->[x,y,class1,class2,...] */
/*     std::map<std::vector<T>,std::vector<std::vector<unsigned int> > > umap; */
/*     std::vector<short> lineMask; */
/*     std::vector<T> pixel(band_opt.size());//pixel with band information */
/*     std::vector<T> refpixel(cband_opt.size());//pixel with reference information */

/*     double oldRowMask=-1;//keep track of row mask to optimize number of line readings */
/*     for(int y=0;y<nrOfRow();++y){ */
/*       Vector2d<T> lineInput(band_opt.size(),nrOfCol()); */
/*       for(unsigned int iband=0;iband<band_opt.size();++iband) */
/*         readData(lineInput[iband],y,band_opt[iband]); */
/*       /\* Vector2d<T> classInput(cband_opt.size(),nrOfCol()); *\/ */
/*       Vector2d<T> classInput(cband_opt.size()); */
/*       for(unsigned int iband=0;iband<cband_opt.size();++iband) */
/*         readData(classInput[iband],y,cband_opt[iband]); */
/*       for(int x=0;x<this->nrOfCol();++x){ */
/*         std::vector<unsigned int> posclass(2+nclass); */
/*         posclass[0]=static_cast<unsigned int>(x); */
/*         posclass[1]=static_cast<unsigned int>(y); */
/*         bool doClassify=true; */
/*         double geox=0; */
/*         double geoy=0; */
/*         if(maskReader.isInit()){ */
/*           bool masked=false; */
/*           //read mask */
/*           double colMask=0; */
/*           double rowMask=0; */
/*           imgWriter.image2geo(x,y,geox,geoy); */
/*           maskReader.geo2image(geox,geoy,colMask,rowMask); */
/*           colMask=static_cast<int>(colMask); */
/*           rowMask=static_cast<int>(rowMask); */
/*           if(rowMask>=0&&rowMask<maskReader.nrOfRow()&&colMask>=0&&colMask<maskReader.nrOfCol()){ */
/*             if(static_cast<int>(rowMask)!=static_cast<int>(oldRowMask)){ */
/*               maskReader.readData(lineMask,static_cast<unsigned int>(rowMask)); */
/*               oldRowMask=rowMask; */
/*             } */
/*             short theMask=0; */
/*             for(short ivalue=0;ivalue<msknodata_opt.size();++ivalue){ */
/*               if(lineMask[colMask]==msknodata_opt[ivalue]){ */
/*                 theMask=lineMask[colMask]; */
/*                 masked=true; */
/*                 break; */
/*               } */
/*             } */
/*             if(masked){ */
/*               double prob=nodata_opt[0]; */
/*               for(unsigned int iclass=0;iclass<nclass;++iclass) */
/*                 imgWriter.writeData(static_cast<unsigned short>(prob),x,y,iclass); */
/*               continue; */
/*             } */
/*           } */
/*         } */
/*         pixel=lineInput.selectCol(x); */
/*         refpixel=classInput.selectCol(x); */
/*         for(size_t refband=0;refband<cband_opt.size();++refband){ */
/*           bool notFound=true; */
/*           for(size_t iclass=0;iclass<class_opt.size();++iclass){ */
/*             if(refpixel[refband]==class_opt[iclass]){ */
/*               ++(posclass[2+iclass]); */
/*               notFound=false; */
/*               break; */
/*             } */
/*           } */
/*           if(notFound){ */
/*             if(class_opt.size()<2) */
/*               ++(posclass[2+1]); */
/*           } */
/*         } */
/*         umap[pixel].push_back(posclass); */
/*       } */
/*     } */
/*     //loop through umap and assign pixel values based on occurrence */
/*     for(auto mapit = umap.begin(); mapit != umap.end(); ++mapit){ */
/*       std::vector<double> fclass(nclass); */
/*       double maxValue=0; */
/*       for(auto tupit = mapit->second.begin(); tupit != mapit->second.end(); ++tupit){ */
/*         for(size_t iclass=0;iclass<nclass;++iclass){ */
/*           double value=(*tupit)[2+iclass]; */
/*           fclass[iclass]+=value; */
/*           if(fclass[iclass]>maxValue) */
/*             maxValue=fclass[iclass]; */
/*         } */
/*       } */
/*       double scale=255.0/maxValue; */
/*       for(size_t iclass=0;iclass<nclass;++iclass) */
/*         fclass[iclass]*=scale; */
/*       for(auto tupit = mapit->second.begin(); tupit != mapit->second.end(); ++tupit){ */
/*         for(unsigned int iclass=0;iclass<nclass;++iclass) */
/*           imgWriter.writeData(static_cast<unsigned short>(fclass[iclass]),(*tupit)[0],(*tupit)[1],iclass); */
/*         if(mask_opt.size()) */
/*           maskReader.close(); */
/*       } */
/*     } */
/*   } */
/*   catch(std::string errorString){ */
/*     std::cerr << errorString << std::endl; */
/*     return(CE_Failure); */
/*   } */
/* } */

/* template<typename T> std::shared_ptr<Jim> Jim::classifySML(JimList& referenceReader, app::AppFactory& app){ */
/*   try{ */
/*     std::shared_ptr<Jim> imgWriter=createImg(); */
/*     classifySML<T>(referenceReader,*imgWriter, app); */
/*     return(imgWriter); */
/*   } */
/*   catch(std::string helpString){ */
/*     std::cerr << helpString << std::endl; */
/*     return(0); */
/*   } */
/* } */

/**
 * @param app application specific option arguments
 * @return output classified raster dataset
 **/
template<typename T> std::shared_ptr<Jim> Jim::classifySML_t(app::AppFactory& app){
  try{
    std::shared_ptr<Jim> imgWriter=createImg();
    classifySML_t<T>(*imgWriter, app);
    return(imgWriter);
  }
  catch(std::string helpString){
    std::cerr << helpString << std::endl;
    throw;
  }
}
