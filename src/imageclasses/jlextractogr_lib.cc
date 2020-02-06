/**********************************************************************
jlextractogr_lib.cc: extract pixel values from raster image from a vector sample
Author(s): Pieter.Kempeneers@ec.europa.eu
Copyright (c) 2016-2019 European Union (Joint Research Centre)
License EUPLv1.2

This file is part of jiplib
***********************************************************************/
#include <assert.h>
#include <string>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <ctime>
#include <vector>
#include <memory>
// #include <boost/filesystem.hpp>
#include <ogr_geometry.h>
#include "imageclasses/Jim.h"
#include "imageclasses/VectorOgr.h"
#include "base/Optionjl.h"
#include "algorithms/StatFactory.h"
#include "apps/AppFactory.h"

using namespace std;
using namespace app;

#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif

namespace rule{
  enum RULE_TYPE {point=0, mean=1, proportion=2, custom=3, min=4, max=5, mode=6, centroid=7, sum=8, median=9, stdev=10, percentile=11, count=12, allpoints=13};
}


/**
 * @param app application specific option arguments
 * @return output Vector
 **/
// make sure to setSpatialFilterRect on vector before entering here
shared_ptr<VectorOgr> Jim::extractOgr(VectorOgr& sampleReader, AppFactory& app){
  shared_ptr<VectorOgr> ogrWriter=VectorOgr::createVector();
  if(extractOgr(sampleReader, *ogrWriter, app)!=OGRERR_NONE){
    std::cerr << "Failed to extract" << std::endl;
  }
  return(ogrWriter);
}

/**
 * @param app application specific option arguments
 * @return output Vector
 **/
shared_ptr<VectorOgr> Jim::extractSample(AppFactory& app){
  shared_ptr<VectorOgr> ogrWriter=VectorOgr::createVector();
  if(extractSample(*ogrWriter, app)!=OGRERR_NONE){
    std::cerr << "Failed to extract" << std::endl;
  }
  return(ogrWriter);
}

/**
 * @param app application specific option arguments
 *
 * @return CE_None if success, CE_Failure if failure
 */
//todo: support multiple layers for writing
//output vector ogrWriter will take spatial reference system of input vector sampleReader
// make sure to setSpatialFilterRect on vector before entering here
CPLErr Jim::extractOgr(VectorOgr& sampleReader, VectorOgr&ogrWriter, AppFactory& app){
  // Optionjl<string> image_opt("i", "input", "Raster input dataset containing band information");
  // Optionjl<string> sample_opt("s", "sample", "OGR vector dataset with features to be extracted from input data. Output will contain features with input band information included.");
  // Optionjl<string> layer_opt("ln", "ln", "Layer name(s) in sample (leave empty to select all)");
  // Optionjl<unsigned int> random_opt("rand", "random", "Create simple random sample of points. Provide number of points to generate");
  // Optionjl<double> grid_opt("grid", "grid", "Create systematic grid of points. Provide cell grid size (in projected units, e.g,. m)");
  Optionjl<string> output_opt("o", "output", "Output sample dataset");
  Optionjl<int> label_opt("label", "label", "Create extra label field with this value");
  Optionjl<std::string> fid_opt("fid", "fid", "Create extra field with field identifier (sequence in which the features have been read");
  Optionjl<string> copyFields_opt("copy", "copy", "Restrict these fields only to copy from input to output vector dataset (default is to copy all fields)");
  Optionjl<int> class_opt("c", "class", "Class(es) in input raster dataset t take into account for the rules mode, proportion and count");
  Optionjl<float> threshold_opt("t", "threshold", "Probability threshold for selecting samples (randomly). Provide probability in percentage (>0) or absolute (<0). Use a single threshold per vector sample layer.  Use value 100 to select all pixels for selected class(es)", 100);
  Optionjl<double> percentile_opt("perc","perc","Percentile value(s) used for rule percentile",95);
  Optionjl<string> ogrformat_opt("f", "oformat", "Output vector dataset format","SQLite");
  Optionjl<unsigned int> access_opt("access", "access", "Access (0: GDAL_OF_READ_ONLY, 1: GDAL_OF_UPDATE)",1);
  Optionjl<std::string> option_opt("co", "co", "Creation option for output file. Multiple options can be specified.");
  Optionjl<string> ftype_opt("ft", "ftype", "Field type (only Real or Integer)", "Real");
  Optionjl<int> band_opt("b", "band", "Band index(es) to extract (0 based). Leave empty to use all bands");
  Optionjl<size_t> plane_opt("p", "plane", "Plane index(es) to extract (0 based). Leave empty to use all bands");
  Optionjl<std::string> bandNames_opt("bn", "bandname", "Band name(s) corresponding to band index(es).");
  Optionjl<std::string> planeNames_opt("bn", "planename", "Plane name(s) corresponding to plane index(es).");
  Optionjl<unsigned short> bstart_opt("sband", "startband", "Start band sequence number");
  Optionjl<unsigned short> bend_opt("eband", "endband", "End band sequence number");
  Optionjl<string> rule_opt("r", "rule", "Rule how to report image information per feature. point (single point within polygon), allpoints (all points within polygon), centroid, mean, stdev, median, proportion, count, min, max, mode, sum, percentile.","centroid");
  Optionjl<double> srcnodata_opt("srcnodata", "srcnodata", "Invalid value(s) for input image");
  Optionjl<int> bndnodata_opt("bndnodata", "bndnodata", "Band(s) in input image to check if pixel is valid (used for srcnodata)", 0);
  Optionjl<string> mask_opt("m", "mask", "Use the the specified file as a validity mask (0 is nodata).");
  Optionjl<double> msknodata_opt("msknodata", "msknodata", "Mask value not to extract.", 0);
  Optionjl<unsigned int> mskband_opt("mskband", "mskband", "Mask band to read (0 indexed)", 0);
  Optionjl<float> polythreshold_opt("tp", "thresholdPolygon", "(absolute) threshold for selecting samples in each polygon");
  Optionjl<short> buffer_opt("buf", "buffer", "Buffer for calculating statistics in geometric units of raster dataset");
  Optionjl<bool> disc_opt("circ", "circular", "Use a circular disc kernel buffer (for vector point sample datasets only, use in combination with buffer option)", false);
  Optionjl<std::string> allCovered_opt("cover", "cover", "Which polygons to include based on coverage (ALL_TOUCHED, ALL_COVERED)", "ALL_TOUCHED");
  Optionjl<unsigned long int>  memory_opt("mem", "mem", "Buffer size (in MB) to read image data blocks in memory",0,1);
  Optionjl<int> s_srs_opt("s_srs", "s_srs", "Spatial reference system of vector dataset (in EPSG)");
  Optionjl<short> verbose_opt("v", "verbose", "Verbose mode if > 0", 0,2);

  bstart_opt.setHide(1);
  bend_opt.setHide(1);
  bndnodata_opt.setHide(1);
  srcnodata_opt.setHide(1);
  mask_opt.setHide(1);
  msknodata_opt.setHide(1);
  mskband_opt.setHide(1);
  polythreshold_opt.setHide(1);
  percentile_opt.setHide(1);
  buffer_opt.setHide(1);
  disc_opt.setHide(1);
  allCovered_opt.setHide(1);
  option_opt.setHide(1);
  memory_opt.setHide(1);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    // doProcess=sample_opt.retrieveOption(app);
    // layer_opt.retrieveOption(app);
    doProcess=output_opt.retrieveOption(app);
    // random_opt.retrieveOption(app);
    // grid_opt.retrieveOption(app);
    class_opt.retrieveOption(app);
    label_opt.retrieveOption(app);
    fid_opt.retrieveOption(app);
    copyFields_opt.retrieveOption(app);
    threshold_opt.retrieveOption(app);
    percentile_opt.retrieveOption(app);
    ogrformat_opt.retrieveOption(app);
    access_opt.retrieveOption(app);
    option_opt.retrieveOption(app);
    ftype_opt.retrieveOption(app);
    band_opt.retrieveOption(app);
    plane_opt.retrieveOption(app);
    bandNames_opt.retrieveOption(app);
    planeNames_opt.retrieveOption(app);
    bstart_opt.retrieveOption(app);
    bend_opt.retrieveOption(app);
    rule_opt.retrieveOption(app);
    bndnodata_opt.retrieveOption(app);
    srcnodata_opt.retrieveOption(app);
    mask_opt.retrieveOption(app);
    msknodata_opt.retrieveOption(app);
    mskband_opt.retrieveOption(app);
    polythreshold_opt.retrieveOption(app);
    buffer_opt.retrieveOption(app);
    disc_opt.retrieveOption(app);
    allCovered_opt.retrieveOption(app);
    memory_opt.retrieveOption(app);
    s_srs_opt.retrieveOption(app);
    verbose_opt.retrieveOption(app);
    if(!doProcess){
      cout << endl;
      std::ostringstream helpStream;
      helpStream << "short option -h shows basic options only, use long option --help to show all options" << std::endl;
      throw(helpStream.str());//help was invoked, stop processing
    }

    // std::vector<std::string> badKeys;
    // app.badKeys(badKeys);
    // if(badKeys.size()){
    //   std::ostringstream errorStream;
    //   if(badKeys.size()>1)
    //     errorStream << "Error: unknown keys: ";
    //   else
    //     errorStream << "Error: unknown key: ";
    //   for(int ikey=0;ikey<badKeys.size();++ikey){
    //     errorStream << badKeys[ikey] << " ";
    //   }
    //   errorStream << std::endl;
    //   throw(errorStream.str());
    // }

    //initialize ruleMap
    std::map<std::string, rule::RULE_TYPE> ruleMap;
    ruleMap["point"]=rule::point;
    ruleMap["centroid"]=rule::centroid;
    ruleMap["mean"]=rule::mean;
    ruleMap["stdev"]=rule::stdev;
    ruleMap["median"]=rule::median;
    ruleMap["proportion"]=rule::proportion;
    ruleMap["count"]=rule::count;
    ruleMap["min"]=rule::min;
    ruleMap["max"]=rule::max;
    ruleMap["custom"]=rule::custom;
    ruleMap["mode"]=rule::mode;
    ruleMap["sum"]=rule::sum;
    ruleMap["percentile"]=rule::percentile;
    ruleMap["allpoints"]=rule::allpoints;

    //initialize fieldMap
    std::map<std::string, std::string> fieldMap;
    fieldMap["point"]="point";
    fieldMap["centroid"]="cntrd";
    fieldMap["mean"]="mean";
    fieldMap["stdev"]="stdev";
    fieldMap["median"]="median";
    fieldMap["proportion"]="prop";
    fieldMap["count"]="count";
    fieldMap["min"]="min";
    fieldMap["max"]="max";
    fieldMap["custom"]="custom";
    fieldMap["mode"]="mode";
    fieldMap["sum"]="sum";
    fieldMap["percentile"]="perc";
    fieldMap["allpoints"]="allp";

    statfactory::StatFactory stat;
    if(srcnodata_opt.size()){
      while(srcnodata_opt.size()<bndnodata_opt.size())
        srcnodata_opt.push_back(srcnodata_opt[0]);
      stat.setNoDataValues(srcnodata_opt);
    }
    Jim maskReader;
    if(mask_opt.size()){
      try{
        //todo: open with resampling, resolution and projection according to input
        maskReader.open(mask_opt[0]);
        if(mskband_opt[0]>=maskReader.nrOfBand()){
          string errorString="Error: illegal mask band";
          throw(errorString);
        }
      }
      catch(string error){
        cerr << error << std::endl;
        throw;
      }
    }
    Vector2d<unsigned int> posdata;

    if(output_opt.empty()){
      std::cerr << "No output dataset provided (use option -o). Use --help for help information";
      return(CE_Failure);
    }
    if(plane_opt.empty()){
      size_t iplane=0;
      while(plane_opt.size()<nrOfPlane())
        plane_opt.push_back(iplane++);
    }
    int nplane=plane_opt.size();
    if(nplane>1){
      if(planeNames_opt.size()<nplane){
        planeNames_opt.clear();
        for(size_t iplane=0;iplane<nplane;++iplane){
          int thePlane=plane_opt[iplane];
          ostringstream planestream;
          planestream << "t" << thePlane;
          planeNames_opt.push_back(planestream.str());
        }
      }
    }

    char **papszOptions=NULL;
    for(std::vector<std::string>::const_iterator optionIt=option_opt.begin();optionIt!=option_opt.end();++optionIt)
      papszOptions=CSLAddString(papszOptions,optionIt->c_str());

    bool initWriter=true;
    if(verbose_opt[0])
      std::cout << "Opening ogrWriter: " << output_opt[0] << " in format " << ogrformat_opt[0] << endl;
    try{
      ogrWriter.open(output_opt[0],ogrformat_opt[0],access_opt[0]);
    }
    catch(std::string errorString){
      if(verbose_opt[0])
        std::cout << "initWriter is false" << endl;
      initWriter=false;
    }

    if(verbose_opt[0])
      std::cout << "number of layers: " << sampleReader.getLayerCount() << endl;

    for(int ilayer=0;ilayer<sampleReader.getLayerCount();++ilayer){
      if(verbose_opt[0])
        std::cout << "getLayer " << ilayer << std::endl;
      OGRLayer *readLayer=sampleReader.getLayer(ilayer);
      if(!readLayer){
        ostringstream ess;
        ess << "Error: could not get layer of sampleReader" << endl;
        throw(ess.str());
      }

      OGRSpatialReference thisSpatialRef=getSpatialRef();

      OGRSpatialReference sampleSpatialRef(sampleReader.getSpatialRef());
      OGRCoordinateTransformation *sample2img = OGRCreateCoordinateTransformation(&sampleSpatialRef, &thisSpatialRef);
      OGRCoordinateTransformation *img2sample = OGRCreateCoordinateTransformation(&thisSpatialRef,&sampleSpatialRef);
      if(verbose_opt[0]){
        std::cout << "spatialref of raster: " << this->getFileName() << std::endl;
        thisSpatialRef.dumpReadable();
        std::cout << "spatialref of vector sample: " << std::endl;
        sampleSpatialRef.dumpReadable();
      }
      if(thisSpatialRef.IsSame(&sampleSpatialRef)){
        if(verbose_opt[0])
          std::cout << "spatial reference of vector sample is same as raster" << std::endl;
        sample2img=0;
        img2sample=0;
      }
      else{
        if(verbose_opt[0])
          std::cout << "spatial reference of vector sample is different from raster, img2sample: " << img2sample << std::endl;
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
      //image bounding box in SRS of the input sample vector sampleReader
      double img_ulx,img_uly,img_lrx,img_lry;
      this->getBoundingBox(img_ulx,img_uly,img_lrx,img_lry,img2sample);
      //layer bounding box in SRS of this image raster
      double layer_ulx;
      double layer_uly;
      double layer_lrx;
      double layer_lry;
      if(verbose_opt[0])
        std::cout << "getExtent " << std::endl;
      sampleReader.getExtent(layer_ulx,layer_uly,layer_lrx,layer_lry,ilayer,sample2img);

      if(verbose_opt[0])
        std::cout<< std::setprecision(12) << "--ulx " << layer_ulx << " --uly " << layer_uly << " --lrx " << layer_lrx   << " --lry " << layer_lry << std::endl;
      if(verbose_opt[0])
        std::cout << "covered: " << allCovered_opt[0] << std::endl;
      //check if rule contains allpoints
      if(find(rule_opt.begin(),rule_opt.end(),"allpoints")!=rule_opt.end()){
        rule_opt.clear();
        rule_opt.push_back("allpoints");
        //allpoints should be the only rule
        //rasterize vector sample
        // ImgReaderOgr sampleReader;
        // VectorOgr sampleReader;
        // std::shared_ptr<Jim> sampleMask=Jim::createImg();
        Jim sampleMask;
        // sampleReader.open(sample_opt[0]);
        //layer bounding box in SRS of this image raster
        // double ulx,uly,lrx,lry;
        // sampleReader.getExtent(layer_ulx,layer_uly,layer_lrx,layer_lry,sample2img,ilayer);
        if(layer_ulx<this->getUlx())
          layer_ulx=this->getUlx();
        if(layer_uly>this->getUly())
          layer_uly=this->getUly();
        if(layer_lrx>this->getLrx())
          layer_lrx=this->getLrx();
        if(layer_lry<this->getLry())
          layer_lry=this->getLry();
        // sampleMask->open(this->nrOfCol(),this->nrOfRow(),1,GDT_Float64);
        sampleMask.open(this->nrOfCol(),this->nrOfRow(),1,GDT_Float64);
        double gt[6];
        this->getGeoTransform(gt);
        // sampleMask->setGeoTransform(gt);
        // sampleMask->setProjection(this->getProjection());
        sampleMask.setGeoTransform(gt);
        sampleMask.setProjection(this->getProjection());
        AppFactory anApp;
        anApp.pushLongOption("ulx",layer_ulx);
        anApp.pushLongOption("uly",layer_uly);
        anApp.pushLongOption("lrx",layer_lrx);
        anApp.pushLongOption("lry",layer_lry);
        sampleMask.crop(sampleMask,anApp);
        // sampleMask.crop(sampleMask,layer_ulx,layer_uly,layer_lrx,layer_lry);
        // vector<double> burnValues(1,1);//burn value is 1 (single band)
        // sampleMask.rasterizeBuf(sampleReader,burnValues,eoption_opt);
        double burnValue=1;
        if(label_opt.size())
          burnValue=label_opt[0];
        // sampleMask->rasterizeBuf(sampleReader,burnValue,layer_opt);
        // sampleMask->pushNoDataValue(0);
        // sampleMask->setFile("/vsimem/mask.tif","GTiff");

        //todo:handle projection transform when dealing with masks!
        sampleMask.rasterizeBuf(sampleReader,burnValue);
        sampleMask.pushNoDataValue(0);
        // sampleMask.setFile("/vsimem/mask.tif","GTiff");
        // app.clearOption("s");
        // app.clearOption("sample");
        // app.setLongOption("sample","/vsimem/mask.tif");
        // sampleReader.close();
        app.setLongOption("mem","0");
        app.setLongOption("class",burnValue);
        app.setLongOption("verbose",verbose_opt[0]);
        CPLErr retValue=extractImg(sampleMask,ogrWriter, app);
        sampleMask.close();
        return retValue;
      }

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
          for(int iband=bstart_opt[ipair];iband<=bend_opt[ipair];++iband)
            band_opt.push_back(iband);
        }
      }
      else if(band_opt.empty()){
        size_t iband=0;
        while(band_opt.size()<nrOfBand())
          band_opt.push_back(iband++);
      }
      int nband=(band_opt.size()) ? band_opt.size() : this->nrOfBand();
      if(class_opt.size()){
        if(nband>1){
          cerr << "Warning: using only first band of multiband image" << endl;
          nband=1;
          band_opt.clear();
          band_opt.push_back(0);
          bandNames_opt.retrieveOption(app);
        }
      }

      if(verbose_opt[0]>1)
        std::cout << "Number of bands in input image: " << this->nrOfBand() << std::endl;

      OGRFieldType fieldType;
      int ogr_typecount=11;//hard coded for now!
      if(verbose_opt[0]>1)
        std::cout << "field and label types can be: ";
      for(int iType = 0; iType < ogr_typecount; ++iType){
        if(verbose_opt[0]>1)
          std::cout << " " << OGRFieldDefn::GetFieldTypeName((OGRFieldType)iType);
        if( OGRFieldDefn::GetFieldTypeName((OGRFieldType)iType) != NULL
            && EQUAL(OGRFieldDefn::GetFieldTypeName((OGRFieldType)iType),
                     ftype_opt[0].c_str()))
          fieldType=(OGRFieldType) iType;
      }
      switch( fieldType ){
      case OFTInteger:
      case OFTReal:
      case OFTRealList:
      case OFTString:
        if(verbose_opt[0]>1)
          std::cout << std::endl << "field type is: " << OGRFieldDefn::GetFieldTypeName(fieldType) << std::endl;
        break;
      default:
        cerr << "field type " << OGRFieldDefn::GetFieldTypeName(fieldType) << " not supported" << std::endl;
        return(CE_Failure);
        break;
      }

      const char* pszMessage;
      // void* pProgressArg=NULL;
      // GDALProgressFunc pfnProgress=GDALTermProgress;
      // double progress=0;
      srand(time(NULL));

      bool sampleIsRaster=false;

      // VectorOgr sampleReaderOgr;
      // ImgWriterOgr sampleWriterOgr;
      VectorOgr sampleWriterOgr;

      Vector2d<int> maskBuffer;
      // if(sample_opt.empty()){
      //   string errorString="Error: no sample dataset provided (use option -s). Use --help for help information";
      //   throw(errorString);
      // }


      // ImgWriterOgr ogrWriter;
      // VectorOgr ogrWriter;
      double vectords_ulx;
      double vectords_uly;
      double vectords_lrx;
      double vectords_lry;
      bool calculateSpatialStatistics=false;

      // if(verbose_opt[0])
      //   std::cout << "opening " << output_opt[0] << " for writing output vector dataset" << std::endl;
      // ogrWriter.open(output_opt[0],ogrformat_opt[0]);
      //if class_opt not set, get number of classes from input image for these rules
      for(int irule=0;irule<rule_opt.size();++irule){
        switch(ruleMap[rule_opt[irule]]){
        case(rule::point):
        case(rule::centroid):
        case(rule::allpoints):
          break;
        case(rule::proportion):
        case(rule::count):
        case(rule::custom):
        case(rule::mode):{
          if(class_opt.empty()){
            int theBand=0;
            double minValue=0;
            double maxValue=0;
            if(band_opt.size())
              theBand=band_opt[0];
            this->getMinMax(minValue,maxValue,theBand);
            int nclass=maxValue-minValue+1;
            if(nclass<0&&nclass<256){
              string errorString="Error: Could not automatically define classes, please set class option";
              throw(errorString);
            }
            for(int iclass=minValue;iclass<=maxValue;++iclass)
              class_opt.push_back(iclass);
          }
        }//deliberate fall through: calculate spatial statistics for all non-point like rules
        default:
          calculateSpatialStatistics=true;
          break;
        }
      }

      //support multiple layers
      // int nlayerRead=sampleReaderOgr.getDataSource()->GetLayerCount();
      // if(layer_opt.empty())
      //   layer_opt.push_back(std::string());
      // int nlayerRead=layer_opt.size();
      unsigned long int ntotalvalid=0;

      // if(verbose_opt[0])
      //   std::cout << "opening " << sample_opt[0] << " layer " << layer_opt[0] << " for reading input vector dataset" << std::endl;
      // sampleReaderOgr.open(sample_opt[0],layer_opt[0]);
      // sampleReaderOgr.readFeatures();//now already done when opening

      // OGRLayer *readLayer=sampleReaderOgr.getLayer();

      // double layer_ulx;
      // double layer_uly;
      // double layer_lrx;
      // double layer_lry;
      // if(verbose_opt[0])
      //   std::cout << "getExtent " << std::endl;
      // sampleReader.getExtent(layer_ulx,layer_uly,layer_lrx,layer_lry,sample2img);

      if(verbose_opt[0])
        std::cout<< std::setprecision(12) << "--ulx " << layer_ulx << " --uly " << layer_uly << " --lrx " << layer_lrx   << " --lry " << layer_lry << std::endl;
      bool hasCoverage=((layer_ulx >= this->getUlx())&&(layer_lrx <= this->getLrx())&&(layer_uly <= this->getUly())&&(layer_lry >= this->getLry()));
      if(!hasCoverage){
        std::cerr << "Warning: raster dataset does not fully cover vector layer " << ilayer << endl;
        if(verbose_opt[0])
          std::cerr << std::setprecision(12) << "--ulx " << getUlx() << " --uly " << getUly() << " --lrx " << getLrx()  << " --lry " << getLry() << std::endl;
      }
      if(layer_ulx>this->getLrx()||layer_lrx<this->getUlx()||layer_lry>this->getUly()||layer_uly<this->getLry()){
        std::cerr << "Warning: raster dataset does not fully coverage of vector layer " << ilayer << endl;
        string errorString="Error: no coverage for layer in raster dataset";
        throw(errorString);
      }

      //align bounding box to input image
      layer_ulx-=fmod(layer_ulx-this->getUlx(),this->getDeltaX());
      layer_lrx+=fmod(this->getLrx()-layer_lrx,this->getDeltaX());
      layer_uly+=fmod(this->getUly()-layer_uly,this->getDeltaY());
      layer_lry-=fmod(layer_lry-this->getLry(),this->getDeltaY());

      //do not read outside input image
      if(layer_ulx<this->getUlx())
        layer_ulx=this->getUlx();
      if(layer_lrx>this->getLrx())
        layer_lrx=this->getLrx();
      if(layer_uly>this->getUly())
        layer_uly=this->getUly();
      if(layer_lry<this->getLry())
        layer_lry=this->getLry();

      //read entire block for coverage in memory
      //todo: use different data types
      vector< vector< Vector2d<float> > > readValuesReal(nplane);
      vector< vector< Vector2d<int> > > readValuesInt(nplane);

      for(size_t iplane=0;iplane<nplane;++iplane){
        switch( fieldType ){
        case OFTInteger:
          readValuesInt[iplane].resize(nband);
          break;
        case OFTReal:
          readValuesReal[iplane].resize(nband);
        default:
          break;
        }
      }
      double layer_uli;
      double layer_ulj;
      double layer_lri;
      double layer_lrj;
      this->geo2image(layer_ulx,layer_uly,layer_uli,layer_ulj);
      this->geo2image(layer_lrx,layer_lry,layer_lri,layer_lrj);

      if(verbose_opt[0])
        std::cout << "reading layer geometry" << std::endl;
      OGRwkbGeometryType layerGeometry=readLayer->GetLayerDefn()->GetGeomType();
      if(verbose_opt[0])
        std::cout << "layer geometry: " << OGRGeometryTypeToName(layerGeometry) << std::endl;

      if(layerGeometry==wkbPoint){
        if(calculateSpatialStatistics){
          if(buffer_opt.size()){
            if(buffer_opt[0]<getDeltaX())
              buffer_opt[0]=getDeltaX();
          }
          else
            buffer_opt.push_back(getDeltaX());
        }
      }

      //extend bounding box with buffer
      //todo: check if safety margin is needed?
      if(buffer_opt.size()){
        //in pixels:
        // layer_uli-=buffer_opt[0];
        // layer_ulj-=buffer_opt[0];
        // layer_lri+=buffer_opt[0];
        // layer_lrj+=buffer_opt[0];
        //in geometric units of raster dataset:
        double bufferI=buffer_opt[0]/getDeltaX();
        double bufferJ=buffer_opt[0]/getDeltaY();
        layer_uli-=bufferI;
        layer_ulj-=bufferJ;
        layer_lri+=bufferI;
        layer_lrj+=bufferJ;
      }

      //we already checked there is coverage
      layer_uli=(layer_uli<0)? 0 : static_cast<int>(layer_uli);
      layer_ulj=(layer_ulj<0)? 0 : static_cast<int>(layer_ulj);
      layer_lri=(layer_lri>=this->nrOfCol())? this->nrOfCol()-1 : static_cast<int>(layer_lri);
      layer_lrj=(layer_lrj>=this->nrOfRow())? this->nrOfRow()-1 : static_cast<int>(layer_lrj);

      //todo: separate between case when data has been read already or opened with noRead true;
      // if(m_data.size()){
      bool layerRead=false;
      size_t maxIndexJ=0;
      size_t maxIndexI=0;
      if(getBlockSize()>=layer_lrj-layer_ulj){
        if(verbose_opt[0])
          std::cout << "blockSize " << getBlockSize() << " >= " << layer_lrj-layer_ulj << std::endl;
        for(int iplane=0;iplane<nplane;++iplane){
          int thePlane=plane_opt[iplane];
          for(int iband=0;iband<nband;++iband){
            int theBand=(band_opt.size()) ? band_opt[iband] : iband;
            if(theBand<0){
              string errorString="Error: illegal band (must be positive and starting from 0)";
              throw(errorString);
            }
            if(theBand>=this->nrOfBand()){
              string errorString="Error: illegal band (must be lower than number of bands in input raster dataset)";
              throw(errorString);
            }
            if(verbose_opt[0])
              cout << "reading image band " << theBand << " block rows " << layer_ulj << "-" << layer_lrj << ", cols " << layer_uli << "-" << layer_lri << endl;
            switch( fieldType ){
            case OFTInteger:
              this->readDataBlock3D(readValuesInt[iplane][iband],layer_uli,layer_lri,layer_ulj,layer_lrj,thePlane,theBand);
              break;
            case OFTReal:
            default:
              this->readDataBlock3D(readValuesReal[iplane][iband],layer_uli,layer_lri,layer_ulj,layer_lrj,thePlane,theBand);
              break;
            }
          }
          layerRead=true;
        }
      }
      else{
        if(verbose_opt[0])
          std::cout << "blockSize " << getBlockSize() << " < " << layer_lrj-layer_ulj << std::endl;
      }
      if(maskReader.isInit()){//mask already read when random is set
        if(maskReader.covers(layer_ulx,layer_uly,layer_lrx,layer_lry,"ALL_COVERED"))
          maskReader.readDataBlock(maskBuffer,layer_uli,layer_lri,layer_ulj,layer_lrj,mskband_opt[0]);
        else{
          string errorString="Error: mask does not entirely cover the geographical layer boundaries";
          throw(errorString);
        }
      }


      // float theThreshold=(threshold_opt.size()==layer_opt.size())? threshold_opt[layerIndex]: threshold_opt[0];
      float theThreshold=threshold_opt[0];

      bool createPolygon=true;
      if(find(rule_opt.begin(),rule_opt.end(),"allpoints")!=rule_opt.end())
        createPolygon=false;

      // if(!ogrWriter.getDataset()){
      // OGRLayer *writeLayer;
      try{
        if(initWriter){
          if(verbose_opt[0])
            std::cout << "initWriter is true" << std::endl;
          if(createPolygon){
            //create polygon
            if(verbose_opt[0])
              std::cout << "create polygons" << std::endl;
            if(verbose_opt[0])
              std::cout << "open ogrWriter for polygons (1)" << std::endl;
            ogrWriter.pushLayer(readLayer->GetName(),readLayer->GetSpatialRef(),wkbPolygon,papszOptions);
            // ostringstream fs;
            // fs << "push layer to ogrWriter with polygons failed ";
            // fs << "layer name: "<< readLayer->GetName() << std::endl;
            // throw(fs.str());
            if(verbose_opt[0])
              std::cout << "pushed layer " << readLayer->GetName() << std::endl;
          }
          else{
            if(verbose_opt[0])
              std::cout << "create points in layer " << readLayer->GetName() << std::endl;
            if(verbose_opt[0])
              std::cout << "open ogrWriter for points (1)" << std::endl;
            ogrWriter.pushLayer(readLayer->GetName(),readLayer->GetSpatialRef(),wkbPoint,papszOptions);
            // ostringstream fs;
            // fs << "push layer to ogrWriter with points failed ";
            // fs << "layer name: "<< readLayer->GetName() << std::endl;
            // throw(fs.str());
          }
          if(verbose_opt[0]){
            std::cout << "ogrWriter opened" << std::endl;
            // writeLayer=ogrWriter.createLayer(readLayer->GetName(), this->getProjection(), wkbPoint, papszOptions);
          }
          if(verbose_opt[0]){
            std::cout << "copy fields" << std::endl;
            for(size_t ifield=0;ifield<copyFields_opt.size();++ifield)
              std::cout << "field " << ifield << ": " << copyFields_opt[ifield] << std::endl;
          }
          ogrWriter.copyFields(sampleReader,copyFields_opt,ilayer);

          if(verbose_opt[0])
            std::cout << "create new fields" << std::endl;
          if(label_opt.size()){
            if(verbose_opt[0])
              std::cout << "create label" << std::endl;
            ogrWriter.createField("label",OFTInteger,ilayer);
          }
          if(fid_opt.size()){
            if(verbose_opt[0])
              std::cout << "create fid" << std::endl;
            if(ogrWriter.createField(fid_opt[0],OFTInteger64,ilayer)!=OGRERR_NONE){
              std::string errorString="Error: could not create fid";
              throw(errorString);
            }
            if(verbose_opt[0])
              std::cout << "fid has been created" << std::endl;
          }

          if(verbose_opt[0])
            std::cout << "checking rules" << std::endl;
          for(int irule=0;irule<rule_opt.size();++irule){
            for(int iplane=0;iplane<nplane;++iplane){
              int thePlane=plane_opt[iplane];
                for(int iband=0;iband<nband;++iband){
                int theBand=(band_opt.size()) ? band_opt[iband] : iband;
                ostringstream fs;
                if(bandNames_opt.size()){
                  if(rule_opt.size()>1)
                    fs << fieldMap[rule_opt[irule]];
                  if(planeNames_opt.size())
                    fs << planeNames_opt[iplane];
                  fs << bandNames_opt[iband];
                }
                else{
                  if(rule_opt.size()>1||nband==1)
                    fs << fieldMap[rule_opt[irule]];
                  if(planeNames_opt.size())
                    fs << planeNames_opt[iplane];
                  if(nband>1)
                    fs << "b" << theBand;
                }
                switch(ruleMap[rule_opt[irule]]){
                case(rule::proportion):
                case(rule::count):{//count for each class
                  for(int iclass=0;iclass<class_opt.size();++iclass){
                    ostringstream fsclass;
                    fsclass << fs.str() << "class" << class_opt[iclass];
                    ogrWriter.createField(fsclass.str(),fieldType,ilayer);
                  }
                  break;
                }
                case(rule::percentile):{//for each percentile
                  for(int iperc=0;iperc<percentile_opt.size();++iperc){
                    ostringstream fsperc;
                    fsperc << fs.str() << percentile_opt[iperc];
                    ogrWriter.createField(fsperc.str(),fieldType,ilayer);
                  }
                  break;
                }
                default:
                  ogrWriter.createField(fs.str(),fieldType,ilayer);
                  break;
                }
              }
            }
          }
          if(verbose_opt[0]){
            std::vector<std::string> fieldnames;
            ogrWriter.getFieldNames(fieldnames,ilayer);
            std::cout << "field names of ogrWriter" << std::endl;
            for(size_t ifield=0;ifield<fieldnames.size();++ifield)
              std::cout << fieldnames[ifield] << std::endl;
            std::cout << endl;
            std::cout << "end of initWriter" << std::endl;
          }
        }
        if(verbose_opt[0])
          std::cout << "after initWriter" << std::endl;
      }
      catch(std::string errorString){
        std::cerr << errorString << "failed to initWriter" << std::endl;
        throw;
      }
      OGRFeature *readFeature;
      unsigned long int ifeature=0;
      // unsigned long int nfeatureLayer=sampleReaderOgr.getFeatureCount();
      unsigned long int nfeatureLayer=sampleReader.getFeatureCount(ilayer);
      if(verbose_opt[0])
        std::cout << "nfeatureLayer: " << nfeatureLayer << std::endl;
      unsigned long int ntotalvalidLayer=0;

      ogrWriter.resize(sampleReader.getFeatureCount(ilayer),ilayer);
      // if(verbose_opt[0])
      //   std::cout << "start processing " << sampleReader.getFeatureCount(ilayer) << " features" << std::endl;
      // progress=0;
      // MyProgressFunc(progress,pszMessage,pProgressArg);
      // readLayer->ResetReading();
      // while( (readFeature = readLayer->GetNextFeature()) != NULL ){

#if JIPLIB_PROCESS_IN_PARALLEL == 1
      if(verbose_opt[0])
        std::cout << "process in parallel" << std::endl;
#pragma omp parallel for
#else
#endif
      for(unsigned int ifeature=0;ifeature<sampleReader.getFeatureCount(ilayer);++ifeature){
        // OGRFeature *readFeature=sampleReaderOgr.getFeatureRef(ifeature);
        OGRFeature *readFeature=sampleReader.cloneFeature(ifeature,ilayer);
        bool validFeature=false;
        if(verbose_opt[0]>2)
          std::cout << "reading feature " << readFeature->GetFID() << std::endl;
        if(theThreshold>0){//percentual value
          double p=static_cast<double>(rand())/(RAND_MAX);
          p*=100.0;
          if(p>theThreshold){
            continue;//do not select for now, go to next feature
          }
        }
        else{//absolute value
          if(threshold_opt.size()==sampleReader.getLayerCount()){
            if(ntotalvalidLayer>=-theThreshold){
              continue;//do not select any more pixels, go to next column feature
            }
          }
          else{
            if(ntotalvalid>=-theThreshold){
              continue;//do not select any more pixels, go to next column feature
            }
          }
        }
        if(verbose_opt[0]>2)
          std::cout << "processing feature " << readFeature->GetFID() << std::endl;
        //get x and y from readFeature
        // double x,y;
        OGRGeometry *readGeometry;
        readGeometry = readFeature->GetGeometryRef();
        if(!readGeometry){
          std::string errorString="Error: geometry is empty";
          throw(errorString);
        }
        //create buffer
        OGRGeometry* poGeometry=readGeometry->clone();
        if(!VectorOgr::transform(poGeometry,sample2img)){
          std::string errorString="Error: coordinate transform not successful";
          throw(errorString);
        }
        if(buffer_opt.size())
          poGeometry=poGeometry->Buffer(buffer_opt[0]);
        if(!poGeometry){
          std::string errorString="Error: po geometry is empty";
          throw(errorString);
        }
        try{
          if(wkbFlatten(poGeometry->getGeometryType()) == wkbPoint ){
            //todo: handle case if m_data is empty
            OGRPoint readPoint = *((OGRPoint *) poGeometry);//readPoint is in SRS of raster

            double i_centre,j_centre;
            this->geo2image(readPoint.getX(),readPoint.getY(),i_centre,j_centre);
            //nearest neighbour
            j_centre=static_cast<int>(j_centre);
            i_centre=static_cast<int>(i_centre);

            double uli=i_centre;//-buffer_opt[0];
            double ulj=j_centre;//-buffer_opt[0];
            double lri=i_centre;//+buffer_opt[0];
            double lrj=j_centre;//+buffer_opt[0];

            //nearest neighbour
            ulj=static_cast<int>(ulj);
            uli=static_cast<int>(uli);
            lrj=static_cast<int>(lrj);
            lri=static_cast<int>(lri);

            //check if j is out of bounds
            // if(static_cast<int>(ulj)<0||static_cast<int>(ulj)>=this->nrOfRow())
            if(static_cast<int>(ulj)<0||static_cast<int>(ulj)>=layer_lrj)
              continue;
            //check if j is out of bounds
            // if(static_cast<int>(uli)<0||static_cast<int>(lri)>=this->nrOfCol())
            if(static_cast<int>(uli)<0||static_cast<int>(lri)>=layer_lri)
              continue;

            OGRPoint ulPoint,urPoint,llPoint,lrPoint;
            OGRPolygon writePolygon;
            OGRPoint writePoint;
            OGRLinearRing writeRing;
            OGRFeature *writePolygonFeature;

            int nPointPolygon=0;
            if(createPolygon){
              if(disc_opt[0]&&buffer_opt.size()){
                //todo: check if this still makes sense with new meaning of buffer
                double radius=buffer_opt[0]*sqrt(this->getDeltaX()*this->getDeltaY());
                unsigned short nstep = 25;
                for(int i=0;i<nstep;++i){
                  OGRPoint aPoint(readPoint);
                  // VectorOgr::transform(&aPoint,sample2img);
                  aPoint.setX(aPoint.getX()+this->getDeltaX()/2.0+radius*cos(2*PI*i/nstep));
                  aPoint.setY(aPoint.getY()-this->getDeltaY()/2.0+radius*sin(2*PI*i/nstep));
                  writeRing.addPoint(&aPoint);
                }
                writePolygon.addRing(&writeRing);
                writePolygon.closeRings();
              }
              else{
                double ulx,uly,lrx,lry;
                this->image2geo(uli,ulj,ulx,uly);
                this->image2geo(lri,lrj,lrx,lry);
                ulPoint.setX(ulx-this->getDeltaX()/2.0);
                ulPoint.setY(uly+this->getDeltaY()/2.0);
                lrPoint.setX(lrx+this->getDeltaX()/2.0);
                lrPoint.setY(lry-this->getDeltaY()/2.0);
                urPoint.setX(lrx+this->getDeltaX()/2.0);
                urPoint.setY(uly+this->getDeltaY()/2.0);
                llPoint.setX(ulx-this->getDeltaX()/2.0);
                llPoint.setY(lry-this->getDeltaY()/2.0);

                writeRing.addPoint(&ulPoint);
                writeRing.addPoint(&urPoint);
                writeRing.addPoint(&lrPoint);
                writeRing.addPoint(&llPoint);
                writePolygon.addRing(&writeRing);
                writePolygon.closeRings();
              }
              //coordinate transform
              if(!VectorOgr::transform(&writePolygon,img2sample)){
                std::string errorString="Error: coordinate transform img2sample not successful";
                throw(errorString);
              }
              writePolygonFeature = ogrWriter.createFeature(ilayer);
              // writePolygonFeature = OGRFeature::CreateFeature(writeLayer->GetLayerDefn());
              if(writePolygonFeature->SetFrom(readFeature)!= OGRERR_NONE)
                cerr << "writing feature failed" << std::endl;
              writePolygonFeature->SetGeometry(&writePolygon);
              if(verbose_opt[0]>1)
                std::cout << "copying new fields write polygon " << std::endl;
              if(verbose_opt[0]>1)
                std::cout << "write feature has " << writePolygonFeature->GetFieldCount() << " fields" << std::endl;

              OGRPoint readPoint;//this readPoint is in SRS of vector layer
              if(find(rule_opt.begin(),rule_opt.end(),"centroid")!=rule_opt.end()){
                if(verbose_opt[0]>1)
                  std::cout << "get centroid" << std::endl;
                writePolygon.Centroid(&readPoint);
                double i,j;
                this->geo2image(readPoint.getX(),readPoint.getY(),i,j,sample2img);
                if(verbose_opt[0]>1)
                  std::cout << "centroid in vector SRS: " << readPoint.getX() << ", " << readPoint.getY() << std::endl;
                int indexJ=static_cast<int>(j-layer_ulj);
                int indexI=static_cast<int>(i-layer_uli);
                bool valid=true;
                valid=valid&&(indexJ>=0);
                valid=valid&&(indexJ<this->nrOfRow());
                valid=valid&&(indexI>=0);
                valid=valid&&(indexI<this->nrOfCol());

                if(valid){
                  if(maskReader.isInit()){
                    double maskI,maskJ;
                    maskReader.geo2image(readPoint.getX(),readPoint.getY(),maskI,maskJ,sample2img);
                    maskI=static_cast<unsigned int>(maskI);
                    maskJ=static_cast<unsigned int>(maskJ);
                    if(maskI>0&&maskI<maskBuffer.nrOfCol()&&maskJ>0&&maskJ<maskBuffer.nrOfRow()){
                      for(int ivalue=0;ivalue<msknodata_opt.size();++ivalue){
                        if(maskBuffer[maskJ][maskI]==msknodata_opt[ivalue]){
                          valid=false;
                          break;
                        }
                      }
                    }
                  }
                }
                for(size_t iplane=0;iplane<nplane;++iplane){
                  if(valid){
                    if(srcnodata_opt.empty())
                      validFeature=true;
                    else{
                      for(int vband=0;vband<bndnodata_opt.size();++vband){
                        switch( fieldType ){
                        case OFTInteger:{
                          int value;
                          value=((readValuesInt[iplane][vband])[indexJ])[indexI];
                          if(value==srcnodata_opt[vband])
                            valid=false;
                          break;
                        }
                        case OFTReal:{
                          double value;
                          value=((readValuesReal[iplane][vband])[indexJ])[indexI];
                          if(value==srcnodata_opt[vband])
                            valid=false;
                          break;
                        }
                        }
                        if(!valid)
                          continue;
                        else
                          validFeature=true;
                      }
                    }
                  }
                  // if(valid){
                  if(validFeature){//replace valid with validFeature!
                    if(label_opt.size())
                      writePolygonFeature->SetField("label",label_opt[0]);
                    if(fid_opt.size())
                      writePolygonFeature->SetField(fid_opt[0].c_str(),static_cast<GIntBig>(ifeature));
                    for(size_t iplane=0;iplane<nplane;++iplane){
                      for(int iband=0;iband<nband;++iband){
                        int theBand=(band_opt.size()) ? band_opt[iband] : iband;
                        //write fields for point on surface and centroid
                        string fieldname;
                        ostringstream fs;
                        if(bandNames_opt.size()){
                          if(rule_opt.size()>1)
                            fs << fieldMap["centroid"];
                          if(planeNames_opt.size())
                            fs << planeNames_opt[iplane];
                          fs << bandNames_opt[iband];
                        }
                        else{
                          if(rule_opt.size()>1||nband==1)
                            fs << fieldMap["centroid"];
                          if(planeNames_opt.size())
                            fs << planeNames_opt[iplane];
                          if(nband>1)
                            fs << "b" << theBand;
                        }
                        fieldname=fs.str();
                        switch( fieldType ){
                        case OFTInteger:
                          writePolygonFeature->SetField(fieldname.c_str(),static_cast<int>(((readValuesInt[iplane][iband])[indexJ])[indexI]));
                          break;
                        case OFTReal:
                          writePolygonFeature->SetField(fieldname.c_str(),((readValuesReal[iplane][iband])[indexJ])[indexI]);
                          break;
                        default://not supported
                          std::string errorString="field type not supported";
                          throw(errorString);
                          break;
                        }
                      }
                    }
                  }
                }
              }//if centroid
              if(find(rule_opt.begin(),rule_opt.end(),"point")!=rule_opt.end()){
                if(writePolygon.PointOnSurface(&readPoint)!=OGRERR_NONE)
                  writePolygon.Centroid(&readPoint);
                double i,j;
                this->geo2image(readPoint.getX(),readPoint.getY(),i,j,sample2img);
                int indexJ=static_cast<int>(j-layer_ulj);
                int indexI=static_cast<int>(i-layer_uli);
                bool valid=true;
                valid=valid&&(indexJ>=0);
                valid=valid&&(indexJ<this->nrOfRow());
                valid=valid&&(indexI>=0);
                valid=valid&&(indexI<this->nrOfCol());

                if(valid){
                  if(maskReader.isInit()){
                    double maskI,maskJ;
                    maskReader.geo2image(readPoint.getX(),readPoint.getY(),maskI,maskJ,sample2img);
                    maskI=static_cast<unsigned int>(maskI);
                    maskJ=static_cast<unsigned int>(maskJ);
                    if(maskI>0&&maskI<maskBuffer.nrOfCol()&&maskJ>0&&maskJ<maskBuffer.nrOfRow()){
                      for(int ivalue=0;ivalue<msknodata_opt.size();++ivalue){
                        if(maskBuffer[indexJ][indexI]==msknodata_opt[ivalue]){
                          valid=false;
                          break;
                        }
                      }
                    }
                  }
                }
                if(valid){
                  if(srcnodata_opt.empty())
                    validFeature=true;
                  else{
                    for(size_t iplane=0;iplane<nplane;++iplane){
                      for(int vband=0;vband<bndnodata_opt.size();++vband){
                        switch( fieldType ){
                        case OFTInteger:{
                          int value;
                          value=((readValuesInt[iplane][vband])[indexJ])[indexI];
                          if(value==srcnodata_opt[vband])
                            valid=false;
                          break;
                        }
                        case OFTReal:{
                          double value;
                          value=((readValuesReal[iplane][vband])[indexJ])[indexI];
                          if(value==srcnodata_opt[vband])
                            valid=false;
                          break;
                        }
                        }
                        if(!valid)
                          continue;
                        else
                          validFeature=true;
                      }
                    }
                  }
                }

                // if(valid){
                if(validFeature){//replaced valid with validFeature!
                  if(label_opt.size())
                    writePolygonFeature->SetField("label",label_opt[0]);
                  if(fid_opt.size())
                    writePolygonFeature->SetField(fid_opt[0].c_str(),static_cast<GIntBig>(ifeature));
                  for(size_t iplane=0;iplane<nplane;++iplane){
                    for(int iband=0;iband<nband;++iband){
                      int theBand=(band_opt.size()) ? band_opt[iband] : iband;
                      //write fields for point on surface and centroid
                      string fieldname;
                      ostringstream fs;
                      if(bandNames_opt.size()){
                        if(rule_opt.size()>1)
                          fs << fieldMap["point"];
                        if(planeNames_opt.size())
                          fs << planeNames_opt[iplane];
                        fs << bandNames_opt[iband];
                      }
                      else{
                        if(rule_opt.size()>1||nband==1)
                          fs << fieldMap["point"];
                        if(planeNames_opt.size())
                          fs << planeNames_opt[iplane];
                        if(nband>1)
                          fs << "b" << theBand;
                      }
                      fieldname=fs.str();
                      switch( fieldType ){
                      case OFTInteger:
                        writePolygonFeature->SetField(fieldname.c_str(),static_cast<int>(((readValuesInt[iplane][iband])[indexJ])[indexI]));
                        break;
                      case OFTReal:
                        writePolygonFeature->SetField(fieldname.c_str(),((readValuesReal[iplane][iband])[indexJ])[indexI]);
                        break;
                      default://not supported
                        std::string errorString="field type not supported";
                        throw(errorString);
                        break;
                      }
                    }
                  }
                }
              }//if point
            }//if createPolygon
            if(calculateSpatialStatistics||!createPolygon){
              vector< Vector2d<double> > polyValues(nrOfPlane());
              vector< vector<double> > polyClassValues(nrOfPlane());
              for(size_t iplane=0;iplane<nplane;++iplane){
                if(class_opt.size()){
                  polyClassValues[iplane].resize(class_opt.size());
                  //initialize
                  for(int iclass=0;iclass<class_opt.size();++iclass)
                    polyClassValues[iplane][iclass]=0;
                }
                else
                  polyValues[iplane].resize(nband);
              }

              OGRPoint thePoint;
              for(int j=ulj;j<=lrj;++j){
                for(int i=uli;i<=lri;++i){
                  // //check if within raster image
                  // if(i<0||i>=this->nrOfCol())
                  //   continue;
                  // if(j<0||j>=this->nrOfRow())
                  //   continue;
                  if(j<0){
                    // std::cerr << "Warning: j is " << j << ", setting to 0" << std::endl;
                    j=0;
                  }
                  if(j>=layer_lrj){
                    // std::cerr << "Warning: j is " << j << " and out of reading block, skipping" << std::endl;
                    continue;
                  }
                  if(i<0){
                    // std::cerr << "Warning: i is " << i << ", setting to 0" << std::endl;
                    i=0;
                  }
                  if(i>=layer_lri){
                    // std::cerr << "Warning: i is " << i << " and out of reading block, skipping" << std::endl;
                    continue;
                  }
                  int indexJ=j-layer_ulj;
                  int indexI=i-layer_uli;
                  if(indexJ<0)
                    continue;
                  if(indexI<0)
                    continue;
                  if(indexJ>=this->nrOfRow())
                    continue;
                  if(indexI>=this->nrOfCol())
                    continue;

                  double theX=0;
                  double theY=0;
                  this->image2geo(i,j,theX,theY);
                  thePoint.setX(theX);
                  thePoint.setY(theY);
                  //todo: check if this still makes sense with new meaning of buffer
                  if(disc_opt[0]&&buffer_opt.size()){
                    if(buffer_opt[0]>0){
                      double radius=buffer_opt[0]*sqrt(this->getDeltaX()*this->getDeltaY());
                      if((theX-readPoint.getX())*(theX-readPoint.getX())+(theY-readPoint.getY())*(theY-readPoint.getY())>radius*radius)
                        continue;
                    }
                  }
                  bool valid=true;

                  if(maskReader.isInit()){
                    double maskI,maskJ;
                    maskReader.geo2image(readPoint.getX(),readPoint.getY(),maskI,maskJ,sample2img);
                    maskI=static_cast<unsigned int>(maskI);
                    maskJ=static_cast<unsigned int>(maskJ);
                    if(maskI>0&&maskI<maskBuffer.nrOfCol()&&maskJ>0&&maskJ<maskBuffer.nrOfRow()){
                      for(int ivalue=0;ivalue<msknodata_opt.size();++ivalue){
                        // if(maskBuffer[indexJ][indexI]==msknodata_opt[ivalue]){
                        if(maskBuffer[maskJ][maskI]==msknodata_opt[ivalue]){
                          valid=false;
                          break;
                        }
                      }
                    }
                  }
                  for(size_t iplane=0;iplane<nplane;++iplane){
                    if(srcnodata_opt.size()){
                      for(int vband=0;vband<bndnodata_opt.size();++vband){
                        switch( fieldType ){
                        case OFTInteger:{
                          int value=((readValuesInt[iplane][vband])[indexJ])[indexI];
                          if(value==srcnodata_opt[vband]){
                            valid=false;
                          }
                          break;
                        }
                        default:{
                          float value=((readValuesReal[iplane][vband])[indexJ])[indexI];
                          if(value==srcnodata_opt[vband]){
                            valid=false;
                          }
                          break;
                        }
                        }
                      }
                    }
                    if(!valid){
                      continue;
                    }
                    else
                      validFeature=true;
                  }

                  ++nPointPolygon;
                  OGRFeature *writePointFeature;
                  // if(valid&&!createPolygon){//write all points
                  if(validFeature&&!createPolygon){//write all points: replaced valid with validFeature!
                    if(polythreshold_opt.size()){
                      if(polythreshold_opt[0]>0){
                        double p=static_cast<double>(rand())/(RAND_MAX);
                        p*=100.0;
                        if(p>polythreshold_opt[0])
                          continue;//do not select for now, go to next feature
                      }
                      else if(nPointPolygon>-polythreshold_opt[0])
                        continue;
                    }
                    //create feature
                    writePointFeature = OGRFeature::CreateFeature(ogrWriter.getLayer(ilayer)->GetLayerDefn());
                    // writePointFeature = OGRFeature::CreateFeature(writeLayer->GetLayerDefn());
                    if(verbose_opt[0]>1)
                      std::cout << "copying fields from point feature " << std::endl;
                    if(writePointFeature->SetFrom(readFeature)!= OGRERR_NONE)
                      cerr << "writing feature failed" << std::endl;
                    if(verbose_opt[0]>1)
                      std::cout << "set geometry as point " << std::endl;
                    //coordinate transform
                    if(!VectorOgr::transform(&thePoint,img2sample)){
                      std::string errorString="Error: coordinate transform img2sample not successful";
                      throw(errorString);
                    }
                    writePointFeature->SetGeometry(&thePoint);
                    assert(wkbFlatten(writePointFeature->GetGeometryRef()->getGeometryType()) == wkbPoint);
                    if(verbose_opt[0]>1){
                      std::cout << "write feature has " << writePointFeature->GetFieldCount() << " fields:" << std::endl;
                      for(int iField=0;iField<writePointFeature->GetFieldCount();++iField){
                        std::string fieldname=ogrWriter.getLayer(ilayer)->GetLayerDefn()->GetFieldDefn(iField)->GetNameRef();
                        // std::string fieldname=writeLayer->GetLayerDefn()->GetFieldDefn(iField)->GetNameRef();
                        cout << fieldname << endl;
                      }
                    }
                  }
                  // if(valid&&class_opt.size()){
                  //   short value=0;
                  //   switch( fieldType ){
                  //   case OFTInteger:
                  //     value=((readValuesInt[0])[indexJ])[indexI];
                  //     break;
                  //   case OFTReal:
                  //     value=((readValuesReal[0])[indexJ])[indexI];
                  //     break;
                  //   }
                  //   for(int iclass=0;iclass<class_opt.size();++iclass){
                  //     if(value==class_opt[iclass])
                  //       polyClassValues[iclass]+=1;
                  //   }
                  // }
                  // if(valid){
                  if(validFeature){//replaced valid with validFeature...
                    if(!createPolygon&&label_opt.size())
                      writePointFeature->SetField("label",label_opt[0]);
                    if(!createPolygon&&fid_opt.size())
                      writePointFeature->SetField(fid_opt[0].c_str(),static_cast<GIntBig>(ifeature));
                    for(size_t iplane=0;iplane<nplane;++iplane){
                      for(int iband=0;iband<nband;++iband){
                        int theBand=(band_opt.size()) ? band_opt[iband] : iband;
                        double value=0;
                        switch( fieldType ){
                        case OFTInteger:
                          value=((readValuesInt[iplane][iband])[indexJ])[indexI];
                          break;
                        case OFTReal:
                          value=((readValuesReal[iplane][iband])[indexJ])[indexI];
                          break;
                        }
                        if(!iband&&class_opt.size()){
                          for(int iclass=0;iclass<class_opt.size();++iclass){
                            if(value==class_opt[iclass])
                              polyClassValues[iplane][iclass]+=1;
                          }
                        }

                        if(verbose_opt[0]>1)
                          std::cout << ": " << value << std::endl;
                        if(!createPolygon){//write all points within polygon
                          string fieldname;
                          ostringstream fs;
                          if(bandNames_opt.size()){
                            if(rule_opt.size()>1)
                              fs << fieldMap["allpoints"];
                            if(planeNames_opt.size())
                              fs << planeNames_opt[iplane];
                            fs << bandNames_opt[iband];
                          }
                          else{
                            if(rule_opt.size()>1||nband==1)
                              fs << fieldMap["allpoints"];
                            if(planeNames_opt.size())
                              fs << planeNames_opt[iplane];
                            if(nband>1)
                              fs << "b" << theBand;
                          }
                          fieldname=fs.str();
                          int fieldIndex=writePointFeature->GetFieldIndex(fieldname.c_str());
                          if(fieldIndex<0){
                            ostringstream ess;
                            ess << "field " << fieldname << " was not found" << endl;
                            throw(ess.str());
                            // return(CE_Failure);
                          }
                          if(verbose_opt[0]>1)
                            std::cout << "set field " << fieldname << " to " << value << std::endl;
                          switch( fieldType ){
                          case OFTInteger:
                          case OFTReal:
                            writePointFeature->SetField(fieldname.c_str(),value);
                            break;
                          default://not supported
                            assert(0);
                            break;
                          }
                        }
                        else{
                          polyValues[iplane][iband].push_back(value);
                        }
                      }//iband
                    }
                  }

                  // if(valid&&!createPolygon){
                  if(validFeature&&!createPolygon){//replaced valid with validFeature!
                    //write feature
                    if(verbose_opt[0]>1)
                      std::cout << "creating point feature" << std::endl;
                    // if(writeLayer->CreateFeature( writePointFeature ) != OGRERR_NONE ){
                    if(ogrWriter.getLayer(ilayer)->CreateFeature( writePointFeature ) != OGRERR_NONE ){
                      std::string errorString="Failed to create feature in ogr vector dataset";
                      throw(errorString);
                    }
                    //destroy feature
                    // OGRFeature::DestroyFeature( writePointFeature );
                    ++ntotalvalid;
                    ++ntotalvalidLayer;
                  }
                }//for in i
              }//for int j

              if(createPolygon){
                //do not create if no points found within polygon
                if(!nPointPolygon){
                  if(verbose_opt[0])
                    cout << "no points found in polygon, continuing" << endl;
                  continue;
                }
                //write field attributes to polygon feature
                for(int irule=0;irule<rule_opt.size();++irule){
                  //skip centroid and point
                  if(ruleMap[rule_opt[irule]]==rule::centroid||ruleMap[rule_opt[irule]]==rule::point)
                    continue;
                  if(!irule&&label_opt.size())
                    writePolygonFeature->SetField("label",label_opt[0]);
                  if(!irule&&fid_opt.size())
                    writePolygonFeature->SetField(fid_opt[0].c_str(),static_cast<GIntBig>(ifeature));
                  for(size_t iplane=0;iplane<nplane;++iplane){
                    for(int iband=0;iband<nband;++iband){
                      int theBand=(band_opt.size()) ? band_opt[iband] : iband;
                      vector< vector<double> > theValue(nrOfPlane());
                      vector< vector<string> > fieldname(nrOfPlane());
                      ostringstream fs;
                      if(bandNames_opt.size()){
                        if(rule_opt.size()>1)
                          fs << fieldMap[rule_opt[irule]];
                        if(planeNames_opt.size())
                          fs << planeNames_opt[iplane];
                        fs << bandNames_opt[iband];
                      }
                      else{
                        if(rule_opt.size()>1||nband==1)
                          fs << fieldMap[rule_opt[irule]];
                        if(planeNames_opt.size())
                          fs << planeNames_opt[iplane];
                        if(nband>1)
                          fs << "b" << theBand;
                      }
                      switch(ruleMap[rule_opt[irule]]){
                      case(rule::proportion)://deliberate fall through
                        stat.normalize_pct(polyClassValues[iplane]);
                      case(rule::count):{//count for each class
                        for(int index=0;index<polyClassValues[iplane].size();++index){
                          theValue[iplane].push_back(polyClassValues[iplane][index]);
                          ostringstream fsclass;
                          fsclass << fs.str() << "class" << class_opt[index];
                          fieldname[iplane].push_back(fsclass.str());
                        }
                        break;
                      }
                      case(rule::mode):{
                        //maximum votes in polygon
                        if(verbose_opt[0])
                          std::cout << "number of points in polygon: " << nPointPolygon << std::endl;
                        //search for class with maximum votes
                        int maxClass=stat.mymin(class_opt);
                        vector<double>::iterator maxit;
                        maxit=stat.mymax(polyClassValues[iplane],polyClassValues[iplane].begin(),polyClassValues[iplane].end());
                        int maxIndex=distance(polyClassValues[iplane].begin(),maxit);
                        maxClass=class_opt[maxIndex];
                        if(verbose_opt[0]>0)
                          std::cout << "maxClass: " << maxClass << std::endl;
                        theValue[iplane].push_back(maxClass);
                        fieldname[iplane].push_back(fs.str());
                        break;
                      }
                      case(rule::mean):
                        theValue[iplane].push_back(stat.mean(polyValues[iplane][iband]));
                        fieldname[iplane].push_back(fs.str());
                        break;
                      case(rule::median):
                        theValue[iplane].push_back(stat.median(polyValues[iplane][iband]));
                        fieldname[iplane].push_back(fs.str());
                        break;
                      case(rule::stdev):
                        theValue[iplane].push_back(sqrt(stat.var(polyValues[iplane][iband])));
                        fieldname[iplane].push_back(fs.str());
                        break;
                      case(rule::percentile):{
                        for(int iperc=0;iperc<percentile_opt.size();++iperc){
                          theValue[iplane].push_back(stat.percentile(polyValues[iplane][iband],polyValues[iplane][iband].begin(),polyValues[iplane][iband].end(),percentile_opt[iperc]));
                          ostringstream fsperc;
                          fsperc << fs.str() << percentile_opt[iperc];
                          fieldname[iplane].push_back(fsperc.str());
                        }
                        break;
                      }
                      case(rule::sum):
                        theValue[iplane].push_back(stat.sum(polyValues[iplane][iband]));
                        fieldname[iplane].push_back(fs.str());
                        break;
                      case(rule::max):
                        theValue[iplane].push_back(stat.mymax(polyValues[iplane][iband]));
                        fieldname[iplane].push_back(fs.str());
                        break;
                      case(rule::min):
                        theValue[iplane].push_back(stat.mymin(polyValues[iplane][iband]));
                        fieldname[iplane].push_back(fs.str());
                        break;
                      case(rule::point):
                      case(rule::centroid):
                        theValue[iplane].push_back(polyValues[iplane][iband].back());
                      fieldname[iplane].push_back(fs.str());
                      break;
                      default://not supported
                        break;
                      }
                      for(int ivalue=0;ivalue<theValue[iplane].size();++ivalue){
                        switch( fieldType ){
                        case OFTInteger:
                          writePolygonFeature->SetField(fieldname[iplane][ivalue].c_str(),static_cast<int>(theValue[iplane][ivalue]));
                          break;
                        case OFTReal:
                          writePolygonFeature->SetField(fieldname[iplane][ivalue].c_str(),theValue[iplane][ivalue]);
                          break;
                        case OFTString:
                          writePolygonFeature->SetField(fieldname[iplane][ivalue].c_str(),type2string<double>(theValue[iplane][ivalue]).c_str());
                          break;
                        default://not supported
                          std::string errorString="field type not supported";
                          throw(errorString);
                          break;
                        }
                      }
                    }
                  }
                }
              }
            }
            if(createPolygon&&validFeature){
              // if(createPolygon){
              //write polygon feature
              //todo: create only in case of valid feature
              if(verbose_opt[0]>1)
                std::cout << "creating polygon feature (1)" << std::endl;
              // if(writeLayer->CreateFeature( writePolygonFeature ) != OGRERR_NONE ){
              //   std::string errorString="Failed to create polygon feature in ogr vector dataset";
              //   throw(errorString);
              // }
              //test: no need to destroy anymore?
              // OGRFeature::DestroyFeature( writePolygonFeature );
              //make sure to use setFeature instead of pushFeature when in processing in parallel!!!
              if(!writePolygonFeature)
                std::cerr << "Warning: NULL feature" << ifeature << std::endl;
              ogrWriter.setFeature(ifeature,writePolygonFeature,ilayer);
              ++ntotalvalid;
              ++ntotalvalidLayer;
            }
          }//for points
          else{//(multi-)polygons
            OGRPolygon readPolygon;//readPolygon is in SRS of raster dataset
            OGRMultiPolygon readMultiPolygon;//readMultiPolygon is in SRS of raster dataset

            //get envelope
            OGREnvelope* psEnvelope=new OGREnvelope();


            if(wkbFlatten(poGeometry->getGeometryType()) == wkbPolygon){
              readPolygon = *((OGRPolygon *) poGeometry);
              readPolygon.closeRings();
              readPolygon.getEnvelope(psEnvelope);
            }
            else if(wkbFlatten(poGeometry->getGeometryType()) == wkbMultiPolygon){
              readMultiPolygon = *((OGRMultiPolygon *) poGeometry);
              readMultiPolygon.closeRings();
              readMultiPolygon.getEnvelope(psEnvelope);
            }
            else{
              ostringstream oss;
              oss << "geometry " << static_cast<std::string>(poGeometry->getGeometryName()) << " not supported";
              throw(oss.str());
            }

            double ulx,uly,lrx,lry;
            double uli,ulj,lri,lrj;
            ulx=psEnvelope->MinX;
            uly=psEnvelope->MaxY;
            lrx=psEnvelope->MaxX;
            lry=psEnvelope->MinY;
            delete psEnvelope;

            //check if feature is covered by input raster dataset
            if(!this->covers(ulx,uly,lrx,lry,allCovered_opt[0])){
              if(verbose_opt[0]>2){
                std::cerr << "Warning: raster does not cover polygon: " << readFeature->GetFID()  <<  std::endl;
                std::cerr << "Envelope polygon in SRS of raster:" << std::endl;
                std::cout << std::setprecision(12) << "--ulx " << ulx << " --uly " << uly << " --lrx " << lrx   << " --lry " << lry << std::endl;
                std::cerr << "Bounding box of raster:" << std::endl;
                std::cerr << std::setprecision(12) << "--ulx " << getUlx() << " --uly " << getUly() << " --lrx " << getLrx()   << " --lry " << getLry() << std::endl;
              }
              continue;
            }
            else if(verbose_opt[0]>2){
              std::cout << "raster does cover polygon: " << readFeature->GetFID()  <<  std::endl;
              std::cout << "bounding box of polygon in SRS of raster: " << readFeature->GetFID()  <<  std::endl;
              std::cout<< std::setprecision(12) << "--ulx " << ulx << " --uly " << uly<< " --lrx " << lrx   << " --lry " << lry << std::endl;
            }
            // if(!this->covers(ulx,uly,lrx,lry,allCovered_opt[0])){
            //   if(verbose_opt[0]>1)
            //     std::cout << "polygon not covered, skipping" << std::endl;
            //   continue;
            // }

            if(!layerRead){//todo (implementation not finished)
              ostringstream oss;
              oss << "implementation not finished";
              throw(oss.str());

              if(verbose_opt[0]>1)
                std::cout << "read data within polygon" << std::endl;
              Jim blockRaster;
              AppFactory anApp;
              anApp.pushLongOption("ulx",ulx);
              anApp.pushLongOption("uly",uly);
              anApp.pushLongOption("lrx",lrx);
              anApp.pushLongOption("lry",lry);
              cropDS(blockRaster,anApp);
              switch( fieldType ){
              case OFTInteger:
                anApp.pushLongOption("otype","Int32");
                cropDS(blockRaster,anApp);
                // for(int iband=0;iband<nband;++iband){
                //   int theBand=(band_opt.size()) ? band_opt[iband] : iband;
                //   blockRaster.readDataBlock(readValuesInt[iband],theBand);
                // }
                break;
              case OFTReal:
              default:
                anApp.pushLongOption("otype","Float32");
                cropDS(blockRaster,anApp);
                // for(int iband=0;iband<nband;++iband){
                //   int theBand=(band_opt.size()) ? band_opt[iband] : iband;
                //   blockRaster.readDataBlock(readValuesReal[iband],theBand);
                // }
                break;
              }
            }
            else{
              if(verbose_opt[0]>1)
                std::cout << "layer has been read already" << std::endl;
            }
            OGRFeature *writePolygonFeature;
            int nPointPolygon=0;
            if(createPolygon){
              if(verbose_opt[0]>2)
                std::cout << "writePolygonFeature in ogrWriter for layer " << ilayer << std::endl;
              // writePolygonFeature = OGRFeature::CreateFeature(writeLayer->GetLayerDefn());
              writePolygonFeature = ogrWriter.createFeature(ilayer);
              //coordinate transform
              if(!VectorOgr::transform(poGeometry,img2sample)){
                std::string errorString="Error: coordinate transform img2sample not successful";
                throw(errorString);
              }
              //writePolygonFeature and readFeature are both of type wkbPolygon
              if(writePolygonFeature->SetFrom(readFeature)!= OGRERR_NONE)
                cerr << "writing feature failed" << std::endl;
              //uncomment if we want to get buffered geometry
              // writePolygonFeature->SetGeometry(poGeometry);
              if(verbose_opt[0]>1)
                std::cout << "copying new fields write polygon " << std::endl;
              if(verbose_opt[0]>1)
                std::cout << "write polygon feature has " << writePolygonFeature->GetFieldCount() << " fields" << std::endl;
            }

            OGRPoint readPoint;//readPoint is in SRS of raster dataset
            if(find(rule_opt.begin(),rule_opt.end(),"centroid")!=rule_opt.end()){
              if(verbose_opt[0]>1)
                std::cout << "get centroid" << std::endl;
              if(wkbFlatten(poGeometry->getGeometryType()) == wkbPolygon)
                readPolygon.Centroid(&readPoint);
              else if(wkbFlatten(poGeometry->getGeometryType()) == wkbMultiPolygon)
                readMultiPolygon.Centroid(&readPoint);

              double i,j;
              // blockRaster.geo2image(readPoint.getX(),readPoint.getY(),i,j,sample2img)
              this->geo2image(readPoint.getX(),readPoint.getY(),i,j);
              if(verbose_opt[0]>1)
                std::cout << "centroid in raster SRS: " << readPoint.getX() << ", " << readPoint.getY() << std::endl;
              int indexJ=static_cast<int>(j-layer_ulj);
              int indexI=static_cast<int>(i-layer_uli);
              // if(m_data.empty()){
              //   //todo: check
              //   int indexJ-=polygon_ulj;
              //   int indexI-=polygon_uli;
              // }
              bool valid=true;
              valid=valid&&(indexJ>=0);
              valid=valid&&(indexJ<this->nrOfRow());
              valid=valid&&(indexI>=0);
              valid=valid&&(indexI<this->nrOfCol());
              if(valid){
                if(maskReader.isInit()){
                  double maskI,maskJ;
                  maskReader.geo2image(readPoint.getX(),readPoint.getY(),maskI,maskJ);
                  maskI=static_cast<unsigned int>(maskI);
                  maskJ=static_cast<unsigned int>(maskJ);
                  if(maskI>0&&maskI<maskBuffer.nrOfCol()&&maskJ>0&&maskJ<maskBuffer.nrOfRow()){
                    for(int ivalue=0;ivalue<msknodata_opt.size();++ivalue){
                      if(maskBuffer[indexJ][indexI]==msknodata_opt[ivalue]){
                        valid=false;
                        break;
                      }
                    }
                  }
                }
              }
              if(valid){
                if(srcnodata_opt.empty())
                  validFeature=true;
                else{
                  for(size_t iplane=0;iplane<nplane;++iplane){
                    for(int vband=0;vband<bndnodata_opt.size();++vband){
                      switch( fieldType ){
                      case OFTInteger:{
                        int value;
                        value=((readValuesInt[iplane][vband])[indexJ])[indexI];
                        if(value==srcnodata_opt[vband])
                          valid=false;
                        break;
                      }
                      case OFTReal:{
                        double value;
                        value=((readValuesReal[iplane][vband])[indexJ])[indexI];
                        if(value==srcnodata_opt[vband])
                          valid=false;
                        break;
                      }
                      }
                      if(!valid)
                        continue;
                      else
                        validFeature=true;
                    }
                  }
                }
              }
              // if(valid){
              if(validFeature){//replace valid with validFeature!
                if(label_opt.size())
                  writePolygonFeature->SetField("label",label_opt[0]);
                if(fid_opt.size())
                  writePolygonFeature->SetField(fid_opt[0].c_str(),static_cast<GIntBig>(ifeature));
                for(size_t iplane=0;iplane<nplane;++iplane){
                  for(int iband=0;iband<nband;++iband){
                    int theBand=(band_opt.size()) ? band_opt[iband] : iband;
                    //write fields for point on surface and centroid
                    string fieldname;
                    ostringstream fs;
                    if(bandNames_opt.size()){
                      if(rule_opt.size()>1)
                        fs << fieldMap["centroid"];
                      if(planeNames_opt.size())
                        fs << planeNames_opt[iplane];
                      fs << bandNames_opt[iband];
                    }
                    else{
                      if(rule_opt.size()>1||nband==1)
                        fs << fieldMap["centroid"];
                      if(planeNames_opt.size())
                        fs << planeNames_opt[iplane];
                      if(nband>1)
                        fs << "b" << theBand;
                    }
                    fieldname=fs.str();
                    switch( fieldType ){
                    case OFTInteger:
                      writePolygonFeature->SetField(fieldname.c_str(),static_cast<int>(((readValuesInt[iplane][iband])[indexJ])[indexI]));
                      break;
                    case OFTReal:
                      writePolygonFeature->SetField(fieldname.c_str(),((readValuesReal[iplane][iband])[indexJ])[indexI]);
                      break;
                    default://not supported
                      std::string errorString="field type not supported";
                      throw(errorString);
                      break;
                    }
                  }
                }
              }
            }
            if(find(rule_opt.begin(),rule_opt.end(),"point")!=rule_opt.end()){
              if(wkbFlatten(poGeometry->getGeometryType()) == wkbPolygon){
                if(readPolygon.PointOnSurface(&readPoint)!=OGRERR_NONE){
                  if(verbose_opt[0]>1)
                    std::cout << "get centroid from readPolygon" << std::endl << std::flush;
                  readPolygon.Centroid(&readPoint);
                }
              }
              else if(wkbFlatten(poGeometry->getGeometryType()) == wkbMultiPolygon){
                // if(readMultiPolygon.PointOnSurface(&readPoint)!=OGRERR_NONE)
                readMultiPolygon.Centroid(&readPoint);
              }
              double i,j;
              this->geo2image(readPoint.getX(),readPoint.getY(),i,j);
              int indexJ=static_cast<int>(j-layer_ulj);
              int indexI=static_cast<int>(i-layer_uli);
              bool valid=true;
              valid=valid&&(indexJ>=0);
              valid=valid&&(indexJ<this->nrOfRow());
              valid=valid&&(indexI>=0);
              valid=valid&&(indexI<this->nrOfCol());
              if(valid){
                if(maskReader.isInit()){
                  double maskI,maskJ;
                  maskReader.geo2image(readPoint.getX(),readPoint.getY(),maskI,maskJ);
                  maskI=static_cast<unsigned int>(maskI);
                  maskJ=static_cast<unsigned int>(maskJ);
                  if(maskI>0&&maskI<maskBuffer.nrOfCol()&&maskJ>0&&maskJ<maskBuffer.nrOfRow()){
                    for(int ivalue=0;ivalue<msknodata_opt.size();++ivalue){
                      if(maskBuffer[indexJ][indexI]==msknodata_opt[ivalue]){
                        valid=false;
                        break;
                      }
                    }
                  }
                }
              }
              for(size_t iplane=0;iplane<nplane;++iplane){
                if(valid){
                  if(srcnodata_opt.empty())
                    validFeature=true;
                  else{
                    for(int vband=0;vband<bndnodata_opt.size();++vband){
                      switch( fieldType ){
                      case OFTInteger:{
                        int value;
                        value=((readValuesInt[iplane][vband])[indexJ])[indexI];
                        if(value==srcnodata_opt[vband])
                          valid=false;
                        break;
                      }
                      case OFTReal:{
                        double value;
                        value=((readValuesReal[iplane][vband])[indexJ])[indexI];
                        if(value==srcnodata_opt[vband])
                          valid=false;
                        break;
                      }
                      }
                      if(!valid)
                        continue;
                      else
                        validFeature=true;
                    }
                  }
                }
              }
              // if(valid){
              if(validFeature){//replace valid with validFeature!
                if(label_opt.size()){
                  writePolygonFeature->SetField("label",label_opt[0]);
                }
                if(fid_opt.size()){
                  writePolygonFeature->SetField(fid_opt[0].c_str(),static_cast<GIntBig>(ifeature));
                }
                for(size_t iplane=0;iplane<nplane;++iplane){
                  for(int iband=0;iband<nband;++iband){
                    int theBand=(band_opt.size()) ? band_opt[iband] : iband;
                    //write fields for point on surface and centroid
                    string fieldname;
                    ostringstream fs;
                    if(bandNames_opt.size()){
                      if(rule_opt.size()>1)
                        fs << fieldMap["point"];
                      if(planeNames_opt.size())
                        fs << planeNames_opt[iplane];
                      fs << bandNames_opt[iband];
                    }
                    else{
                      if(rule_opt.size()>1||nband==1)
                        fs << fieldMap["point"];
                      if(planeNames_opt.size())
                        fs << planeNames_opt[iplane];
                      if(nband>1)
                        fs << "b" << theBand;
                    }
                    fieldname=fs.str();
                    switch( fieldType ){
                    case OFTInteger:
                      writePolygonFeature->SetField(fieldname.c_str(),static_cast<int>(((readValuesInt[iplane][iband])[indexJ])[indexI]));
                      break;
                    case OFTReal:
                      writePolygonFeature->SetField(fieldname.c_str(), ((readValuesReal[iplane][iband])[indexJ])[indexI]);
                      break;
                    default://not supported
                      std::string errorString="field type not supported";
                      throw(errorString);
                      break;
                    }
                  }
                }
              }
            }
            if(calculateSpatialStatistics||ruleMap[rule_opt[0]]==rule::allpoints){
              this->geo2image(ulx,uly,uli,ulj);
              this->geo2image(lrx,lry,lri,lrj);
              if(verbose_opt[0]>1){
                std::cout << "calculated uli=" << uli << "and ulj" << ulj << " from ulx=" << ulx << "and uly=" << uly << std::endl;
                std::cout << "calculated lri=" << lri << "and lrj" << lrj << " from lrx=" << lrx << "and lry=" << lry << std::endl;
                std::cout << "bounding box for polygon feature " << ifeature << ": " << uli << " " << ulj << " " << lri << " " << lrj << std::endl;
              }
              //nearest neighbour
              ulj=static_cast<int>(ulj);
              uli=static_cast<int>(uli);
              lrj=static_cast<int>(lrj);
              lri=static_cast<int>(lri);
              //iterate through all pixels
              if(verbose_opt[0]>1)
                std::cout << "bounding box for polygon feature " << ifeature << ": " << uli << " " << ulj << " " << lri << " " << lrj << std::endl;

              if(uli<0)
                uli=0;
              if(lri<0)
                lri=0;
              // if(uli>=this->nrOfCol())
              //   uli=this->nrOfCol()-1;
              if(uli>=layer_lri)
                uli=layer_lri-1;
              // if(lri>=this->nrOfCol())
              //   lri=this->nrOfCol()-1;
              if(lri>=layer_lri)
                lri=layer_lri-1;
              if(ulj<0)
                ulj=0;
              if(lrj<0)
                lrj=0;
              // if(ulj>=this->nrOfRow())
              //   ulj=this->nrOfRow()-1;
              if(ulj>=layer_lrj)
                ulj=layer_lrj-1;
              // if(lrj>=this->nrOfRow())
              //   lrj=this->nrOfRow()-1;
              if(lrj>=layer_lrj)
                lrj=layer_lrj-1;

              if(verbose_opt[0]>1)
                std::cout << "bounding box for polygon feature after check " << ifeature << ": " << uli << " " << ulj << " " << lri << " " << lrj << std::endl;
              vector< Vector2d<double> > polyValues(nrOfPlane());
              vector< vector<double> > polyClassValues(nrOfPlane());

              for(size_t iplane=0;iplane<nplane;++iplane){
                if(class_opt.size()){
                  polyClassValues[iplane].resize(class_opt.size());
                  //initialize
                  for(int iclass=0;iclass<class_opt.size();++iclass)
                    polyClassValues[iplane][iclass]=0;
                }
                else
                  polyValues[iplane].resize(nband);
              }

              OGRPoint thePoint;//in SRS of raster dataset
              for(int j=ulj;j<=lrj;++j){
                for(int i=uli;i<=lri;++i){
                  // //check if within raster image
                  // if(i<0||i>=this->nrOfCol())
                  //   continue;
                  // if(j<0||j>=this->nrOfRow())
                  //   continue;
                  //check if within read block of raster image
                  if(j<0){
                    std::cerr << "Warning: j is " << j << ", setting to 0" << std::endl;
                    j=0;
                  }
                  if(j>=layer_lrj){
                    std::cerr << "Warning: j is " << j << " and out of reading block, skipping" << std::endl;
                    continue;
                  }
                  if(i<0){
                    std::cerr << "Warning: i is " << i << ", setting to 0" << std::endl;
                    i=0;
                  }
                  if(i>=layer_lri){
                    std::cerr << "Warning: i is " << i << " and out of reading block, skipping" << std::endl;
                    continue;
                  }
                  int indexJ=j-layer_ulj;
                  int indexI=i-layer_uli;
                  if(indexJ<0||indexJ>=this->nrOfRow()||indexI<0||indexI>=this->nrOfCol())
                    continue;
                  double theX=0;
                  double theY=0;
                  this->image2geo(i,j,theX,theY);
                  thePoint.setX(theX);
                  thePoint.setY(theY);
                  //check if point is on surface
                  if(wkbFlatten(poGeometry->getGeometryType()) == wkbPolygon){
                    if(!readPolygon.Contains(&thePoint))//readPolygon is in SRS of raster dataset
                      continue;
                  }
                  else if(wkbFlatten(poGeometry->getGeometryType()) == wkbMultiPolygon){
                    if(!readMultiPolygon.Contains(&thePoint))//readMultiPolygon is in SRS of raster dataset
                      continue;
                  }

                  bool valid=true;
                  double maskI,maskJ;
                  if(maskReader.isInit()){
                    maskReader.geo2image(readPoint.getX(),readPoint.getY(),maskI,maskJ);
                    maskI=static_cast<unsigned int>(maskI);
                    maskJ=static_cast<unsigned int>(maskJ);
                    if(maskI>0&&maskI<maskBuffer.nrOfCol()&&maskJ>0&&maskJ<maskBuffer.nrOfRow()){
                      for(int ivalue=0;ivalue<msknodata_opt.size();++ivalue){
                        if(maskBuffer[indexJ][indexI]==msknodata_opt[ivalue]){
                          valid=false;
                          break;
                        }
                      }
                    }
                  }
                  for(size_t iplane=0;iplane<nplane;++iplane){
                    if(srcnodata_opt.size()){
                      for(int vband=0;vband<bndnodata_opt.size();++vband){
                        switch( fieldType ){
                        case OFTInteger:{
                          int value=((readValuesInt[iplane][vband])[indexJ])[indexI];
                          if(value==srcnodata_opt[vband]){
                            valid=false;
                          }
                          break;
                        }
                        default:{
                          float value=((readValuesReal[iplane][vband])[indexJ])[indexI];
                          if(value==srcnodata_opt[vband]){
                            valid=false;
                          }
                          break;
                        }
                        }
                      }
                    }
                    if(!valid)
                      continue;
                    else
                      validFeature=true;
                  }

                  if(verbose_opt[0]>1)
                    std::cout << "point is on surface: " << thePoint.getX() << "," << thePoint.getY() << std::endl;
                  ++nPointPolygon;

                  OGRFeature *writePointFeature;
                  if(!createPolygon){//write all points within polygon
                    if(verbose_opt[0]>1)
                      std::cout << "do not create polygon" << std::endl;
                    if(polythreshold_opt.size()){
                      if(polythreshold_opt[0]>0){
                        double p=static_cast<double>(rand())/(RAND_MAX);
                        p*=100.0;
                        if(p>polythreshold_opt[0])
                          continue;//do not select for now, go to next feature
                      }
                      else if(nPointPolygon>-polythreshold_opt[0])
                        continue;
                    }
                    // if(polythreshold_opt.size())
                    //   if(nPointPolygon>=polythreshold_opt[0])
                    //     continue;
                    // if(threshold_opt.size()){
                    //   if(threshold_opt[0]<=0){
                    //     if(ntotalvalid>=-threshold_opt[0])
                    //       continue;
                    //   }
                    //   else if(threshold_opt[0]<100){
                    //     double p=static_cast<double>(rand())/(RAND_MAX);
                    //     p*=100.0;
                    //     if(p>threshold_opt[0]){
                    //       continue;//do not select for now, go to next feature
                    //     }
                    //   }
                    // }
                    //create feature
                    writePointFeature = OGRFeature::CreateFeature(ogrWriter.getLayer(ilayer)->GetLayerDefn());
                    // writePointFeature = OGRFeature::CreateFeature(writeLayer->GetLayerDefn());
                    if(verbose_opt[0]>1)
                      std::cout << "copying fields from polygons " << std::endl;
                    if(writePointFeature->SetFrom(readFeature)!= OGRERR_NONE)
                      cerr << "writing feature failed" << std::endl;
                    if(verbose_opt[0]>1)
                      std::cout << "set geometry as point in SRS of vector layer" << std::endl;
                    VectorOgr::transform(&thePoint,img2sample);
                    writePointFeature->SetGeometry(&thePoint);
                    assert(wkbFlatten(writePointFeature->GetGeometryRef()->getGeometryType()) == wkbPoint);
                    if(verbose_opt[0]>1){
                      std::cout << "write point feature has " << writePointFeature->GetFieldCount() << " fields:" << std::endl;
                      for(int iField=0;iField<writePointFeature->GetFieldCount();++iField){
                        std::string fieldname=ogrWriter.getLayer(ilayer)->GetLayerDefn()->GetFieldDefn(iField)->GetNameRef();
                        // std::string fieldname=writeLayer->GetLayerDefn()->GetFieldDefn(iField)->GetNameRef();
                        cout << fieldname << endl;
                      }
                    }
                  }
                  // if(class_opt.size()){
                  //   short value=0;
                  //   switch( fieldType ){
                  //   case OFTInteger:
                  //     value=((readValuesInt[0])[indexJ])[indexI];
                  //     break;
                  //   case OFTReal:
                  //     value=((readValuesReal[0])[indexJ])[indexI];
                  //     break;
                  //   }
                  //   for(int iclass=0;iclass<class_opt.size();++iclass){
                  //     if(value==class_opt[iclass])
                  //       polyClassValues[iclass]+=1;
                  //   }
                  // }
                  // else{

                  if(!createPolygon&&label_opt.size()){
                    if(verbose_opt[0]>1)
                      std::cout << "set field label" << std::endl;
                    writePointFeature->SetField("label",label_opt[0]);
                  }

                  if(!createPolygon&&fid_opt.size())
                    writePointFeature->SetField(fid_opt[0].c_str(),static_cast<GIntBig>(ifeature));
                  for(size_t iplane=0;iplane<nplane;++iplane){
                    for(int iband=0;iband<nband;++iband){
                      int theBand=(band_opt.size()) ? band_opt[iband] : iband;
                      double value=0;
                      switch( fieldType ){
                      case OFTInteger:
                        value=((readValuesInt[iplane][iband])[indexJ])[indexI];
                        break;
                      case OFTReal:
                        value=((readValuesReal[iplane][iband])[indexJ])[indexI];
                        break;
                      }
                      if(!iband&&class_opt.size()){
                        for(int iclass=0;iclass<class_opt.size();++iclass){
                          if(value==class_opt[iclass])
                            polyClassValues[iplane][iclass]+=1;
                        }
                      }

                      if(verbose_opt[0]>1)
                        std::cout << ": " << value << std::endl;
                      if(!createPolygon){//write all points within polygon
                        string fieldname;
                        ostringstream fs;
                        if(bandNames_opt.size()){
                          if(rule_opt.size()>1)
                            fs << fieldMap["allpoints"];
                          if(planeNames_opt.size())
                            fs << planeNames_opt[iplane];
                          fs << bandNames_opt[iband];
                        }
                        else{
                          if(rule_opt.size()>1||nband==1)
                            fs << fieldMap["allpoints"];
                          if(planeNames_opt.size())
                            fs << planeNames_opt[iplane];
                          if(nband>1)
                            fs << "b" << theBand;
                        }
                        fieldname=fs.str();
                        int fieldIndex=writePointFeature->GetFieldIndex(fieldname.c_str());
                        if(fieldIndex<0){
                          ostringstream ess;
                          ess << "field " << fieldname << " was not found" << endl;
                          throw(ess.str());
                          // cerr << "field " << fieldname << " was not found" << endl;
                          // return(CE_Failure);
                        }
                        if(verbose_opt[0]>1)
                          std::cout << "set field " << fieldname << " to " << value << std::endl;
                        switch( fieldType ){
                        case OFTInteger:
                        case OFTReal:
                          writePointFeature->SetField(fieldname.c_str(),value);
                          break;
                        default://not supported
                          assert(0);
                          break;
                        }
                      }
                      else{
                        polyValues[iplane][iband].push_back(value);
                      }
                    }//iband
                  }//iplane

                  if(!createPolygon){
                    //todo: only if valid feature?
                    //write feature
                    if(verbose_opt[0])
                      std::cout << "creating point feature " << ntotalvalidLayer << std::endl;
                    // if(writeLayer->CreateFeature( writePointFeature ) != OGRERR_NONE ){
                    if(ogrWriter.getLayer(ilayer)->CreateFeature( writePointFeature ) != OGRERR_NONE ){
                      std::string errorString="Failed to create feature in ogr vector dataset";
                      throw(errorString);
                    }
                    //destroy feature
                    // OGRFeature::DestroyFeature( writePointFeature );
                    ++ntotalvalid;
                    ++ntotalvalidLayer;
                  }
                }//for in i
              }//for int j
              if(createPolygon){
                //do not create if no points found within polygon
                if(!nPointPolygon){
                  if(verbose_opt[0])
                    cout << "no points found in polygon, continuing" << endl;
                  continue;
                }
                //write field attributes to polygon feature
                for(int irule=0;irule<rule_opt.size();++irule){
                  //skip centroid and point
                  if(ruleMap[rule_opt[irule]]==rule::centroid||ruleMap[rule_opt[irule]]==rule::point)
                    continue;
                  if(!irule&&label_opt.size())
                    writePolygonFeature->SetField("label",label_opt[0]);
                  if(!irule&&fid_opt.size())
                    writePolygonFeature->SetField(fid_opt[0].c_str(),static_cast<GIntBig>(ifeature));
                  for(size_t iplane=0;iplane<nplane;++iplane){
                    for(int iband=0;iband<nband;++iband){
                      int theBand=(band_opt.size()) ? band_opt[iband] : iband;
                      vector< vector<double> > theValue(nrOfPlane());
                      vector< vector<string> > fieldname(nrOfPlane());
                      ostringstream fs;
                      if(bandNames_opt.size()){
                        if(rule_opt.size()>1)
                          fs << fieldMap[rule_opt[irule]];
                        if(planeNames_opt.size())
                          fs << planeNames_opt[iplane];
                        fs << bandNames_opt[iband];
                      }
                      else{
                        if(rule_opt.size()>1||nband==1)
                          fs << fieldMap[rule_opt[irule]];
                        if(planeNames_opt.size())
                          fs << planeNames_opt[iplane];
                        if(nband>1)
                          fs << "b" << theBand;
                      }
                      switch(ruleMap[rule_opt[irule]]){
                      case(rule::proportion):
                        stat.normalize_pct(polyClassValues[iplane]);
                      case(rule::count):{//count for each class
                        for(int index=0;index<polyClassValues[iplane].size();++index){
                          theValue[iplane].push_back(polyClassValues[iplane][index]);
                          ostringstream fsclass;
                          fsclass << fs.str() << "class" << class_opt[index];
                          fieldname[iplane].push_back(fsclass.str());
                        }
                        break;
                      }
                      case(rule::mode):{
                        //maximum votes in polygon
                        if(verbose_opt[0])
                          std::cout << "number of points in polygon: " << nPointPolygon << std::endl;
                        //search for class with maximum votes
                        int maxClass=stat.mymin(class_opt);
                        vector<double>::iterator maxit;
                        maxit=stat.mymax(polyClassValues[iplane],polyClassValues[iplane].begin(),polyClassValues[iplane].end());
                        int maxIndex=distance(polyClassValues[iplane].begin(),maxit);
                        maxClass=class_opt[maxIndex];
                        if(verbose_opt[0]>0)
                          std::cout << "maxClass: " << maxClass << std::endl;
                        theValue[iplane].push_back(maxClass);
                        fieldname[iplane].push_back(fs.str());
                        break;
                      }
                      case(rule::mean):
                        theValue[iplane].push_back(stat.mean(polyValues[iplane][iband]));
                        fieldname[iplane].push_back(fs.str());
                        break;
                      case(rule::median):
                        theValue[iplane].push_back(stat.median(polyValues[iplane][iband]));
                        fieldname[iplane].push_back(fs.str());
                        break;
                      case(rule::stdev):
                        theValue[iplane].push_back(sqrt(stat.var(polyValues[iplane][iband])));
                        fieldname[iplane].push_back(fs.str());
                        break;
                      case(rule::percentile):{
                        for(int iperc=0;iperc<percentile_opt.size();++iperc){
                          theValue[iplane].push_back(stat.percentile(polyValues[iplane][iband],polyValues[iplane][iband].begin(),polyValues[iplane][iband].end(),percentile_opt[iperc]));
                          ostringstream fsperc;
                          fsperc << fs.str() << percentile_opt[iperc];
                          fieldname[iplane].push_back(fsperc.str());
                        }
                        break;
                      }
                      case(rule::sum):
                        theValue[iplane].push_back(stat.sum(polyValues[iplane][iband]));
                        fieldname[iplane].push_back(fs.str());
                        break;
                      case(rule::max):
                        theValue[iplane].push_back(stat.mymax(polyValues[iplane][iband]));
                        fieldname[iplane].push_back(fs.str());
                        break;
                      case(rule::min):
                        theValue[iplane].push_back(stat.mymin(polyValues[iplane][iband]));
                        fieldname[iplane].push_back(fs.str());
                        break;
                      case(rule::centroid):
                      case(rule::point):
                        theValue[iplane].push_back(polyValues[iplane][iband].back());
                      fieldname[iplane].push_back(fs.str());
                      break;
                      default://not supported
                        break;
                      }
                      for(int ivalue=0;ivalue<theValue[iplane].size();++ivalue){
                        switch( fieldType ){
                        case OFTInteger:
                          writePolygonFeature->SetField(fieldname[iplane][ivalue].c_str(),static_cast<int>(theValue[iplane][ivalue]));
                          break;
                        case OFTReal:
                          writePolygonFeature->SetField(fieldname[iplane][ivalue].c_str(),theValue[iplane][ivalue]);
                          break;
                        case OFTString:
                          writePolygonFeature->SetField(fieldname[iplane][ivalue].c_str(),type2string<double>(theValue[iplane][ivalue]).c_str());
                          break;
                        default://not supported
                          std::string errorString="field type not supported";
                          throw(errorString);
                          break;
                        }
                      }
                    }
                  }
                }
              }
            }
            if(createPolygon&&validFeature){
              //todo: only create if valid feature?
              //write polygon feature
              if(verbose_opt[0]>1)
                std::cout << "creating polygon feature (2)" << std::endl;
              // if(writeLayer->CreateFeature( writePolygonFeature ) != OGRERR_NONE ){
              //   std::string errorString="Failed to create polygon feature in ogr vector dataset";
              //   throw(errorString);
              // }
              //test: no need to destroy anymore?
              // OGRFeature::DestroyFeature( writePolygonFeature );
              //make sure to use setFeature instead of pushFeature when in processing in parallel!!!
              if(verbose_opt[0]){
                std::cout << "set feature " << ifeature << " with address " << writePolygonFeature << " from " << ogrWriter.getFeatureCount() << "in layer " << ilayer << std::endl;
                std::cout << "writePolygonFeature: " << writePolygonFeature << " (" << ifeature << " from " << sampleReader.getFeatureCount(ilayer) << ")" << std::endl;
              }
              if(!writePolygonFeature)
                std::cerr << "Warning: NULL feature" << ifeature << std::endl;
              ogrWriter.setFeature(ifeature,writePolygonFeature,ilayer);
              ++ntotalvalid;
              ++ntotalvalidLayer;
              if(verbose_opt[0])
                std::cout << "ntotalvalidLayer: " << ntotalvalidLayer << std::endl;
            }
          }
          // ++ifeature;
          // if(theThreshold>0){
          //   if(threshold_opt.size()==sampleReader.getLayerCount())
          //     progress=(100.0/theThreshold)*static_cast<float>(ntotalvalidLayer)/nfeatureLayer;
          //   else
          //     progress=static_cast<float>(ntotalvalidLayer)/nfeatureLayer;
          // }
          // else
          //   progress=static_cast<float>(ifeature+1)/(-theThreshold);
          // MyProgressFunc(progress,pszMessage,pProgressArg);
        }
        catch(std::string e){
          std::cout << e << std::endl;
          continue;
        }
        catch(int npoint){
          if(verbose_opt[0])
            std::cout << "number of points read in polygon: " << npoint << std::endl;
          continue;
        }
        catch(...){
          std::cout << "Error: something went wrong in extractOgr" << std::endl;
        }
      }
      // if(rbox_opt[0]>0||cbox_opt[0]>0)
      //   boxWriter.close();
      // progress=1.0;
      // MyProgressFunc(progress,pszMessage,pProgressArg);
      if(verbose_opt[0])
        std::cout << "number of valid points in layer: " << ntotalvalidLayer << std::endl;
      if(verbose_opt[0])
        std::cout << "number of valid points in all layers: " << ntotalvalid<< std::endl;
    }//for int ilayer

    if(papszOptions)
      CSLDestroy(papszOptions);

    if(maskReader.isInit()){
      maskReader.close();
    }
    return(CE_None);
  }
  catch(string predefinedString){
    std::cout << predefinedString << std::endl;
    throw;
  }
}

/**
 * @param app application specific option arguments
 *
 * @return CE_None if success, CE_Failure if failure
 */
//todo: support multiple layers for writing
CPLErr Jim::extractSample(VectorOgr& ogrWriter, AppFactory& app){
  // Optionjl<string> image_opt("i", "input", "Raster input dataset containing band information");
  // Optionjl<string> sample_opt("s", "sample", "OGR vector dataset with features to be extracted from input data. Output will contain features with input band information included.");
  Optionjl<string> layer_opt("ln", "ln", "Layer name of output vector dataset");
  Optionjl<unsigned int> random_opt("rand", "random", "Create simple random sample of points. Provide number of points to generate");
  Optionjl<double> grid_opt("grid", "grid", "Create systematic grid of points. Provide cell grid size (in projected units, e.g,. m)");
  Optionjl<string> output_opt("o", "output", "Output sample dataset");
  Optionjl<int> label_opt("label", "label", "Create extra label field with this value");
  Optionjl<std::string> fid_opt("fid", "fid", "Create extra field with field identifier (sequence in which the features have been read");
  Optionjl<int> class_opt("c", "class", "Class(es) to extract from input sample image. Leave empty to extract all valid data pixels from sample dataset. Make sure to set classes if rule is set to mode, proportion or count");
  Optionjl<float> threshold_opt("t", "threshold", "Probability threshold for selecting samples (randomly). Provide probability in percentage (>0) or absolute (<0). Use a single threshold per vector sample layer.  Use value 100 to select all pixels for selected class(es)", 100);
  Optionjl<double> percentile_opt("perc","perc","Percentile value(s) used for rule percentile",95);
  Optionjl<string> ogrformat_opt("f", "oformat", "Output sample dataset format","SQLite");
  Optionjl<std::string> option_opt("co", "co", "Creation option for output file. Multiple options can be specified.");
  Optionjl<string> ftype_opt("ft", "ftype", "Field type (only Real or Integer)", "Real");
  Optionjl<int> band_opt("b", "band", "Band index(es) to extract (0 based). Leave empty to use all bands");
  Optionjl<std::string> bandNames_opt("bn", "bandname", "Band name(s) corresponding to band index(es). Leave empty to use all bands");
  Optionjl<unsigned short> bstart_opt("sband", "startband", "Start band sequence number");
  Optionjl<unsigned short> bend_opt("eband", "endband", "End band sequence number");
  Optionjl<string> rule_opt("r", "rule", "Rule how to report image information per feature. point (single point within polygon), allpoints (all points within polygon), centroid, mean, stdev, median, proportion, count, min, max, mode, sum, percentile.","centroid");
  Optionjl<double> srcnodata_opt("srcnodata", "srcnodata", "Invalid value(s) for input image");
  Optionjl<int> bndnodata_opt("bndnodata", "bndnodata", "Band(s) in input image to check if pixel is valid (used for srcnodata)", 0);
  Optionjl<string> mask_opt("m", "mask", "Use the the specified file as a validity mask (0 is nodata).");
  Optionjl<double> msknodata_opt("msknodata", "msknodata", "Mask value not to consider for crop.", 0);
  Optionjl<unsigned int> mskband_opt("mskband", "mskband", "Mask band to read (0 indexed)", 0);
  Optionjl<float> polythreshold_opt("tp", "thresholdPolygon", "(absolute) threshold for selecting samples in each polygon");
  Optionjl<short> buffer_opt("buf", "buffer", "Buffer for calculating statistics for point features (in geometric units of raster dataset) ");
  Optionjl<bool> disc_opt("circ", "circular", "Use a circular disc kernel buffer (for vector point sample datasets only, use in combination with buffer option)", false);
  Optionjl<std::string> allCovered_opt("cover", "cover", "Which polygons to include based on coverage (ALL_TOUCHED, ALL_COVERED)", "ALL_TOUCHED");
  Optionjl<unsigned long int>  memory_opt("mem", "mem", "Buffer size (in MB) to read image data blocks in memory",0,1);
  Optionjl<short> verbose_opt("v", "verbose", "Verbose mode if > 0", 0,2);

  bstart_opt.setHide(1);
  bend_opt.setHide(1);
  bndnodata_opt.setHide(1);
  srcnodata_opt.setHide(1);
  mask_opt.setHide(1);
  msknodata_opt.setHide(1);
  mskband_opt.setHide(1);
  polythreshold_opt.setHide(1);
  percentile_opt.setHide(1);
  buffer_opt.setHide(1);
  disc_opt.setHide(1);
  allCovered_opt.setHide(1);
  option_opt.setHide(1);
  memory_opt.setHide(1);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    // doProcess=sample_opt.retrieveOption(app);
    layer_opt.retrieveOption(app);
    doProcess=output_opt.retrieveOption(app);
    random_opt.retrieveOption(app);
    grid_opt.retrieveOption(app);
    class_opt.retrieveOption(app);
    label_opt.retrieveOption(app);
    fid_opt.retrieveOption(app);
    threshold_opt.retrieveOption(app);
    percentile_opt.retrieveOption(app);
    ogrformat_opt.retrieveOption(app);
    option_opt.retrieveOption(app);
    ftype_opt.retrieveOption(app);
    band_opt.retrieveOption(app);
    bandNames_opt.retrieveOption(app);
    bstart_opt.retrieveOption(app);
    bend_opt.retrieveOption(app);
    rule_opt.retrieveOption(app);
    bndnodata_opt.retrieveOption(app);
    srcnodata_opt.retrieveOption(app);
    mask_opt.retrieveOption(app);
    msknodata_opt.retrieveOption(app);
    mskband_opt.retrieveOption(app);
    polythreshold_opt.retrieveOption(app);
    buffer_opt.retrieveOption(app);
    disc_opt.retrieveOption(app);
    allCovered_opt.retrieveOption(app);
    memory_opt.retrieveOption(app);
    verbose_opt.retrieveOption(app);
    if(!doProcess){
      cout << endl;
      std::ostringstream helpStream;
      helpStream << "short option -h shows basic options only, use long option --help to show all options" << std::endl;
      throw(helpStream.str());//help was invoked, stop processing
    }

    // std::vector<std::string> badKeys;
    // app.badKeys(badKeys);
    // if(badKeys.size()){
    //   std::ostringstream errorStream;
    //   if(badKeys.size()>1)
    //     errorStream << "Error: unknown keys: ";
    //   else
    //     errorStream << "Error: unknown key: ";
    //   for(int ikey=0;ikey<badKeys.size();++ikey){
    //     errorStream << badKeys[ikey] << " ";
    //   }
    //   errorStream << std::endl;
    //   throw(errorStream.str());
    // }

    //initialize ruleMap
    std::map<std::string, rule::RULE_TYPE> ruleMap;
    ruleMap["point"]=rule::point;
    ruleMap["centroid"]=rule::centroid;
    ruleMap["mean"]=rule::mean;
    ruleMap["stdev"]=rule::stdev;
    ruleMap["median"]=rule::median;
    ruleMap["proportion"]=rule::proportion;
    ruleMap["count"]=rule::count;
    ruleMap["min"]=rule::min;
    ruleMap["max"]=rule::max;
    ruleMap["custom"]=rule::custom;
    ruleMap["mode"]=rule::mode;
    ruleMap["sum"]=rule::sum;
    ruleMap["percentile"]=rule::percentile;
    ruleMap["allpoints"]=rule::allpoints;

    //initialize fieldMap
    std::map<std::string, std::string> fieldMap;
    fieldMap["point"]="point";
    fieldMap["centroid"]="cntrd";
    fieldMap["mean"]="mean";
    fieldMap["stdev"]="stdev";
    fieldMap["median"]="median";
    fieldMap["proportion"]="prop";
    fieldMap["count"]="count";
    fieldMap["min"]="min";
    fieldMap["max"]="max";
    fieldMap["custom"]="custom";
    fieldMap["mode"]="mode";
    fieldMap["sum"]="sum";
    fieldMap["percentile"]="perc";
    fieldMap["allpoints"]="allp";

    statfactory::StatFactory stat;
    if(srcnodata_opt.size()){
      while(srcnodata_opt.size()<bndnodata_opt.size())
        srcnodata_opt.push_back(srcnodata_opt[0]);
      stat.setNoDataValues(srcnodata_opt);
    }
    Jim maskReader;
    if(mask_opt.size()){
      try{
        //todo: open with resampling, resolution and projection according to input
        maskReader.open(mask_opt[0]);
        if(mskband_opt[0]>=maskReader.nrOfBand()){
          string errorString="Error: illegal mask band";
          throw(errorString);
        }
      }
      catch(string error){
        cerr << error << std::endl;
        throw;
      }
    }
    Vector2d<unsigned int> posdata;

    if(output_opt.empty()){
      std::cerr << "No output dataset provided (use option -o). Use --help for help information";
      return(CE_Failure);
    }

    //check if rule contains allpoints
    if(find(rule_opt.begin(),rule_opt.end(),"allpoints")!=rule_opt.end()){
      string errorstring="Error: allpoints not supported for random and grid sampling";
      throw(errorstring);
    }
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
        for(int iband=bstart_opt[ipair];iband<=bend_opt[ipair];++iband)
          band_opt.push_back(iband);
      }
    }
    else if(band_opt.empty()){
      size_t iband=0;
      while(band_opt.size()<nrOfBand())
        band_opt.push_back(iband++);
    }
    int nband=(band_opt.size()) ? band_opt.size() : this->nrOfBand();
    if(class_opt.size()){
      if(nband>1){
        cerr << "Warning: using only first band of multiband image" << endl;
        nband=1;
        band_opt.clear();
        band_opt.push_back(0);
      }
    }

    if(verbose_opt[0]>1)
      std::cout << "Number of bands in input image: " << this->nrOfBand() << std::endl;

    OGRFieldType fieldType;
    int ogr_typecount=11;//hard coded for now!
    if(verbose_opt[0]>1)
      std::cout << "field and label types can be: ";
    for(int iType = 0; iType < ogr_typecount; ++iType){
      if(verbose_opt[0]>1)
        std::cout << " " << OGRFieldDefn::GetFieldTypeName((OGRFieldType)iType);
      if( OGRFieldDefn::GetFieldTypeName((OGRFieldType)iType) != NULL
          && EQUAL(OGRFieldDefn::GetFieldTypeName((OGRFieldType)iType),
                   ftype_opt[0].c_str()))
        fieldType=(OGRFieldType) iType;
    }
    switch( fieldType ){
    case OFTInteger:
    case OFTReal:
    case OFTRealList:
    case OFTString:
      if(verbose_opt[0]>1)
        std::cout << std::endl << "field type is: " << OGRFieldDefn::GetFieldTypeName(fieldType) << std::endl;
      break;
    default:
      cerr << "field type " << OGRFieldDefn::GetFieldTypeName(fieldType) << " not supported" << std::endl;
      return(CE_Failure);
      break;
    }

    const char* pszMessage;
    void* pProgressArg=NULL;
    GDALProgressFunc pfnProgress=GDALTermProgress;
    double progress=0;
    srand(time(NULL));

    bool sampleIsRaster=false;

    VectorOgr sampleReaderOgr;
    // ImgWriterOgr sampleWriterOgr;
    VectorOgr sampleWriterOgr;

    Vector2d<int> maskBuffer;
    // sampleWriterOgr.open("/vsimem/virtual",ogrformat_opt[0]);

    char **papszOptions=NULL;
    for(std::vector<std::string>::const_iterator optionIt=option_opt.begin();optionIt!=option_opt.end();++optionIt)
      papszOptions=CSLAddString(papszOptions,optionIt->c_str());

    char **vsiOptions=NULL;
    std::string overwriteOption="OVERWRITE=YES";
    vsiOptions=CSLAddString(vsiOptions,overwriteOption.c_str());
    std::string vsifn;
    if(random_opt.size()){
      //create simple random sampling within boundary
      double ulx,uly,lrx,lry;
      this->getBoundingBox(ulx,uly,lrx,lry);
      vsifn="/vsimem/random.sqlite";
      if(random_opt[0]>0){
        if(verbose_opt[0])
          std::cout << "Opening " << vsifn << endl;
        try{
          sampleWriterOgr.open(vsifn,"SQLite");
        }
        catch(std::string errorString){
          std::cerr << errorString << endl;
          std::cerr << "Warning: could not open sampleWriterOgr" << endl;
          throw;
          //todo: check if error handling is needed
        }
        if(verbose_opt[0])
          std::cout << "Pushing layer random in " << vsifn << endl;
        sampleWriterOgr.pushLayer("random",this->getProjectionRef(),wkbPoint,vsiOptions);
        if(verbose_opt[0])
          std::cout << "Pushed layer random in " << vsifn << endl;
        // sampleWriterOgr.open(vsifn,"random","ESRI Shapefile", "wkbPoint");
      }
      if(maskReader.isInit()){
        if(verbose_opt[0])
          std::cout << "read data block from maskReader" << std::endl;
        double maskULI,maskULJ,maskLRI,maskLRJ;
        maskReader.geo2image(ulx,uly,maskULI,maskULJ);
        maskReader.geo2image(lrx,lry,maskLRI,maskLRJ);
        maskULI=(maskULI>0)? maskULI : 0;
        maskULJ=(maskULJ>0)? maskULJ : 0;
        maskLRI=(maskLRI<maskReader.nrOfCol())? maskLRI : maskReader.nrOfCol()-1;
        maskLRJ=(maskLRJ<maskReader.nrOfRow())? maskLRJ : maskReader.nrOfRow()-1;
        maskReader.readDataBlock(maskBuffer,maskULI,maskLRI,maskULJ,maskLRJ,mskband_opt[0]);
      }
      OGRPoint pt;
      unsigned int ipoint=0;
      unsigned int trials=0;
      double maxTrials=nrOfCol()*nrOfRow();
      size_t outOfRegion=0;
      while(ipoint<random_opt[0]&&trials<=maxTrials){
        ++trials;
        double theX=ulx+static_cast<double>(rand())/(RAND_MAX)*(lrx-ulx);
        double theY=uly-static_cast<double>(rand())/(RAND_MAX)*(uly-lry);
        //check if point is valid
        bool valid=true;
        //test
        double maskI,maskJ;
        if(maskReader.isInit()){
          maskReader.geo2image(theX,theY,maskI,maskJ);
          maskI=static_cast<unsigned int>(maskI);
          maskJ=static_cast<unsigned int>(maskJ);
          if(maskI<0||maskI>=maskBuffer.nrOfCol()){
            ++outOfRegion;
            continue;
          }
          if(maskJ<0||maskJ>=maskBuffer.nrOfRow()){
            ++outOfRegion;
            continue;
          }
          for(int ivalue=0;ivalue<msknodata_opt.size();++ivalue){
            if(maskBuffer[maskJ][maskI]==msknodata_opt[ivalue]){
              valid=false;
              break;
            }
          }
        }
        if(!valid)
          continue;
        else
          ++ipoint;
        pt.setX(theX);
        pt.setY(theY);
        // OGRFeature *pointFeature;
        OGRFeature *pointFeature=sampleWriterOgr.createFeature();
        // pointFeature=sampleWriterOgr.createFeature();
        pointFeature->SetGeometry( &pt );
        sampleWriterOgr.pushFeature(pointFeature);
        // if(sampleWriterOgr.createFeature(pointFeature) != OGRERR_NONE ){
        //   string errorString="Error: failed to create feature in vector dataset";
        //   throw(errorString);
        // }
        // OGRFeature::DestroyFeature(pointFeature);
      }
      if(verbose_opt[0])
        std::cout << "out of region: " << outOfRegion << " from " << trials << " trials"<< std::endl;
      if(!ipoint){
        ostringstream ess;
        ess << "Error: no random point created afer " << trials << " trials" << endl;
        throw(ess.str());
      }
      else if(trials>=maxTrials)
        std::cerr << "Warning: maximum number of trials reached (" << maxTrials << ")" << std::endl;
      else if(verbose_opt[0])
        std::cout << "number of points created from " << trials << " trials: " << ipoint << std::endl;
      sampleWriterOgr.write();
      sampleWriterOgr.close();
      // sample_opt.push_back(vsifn);
    }
    else if(grid_opt.size()){
      bool initSample=true;
      //create systematic grid of points
      double ulx,uly,lrx,lry;
      this->getBoundingBox(ulx,uly,lrx,lry);
      vsifn="/vsimem/grid.sqlite";
      if(uly-grid_opt[0]/2<lry&&ulx+grid_opt[0]/2>lrx){
        string errorString="Error: grid distance too large";
        throw(errorString);
      }
      else if(grid_opt[0]>0){
        if(verbose_opt[0])
          std::cout << "Opening " << vsifn << endl;
        try{
          sampleWriterOgr.open(vsifn,"SQLite");
        }
        catch(std::string errorString){
          initSample=false;
          std::cerr << errorString << endl;
          std::cerr << "Warning: could not open sampleWriterOgr" << endl;
          //todo: check if error handling is needed
        }
        if(verbose_opt[0])
          std::cout << "Pushing layer grid in " << vsifn << endl;
        if(initSample){
          sampleWriterOgr.pushLayer("grid",this->getProjectionRef(),wkbPoint,vsiOptions);
        }
        if(verbose_opt[0])
          std::cout << "Pushed layer grid in " << vsifn << endl;
        // sampleWriterOgr.open(vsifn,"grid","ESRI Shapefile", "wkbPoint");
        // sampleWriterOgr.createLayer("points", this->getProjection(), wkbPoint, papszOptions);
      }
      else{
        string errorString="Error: grid distance must be strictly positive number";
        throw(errorString);
      }
      OGRPoint pt;
      unsigned int ipoint=0;
      for(double theY=uly-grid_opt[0]/2;theY>lry;theY-=grid_opt[0]){
        for(double theX=ulx+grid_opt[0]/2;theX<lrx;theX+=grid_opt[0]){
          OGRFeature *pointFeature;
          pointFeature=sampleWriterOgr.createFeature();
          pt.setX(theX);
          pt.setY(theY);
          pointFeature->SetGeometry( &pt );
          sampleWriterOgr.pushFeature(pointFeature);
          // if(sampleWriterOgr.createFeature(pointFeature) != OGRERR_NONE ){
          //   string errorString="Failed to create feature in vector dataset";
          //   throw(errorString);
          // }
          ++ipoint;
          // OGRFeature::DestroyFeature(pointFeature);
        }
      }
      if(!ipoint){
        string errorString="Error: no points created in grid";
        throw(errorString);
      }
      sampleWriterOgr.write();
      sampleWriterOgr.close();
      // sample_opt.push_back(vsifn);
    }
    else{
      string errorString="Error: no random nor grid option provided. Use --help for help information";
      throw(errorString);
    }
    // if(verbose_opt[0])
    //   std::cout << "number of features in sample: " << sampleReaderOgr.getFeatureCount() << std::endl;


    // ImgWriterOgr ogrWriter;
    // VectorOgr ogrWriter;
    double vectords_ulx;
    double vectords_uly;
    double vectords_lrx;
    double vectords_lry;
    bool calculateSpatialStatistics=false;

    // if(verbose_opt[0])
    //   std::cout << "opening " << output_opt[0] << " for writing output vector dataset" << std::endl;
    // ogrWriter.open(output_opt[0],ogrformat_opt[0]);
    //if class_opt not set, get number of classes from input image for these rules
    for(int irule=0;irule<rule_opt.size();++irule){
      switch(ruleMap[rule_opt[irule]]){
      case(rule::point):
      case(rule::centroid):
      case(rule::allpoints):
        break;
      case(rule::proportion):
      case(rule::count):
      case(rule::custom):
      case(rule::mode):{
        if(class_opt.empty()){
          int theBand=0;
          double minValue=0;
          double maxValue=0;
          if(band_opt.size())
            theBand=band_opt[0];
          this->getMinMax(minValue,maxValue,theBand);
          int nclass=maxValue-minValue+1;
          if(nclass<0&&nclass<256){
            string errorString="Error: Could not automatically define classes, please set class option";
            throw(errorString);
          }
          for(int iclass=minValue;iclass<=maxValue;++iclass)
            class_opt.push_back(iclass);
        }
      }//deliberate fall through: calculate spatial statistics for all non-point like rules
      default:
        calculateSpatialStatistics=true;
        break;
      }
    }

    //support multiple layers
    // int nlayerRead=sampleReaderOgr.getDataSource()->GetLayerCount();
    // if(layer_opt.empty())
    //   layer_opt.push_back(std::string());
    int nlayerRead=1;
    unsigned long int ntotalvalid=0;

    if(verbose_opt[0])
      std::cout << "opening " << vsifn << " for reading input vector dataset" << std::endl;
    sampleReaderOgr.open(vsifn);
    // sampleReaderOgr.readFeatures();//now already done when opening

    OGRLayer *readLayer=sampleReaderOgr.getLayer();

    double layer_ulx;
    double layer_uly;
    double layer_lrx;
    double layer_lry;
    sampleReaderOgr.getExtent(layer_ulx,layer_uly,layer_lrx,layer_lry);
    if(verbose_opt[0])
      std::cout << "--ulx " << layer_ulx<< " --uly " << layer_uly<< " --lrx " << layer_lrx   << " --lry " << layer_lry << std::endl;
    // bool hasCoverage=((layer_ulx >= this->getLrx())&&(layer_lrx <= this->getUlx())&&(layer_lry >= this->getUly())&&(layer_uly <= this->getLry()));
    bool hasCoverage=this->covers(layer_ulx,layer_uly,layer_lrx,layer_lry);
    if(!hasCoverage)
      std::cerr << "Warning: No coverage for layer in " << vsifn << endl;

    //align bounding box to input image
    layer_ulx-=fmod(layer_ulx-this->getUlx(),this->getDeltaX());
    layer_lrx+=fmod(this->getLrx()-layer_lrx,this->getDeltaX());
    layer_uly+=fmod(this->getUly()-layer_uly,this->getDeltaY());
    layer_lry-=fmod(layer_lry-this->getLry(),this->getDeltaY());

    //do not read outside input image
    if(layer_ulx<this->getUlx())
      layer_ulx=this->getUlx();
    if(layer_lrx>this->getLrx())
      layer_lrx=this->getLrx();
    if(layer_uly>this->getUly())
      layer_uly=this->getUly();
    if(layer_lry<this->getLry())
      layer_lry=this->getLry();

    //read entire block for coverage in memory
    //todo: use different data types
    vector< Vector2d<float> > readValuesReal(nband);
    vector< Vector2d<int> > readValuesInt(nband);

    double layer_uli;
    double layer_ulj;
    double layer_lri;
    double layer_lrj;
    this->geo2image(layer_ulx,layer_uly,layer_uli,layer_ulj);
    this->geo2image(layer_lrx,layer_lry,layer_lri,layer_lrj);

    if(verbose_opt[0])
      std::cout << "reading layer geometry" << std::endl;
    OGRwkbGeometryType layerGeometry=readLayer->GetLayerDefn()->GetGeomType();
    if(verbose_opt[0])
      std::cout << "layer geometry: " << OGRGeometryTypeToName(layerGeometry) << std::endl;

    if(layerGeometry==wkbPoint){
      if(verbose_opt[0])
        std::cout << "layerGeometry is wkbPoint" << std::endl;
      if(calculateSpatialStatistics){
        if(verbose_opt[0])
          std::cout << "calculateSpatialStatistics is true" << std::endl;
        // if(buffer_opt.size()){
        //   if(buffer_opt[0]<getDeltaX())
        //     buffer_opt[0]=getDeltaX();
        // }
        // else
        //   buffer_opt.push_back(getDeltaX());
      }
    }

    //extend bounding box with buffer
    if(buffer_opt.size()){
      layer_uli-=buffer_opt[0]/getDeltaX();
      layer_ulj-=buffer_opt[0]/getDeltaY();
      layer_lri+=buffer_opt[0]/getDeltaX();
      layer_lrj+=buffer_opt[0]/getDeltaY();
    }

    //we already checked there is coverage
    layer_uli=(layer_uli<0)? 0 : static_cast<int>(layer_uli);
    layer_ulj=(layer_ulj<0)? 0 : static_cast<int>(layer_ulj);
    layer_lri=(layer_lri>=this->nrOfCol())? this->nrOfCol()-1 : static_cast<int>(layer_lri);
    layer_lrj=(layer_lrj>=this->nrOfRow())? this->nrOfRow()-1 : static_cast<int>(layer_lrj);

    for(int iband=0;iband<nband;++iband){
      int theBand=(band_opt.size()) ? band_opt[iband] : iband;
      if(theBand<0){
        ostringstream ess;
        ess << "Error: illegal band " << theBand << " (must be positive and starting from 0)" << endl;
        throw(ess.str());
      }
      if(theBand>=this->nrOfBand()){
        ostringstream ess;
        ess << "Error: illegal band " << theBand << " (must be lower than number of bands in input raster dataset)" << endl;
        throw(ess.str());
      }
      switch( fieldType ){
      case OFTInteger:
        this->readDataBlock(readValuesInt[iband],layer_uli,layer_lri,layer_ulj,layer_lrj,theBand);
        break;
      case OFTReal:
      default:
        this->readDataBlock(readValuesReal[iband],layer_uli,layer_lri,layer_ulj,layer_lrj,theBand);
        break;
      }
    }
    if(maskReader.isInit()&&random_opt.empty()){//mask already read when random is set
      if(maskReader.covers(layer_ulx,layer_uly,layer_lrx,layer_lry,"ALL_COVERED"))
        maskReader.readDataBlock(maskBuffer,layer_uli,layer_lri,layer_ulj,layer_lrj,mskband_opt[0]);
      else{
        string errorString="Error: mask does not entirely cover the geographical layer boundaries";
        throw(errorString);
      }
    }


    // float theThreshold=(threshold_opt.size()==layer_opt.size())? threshold_opt[layerIndex]: threshold_opt[0];
    float theThreshold=threshold_opt[0];

    bool createPolygon=true;
    if(find(rule_opt.begin(),rule_opt.end(),"allpoints")!=rule_opt.end())
      createPolygon=false;

    // OGRLayer *writeLayer;
    std::string layername=(layer_opt.size())? layer_opt[0] : readLayer->GetName();
    bool initWriter=true;
    if(createPolygon){
      //create polygon
      if(verbose_opt[0])
        std::cout << "create polygons" << std::endl;
      if(verbose_opt[0])
        std::cout << "open ogrWriter for polygons (2)" << std::endl;

      try{
        ogrWriter.open(output_opt[0],ogrformat_opt[0]);
      }
      catch(std::string errorString){
        initWriter=false;
        std::cerr << errorString << endl;
        std::cerr << "Warning: could not open " << output_opt[0] << endl;
      }
      if(initWriter){
        ogrWriter.pushLayer(layername,this->getProjectionRef(),wkbPolygon,papszOptions);
      }
    }
    else{
      if(verbose_opt[0])
        std::cout << "create points in layer " << readLayer->GetName() << std::endl;
      if(verbose_opt[0])
        std::cout << "open ogrWriter for points (2)" << std::endl;
      try{
        ogrWriter.open(output_opt[0],ogrformat_opt[0]);
      }
      catch(std::string errorString){
        initWriter=false;
        std::cerr << errorString << endl;
        std::cerr << "Warning: could not open " << output_opt[0] << endl;
      }
      if(initWriter){
        ogrWriter.pushLayer(layername,this->getProjectionRef(),wkbPoint,papszOptions);
      }
    }
    if(verbose_opt[0]){
      std::cout << "ogrWriter opened" << std::endl;
      // writeLayer=ogrWriter.createLayer(readLayer->GetName(), this->getProjection(), wkbPoint, papszOptions);
    }
    if(verbose_opt[0])
      std::cout << "copy fields" << std::endl;
    ogrWriter.copyFields(sampleReaderOgr);

    if(verbose_opt[0])
      std::cout << "create new fields" << std::endl;
    if(label_opt.size())
      ogrWriter.createField("label",OFTInteger);
    if(fid_opt.size())
      ogrWriter.createField(fid_opt[0],OFTInteger64,0);

    for(int irule=0;irule<rule_opt.size();++irule){
      for(int iband=0;iband<nband;++iband){
        int theBand=(band_opt.size()) ? band_opt[iband] : iband;
        ostringstream fs;
        if(bandNames_opt.size()){
          if(rule_opt.size()>1)
            fs << fieldMap[rule_opt[irule]];
          fs << bandNames_opt[iband];
        }
        else{
          if(rule_opt.size()>1||nband==1)
            fs << fieldMap[rule_opt[irule]];
          if(nband>1)
            fs << "b" << theBand;
        }
        switch(ruleMap[rule_opt[irule]]){
        case(rule::proportion):
        case(rule::count):{//count for each class
          for(int iclass=0;iclass<class_opt.size();++iclass){
            ostringstream fsclass;
            fsclass << fs.str() << "class" << class_opt[iclass];
            ogrWriter.createField(fsclass.str(),fieldType);
          }
          break;
        }
        case(rule::percentile):{//for each percentile
          for(int iperc=0;iperc<percentile_opt.size();++iperc){
            ostringstream fsperc;
            fsperc << fs.str() << percentile_opt[iperc];
            ogrWriter.createField(fsperc.str(),fieldType);
          }
          break;
        }
        default:
          ogrWriter.createField(fs.str(),fieldType);
          break;
        }
      }
    }
    OGRFeature *readFeature;
    unsigned long int ifeature=0;
    unsigned long int nfeatureLayer=sampleReaderOgr.getFeatureCount();
    unsigned long int ntotalvalidLayer=0;


    ogrWriter.resize(sampleReaderOgr.getFeatureCount());
    if(verbose_opt[0])
      std::cout << "start processing " << sampleReaderOgr.getFeatureCount() << " features" << std::endl;
    progress=0;
    MyProgressFunc(progress,pszMessage,pProgressArg);
    // readLayer->ResetReading();
    // while( (readFeature = readLayer->GetNextFeature()) != NULL ){

#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
    for(unsigned int ifeature=0;ifeature<sampleReaderOgr.getFeatureCount();++ifeature){
      // OGRFeature *readFeature=sampleReaderOgr.getFeatureRef(ifeature);
      OGRFeature *readFeature=sampleReaderOgr.cloneFeature(ifeature);
      bool validFeature=false;
      if(verbose_opt[0]>2)
        std::cout << "reading feature " << readFeature->GetFID() << std::endl;
      if(theThreshold>0){//percentual value
        double p=static_cast<double>(rand())/(RAND_MAX);
        p*=100.0;
        if(p>theThreshold){
          continue;//do not select for now, go to next feature
        }
      }
      else{//absolute value
        if(threshold_opt.size()==sampleReaderOgr.getLayerCount()){
          if(ntotalvalidLayer>=-theThreshold){
            continue;//do not select any more pixels, go to next column feature
          }
        }
        else{
          if(ntotalvalid>=-theThreshold){
            continue;//do not select any more pixels, go to next column feature
          }
        }
      }
      if(verbose_opt[0]>2)
        std::cout << "processing feature " << readFeature->GetFID() << std::endl;
      //get x and y from readFeature
      // double x,y;
      OGRGeometry *poGeometry;
      poGeometry = readFeature->GetGeometryRef();
      assert(poGeometry!=NULL);
      try{
        if(wkbFlatten(poGeometry->getGeometryType()) == wkbPoint ){
          OGRPoint readPoint = *((OGRPoint *) poGeometry);
          double i_centre,j_centre;
          this->geo2image(readPoint.getX(),readPoint.getY(),i_centre,j_centre);
          //nearest neighbour
          j_centre=static_cast<int>(j_centre);
          i_centre=static_cast<int>(i_centre);

          double bufferI=buffer_opt.size() ? buffer_opt[0]/getDeltaX() : 1;
          double bufferJ=buffer_opt.size() ? buffer_opt[0]/getDeltaY() : 1;
          double uli=i_centre-bufferI;
          double ulj=j_centre-bufferJ;
          double lri=i_centre+bufferI;
          double lrj=j_centre+bufferJ;
          // double uli=i_centre-buffer_opt[0];
          // double ulj=j_centre-buffer_opt[0];
          // double lri=i_centre+buffer_opt[0];
          // double lrj=j_centre+buffer_opt[0];

          //nearest neighbour
          ulj=static_cast<int>(ulj);
          uli=static_cast<int>(uli);
          lrj=static_cast<int>(lrj);
          lri=static_cast<int>(lri);

          //check if j is out of bounds
          if(static_cast<int>(ulj)<0||static_cast<int>(ulj)>=this->nrOfRow())
            continue;
          //check if j is out of bounds
          if(static_cast<int>(uli)<0||static_cast<int>(lri)>=this->nrOfCol())
            continue;

          OGRPoint ulPoint,urPoint,llPoint,lrPoint;
          OGRPolygon writePolygon;
          OGRPoint writePoint;
          OGRLinearRing writeRing;
          OGRFeature *writePolygonFeature;

          int nPointPolygon=0;
          if(createPolygon){
            if(disc_opt[0]){
              double radius=buffer_opt.size() ? buffer_opt[0] : sqrt(this->getDeltaX()*this->getDeltaY());
              unsigned short nstep = 25;
              for(int i=0;i<nstep;++i){
                OGRPoint aPoint;
                aPoint.setX(readPoint.getX()+this->getDeltaX()/2.0+radius*cos(2*PI*i/nstep));
                aPoint.setY(readPoint.getY()-this->getDeltaY()/2.0+radius*sin(2*PI*i/nstep));
                writeRing.addPoint(&aPoint);
              }
              writePolygon.addRing(&writeRing);
              writePolygon.closeRings();
            }
            else{
              double ulx,uly,lrx,lry;
              this->image2geo(uli,ulj,ulx,uly);
              this->image2geo(lri,lrj,lrx,lry);
              ulPoint.setX(ulx-this->getDeltaX()/2.0);
              ulPoint.setY(uly+this->getDeltaY()/2.0);
              lrPoint.setX(lrx+this->getDeltaX()/2.0);
              lrPoint.setY(lry-this->getDeltaY()/2.0);
              urPoint.setX(lrx+this->getDeltaX()/2.0);
              urPoint.setY(uly+this->getDeltaY()/2.0);
              llPoint.setX(ulx-this->getDeltaX()/2.0);
              llPoint.setY(lry-this->getDeltaY()/2.0);

              writeRing.addPoint(&ulPoint);
              writeRing.addPoint(&urPoint);
              writeRing.addPoint(&lrPoint);
              writeRing.addPoint(&llPoint);
              writePolygon.addRing(&writeRing);
              writePolygon.closeRings();
            }
            writePolygonFeature = ogrWriter.createFeature();
            // writePolygonFeature = OGRFeature::CreateFeature(writeLayer->GetLayerDefn());
            if(writePolygonFeature->SetFrom(readFeature)!= OGRERR_NONE)
              cerr << "writing feature failed" << std::endl;
            writePolygonFeature->SetGeometry(&writePolygon);
            if(verbose_opt[0]>1)
              std::cout << "copying new fields write polygon " << std::endl;
            if(verbose_opt[0]>1)
              std::cout << "write feature has " << writePolygonFeature->GetFieldCount() << " fields" << std::endl;

            OGRPoint readPoint;
            if(find(rule_opt.begin(),rule_opt.end(),"centroid")!=rule_opt.end()){
              if(verbose_opt[0]>1)
                std::cout << "get centroid" << std::endl;
              writePolygon.Centroid(&readPoint);
              double i,j;
              this->geo2image(readPoint.getX(),readPoint.getY(),i,j);
              int indexJ=static_cast<int>(j-layer_ulj);
              int indexI=static_cast<int>(i-layer_uli);
              bool valid=true;
              valid=valid&&(indexJ>=0);
              valid=valid&&(indexJ<this->nrOfRow());
              valid=valid&&(indexI>=0);
              valid=valid&&(indexI<this->nrOfCol());

              if(valid){
                if(maskReader.isInit()){
                  double maskI,maskJ;
                  maskReader.geo2image(readPoint.getX(),readPoint.getY(),maskI,maskJ);
                  maskI=static_cast<unsigned int>(maskI);
                  maskJ=static_cast<unsigned int>(maskJ);
                  if(maskI>0&&maskI<maskBuffer.nrOfCol()&&maskJ>0&&maskJ<maskBuffer.nrOfRow()){
                    for(int ivalue=0;ivalue<msknodata_opt.size();++ivalue){
                      if(maskBuffer[maskJ][maskI]==msknodata_opt[ivalue]){
                        valid=false;
                        break;
                      }
                    }
                  }
                }
              }
              if(valid){
                if(srcnodata_opt.empty())
                  validFeature=true;
                else{
                  for(int vband=0;vband<bndnodata_opt.size();++vband){
                    switch( fieldType ){
                    case OFTInteger:{
                      int value;
                      value=((readValuesInt[vband])[indexJ])[indexI];
                      if(value==srcnodata_opt[vband])
                        valid=false;
                      break;
                    }
                    case OFTReal:{
                      double value;
                      value=((readValuesReal[vband])[indexJ])[indexI];
                      if(value==srcnodata_opt[vband])
                        valid=false;
                      break;
                    }
                    }
                    if(!valid)
                      continue;
                    else
                      validFeature=true;
                  }
                }
              }
              if(valid){
                if(label_opt.size())
                  writePolygonFeature->SetField("label",label_opt[0]);
                if(fid_opt.size())
                  writePolygonFeature->SetField(fid_opt[0].c_str(),static_cast<GIntBig>(ifeature));
                for(int iband=0;iband<nband;++iband){
                  int theBand=(band_opt.size()) ? band_opt[iband] : iband;
                  //write fields for point on surface and centroid
                  string fieldname;
                  ostringstream fs;
                  if(bandNames_opt.size()){
                    if(rule_opt.size()>1)
                      fs << fieldMap["centroid"];
                    fs << bandNames_opt[iband];
                  }
                  else{
                    if(rule_opt.size()>1||nband==1)
                      fs << fieldMap["centroid"];
                    if(nband>1)
                      fs << "b" << theBand;
                  }
                  fieldname=fs.str();
                  switch( fieldType ){
                  case OFTInteger:
                    writePolygonFeature->SetField(fieldname.c_str(),static_cast<int>(((readValuesInt[iband])[indexJ])[indexI]));
                    break;
                  case OFTReal:
                    writePolygonFeature->SetField(fieldname.c_str(),((readValuesReal[iband])[indexJ])[indexI]);
                    break;
                  default://not supported
                    std::string errorString="field type not supported";
                    throw(errorString);
                    break;
                  }
                }
              }
            }//if centroid
            if(find(rule_opt.begin(),rule_opt.end(),"point")!=rule_opt.end()){
              if(verbose_opt[0]>1)
                std::cout << "get point on surface" << std::endl;
              if(writePolygon.PointOnSurface(&readPoint)!=OGRERR_NONE)
                writePolygon.Centroid(&readPoint);
              double i,j;
              this->geo2image(readPoint.getX(),readPoint.getY(),i,j);
              int indexJ=static_cast<int>(j-layer_ulj);
              int indexI=static_cast<int>(i-layer_uli);
              bool valid=true;
              valid=valid&&(indexJ>=0);
              valid=valid&&(indexJ<this->nrOfRow());
              valid=valid&&(indexI>=0);
              valid=valid&&(indexI<this->nrOfCol());

              if(valid){
                if(maskReader.isInit()){
                  double maskI,maskJ;
                  maskReader.geo2image(readPoint.getX(),readPoint.getY(),maskI,maskJ);
                  maskI=static_cast<unsigned int>(maskI);
                  maskJ=static_cast<unsigned int>(maskJ);
                  if(maskI>0&&maskI<maskBuffer.nrOfCol()&&maskJ>0&&maskJ<maskBuffer.nrOfRow()){
                    for(int ivalue=0;ivalue<msknodata_opt.size();++ivalue){
                      if(maskBuffer[indexJ][indexI]==msknodata_opt[ivalue]){
                        valid=false;
                        break;
                      }
                    }
                  }
                }
              }
              if(valid){
                if(srcnodata_opt.empty())
                  validFeature=true;
                else{
                  for(int vband=0;vband<bndnodata_opt.size();++vband){
                    switch( fieldType ){
                    case OFTInteger:{
                      int value;
                      value=((readValuesInt[vband])[indexJ])[indexI];
                      if(value==srcnodata_opt[vband])
                        valid=false;
                      break;
                    }
                    case OFTReal:{
                      double value;
                      value=((readValuesReal[vband])[indexJ])[indexI];
                      if(value==srcnodata_opt[vband])
                        valid=false;
                      break;
                    }
                    }
                    if(!valid)
                      continue;
                    else
                      validFeature=true;
                  }
                }
              }
              if(valid){
                if(label_opt.size())
                  writePolygonFeature->SetField("label",label_opt[0]);
                if(fid_opt.size())
                  writePolygonFeature->SetField(fid_opt[0].c_str(),static_cast<GIntBig>(ifeature));
                for(int iband=0;iband<nband;++iband){
                  int theBand=(band_opt.size()) ? band_opt[iband] : iband;
                  //write fields for point on surface and centroid
                  string fieldname;
                  ostringstream fs;
                  if(bandNames_opt.size()){
                    if(rule_opt.size()>1)
                      fs << fieldMap["point"];
                    fs << bandNames_opt[iband];
                  }
                  else{
                    if(rule_opt.size()>1||nband==1)
                      fs << fieldMap["point"];
                    if(nband>1)
                      fs << "b" << theBand;
                  }
                  fieldname=fs.str();
                  switch( fieldType ){
                  case OFTInteger:
                    writePolygonFeature->SetField(fieldname.c_str(),static_cast<int>(((readValuesInt[iband])[indexJ])[indexI]));
                    break;
                  case OFTReal:
                    writePolygonFeature->SetField(fieldname.c_str(),((readValuesReal[iband])[indexJ])[indexI]);
                    break;
                  default://not supported
                    std::string errorString="field type not supported";
                    throw(errorString);
                    break;
                  }
                }
              }
            }//if point
          }//if createPolygon

          if(calculateSpatialStatistics||!createPolygon){
            Vector2d<double> polyValues;
            vector<double> polyClassValues;

            if(class_opt.size()){
              polyClassValues.resize(class_opt.size());
              //initialize
              for(int iclass=0;iclass<class_opt.size();++iclass)
                polyClassValues[iclass]=0;
            }
            else
              polyValues.resize(nband);

            OGRPoint thePoint;
            for(int j=ulj;j<=lrj;++j){
              for(int i=uli;i<=lri;++i){
                //check if within raster image
                if(i<0||i>=this->nrOfCol())
                  continue;
                if(j<0||j>=this->nrOfRow())
                  continue;
                int indexJ=j-layer_ulj;
                int indexI=i-layer_uli;
                if(indexJ<0)
                  continue;
                if(indexI<0)
                  continue;
                // if(indexJ>=this->nrOfRow())
                //   indexJ=this->nrOfRow()-1;
                // if(indexI>=this->nrOfCol())
                //   indexI=this->nrOfCol()-1;
                if(indexJ>=this->nrOfRow())
                  continue;
                if(indexI>=this->nrOfCol())
                  continue;

                double theX=0;
                double theY=0;
                this->image2geo(i,j,theX,theY);
                thePoint.setX(theX);
                thePoint.setY(theY);
                if(disc_opt[0]&&buffer_opt.size()){
                  double radius=buffer_opt.size() ? buffer_opt[0] : sqrt(this->getDeltaX()*this->getDeltaY());
                  if((theX-readPoint.getX())*(theX-readPoint.getX())+(theY-readPoint.getY())*(theY-readPoint.getY())>radius*radius)
                    continue;
                }
                bool valid=true;

                if(maskReader.isInit()){
                  double maskI,maskJ;
                  maskReader.geo2image(readPoint.getX(),readPoint.getY(),maskI,maskJ);
                  maskI=static_cast<unsigned int>(maskI);
                  maskJ=static_cast<unsigned int>(maskJ);
                  if(maskI>0&&maskI<maskBuffer.nrOfCol()&&maskJ>0&&maskJ<maskBuffer.nrOfRow()){
                    for(int ivalue=0;ivalue<msknodata_opt.size();++ivalue){
                      // if(maskBuffer[indexJ][indexI]==msknodata_opt[ivalue]){
                      if(maskBuffer[maskJ][maskI]==msknodata_opt[ivalue]){
                        valid=false;
                        break;
                      }
                    }
                  }
                }
                if(srcnodata_opt.size()){
                  for(int vband=0;vband<bndnodata_opt.size();++vband){
                    switch( fieldType ){
                    case OFTInteger:{
                      int value=((readValuesInt[vband])[indexJ])[indexI];
                      if(value==srcnodata_opt[vband]){
                        valid=false;
                      }
                      break;
                    }
                    default:{
                      float value=((readValuesReal[vband])[indexJ])[indexI];
                      if(value==srcnodata_opt[vband]){
                        valid=false;
                      }
                      break;
                    }
                    }
                  }
                }
                if(!valid)
                  continue;
                else
                  validFeature=true;

                ++nPointPolygon;
                OGRFeature *writePointFeature;
                if(valid&&!createPolygon){//write all points
                  if(polythreshold_opt.size()){
                    if(polythreshold_opt[0]>0){
                      double p=static_cast<double>(rand())/(RAND_MAX);
                      p*=100.0;
                      if(p>polythreshold_opt[0])
                        continue;//do not select for now, go to next feature
                    }
                    else if(nPointPolygon>-polythreshold_opt[0])
                      continue;
                  }
                  //create feature
                  writePointFeature = OGRFeature::CreateFeature(ogrWriter.getLayer()->GetLayerDefn());
                  // writePointFeature = OGRFeature::CreateFeature(writeLayer->GetLayerDefn());
                  if(verbose_opt[0]>1)
                    std::cout << "copying fields from point feature " << std::endl;
                  if(writePointFeature->SetFrom(readFeature)!= OGRERR_NONE)
                    cerr << "writing feature failed" << std::endl;
                  if(verbose_opt[0]>1)
                    std::cout << "set geometry as point " << std::endl;
                  writePointFeature->SetGeometry(&thePoint);
                  assert(wkbFlatten(writePointFeature->GetGeometryRef()->getGeometryType()) == wkbPoint);
                  if(verbose_opt[0]>1){
                    std::cout << "write feature has " << writePointFeature->GetFieldCount() << " fields:" << std::endl;
                    for(int iField=0;iField<writePointFeature->GetFieldCount();++iField){
                      std::string fieldname=ogrWriter.getLayer()->GetLayerDefn()->GetFieldDefn(iField)->GetNameRef();
                      // std::string fieldname=writeLayer->GetLayerDefn()->GetFieldDefn(iField)->GetNameRef();
                      cout << fieldname << endl;
                    }
                  }
                }
                // if(valid&&class_opt.size()){
                //   short value=0;
                //   switch( fieldType ){
                //   case OFTInteger:
                //     value=((readValuesInt[0])[indexJ])[indexI];
                //     break;
                //   case OFTReal:
                //     value=((readValuesReal[0])[indexJ])[indexI];
                //     break;
                //   }
                //   for(int iclass=0;iclass<class_opt.size();++iclass){
                //     if(value==class_opt[iclass])
                //       polyClassValues[iclass]+=1;
                //   }
                // }
                if(valid){
                  if(!createPolygon&&label_opt.size())
                    writePointFeature->SetField("label",label_opt[0]);
                  if(!createPolygon&&fid_opt.size())
                    writePointFeature->SetField(fid_opt[0].c_str(),static_cast<GIntBig>(ifeature));
                  for(int iband=0;iband<nband;++iband){
                    int theBand=(band_opt.size()) ? band_opt[iband] : iband;
                    double value=0;
                    switch( fieldType ){
                    case OFTInteger:
                      value=((readValuesInt[iband])[indexJ])[indexI];
                      break;
                    case OFTReal:
                      value=((readValuesReal[iband])[indexJ])[indexI];
                      break;
                    }
                    if(!iband&&class_opt.size()){
                      for(int iclass=0;iclass<class_opt.size();++iclass){
                        if(value==class_opt[iclass])
                          polyClassValues[iclass]+=1;
                      }
                    }

                    if(verbose_opt[0]>1)
                      std::cout << ": " << value << std::endl;
                    if(!createPolygon){//write all points within polygon
                      string fieldname;
                      ostringstream fs;
                      if(bandNames_opt.size()){
                        if(rule_opt.size()>1)
                          fs << fieldMap["allpoints"];
                        fs << bandNames_opt[iband];
                      }
                      else{
                        if(rule_opt.size()>1||nband==1)
                          fs << fieldMap["allpoints"];
                        if(nband>1)
                          fs << "b" << theBand;
                      }
                      fieldname=fs.str();
                      int fieldIndex=writePointFeature->GetFieldIndex(fieldname.c_str());
                      if(fieldIndex<0){
                        ostringstream ess;
                        ess << "field " << fieldname << " was not found" << endl;
                        throw(ess.str());
                        // return(CE_Failure);
                      }
                      if(verbose_opt[0]>1)
                        std::cout << "set field " << fieldname << " to " << value << std::endl;
                      switch( fieldType ){
                      case OFTInteger:
                      case OFTReal:
                        writePointFeature->SetField(fieldname.c_str(),value);
                        break;
                      default://not supported
                        assert(0);
                        break;
                      }
                    }
                    else{
                      polyValues[iband].push_back(value);
                    }
                  }//iband
                }
                if(valid&&!createPolygon){
                  //write feature
                  if(verbose_opt[0]>1)
                    std::cout << "creating point feature" << std::endl;
                  // if(writeLayer->CreateFeature( writePointFeature ) != OGRERR_NONE ){
                  if(ogrWriter.getLayer()->CreateFeature( writePointFeature ) != OGRERR_NONE ){
                    std::string errorString="Failed to create feature in ogr vector dataset";
                    throw(errorString);
                  }
                  //destroy feature
                  // OGRFeature::DestroyFeature( writePointFeature );
                  ++ntotalvalid;
                  ++ntotalvalidLayer;
                }
              }//for in i
            }//for int j

            if(createPolygon){
              //do not create if no points found within polygon
              if(!nPointPolygon){
                if(verbose_opt[0])
                  cout << "no points found in polygon, continuing" << endl;
                continue;
              }
              //write field attributes to polygon feature
              for(int irule=0;irule<rule_opt.size();++irule){
                //skip centroid and point
                if(ruleMap[rule_opt[irule]]==rule::centroid||ruleMap[rule_opt[irule]]==rule::point)
                  continue;
                if(!irule&&label_opt.size())
                  writePolygonFeature->SetField("label",label_opt[0]);
                if(!irule&&fid_opt.size())
                  writePolygonFeature->SetField(fid_opt[0].c_str(),static_cast<GIntBig>(ifeature));
                for(int iband=0;iband<nband;++iband){
                  int theBand=(band_opt.size()) ? band_opt[iband] : iband;
                  vector<double> theValue;
                  vector<string> fieldname;
                  ostringstream fs;
                  if(bandNames_opt.size()){
                    if(rule_opt.size()>1)
                      fs << fieldMap[rule_opt[irule]];
                    fs << bandNames_opt[iband];
                  }
                  else{
                    if(rule_opt.size()>1||nband==1)
                      fs << fieldMap[rule_opt[irule]];
                    if(nband>1)
                      fs << "b" << theBand;
                  }
                  switch(ruleMap[rule_opt[irule]]){
                  case(rule::proportion)://deliberate fall through
                    stat.normalize_pct(polyClassValues);
                  case(rule::count):{//count for each class
                    for(int index=0;index<polyClassValues.size();++index){
                      theValue.push_back(polyClassValues[index]);
                      ostringstream fsclass;
                      fsclass << fs.str() << "class" << class_opt[index];
                      fieldname.push_back(fsclass.str());
                    }
                    break;
                  }
                  case(rule::mode):{
                    //maximum votes in polygon
                    if(verbose_opt[0])
                      std::cout << "number of points in polygon: " << nPointPolygon << std::endl;
                    //search for class with maximum votes
                    int maxClass=stat.mymin(class_opt);
                    vector<double>::iterator maxit;
                    maxit=stat.mymax(polyClassValues,polyClassValues.begin(),polyClassValues.end());
                    int maxIndex=distance(polyClassValues.begin(),maxit);
                    maxClass=class_opt[maxIndex];
                    if(verbose_opt[0]>0)
                      std::cout << "maxClass: " << maxClass << std::endl;
                    theValue.push_back(maxClass);
                    fieldname.push_back(fs.str());
                    break;
                  }
                  case(rule::mean):
                    theValue.push_back(stat.mean(polyValues[iband]));
                    fieldname.push_back(fs.str());
                    break;
                  case(rule::median):
                    theValue.push_back(stat.median(polyValues[iband]));
                    fieldname.push_back(fs.str());
                    break;
                  case(rule::stdev):
                    theValue.push_back(sqrt(stat.var(polyValues[iband])));
                    fieldname.push_back(fs.str());
                    break;
                  case(rule::percentile):{
                    for(int iperc=0;iperc<percentile_opt.size();++iperc){
                      theValue.push_back(stat.percentile(polyValues[iband],polyValues[iband].begin(),polyValues[iband].end(),percentile_opt[iperc]));
                      ostringstream fsperc;
                      fsperc << fs.str() << percentile_opt[iperc];
                      fieldname.push_back(fsperc.str());
                    }
                    break;
                  }
                  case(rule::sum):
                    theValue.push_back(stat.sum(polyValues[iband]));
                    fieldname.push_back(fs.str());
                    break;
                  case(rule::max):
                    theValue.push_back(stat.mymax(polyValues[iband]));
                    fieldname.push_back(fs.str());
                    break;
                  case(rule::min):
                    theValue.push_back(stat.mymin(polyValues[iband]));
                    fieldname.push_back(fs.str());
                    break;
                  case(rule::point):
                  case(rule::centroid):
                    theValue.push_back(polyValues[iband].back());
                  fieldname.push_back(fs.str());
                  break;
                  default://not supported
                    break;
                  }
                  for(int ivalue=0;ivalue<theValue.size();++ivalue){
                    switch( fieldType ){
                    case OFTInteger:
                      writePolygonFeature->SetField(fieldname[ivalue].c_str(),static_cast<int>(theValue[ivalue]));
                      break;
                    case OFTReal:
                      writePolygonFeature->SetField(fieldname[ivalue].c_str(),theValue[ivalue]);
                      break;
                    case OFTString:
                      writePolygonFeature->SetField(fieldname[ivalue].c_str(),type2string<double>(theValue[ivalue]).c_str());
                      break;
                    default://not supported
                      std::string errorString="field type not supported";
                      throw(errorString);
                      break;
                    }
                  }
                }
              }
            }
          }
          if(createPolygon&&validFeature){
            // if(createPolygon){
            //write polygon feature
            //todo: create only in case of valid feature
            if(verbose_opt[0]>1)
              std::cout << "creating polygon feature (1)" << std::endl;
            // if(writeLayer->CreateFeature( writePolygonFeature ) != OGRERR_NONE ){
            //   std::string errorString="Failed to create polygon feature in ogr vector dataset";
            //   throw(errorString);
            // }
            //test: no need to destroy anymore?
            // OGRFeature::DestroyFeature( writePolygonFeature );
            //make sure to use setFeature instead of pushFeature when in processing in parallel!!!
            if(!writePolygonFeature)
              std::cerr << "Warning: NULL feature" << ifeature << std::endl;
            ogrWriter.setFeature(ifeature,writePolygonFeature);
            ++ntotalvalid;
            ++ntotalvalidLayer;
          }
        }
        else{
          OGRPolygon readPolygon;
          OGRMultiPolygon readMultiPolygon;

          //get envelope
          OGREnvelope* psEnvelope=new OGREnvelope();

          if(wkbFlatten(poGeometry->getGeometryType()) == wkbPolygon){
            readPolygon = *((OGRPolygon *) poGeometry);
            readPolygon.closeRings();
            readPolygon.getEnvelope(psEnvelope);
          }
          else if(wkbFlatten(poGeometry->getGeometryType()) == wkbMultiPolygon){
            readMultiPolygon = *((OGRMultiPolygon *) poGeometry);
            readMultiPolygon.closeRings();
            readMultiPolygon.getEnvelope(psEnvelope);
          }
          else{
            std::string test;
            test=poGeometry->getGeometryName();
            ostringstream oss;
            oss << "geometry " << test << " not supported";
            throw(oss.str());
          }

          double ulx,uly,lrx,lry;
          double uli,ulj,lri,lrj;
          ulx=psEnvelope->MinX;
          uly=psEnvelope->MaxY;
          lrx=psEnvelope->MaxX;
          lry=psEnvelope->MinY;
          delete psEnvelope;

          //check if feature is covered by input raster dataset
          if(!this->covers(ulx,uly,lrx,lry,allCovered_opt[0])){
            if(verbose_opt[0]>2){
              std::cout << "Warning: raster does not cover polygon: " << readFeature->GetFID()  <<  std::endl;
              std::cout<< std::setprecision(12) << "--ulx " << ulx << " --uly " << uly << " --lrx " << lrx   << " --lry " << lry << std::endl;
            }
            continue;
          }
          else if(verbose_opt[0]>2){
              std::cout << "raster does cover polygon: " << readFeature->GetFID()  <<  std::endl;
              std::cout<< std::setprecision(12) << "--ulx " << ulx << " --uly " << uly << " --lrx " << lrx   << " --lry " << lry << std::endl;
          }
          OGRFeature *writePolygonFeature;
          int nPointPolygon=0;
          if(createPolygon){
            // writePolygonFeature = OGRFeature::CreateFeature(writeLayer->GetLayerDefn());
            writePolygonFeature = ogrWriter.createFeature();
            writePolygonFeature->SetGeometry(poGeometry);
            //writePolygonFeature and readFeature are both of type wkbPolygon
            if(writePolygonFeature->SetFrom(readFeature)!= OGRERR_NONE)
              cerr << "writing feature failed" << std::endl;
            if(verbose_opt[0]>1)
              std::cout << "copying new fields write polygon " << std::endl;
            if(verbose_opt[0]>1)
              std::cout << "write polygon feature has " << writePolygonFeature->GetFieldCount() << " fields" << std::endl;
          }

          OGRPoint readPoint;
          if(find(rule_opt.begin(),rule_opt.end(),"centroid")!=rule_opt.end()){
            if(verbose_opt[0]>1)
              std::cout << "get centroid" << std::endl;
            if(wkbFlatten(poGeometry->getGeometryType()) == wkbPolygon)
              readPolygon.Centroid(&readPoint);
            else if(wkbFlatten(poGeometry->getGeometryType()) == wkbMultiPolygon)
              readMultiPolygon.Centroid(&readPoint);

            double i,j;
            this->geo2image(readPoint.getX(),readPoint.getY(),i,j);
            int indexJ=static_cast<int>(j-layer_ulj);
            int indexI=static_cast<int>(i-layer_uli);
            bool valid=true;
            valid=valid&&(indexJ>=0);
            valid=valid&&(indexJ<this->nrOfRow());
            valid=valid&&(indexI>=0);
            valid=valid&&(indexI<this->nrOfCol());
            if(valid){
              if(maskReader.isInit()){
                double maskI,maskJ;
                maskReader.geo2image(readPoint.getX(),readPoint.getY(),maskI,maskJ);
                maskI=static_cast<unsigned int>(maskI);
                maskJ=static_cast<unsigned int>(maskJ);
                if(maskI>0&&maskI<maskBuffer.nrOfCol()&&maskJ>0&&maskJ<maskBuffer.nrOfRow()){
                  for(int ivalue=0;ivalue<msknodata_opt.size();++ivalue){
                    if(maskBuffer[indexJ][indexI]==msknodata_opt[ivalue]){
                      valid=false;
                      break;
                    }
                  }
                }
              }
            }
            if(valid){
              if(srcnodata_opt.empty())
                validFeature=true;
              else{
                for(int vband=0;vband<bndnodata_opt.size();++vband){
                  switch( fieldType ){
                  case OFTInteger:{
                    int value;
                    value=((readValuesInt[vband])[indexJ])[indexI];
                    if(value==srcnodata_opt[vband])
                      valid=false;
                    break;
                  }
                  case OFTReal:{
                    double value;
                    value=((readValuesReal[vband])[indexJ])[indexI];
                    if(value==srcnodata_opt[vband])
                      valid=false;
                    break;
                  }
                  }
                  if(!valid)
                    continue;
                  else
                    validFeature=true;
                }
              }
            }
            if(valid){
              if(label_opt.size())
                writePolygonFeature->SetField("label",label_opt[0]);
              if(fid_opt.size())
                writePolygonFeature->SetField(fid_opt[0].c_str(),static_cast<GIntBig>(ifeature));
              for(int iband=0;iband<nband;++iband){
                int theBand=(band_opt.size()) ? band_opt[iband] : iband;
                //write fields for point on surface and centroid
                string fieldname;
                ostringstream fs;
                if(bandNames_opt.size()){
                  if(rule_opt.size()>1)
                    fs << fieldMap["centroid"];
                  fs << bandNames_opt[iband];
                }
                else{
                  if(rule_opt.size()>1||nband==1)
                    fs << fieldMap["centroid"];
                  if(nband>1)
                    fs << "b" << theBand;
                }
                fieldname=fs.str();
                switch( fieldType ){
                case OFTInteger:
                  writePolygonFeature->SetField(fieldname.c_str(),static_cast<int>(((readValuesInt[iband])[indexJ])[indexI]));
                  break;
                case OFTReal:
                  writePolygonFeature->SetField(fieldname.c_str(),((readValuesReal[iband])[indexJ])[indexI]);
                  break;
                default://not supported
                  std::string errorString="field type not supported";
                  throw(errorString);
                  break;
                }
              }
            }
          }
          if(find(rule_opt.begin(),rule_opt.end(),"point")!=rule_opt.end()){
            if(verbose_opt[0]>1)
              std::cout << "get point on surface" << std::endl;
            if(wkbFlatten(poGeometry->getGeometryType()) == wkbPolygon){
              if(readPolygon.PointOnSurface(&readPoint)!=OGRERR_NONE)
                readPolygon.Centroid(&readPoint);
            }
            else if(wkbFlatten(poGeometry->getGeometryType()) == wkbMultiPolygon){
              // if(readMultiPolygon.PointOnSurface(&readPoint)!=OGRERR_NONE)
              readMultiPolygon.Centroid(&readPoint);
            }
            double i,j;
            this->geo2image(readPoint.getX(),readPoint.getY(),i,j);
            int indexJ=static_cast<int>(j-layer_ulj);
            int indexI=static_cast<int>(i-layer_uli);
            bool valid=true;
            valid=valid&&(indexJ>=0);
            valid=valid&&(indexJ<this->nrOfRow());
            valid=valid&&(indexI>=0);
            valid=valid&&(indexI<this->nrOfCol());
            if(valid){
              if(maskReader.isInit()){
                double maskI,maskJ;
                maskReader.geo2image(readPoint.getX(),readPoint.getY(),maskI,maskJ);
                maskI=static_cast<unsigned int>(maskI);
                maskJ=static_cast<unsigned int>(maskJ);
                if(maskI>0&&maskI<maskBuffer.nrOfCol()&&maskJ>0&&maskJ<maskBuffer.nrOfRow()){
                  for(int ivalue=0;ivalue<msknodata_opt.size();++ivalue){
                    if(maskBuffer[indexJ][indexI]==msknodata_opt[ivalue]){
                      valid=false;
                      break;
                    }
                  }
                }
              }
            }
            if(valid){
              if(srcnodata_opt.empty())
                validFeature=true;
              else{
                for(int vband=0;vband<bndnodata_opt.size();++vband){
                  switch( fieldType ){
                  case OFTInteger:{
                    int value;
                    value=((readValuesInt[vband])[indexJ])[indexI];
                    if(value==srcnodata_opt[vband])
                      valid=false;
                    break;
                  }
                  case OFTReal:{
                    double value;
                    value=((readValuesReal[vband])[indexJ])[indexI];
                    if(value==srcnodata_opt[vband])
                      valid=false;
                    break;
                  }
                  }
                  if(!valid)
                    continue;
                  else
                    validFeature=true;
                }
              }
            }
            if(valid){
              if(label_opt.size())
                writePolygonFeature->SetField("label",label_opt[0]);
              if(fid_opt.size())
                writePolygonFeature->SetField(fid_opt[0].c_str(),static_cast<GIntBig>(ifeature));
              for(int iband=0;iband<nband;++iband){
                int theBand=(band_opt.size()) ? band_opt[iband] : iband;
                //write fields for point on surface and centroid
                string fieldname;
                ostringstream fs;
                if(bandNames_opt.size()){
                  if(rule_opt.size()>1)
                    fs << fieldMap["point"];
                  fs << bandNames_opt[iband];
                }
                else{
                  if(rule_opt.size()>1||nband==1)
                    fs << fieldMap["point"];
                  if(nband>1)
                    fs << "b" << theBand;
                }
                fieldname=fs.str();
                switch( fieldType ){
                case OFTInteger:
                  writePolygonFeature->SetField(fieldname.c_str(),static_cast<int>(((readValuesInt[iband])[indexJ])[indexI]));
                  break;
                case OFTReal:
                  writePolygonFeature->SetField(fieldname.c_str(),((readValuesReal[iband])[indexJ])[indexI]);
                  break;
                default://not supported
                  std::string errorString="field type not supported";
                  throw(errorString);
                  break;
                }
              }
            }
          }
          if(calculateSpatialStatistics||ruleMap[rule_opt[0]]==rule::allpoints){
            this->geo2image(ulx,uly,uli,ulj);
            this->geo2image(lrx,lry,lri,lrj);
            //nearest neighbour
            ulj=static_cast<int>(ulj);
            uli=static_cast<int>(uli);
            lrj=static_cast<int>(lrj);
            lri=static_cast<int>(lri);
            //iterate through all pixels
            if(verbose_opt[0]>1)
              std::cout << "bounding box for polygon feature " << ifeature << ": " << uli << " " << ulj << " " << lri << " " << lrj << std::endl;

            if(uli<0)
              uli=0;
            if(lri<0)
              lri=0;
            // if(uli>=this->nrOfCol())
            //   uli=this->nrOfCol()-1;
            if(uli>=layer_lri)
              uli=layer_lri-1;
            // if(lri>=this->nrOfCol())
            //   lri=this->nrOfCol()-1;
            if(lri>=layer_lri)
              lri=layer_lri-1;
            if(ulj<0)
              ulj=0;
            if(lrj<0)
              lrj=0;
            // if(ulj>=this->nrOfRow())
            //   ulj=this->nrOfRow()-1;
            if(ulj>=layer_lrj)
              ulj=layer_lrj-1;
            // if(lrj>=this->nrOfRow())
            //   lrj=this->nrOfRow()-1;
            if(lrj>=layer_lrj)
              lrj=layer_lrj-1;

            if(verbose_opt[0]>1)
              std::cout << "bounding box for polygon feature after check " << ifeature << ": " << uli << " " << ulj << " " << lri << " " << lrj << std::endl;
            Vector2d<double> polyValues;
            vector<double> polyClassValues;

            if(class_opt.size()){
              polyClassValues.resize(class_opt.size());
              //initialize
              for(int iclass=0;iclass<class_opt.size();++iclass)
                polyClassValues[iclass]=0;
            }
            else
              polyValues.resize(nband);

            OGRPoint thePoint;
            for(int j=ulj;j<=lrj;++j){
              for(int i=uli;i<=lri;++i){
                //check if within raster image
                // if(i<0||i>=this->nrOfCol())
                if(i<0||i>=layer_lri)
                  continue;
                // if(j<0||j>=this->nrOfRow())
                //   continue;
                if(j<0||j>=layer_lrj)
                  continue;
                int indexJ=j-layer_ulj;
                int indexI=i-layer_uli;
                //test
                if(indexJ<0||indexJ>=this->nrOfRow()||indexI<0||indexI>=this->nrOfCol())
                  continue;

                double theX=0;
                double theY=0;
                this->image2geo(i,j,theX,theY);
                thePoint.setX(theX);
                thePoint.setY(theY);
                //check if point is on surface
                if(wkbFlatten(poGeometry->getGeometryType()) == wkbPolygon){
                  if(!readPolygon.Contains(&thePoint))
                    continue;
                }
                else if(wkbFlatten(poGeometry->getGeometryType()) == wkbMultiPolygon){
                  if(!readMultiPolygon.Contains(&thePoint))
                    continue;
                }

                bool valid=true;
                double maskI,maskJ;
                if(maskReader.isInit()){
                  maskReader.geo2image(readPoint.getX(),readPoint.getY(),maskI,maskJ);
                  maskI=static_cast<unsigned int>(maskI);
                  maskJ=static_cast<unsigned int>(maskJ);
                  if(maskI>0&&maskI<maskBuffer.nrOfCol()&&maskJ>0&&maskJ<maskBuffer.nrOfRow()){
                    for(int ivalue=0;ivalue<msknodata_opt.size();++ivalue){
                      if(maskBuffer[indexJ][indexI]==msknodata_opt[ivalue]){
                        valid=false;
                        break;
                      }
                    }
                  }
                }
                if(srcnodata_opt.size()){
                  for(int vband=0;vband<bndnodata_opt.size();++vband){
                    switch( fieldType ){
                    case OFTInteger:{
                      int value=((readValuesInt[vband])[indexJ])[indexI];
                      if(value==srcnodata_opt[vband]){
                        valid=false;
                      }
                      break;
                    }
                    default:{
                      float value=((readValuesReal[vband])[indexJ])[indexI];
                      if(value==srcnodata_opt[vband]){
                        valid=false;
                      }
                      break;
                    }
                    }
                  }
                }
                if(!valid)
                  continue;
                else
                  validFeature=true;

                if(verbose_opt[0]>1)
                  std::cout << "point is on surface: " << thePoint.getX() << "," << thePoint.getY() << std::endl;
                ++nPointPolygon;

                OGRFeature *writePointFeature;
                if(!createPolygon){//write all points within polygon
                  if(polythreshold_opt.size()){
                    if(polythreshold_opt[0]>0){
                      double p=static_cast<double>(rand())/(RAND_MAX);
                      p*=100.0;
                      if(p>polythreshold_opt[0])
                        continue;//do not select for now, go to next feature
                    }
                    else if(nPointPolygon>-polythreshold_opt[0])
                      continue;
                  }
                  // if(polythreshold_opt.size())
                  //   if(nPointPolygon>=polythreshold_opt[0])
                  //     continue;
                  // if(threshold_opt.size()){
                  //   if(threshold_opt[0]<=0){
                  //     if(ntotalvalid>=-threshold_opt[0])
                  //       continue;
                  //   }
                  //   else if(threshold_opt[0]<100){
                  //     double p=static_cast<double>(rand())/(RAND_MAX);
                  //     p*=100.0;
                  //     if(p>threshold_opt[0]){
                  //       continue;//do not select for now, go to next feature
                  //     }
                  //   }
                  // }
                  //create feature
                  writePointFeature = OGRFeature::CreateFeature(ogrWriter.getLayer()->GetLayerDefn());
                  // writePointFeature = OGRFeature::CreateFeature(writeLayer->GetLayerDefn());
                  if(verbose_opt[0]>1)
                    std::cout << "copying fields from polygons " << std::endl;
                  if(writePointFeature->SetFrom(readFeature)!= OGRERR_NONE)
                    cerr << "writing feature failed" << std::endl;
                  if(verbose_opt[0]>1)
                    std::cout << "set geometry as point " << std::endl;
                  writePointFeature->SetGeometry(&thePoint);
                  assert(wkbFlatten(writePointFeature->GetGeometryRef()->getGeometryType()) == wkbPoint);
                  if(verbose_opt[0]>1){
                    std::cout << "write point feature has " << writePointFeature->GetFieldCount() << " fields:" << std::endl;
                    for(int iField=0;iField<writePointFeature->GetFieldCount();++iField){
                      std::string fieldname=ogrWriter.getLayer()->GetLayerDefn()->GetFieldDefn(iField)->GetNameRef();
                      // std::string fieldname=writeLayer->GetLayerDefn()->GetFieldDefn(iField)->GetNameRef();
                      cout << fieldname << endl;
                    }
                  }
                }
                // if(class_opt.size()){
                //   short value=0;
                //   switch( fieldType ){
                //   case OFTInteger:
                //     value=((readValuesInt[0])[indexJ])[indexI];
                //     break;
                //   case OFTReal:
                //     value=((readValuesReal[0])[indexJ])[indexI];
                //     break;
                //   }
                //   for(int iclass=0;iclass<class_opt.size();++iclass){
                //     if(value==class_opt[iclass])
                //       polyClassValues[iclass]+=1;
                //   }
                // }
                // else{
                if(!createPolygon&&label_opt.size())
                  writePointFeature->SetField("label",label_opt[0]);
                if(!createPolygon&&fid_opt.size())
                  writePointFeature->SetField(fid_opt[0].c_str(),static_cast<GIntBig>(ifeature));
                for(int iband=0;iband<nband;++iband){
                  int theBand=(band_opt.size()) ? band_opt[iband] : iband;
                  double value=0;
                  switch( fieldType ){
                  case OFTInteger:
                    value=((readValuesInt[iband])[indexJ])[indexI];
                    break;
                  case OFTReal:
                    value=((readValuesReal[iband])[indexJ])[indexI];
                    break;
                  }

                  if(!iband&&class_opt.size()){
                    for(int iclass=0;iclass<class_opt.size();++iclass){
                      if(value==class_opt[iclass])
                        polyClassValues[iclass]+=1;
                    }
                  }
                  if(verbose_opt[0]>1)
                    std::cout << ": " << value << std::endl;
                  if(!createPolygon){//write all points within polygon
                    string fieldname;
                    ostringstream fs;
                    if(bandNames_opt.size()){
                      if(rule_opt.size()>1)
                        fs << fieldMap["allpoints"];
                      fs << bandNames_opt[iband];
                    }
                    else{
                      if(rule_opt.size()>1||nband==1)
                        fs << fieldMap["allpoints"];
                      if(nband>1)
                        fs << "b" << theBand;
                    }
                    fieldname=fs.str();
                    int fieldIndex=writePointFeature->GetFieldIndex(fieldname.c_str());
                    if(fieldIndex<0){
                      ostringstream ess;
                      ess << "field " << fieldname << " was not found" << endl;
                      throw(ess.str());
                      // cerr << "field " << fieldname << " was not found" << endl;
                      // return(CE_Failure);
                    }
                    if(verbose_opt[0]>1)
                      std::cout << "set field " << fieldname << " to " << value << std::endl;
                    switch( fieldType ){
                    case OFTInteger:
                    case OFTReal:
                      writePointFeature->SetField(fieldname.c_str(),value);
                      break;
                    default://not supported
                      assert(0);
                      break;
                    }
                  }
                  else{
                    polyValues[iband].push_back(value);
                  }
                }//iband
                if(!createPolygon){
                  //todo: only if valid feature?
                  //write feature
                  if(verbose_opt[0])
                    std::cout << "creating point feature " << ntotalvalidLayer << std::endl;
                  // if(writeLayer->CreateFeature( writePointFeature ) != OGRERR_NONE ){
                  if(ogrWriter.getLayer()->CreateFeature( writePointFeature ) != OGRERR_NONE ){
                    std::string errorString="Failed to create feature in ogr vector dataset";
                    throw(errorString);
                  }
                  //destroy feature
                  // OGRFeature::DestroyFeature( writePointFeature );
                  ++ntotalvalid;
                  ++ntotalvalidLayer;
                }
              }//for in i
            }//for int j
            if(createPolygon){
              //do not create if no points found within polygon
              if(!nPointPolygon){
                if(verbose_opt[0])
                  cout << "no points found in polygon, continuing" << endl;
                continue;
              }
              //write field attributes to polygon feature
              for(int irule=0;irule<rule_opt.size();++irule){
                //skip centroid and point
                if(ruleMap[rule_opt[irule]]==rule::centroid||ruleMap[rule_opt[irule]]==rule::point)
                  continue;
                if(!irule&&label_opt.size())
                  writePolygonFeature->SetField("label",label_opt[0]);
                if(!irule&&fid_opt.size())
                  writePolygonFeature->SetField(fid_opt[0].c_str(),static_cast<GIntBig>(ifeature));
                for(int iband=0;iband<nband;++iband){
                  int theBand=(band_opt.size()) ? band_opt[iband] : iband;
                  vector<double> theValue;
                  vector<string> fieldname;
                  ostringstream fs;
                  if(bandNames_opt.size()){
                    if(rule_opt.size()>1)
                      fs << fieldMap[rule_opt[irule]];
                    fs << bandNames_opt[iband];
                  }
                  else{
                    if(rule_opt.size()>1||nband==1)
                      fs << fieldMap[rule_opt[irule]];
                    if(nband>1)
                      fs << "b" << theBand;
                  }
                  switch(ruleMap[rule_opt[irule]]){
                  case(rule::proportion):
                    stat.normalize_pct(polyClassValues);
                  case(rule::count):{//count for each class
                    for(int index=0;index<polyClassValues.size();++index){
                      theValue.push_back(polyClassValues[index]);
                      ostringstream fsclass;
                      fsclass << fs.str() << "class" << class_opt[index];
                      fieldname.push_back(fsclass.str());
                    }
                    break;
                  }
                  case(rule::mode):{
                    //maximum votes in polygon
                    if(verbose_opt[0])
                      std::cout << "number of points in polygon: " << nPointPolygon << std::endl;
                    //search for class with maximum votes
                    int maxClass=stat.mymin(class_opt);
                    vector<double>::iterator maxit;
                    maxit=stat.mymax(polyClassValues,polyClassValues.begin(),polyClassValues.end());
                    int maxIndex=distance(polyClassValues.begin(),maxit);
                    maxClass=class_opt[maxIndex];
                    if(verbose_opt[0]>0)
                      std::cout << "maxClass: " << maxClass << std::endl;
                    theValue.push_back(maxClass);
                    fieldname.push_back(fs.str());
                    break;
                  }
                  case(rule::mean):
                    theValue.push_back(stat.mean(polyValues[iband]));
                    fieldname.push_back(fs.str());
                    break;
                  case(rule::median):
                    theValue.push_back(stat.median(polyValues[iband]));
                    fieldname.push_back(fs.str());
                    break;
                  case(rule::stdev):
                    theValue.push_back(sqrt(stat.var(polyValues[iband])));
                    fieldname.push_back(fs.str());
                    break;
                  case(rule::percentile):{
                    for(int iperc=0;iperc<percentile_opt.size();++iperc){
                      theValue.push_back(stat.percentile(polyValues[iband],polyValues[iband].begin(),polyValues[iband].end(),percentile_opt[iperc]));
                      ostringstream fsperc;
                      fsperc << fs.str() << percentile_opt[iperc];
                      fieldname.push_back(fsperc.str());
                    }
                    break;
                  }
                  case(rule::sum):
                    theValue.push_back(stat.sum(polyValues[iband]));
                    fieldname.push_back(fs.str());
                    break;
                  case(rule::max):
                    theValue.push_back(stat.mymax(polyValues[iband]));
                    fieldname.push_back(fs.str());
                    break;
                  case(rule::min):
                    theValue.push_back(stat.mymin(polyValues[iband]));
                    fieldname.push_back(fs.str());
                    break;
                  case(rule::centroid):
                  case(rule::point):
                    theValue.push_back(polyValues[iband].back());
                  fieldname.push_back(fs.str());
                  break;
                  default://not supported
                    break;
                  }
                  for(int ivalue=0;ivalue<theValue.size();++ivalue){
                    switch( fieldType ){
                    case OFTInteger:
                      writePolygonFeature->SetField(fieldname[ivalue].c_str(),static_cast<int>(theValue[ivalue]));
                      break;
                    case OFTReal:
                      writePolygonFeature->SetField(fieldname[ivalue].c_str(),theValue[ivalue]);
                      break;
                    case OFTString:
                      writePolygonFeature->SetField(fieldname[ivalue].c_str(),type2string<double>(theValue[ivalue]).c_str());
                      break;
                    default://not supported
                      std::string errorString="field type not supported";
                      throw(errorString);
                      break;
                    }
                  }
                }
              }
            }
          }
          if(createPolygon&&validFeature){
            //todo: only create if valid feature?
            //write polygon feature
            if(verbose_opt[0]>1)
              std::cout << "creating polygon feature (2)" << std::endl;
            // if(writeLayer->CreateFeature( writePolygonFeature ) != OGRERR_NONE ){
            //   std::string errorString="Failed to create polygon feature in ogr vector dataset";
            //   throw(errorString);
            // }
            //test: no need to destroy anymore?
            // OGRFeature::DestroyFeature( writePolygonFeature );
            //make sure to use setFeature instead of pushFeature when in processing in parallel!!!
            if(verbose_opt[0])
              std::cout << "set feature " << ifeature << std::endl;
            if(!writePolygonFeature)
              std::cerr << "Warning: NULL feature" << ifeature << std::endl;
            ogrWriter.setFeature(ifeature,writePolygonFeature);
            ++ntotalvalid;
            ++ntotalvalidLayer;
          }
        }
        // ++ifeature;
        if(theThreshold>0){
          if(threshold_opt.size()==sampleReaderOgr.getLayerCount())
            progress=(100.0/theThreshold)*static_cast<float>(ntotalvalidLayer)/nfeatureLayer;
          else
            progress=static_cast<float>(ntotalvalidLayer)/nfeatureLayer;
        }
        else
          progress=static_cast<float>(ifeature+1)/(-theThreshold);
        MyProgressFunc(progress,pszMessage,pProgressArg);
      }
      catch(std::string e){
        std::cout << e << std::endl;
        continue;
      }
      catch(int npoint){
        if(verbose_opt[0])
          std::cout << "number of points read in polygon: " << npoint << std::endl;
        continue;
      }
    }
    // if(rbox_opt[0]>0||cbox_opt[0]>0)
    //   boxWriter.close();
    progress=1.0;
    MyProgressFunc(progress,pszMessage,pProgressArg);
    if(verbose_opt[0])
      std::cout << "number of valid points in layer: " << ntotalvalidLayer << std::endl;
    if(verbose_opt[0])
      std::cout << "number of valid points in all layers: " << ntotalvalid<< std::endl;
    sampleReaderOgr.close();

    if(papszOptions)
      CSLDestroy(papszOptions);
    // ogrWriter.write();
    // ogrWriter.close();
    progress=1.0;
    MyProgressFunc(progress,pszMessage,pProgressArg);
    // this->close();
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
 * @param app application specific option arguments
 *
 * @return CE_None if success, CE_Failure if failure
 */
//todo: support multiple layers for writing
//output vector ogrWriter will take spatial reference system of input vector sampleReader
// make sure to setSpatialFilterRect on vector before entering here
//todo: extract each Jim individually and store in tmpWriter and then join the tmpWriter in ogrWriter if different number of bands (or this is behavior of ImgMultiList?)
CPLErr JimList::extractOgr(VectorOgr& sampleReader, VectorOgr& ogrWriter, AppFactory& app){

  Optionjl<short> verbose_opt("v", "verbose", "Verbose mode if > 0", 0,2);
  Optionjl<std::string> combine_opt("combine", "combine", "Combine results of extract by append or join (default is append)","append");
  Optionjl<std::string> bandNames_opt("bn", "bandname", "Band name(s) corresponding to band index(es).");

  try{
    bool doProcess;//stop process when program was invoked with help option (-h --help)
    try{
      doProcess=combine_opt.retrieveOption(app);
      bandNames_opt.retrieveOption(app);
      verbose_opt.retrieveOption(app);
    }
    catch(string predefinedString){
      string errorstring="Error: could not get combine rule";
      throw(errorstring);
    }
    bool append=true;
    if(combine_opt[0]!="append")
      append=false;
    app.clearOption("combine");
    app::AppFactory extractApp(app);
    std::list<std::shared_ptr<Jim> >::const_iterator imit=begin();
    CPLErr result=CE_None;
    size_t iband=0;

    std::string ogrFilename;
    if(!append){//join
      if(verbose_opt[0])
        std::cout << "We are in join, first image" << std::endl;
      extractApp.setLongOption("fid","fid");
      extractApp.setLongOption("co","OVERWRITE=YES");
      extractApp.clearOption("bandname");
      if(bandNames_opt.size())
        extractApp.setLongOption("bandname",bandNames_opt[iband]);

      if(!((*imit)->isGeoRef())){
        std::cerr << "Warning: input image is not georeferenced" << std::endl;
        // string errorstring="Warning: input image is not georeferenced";
        // throw(errorstring);
      }
      if((*imit)->extractOgr(sampleReader,ogrWriter,extractApp)!=CE_None){
        string errorstring="Error: could not extractOgr";
        throw(errorstring);
      }
      ogrFilename=ogrWriter.getFileName();
      if(verbose_opt[0])
        std::cout << "Writing vector dataset " << ogrFilename << std::endl;
      ogrWriter.write();
      ogrWriter.close();
      ++imit;
      ++iband;
    }
    else if(verbose_opt[0])
      std::cout << "We are in append" << std::endl;
    while(imit!=end()){
      try{
        if(!((*imit)->isGeoRef())){
          std::cerr << "Warning: input image is not georeferenced" << std::endl;
          // string errorstring="Warning: input image is not georeferenced";
          // throw(errorstring);
        }
        if(append){
          if(verbose_opt[0])
            std::cout << "We are in append"<< std::endl;
          if(verbose_opt[0]){
            std::cout << "bbox of sampleReader: " << " -ulx " << sampleReader.getUlx()<< " -uly " << sampleReader.getUly()<< " -lrx " << sampleReader.getLrx()   << " -lry " << sampleReader.getLry() << std::endl;
            std::cout << "bbox of raster image: " << " -ulx " << (*imit)->getUlx()<< " -uly " << (*imit)->getUly()<< " -lrx " << (*imit)->getLrx()   << " -lry " << (*imit)->getLry() << std::endl;
          }
          (*imit)->extractOgr(sampleReader,ogrWriter,extractApp);
        }
        else{//join
          if(verbose_opt[0])
            std::cout << "We are in join, band " << iband << std::endl;

          VectorOgr v1=VectorOgr(ogrFilename);
          VectorOgr v2;
          extractApp.clearOption("bandname");
          if(bandNames_opt.size())
            extractApp.setLongOption("bandname",bandNames_opt[iband]);
          extractApp.setLongOption("output","/vsimem/v2.sqlite");
          //*imit is ImageProcess
          (*imit)->extractOgr(sampleReader, v2, extractApp);
          v2.write();
          app::AppFactory joinApp(extractApp);
          joinApp.setLongOption("output",ogrFilename);
          joinApp.setLongOption("key","fid");
          VectorOgr nextWriter;
          if(verbose_opt[0])
            std::cout << "joining vectors" << std::endl;
          v1.join(v2,nextWriter,joinApp);
          v1.close();
          v2.close();
          if(verbose_opt[0])
            std::cout << "Writing vector nextWriter " << nextWriter.getFileName() << std::endl;
          nextWriter.write();
          nextWriter.close();
          ++iband;
        }
      }
      catch(string errorString){
        std::cout << errorString << ", continuing with next image"<< std::endl;
        ++imit;
        continue;
      }
      ++imit;
    }
    return(CE_None);
  }
  catch(string errorString){
    std::cout << errorString << std::endl;
    throw;
  }
}

/**
 * @param app application specific option arguments
 * @return output Vector
 **/
// make sure to setSpatialFilterRect on vector before entering here
shared_ptr<VectorOgr> JimList::extractOgr(VectorOgr& sampleReader, AppFactory& app){
  shared_ptr<VectorOgr> ogrWriter=VectorOgr::createVector();
  if(extractOgr(sampleReader, *ogrWriter, app)!=OGRERR_NONE){
    std::cerr << "Failed to extract" << std::endl;
  }
  return(ogrWriter);
}

size_t JimList::extractOgrMem(VectorOgr& sampleReader, vector<unsigned char> &vbytes, AppFactory& app){
  try{
    size_t filesize=0;
    Optionjl<string> output_opt("o", "output", "Output sample dataset","/vsimem/extractmem");
    Optionjl<string> ogrformat_opt("f", "oformat", "Output sample dataset format","SQLite");
    output_opt.clear();
    ogrformat_opt.clear();
    output_opt.retrieveOption(app);
    ogrformat_opt.retrieveOption(app);

    VectorOgr ogrWriter;
    std::list<std::shared_ptr<Jim> >::const_iterator imit=begin();
    for(imit=begin();imit!=end();++imit){
      if(!((*imit)->isGeoRef())){
        std::cerr << "Warning: input image is not georeferenced" << std::endl;
        // string errorstring="Error: input image is not georeferenced";
        // throw(errorstring);
      }
      if((*imit)->extractOgr(sampleReader,ogrWriter,app)!=CE_None){
        string errorstring="Error: extractOgr failed";
        throw(errorstring);
      }
    }
    ogrWriter.write();
    ogrWriter.close();
    return(ogrWriter.serialize(vbytes));
  }
  catch(string errorString){
    std::cout << errorString << std::endl;
    throw;
  }
}
