/**********************************************************************
jlpolygonize_lib.cc: program to make vector file from raster image
Author(s): Pieter.Kempeneers@ec.europa.eu
Copyright (c) 2016-2018 European Union (Joint Research Centre)
License EUPLv1.2

This file is part of jiplib
***********************************************************************/
#include "cpl_string.h"
#include "gdal_priv.h"
#include "gdal.h"
#include "ogrsf_frmts.h"
extern "C" {
#include "gdal_alg.h"
#include "ogr_api.h"
}
#include <config_jiplib.h>
#include "base/Optionjl.h"
#include "Jim.h"
#include "VectorOgr.h"

using namespace std;

shared_ptr<VectorOgr> Jim::polygonize(app::AppFactory& app, std::shared_ptr<Jim> mask){
  shared_ptr<VectorOgr> ogrWriter=VectorOgr::createVector();
  polygonize(*ogrWriter, app, mask);
  return(ogrWriter);
}

void Jim::polygonize(VectorOgr&ogrWriter, app::AppFactory &theApp, std::shared_ptr<Jim> mask){
  Optionjl<string> output_opt("o", "output", "Output vector file");
  Optionjl<string> layername_opt("ln", "ln", "Output layer name","polygonize");
  Optionjl<string> ogrformat_opt("f", "f", "Output OGR file format","SQLite");
  Optionjl<string> option_opt("co", "co", "Creation option for output file. Multiple options can be specified.");
  Optionjl<string> fname_opt("n", "name", "the field name of the output layer", "DN");
  Optionjl<double> nodata_opt("nodata", "nodata", "Disgard this nodata value when creating polygons.");
  Optionjl<short> verbose_opt("verbose", "verbose", "verbose output",0,2);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=output_opt.retrieveOption(theApp);
    layername_opt.retrieveOption(theApp);
    ogrformat_opt.retrieveOption(theApp);
    option_opt.retrieveOption(theApp);
    fname_opt.retrieveOption(theApp);
    nodata_opt.retrieveOption(theApp);
    verbose_opt.retrieveOption(theApp);
  }
  catch(std::string predefinedString){
    std::cout << predefinedString << std::endl;
  }
  if(!doProcess){
    std::cout << std::endl;
    std::ostringstream helpStream;
    helpStream << "exception thrown due to help info";
    throw(helpStream.str());//help was invoked, stop processing
  }

  std::vector<std::string> badKeys;
  theApp.badKeys(badKeys);
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

  const char *pszFormat = "MEM";
  GDALDriver *poDriverMem = GetGDALDriverManager()->GetDriverByName(pszFormat);

  GDALDataset *poDatasetMask = NULL;
  GDALRasterBand *poMaskBandMem = NULL;
  GDALDataset *poDatasetMem = NULL;
  GDALRasterBand *poBandMem = NULL;
  if( poDriverMem ){
    if(mask){
      poDatasetMask = (GDALDataset *)poDriverMem->Create("memmask",mask->nrOfCol(),mask->nrOfRow(),1,mask->getGDALDataType(),NULL);
      if(poDatasetMask){
        if(verbose_opt[0])
          cout << "get raster band from mask" << endl;
        std::vector<double> maskGT(6);
        getGeoTransform(maskGT);
        poDatasetMask->SetGeoTransform(&maskGT[0]);
        poMaskBandMem = poDatasetMask->GetRasterBand(1);
        if( poMaskBandMem->RasterIO(GF_Write, 0, 0, mask->nrOfCol(), mask->nrOfRow(), mask->getDataPointer(0), mask->nrOfCol(), mask->nrOfRow(), mask->getGDALDataType(), 0, 0, NULL) != CE_None ){
          cerr << CPLGetLastErrorMsg() << endl;
        }
      }
      else{
        std::cerr << "Error: clould not create memory dataset for mask datatset" << std::endl;
        throw;
      }
    }
    poDatasetMem = (GDALDataset *)poDriverMem->Create("memband",nrOfCol(),nrOfRow(),1,getGDALDataType(),NULL);
    if(poDatasetMem){
      //printf("Copy MemoryBand to memband\n"); fflush(stdout);
      std::vector<double> sourceGT(6);
      getGeoTransform(sourceGT);
      poDatasetMem->SetGeoTransform(&sourceGT[0]);
      poBandMem = poDatasetMem->GetRasterBand(1);
      if( poBandMem->RasterIO(GF_Write, 0, 0, nrOfCol(), nrOfRow(), m_data[0], nrOfCol(), nrOfRow(), getGDALDataType(), 0, 0, NULL) == CE_None ){
        // Quality parameters for warping operation
        if(nodata_opt.size())
          poBandMem->SetNoDataValue(nodata_opt[0]);

        char **papszOptions=NULL;
        for(std::vector<std::string>::const_iterator optionIt=option_opt.begin();optionIt!=option_opt.end();++optionIt)
          papszOptions=CSLAddString(papszOptions,optionIt->c_str());

        if(verbose_opt[0])
          std::cout << "Opening ogrWriter: " << output_opt[0] << " in format " << ogrformat_opt[0] << endl;
        ogrWriter.open(output_opt[0],ogrformat_opt[0]);
        ogrWriter.pushLayer(layername_opt[0],getProjection(),wkbUnknown,papszOptions);

        if(verbose_opt[0])
          cout << "projection: " << getProjection() << endl;
        ogrWriter.createField(fname_opt[0],OFTInteger);

        if(verbose_opt[0])
          cout << "GDALPolygonize started..." << endl;

        int index=ogrWriter.getLayer()->GetLayerDefn()->GetFieldIndex(fname_opt[0].c_str());
        double dfComplete=0.0;
        const char* pszMessage;
        void* pProgressArg=NULL;
        GDALProgressFunc pfnProgress=GDALTermProgress;
        pfnProgress(dfComplete,pszMessage,pProgressArg);
        if(GDALPolygonize((GDALRasterBandH)poBandMem, (GDALRasterBandH)poMaskBandMem, (OGRLayerH)ogrWriter.getLayer(),index,NULL,pfnProgress,pProgressArg)!=CE_None){
          cerr << CPLGetLastErrorMsg() << endl;
          throw;
        }
        else{
          dfComplete=1.0;
          pfnProgress(dfComplete,pszMessage,pProgressArg);
        }
        if(verbose_opt[0])
          cout << "number of features: " << OGR_L_GetFeatureCount((OGRLayerH)ogrWriter.getLayer(),TRUE) << endl;
      }
    }
    else{
      std::cerr << "Error: clould not create memory dataset for input datatset" << std::endl;
      throw;
    }
  }
  else{
    std::cerr << "Error: clould not create memory driver" << std::endl;
    throw;
  }
}
