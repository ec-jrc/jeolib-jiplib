/**********************************************************************
jlwarp.cc: warp Jim using GDAL warp algorithm
Author(s): Pieter.Kempeneers@ec.europa.eu
Copyright (c) 2016-2018 European Union (Joint Research Centre)
License EUPLv1.2

This file is part of jiplib
***********************************************************************/
#include <iostream>
#include "ogr_spatialref.h"
extern "C" {
#include "gdal_alg.h"
#include "gdalwarper.h"
}
#include <config_jiplib.h>
#include "Jim.h"
#include "base/Optionjl.h"

using namespace std;
///convert Jim image in memory returning Jim image (alias for crop)
std::shared_ptr<Jim> Jim::warp(app::AppFactory& theApp){
  std::shared_ptr<Jim> imgWriter=Jim::createImg();
  warp(*imgWriter, theApp);
  return(imgWriter);
}

///convert Jim image in memory returning Jim image (alias for crop)
void Jim::warp(Jim& imgWriter, app::AppFactory &theApp){
  Optionjl<std::string> sourceSRS_opt("s_srs", "s_srs", "Source spatial reference for the input file, e.g., epsg:3035 to use European projection and force to European grid");
  Optionjl<std::string> targetSRS_opt("t_srs", "t_srs", "Target spatial reference for the output file, e.g., epsg:3035 to use European projection and force to European grid");
  Optionjl<std::string> resample_opt("r", "resample", "resample: GRIORA_NearestNeighbour|GRIORA_Bilinear|GRIORA_Cubic|GRIORA_CubicSpline|GRIORA_Lanczos|GRIORA_Average|GRIORA_Average|GRIORA_Gauss (check http://www.gdal.org/gdal_8h.html#a640ada511cbddeefac67c548e009d5a)","GRIORA_NearestNeighbour");
  Optionjl<double> nodata_opt("nodata", "nodata", "Nodata value to put in image.",0);
  Optionjl<std::string> warp_opt("wo", "wo", "Warp option(s). Multiple options can be specified.");
  Optionjl<std::string>  otype_opt("ot", "otype", "Data type for output image ({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}). Empty string: inherit type from input image");
  Optionjl<short> verbose_opt("verbose", "verbose", "verbose output",0,2);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=sourceSRS_opt.retrieveOption(theApp);
    targetSRS_opt.retrieveOption(theApp);
    resample_opt.retrieveOption(theApp);
    nodata_opt.retrieveOption(theApp);
    warp_opt.retrieveOption(theApp);
    otype_opt.retrieveOption(theApp);
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

  GDALDataType theType=getGDALDataType();
  if(otype_opt.size()){
    theType=string2GDAL(otype_opt[0]);
    if(theType==GDT_Unknown)
      std::cout << "Warning: unknown output pixel type: " << otype_opt[0] << ", using input type as default" << std::endl;
  }
  if(verbose_opt[0])
    std::cout << "Output pixel type:  " << GDALGetDataTypeName(theType) << std::endl;

  OGRSpatialReference sourceSpatialRef;
  OGRSpatialReference targetSpatialRef;
  sourceSpatialRef.SetFromUserInput(getProjectionRef().c_str());
  targetSpatialRef.SetFromUserInput(targetSRS_opt[0].c_str());
  if(sourceSpatialRef.IsSame(&targetSpatialRef)){
    if(verbose_opt[0])
      std::cout << "source SRS is same as target SRS, no warp is needed (just converting):  " << GDALGetDataTypeName(theType) << std::endl;
    app::AppFactory convertApp(theApp);
    convertApp.clearOption("t_srs");
    convertApp.clearOption("resample");
    Jim::convert(imgWriter,convertApp);
  }
  // char *sourceWKT=getProjectionRef().c_str();
  char *targetWKT=0;
  targetSpatialRef.exportToWkt(&targetWKT);
  std::vector<double> sourceGT(6);
  std::vector<double> targetGT(6);
  getGeoTransform(sourceGT);
  //todo: warp first band to open imgWriter and consecutive bands in parallel
  for(int iband=0;iband<nrOfBand();++iband){
    if(verbose_opt[0])
      std::cout << "warping band " << iband << std::endl;
    const char *pszFormat = "MEM";
    GDALDriver *poDriverMem = GetGDALDriverManager()->GetDriverByName(pszFormat);
    GDALDataset *poDatasetMem = NULL;
    GDALRasterBand *poBandMem = NULL;
    if( poDriverMem ){
      poDatasetMem = (GDALDataset *)poDriverMem->Create("memband",nrOfCol(),nrOfRow(),1,getGDALDataType(),NULL);
      if( poDatasetMem ){
        //printf("Copy MemoryBand to memband\n"); fflush(stdout);
        poDatasetMem->SetGeoTransform(&sourceGT[0]);
        poBandMem = poDatasetMem->GetRasterBand(1);
        if( poBandMem->RasterIO(GF_Write, 0, 0, nrOfCol(), nrOfRow(), m_data[iband], nrOfCol(), nrOfRow(), getGDALDataType(), 0, 0, NULL) == CE_None ){
          // Quality parameters for warping operation
          m_resample=getGDALResample(resample_opt[0]);
          GDALResampleAlg eResampleAlg = (GDALResampleAlg)m_resample;
          poBandMem->SetNoDataValue(nodata_opt[0]);
          double dfWarpMemoryLimit = 0.0;
          double dfMaxError = 0.0;
          GDALProgressFunc pfnProgress = NULL;
          void *pProgressArg = NULL;
          GDALWarpOptions *psOptions = GDALCreateWarpOptions();

          // Create a transformation object from the source to destination coordinate system
          poDatasetMem->SetGeoTransform(&sourceGT[0]);
          void *hTransformArg = GDALCreateGenImgProjTransformer(poDatasetMem, getProjectionRef().c_str(), NULL, targetWKT, TRUE, 1000.0, 0);
          if( hTransformArg != NULL ){
            // Get approximate output definition
            int nPixels, nLines;
            if( GDALSuggestedWarpOutput(poDatasetMem, GDALGenImgProjTransform, hTransformArg, &targetGT[0], &nPixels, &nLines) == CE_None ){
              GDALDestroyGenImgProjTransformer(hTransformArg);

              // Create the output memory band
              GDALDataset *poDatasetOut = (GDALDataset *)poDriverMem->Create("outband",nPixels,nLines,1,theType,NULL);
              if( poDatasetOut != NULL ){
                // Write out the projection definition
                poDatasetOut->SetProjection(targetWKT);
                poDatasetOut->SetGeoTransform(&targetGT[0]);

                psOptions->papszWarpOptions = CSLSetNameValue(psOptions->papszWarpOptions,"INIT_DEST",type2string<double>(nodata_opt[0]).c_str());
                // psOptions->GDALWarpOptions  = CSLSetNameValue(psOptions->papszWarpOptions,"eResampleAlg",resample_opt[0]);
                psOptions->papszWarpOptions = CSLSetNameValue(psOptions->papszWarpOptions,"eResampleAlg",resample_opt[0].c_str());
                // Perform the reprojection
                if( GDALReprojectImage(poDatasetMem, getProjectionRef().c_str(), poDatasetOut, targetWKT, eResampleAlg, dfWarpMemoryLimit, dfMaxError, pfnProgress, pProgressArg, psOptions) == CE_None ){
                  // Copy pixels to pout
                  GDALRasterBand *poBandOut = poDatasetOut->GetRasterBand(1);
                  if( poBandOut ){
                    if(!iband){
                      // app::AppFactory writerApp;
                      // writerApp.setLongOption("ncol",poBandOut->GetXSize());
                      // writerApp.setLongOption("nrow",poBandOut->GetYSize());
                      // writerApp.setLongOption("nband",nrOfBand());
                      // if(otype_opt.size())
                      //   writerApp.setLongOption("otype",otype_opt[0]);
                      // imgWriter.open(writerApp);
                      imgWriter.open(poBandOut->GetXSize(),poBandOut->GetYSize(),nrOfBand(),theType);
                      imgWriter.setGeoTransform(targetGT);
                      imgWriter.setProjection(targetWKT);
                      imgWriter.setNoData(nodata_opt);
                    }
                    if( poBandOut->RasterIO(GF_Read, 0, 0, imgWriter.nrOfCol(), imgWriter.nrOfRow(), imgWriter.getDataPointer(iband), imgWriter.nrOfCol(), imgWriter.nrOfRow(), imgWriter.getGDALDataType(), 0, 0, NULL) != CE_None ){
                      std::string errorString="Error: could not read band from RasterIO";
                      std::cerr << errorString << std::endl;
                      throw(errorString);
                    }
                  }
                  GDALClose(poDatasetOut);
                  if(psOptions->papszWarpOptions) CSLDestroy(psOptions->papszWarpOptions);
                  psOptions->papszWarpOptions = NULL;
                }
              }
              GDALDestroyWarpOptions(psOptions);
            }
          }
          GDALClose(poDatasetMem);
        }
      }
    }
  }
}
