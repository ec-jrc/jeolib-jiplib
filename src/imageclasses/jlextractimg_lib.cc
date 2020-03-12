/**********************************************************************
jlextractimg_lib.cc: extract pixel values from raster image using a raster sample
Author(s): Pieter.Kempeneers@ec.europa.eu
Copyright (c) 2016-2020 European Union (Joint Research Centre)
License EUPLv1.2

This file is part of jiplib
***********************************************************************/
#include <assert.h>
// #include <math.h>
#include <cmath>
#include <stdlib.h>
#include <sstream>
#include <string>
#include <algorithm>
#include <random>
#include <ctime>
#include <vector>
#include "imageclasses/Jim.h"
#include "imageclasses/VectorOgr.h"
#include "base/Optionjl.h"
#include "algorithms/StatFactory.h"
#include "apps/AppFactory.h"
#include "jlextractimg_lib.h"

using namespace std;
using namespace app;

#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif


/**
 * @param app application specific option arguments
 * @return output Vector
 **/
shared_ptr<VectorOgr> Jim::extractImg(Jim& classReader, AppFactory& app){
  shared_ptr<VectorOgr> ogrWriter=VectorOgr::createVector();
  extractImg(classReader, *ogrWriter, app);
  return(ogrWriter);
}

void Jim::extractImg(Jim& classReader, VectorOgr& ogrWriter, app::AppFactory& app){
  if(classReader.getDataType()!=GDT_Byte){
    std::ostringstream errorStream;
    errorStream << "Error: data type must be GDT_Byte" << std::endl;
    throw(errorStream.str());
  }
  switch(getDataType()){
  case(GDT_Byte):
    extractImg_t<unsigned char>(classReader, ogrWriter, app);
    break;
  case(GDT_Int16):
    extractImg_t<short>(classReader, ogrWriter, app);
    break;
  case(GDT_UInt16):
    extractImg_t<unsigned short>(classReader, ogrWriter, app);
    break;
  case(GDT_Int32):
    extractImg_t<int>(classReader, ogrWriter, app);
    break;
  case(GDT_UInt32):
    extractImg_t<unsigned int>(classReader, ogrWriter, app);
    break;
  case(GDT_Float32):
    extractImg_t<float>(classReader, ogrWriter, app);
    break;
  case(GDT_Float64):
    extractImg_t<double>(classReader, ogrWriter, app);
    break;
  default:
    std::string errorString="Error: data type not supported";
    throw(errorString);
    break;
  }
}


// CPLErr Jim::extractImgOld(Jim& classReader, VectorOgr& ogrWriter, app::AppFactory& app){
//   // Optionjl<string> sample_opt("s", "sample", "Raster dataset with features to be extracted from input data. Output will contain features with input band information included.");
//   Optionjl<string> output_opt("o", "output", "Output sample dataset");
//   Optionjl<std::string> layer_opt("ln", "ln", "output layer name","sample");
//   Optionjl<int> class_opt("c", "class", "Class(es) to extract from input sample image. Leave empty to extract all valid data pixels from sample dataset");
//   Optionjl<float> threshold_opt("t", "threshold", "Probability threshold for selecting samples (randomly). Provide probability in percentage (>0) or absolute (<0). If using raster land cover maps as a sample dataset, you can provide a threshold value for each class (e.g. -t 80 -t 60). Use value 100 to select all pixels for selected class(es)", 100);
//   Optionjl<string> ogrformat_opt("f", "oformat", "Output sample dataset format","SQLite");
//   Optionjl<std::string> option_opt("co", "co", "Creation option for output file. Multiple options can be specified.");
//   Optionjl<string> ftype_opt("ft", "ftype", "Field type (only Real or Integer)", "Real");
//   Optionjl<string> ltype_opt("lt", "ltype", "Label type: In16 or String", "Integer");
//   Optionjl<unsigned int> band_opt("b", "band", "Band index(es) to extract (0 based). Leave empty to use all bands");
//   Optionjl<string> bandNames_opt("bn", "bandname", "Band name(s) corresponding to band index(es)","b");
//   Optionjl<unsigned short> bstart_opt("sband", "startband", "Start band sequence number");
//   Optionjl<unsigned short> bend_opt("eband", "endband", "End band sequence number");
//   Optionjl<double> srcnodata_opt("srcnodata", "srcnodata", "Invalid value(s) for input image");
//   Optionjl<unsigned int> bndnodata_opt("bndnodata", "bndnodata", "Band in input image to check if pixel is valid (used for srcnodata)", 0);
//   Optionjl<string> label_opt("cn", "cname", "Name of the class label in the output vector dataset", "label");
//   Optionjl<std::string> fid_opt("fid", "fid", "Create extra field with field identifier (sequence in which the features have been read");
//   Optionjl<short> down_opt("down", "down", "Down sampling factor", 1);
//   Optionjl<unsigned long int>  memory_opt("mem", "mem", "Buffer size (in MB) to read image data blocks in memory",0,1);
//   Optionjl<short> verbose_opt("v", "verbose", "Verbose mode if > 0", 0,2);

//   bstart_opt.setHide(1);
//   bend_opt.setHide(1);
//   bndnodata_opt.setHide(1);
//   srcnodata_opt.setHide(1);
//   label_opt.setHide(1);
//   fid_opt.retrieveOption(app);
//   down_opt.setHide(1);
//   option_opt.setHide(1);
//   memory_opt.setHide(1);

//   bool doProcess;//stop process when program was invoked with help option (-h --help)
//   try{
//     // doProcess=sample_opt.retrieveOption(app);
//     doProcess=output_opt.retrieveOption(app);
//     layer_opt.retrieveOption(app);
//     class_opt.retrieveOption(app);
//     threshold_opt.retrieveOption(app);
//     ogrformat_opt.retrieveOption(app);
//     option_opt.retrieveOption(app);
//     ftype_opt.retrieveOption(app);
//     ltype_opt.retrieveOption(app);
//     band_opt.retrieveOption(app);
//     bandNames_opt.retrieveOption(app);
//     bstart_opt.retrieveOption(app);
//     bend_opt.retrieveOption(app);
//     bndnodata_opt.retrieveOption(app);
//     srcnodata_opt.retrieveOption(app);
//     label_opt.retrieveOption(app);
//     fid_opt.retrieveOption(app);
//     down_opt.retrieveOption(app);
//     memory_opt.retrieveOption(app);
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
//     statfactory::StatFactory stat;
//     if(srcnodata_opt.size()){
//       while(srcnodata_opt.size()<bndnodata_opt.size())
//         srcnodata_opt.push_back(srcnodata_opt[0]);
//       stat.setNoDataValues(srcnodata_opt);
//     }
//     Vector2d<unsigned int> posdata;
//     unsigned long int nsample=0;
//     unsigned long int ntotalvalid=0;
//     unsigned long int ntotalinvalid=0;

//     map<int,unsigned long int> nvalid;
//     map<int,unsigned long int> ninvalid;
//     // vector<unsigned long int> nvalid(class_opt.size());
//     // vector<unsigned long int> ninvalid(class_opt.size());
//     // if(class_opt.empty()){
//     //   nvalid.resize(256);
//     //   ninvalid.resize(256);
//     // }
//     // for(int it=0;it<nvalid.size();++it){
//     //   nvalid[it]=0;
//     //   ninvalid[it]=0;
//     // }

//     map <int,short> classmap;//class->index
//     for(int iclass=0;iclass<class_opt.size();++iclass){
//       nvalid[class_opt[iclass]]=0;
//       ninvalid[class_opt[iclass]]=0;
//       classmap[class_opt[iclass]]=iclass;
//     }

//     // Jim imgReader;
//     // if(image_opt.empty()){
//     //   std::cerr << "No image dataset provided (use option -i). Use --help for help information";
//     //     exit(0);
//     // }
//     if(output_opt.empty()){
//       string errorstring="Error: No output dataset provided (use option -o). Use --help for help information";
//       throw(errorstring);
//     }
//     // try{
//     //   imgReader.open(image_opt[0],memory_opt[0]);
//     // }
//     // catch(std::string errorstring){
//     //   std::cout << errorstring << std::endl;
//     //   exit(0);
//     // }

//     //convert start and end band options to vector of band indexes
//     if(bstart_opt.size()){
//       if(bend_opt.size()!=bstart_opt.size()){
//         string errorstring="Error: options for start and end band indexes must be provided as pairs, missing end band";
//         throw(errorstring);
//       }
//       band_opt.clear();
//       for(int ipair=0;ipair<bstart_opt.size();++ipair){
//         if(bend_opt[ipair]<=bstart_opt[ipair]){
//           string errorstring="Error: index for end band must be smaller then start band";
//           throw(errorstring);
//         }
//         for(int iband=bstart_opt[ipair];iband<=bend_opt[ipair];++iband)
//           band_opt.push_back(iband);
//       }
//     }
//     int nband=(band_opt.size()) ? band_opt.size() : this->nrOfBand();

//     if(bandNames_opt.size()<nband){
//       std::string bandString=bandNames_opt[0];
//       bandNames_opt.clear();
//       bandNames_opt.resize(nband);
//       for(int iband=0;iband<nband;++iband){
//         int theBand=(band_opt.size()) ? band_opt[iband] : iband;
//         ostringstream fs;
//         fs << bandString << theBand;
//         bandNames_opt[iband]=fs.str();
//       }
//     }

//     if(verbose_opt[0])
//       std::cout << bandNames_opt << std::endl;
//     if(verbose_opt[0]>1)
//       std::cout << "Number of bands in input image: " << this->nrOfBand() << std::endl;

//     OGRFieldType fieldType;
//     OGRFieldType labelType;
//     int ogr_typecount=11;//hard coded for now!
//     if(verbose_opt[0]>1)
//       std::cout << "field and label types can be: ";
//     for(int iType = 0; iType < ogr_typecount; ++iType){
//       if(verbose_opt[0]>1)
//         std::cout << " " << OGRFieldDefn::GetFieldTypeName((OGRFieldType)iType);
//       if( OGRFieldDefn::GetFieldTypeName((OGRFieldType)iType) != NULL
//           && EQUAL(OGRFieldDefn::GetFieldTypeName((OGRFieldType)iType),
//                    ftype_opt[0].c_str()))
//         fieldType=(OGRFieldType) iType;
//       if( OGRFieldDefn::GetFieldTypeName((OGRFieldType)iType) != NULL
//           && EQUAL(OGRFieldDefn::GetFieldTypeName((OGRFieldType)iType),
//                    ltype_opt[0].c_str()))
//         labelType=(OGRFieldType) iType;
//     }
//     switch( fieldType ){
//     case OFTInteger:
//     case OFTReal:
//     case OFTRealList:
//     case OFTString:
//       if(verbose_opt[0]>1)
//         std::cout << std::endl << "field type is: " << OGRFieldDefn::GetFieldTypeName(fieldType) << std::endl;
//       break;
//     default:
//       cerr << "field type " << OGRFieldDefn::GetFieldTypeName(fieldType) << " not supported" << std::endl;
//       exit(0);
//       break;
//     }
//     switch( labelType ){
//     case OFTInteger:
//     case OFTReal:
//     case OFTRealList:
//     case OFTString:
//       if(verbose_opt[0]>1)
//         std::cout << std::endl << "label type is: " << OGRFieldDefn::GetFieldTypeName(labelType) << std::endl;
//       break;
//     default:
//       cerr << "label type " << OGRFieldDefn::GetFieldTypeName(labelType) << " not supported" << std::endl;
//       exit(0);
//       break;
//     }

//     const char* pszMessage;
//     void* pProgressArg=NULL;
//     GDALProgressFunc pfnProgress=GDALTermProgress;
//     double progress=0;
//     srand(time(NULL));

//     bool sampleIsRaster=true;

//     // Jim classReader;
//     // ImgWriterOgr sampleWriterOgr;
//     // VectorOgr sampleWriterOgr;

//     // if(sample_opt.size()){
//     //   try{
//     //     classReader.open(sample_opt[0],memory_opt[0]);
//     //   }
//     //   catch(string errorString){
//     //     //todo: sampleIsRaster will not work from GDAL 2.0!!?? (unification of driver for raster and vector datasets)
//     //     sampleIsRaster=false;
//     //   }
//     // }
//     // else{
//     //   std::cerr << "No raster sample dataset provided (use option -s filename). Use --help for help information";
//     //   exit(1);
//     // }

//     char **papszOptions=NULL;
//     for(std::vector<std::string>::const_iterator optionIt=option_opt.begin();optionIt!=option_opt.end();++optionIt)
//       papszOptions=CSLAddString(papszOptions,optionIt->c_str());

//     // OGRSpatialReference classSRS(classReader.getProjectionRef().c_str());
//     OGRSpatialReference classSpatialRef;
//     OGRSpatialReference thisSpatialRef;
//     // OGRSpatialReference thisSRS;

//     classSpatialRef=(classReader.getSpatialRef());
//     thisSpatialRef=getSpatialRef();
//     OGRCoordinateTransformation *ref2img= OGRCreateCoordinateTransformation(&classSpatialRef, &thisSpatialRef);
//     if(thisSpatialRef.IsSame(&classSpatialRef)){
//       // img2ref=0;
//       ref2img=0;
//     }
//     else if(!ref2img){
//       std::ostringstream errorStream;
//       errorStream << "Error: cannot create OGRCoordinateTransformation" << std::endl;
//       throw(errorStream.str());
//     }
//     // std::vector<double> xvector;
//     // std::vector<double> yvector;
//     // std::vector<double> ivector;
//     // std::vector<double> jvector;
//     // if(thisSRS.IsSame(classSpatialRef))
//     //   classSpatialRef=0;//no need to transform
//     // else{
//     //   xvector.resize(classReader.nrOfCol()/down_opt[0]*classReader.nrOfRow()/down_opt[0]);
//     //   yvector.resize(classReader.nrOfCol()/down_opt[0]*classReader.nrOfRow()/down_opt[0]);
//     //   ivector.resize(classReader.nrOfCol()/down_opt[0]*classReader.nrOfRow()/down_opt[0]);
//     //   jvector.resize(classReader.nrOfCol()/down_opt[0]*classReader.nrOfRow()/down_opt[0]);
//     //   for(int irow=0;irow<classReader.nrOfRow();++irow){
//     //     // if(irow%down_opt[0])
//     //     //   continue;
//     //     for(int icol=0;icol<classReader.nrOfCol();++icol){
//     //       // if(icol%down_opt[0])
//     //       //   continue;
//     //       classReader.image2geo(icol,irow,xvector[irow/down_opt[0]*classReader.nrOfCol()/down_opt[0]+icol/down_opt[0]],yvector[irow/down_opt[0]*classReader.nrOfCol()/down_opt[0]+icol/down_opt[0]]);
//     //       //test
//     //       if(irow==100&&icol==100||irow==200&&icol==200)
//     //         std::cout << "icol,irow,xclass,yclass:" << icol << "," << irow << "," << xvector[irow/down_opt[0]*classReader.nrOfCol()/down_opt[0]+icol/down_opt[0]] << "," << yvector[irow/down_opt[0]*classReader.nrOfCol()/down_opt[0]+icol/down_opt[0]] << std::endl;
//     //     }
//     //   }
//     //   //transform
//     //   OGRCoordinateTransformation *poCT = OGRCreateCoordinateTransformation(classSpatialRef,thisSpatialRef);
//     //   if( !poCT ){
//     //     std::ostringstream errorStream;
//     //     errorStream << "Error: cannot create OGRCoordinateTransformation" << std::endl;
//     //     throw(errorStream.str());
//     //   }
//     //   if(!poCT->Transform(xvector.size(),&(xvector[0]),&(yvector[0]))){
//     //     std::ostringstream errorStream;
//     //     errorStream << "Error: cannot apply OGRCoordinateTransformation" << std::endl;
//     //     throw(errorStream.str());
//     //   }
//     //   for(int irow=0;irow<classReader.nrOfRow();++irow){
//     //     // if(irow%down_opt[0])
//     //     //   continue;
//     //     for(int icol=0;icol<classReader.nrOfCol();++icol){
//     //       // if(icol%down_opt[0])
//     //       //   continue;
//     //       geo2image(xvector[irow/down_opt[0]*classReader.nrOfCol()/down_opt[0]+icol/down_opt[0]],yvector[irow/down_opt[0]*classReader.nrOfCol()/down_opt[0]+icol/down_opt[0]],ivector[irow/down_opt[0]*classReader.nrOfCol()/down_opt[0]+icol/down_opt[0]],jvector[irow/down_opt[0]*classReader.nrOfCol()/down_opt[0]+icol/down_opt[0]]);
//     //       //test
//     //       // if(!irow%100&&!icol%100)
//     //       if(irow==100&&icol==100||irow==200&&icol==200)
//     //         std::cout << "ximg,yimg,iimg,jimg:" << xvector[irow/down_opt[0]*classReader.nrOfCol()/down_opt[0]+icol/down_opt[0]] << "," << yvector[irow/down_opt[0]*classReader.nrOfCol()/down_opt[0]+icol/down_opt[0]] << "," << ivector[irow/down_opt[0]*classReader.nrOfCol()/down_opt[0]+icol/down_opt[0]] << "," << jvector[irow/down_opt[0]*classReader.nrOfCol()/down_opt[0]+icol/down_opt[0]] << std::endl;
//     //     }
//     //   }
//     // 
//     if(sampleIsRaster){
//       if(class_opt.empty()){
//         // ImgWriterOgr ogrWriter;
//         // VectorOgr ogrWriter;
//         // if(sample_opt.empty()){
//         //   string errorString="Error: sample raster dataset is missing";
//         //   throw(errorString);
//         // }
//         // classReader.open(sample_opt[0],memory_opt[0]);
//         // vector<int> classBuffer(classReader.nrOfCol());
//         stl::vector<double> classBuffer(classReader.nrOfCol());
//         Vector2d<double> imgBuffer(nband,this->nrOfCol());//[band][col]
//         // vector<double> imgBuffer(nband);//[band]
//         stl::vector<double> sample(2+nband);//x,y,band values
//         Vector2d<double> writeBuffer;
//         vector<int> writeBufferClass;
//         vector<int> selectedClass;
//         Vector2d<double> selectedBuffer;
//         int irow=0;
//         int icol=0;
//         if(verbose_opt[0]>1)
//           std::cout << "extracting sample from image..." << std::endl;
//         progress=0;
//         MyProgressFunc(progress,pszMessage,pProgressArg);
//         for(irow=0;irow<classReader.nrOfRow();++irow){
//           if(irow%down_opt[0])
//             continue;
//           classReader.readData(classBuffer,irow);
//           double x=0;//geo x coordinate
//           double y=0;//geo y coordinate
//           double iimg=0;//image x-coordinate in img image
//           double jimg=0;//image y-coordinate in img image

//           //find col in img
//           classReader.image2geo(icol,irow,x,y);
//           this->geo2image(x,y,iimg,jimg,ref2img);
//           // iimg=ivector[irow/down_opt[0]*classReader.nrOfCol()/down_opt[0]+icol/down_opt[0]];
//           // jimg=jvector[irow/down_opt[0]*classReader.nrOfCol()/down_opt[0]+icol/down_opt[0]];
//           //nearest neighbour
//           if(static_cast<int>(jimg)<0||static_cast<int>(jimg)>=this->nrOfRow())
//             continue;
//           for(int iband=0;iband<nband;++iband){
//             int theBand=(band_opt.size()) ? band_opt[iband] : iband;
//             this->readData(imgBuffer[iband],static_cast<int>(jimg),theBand);
//           }
//           for(icol=0;icol<classReader.nrOfCol();++icol){
//             if(icol%down_opt[0])
//               continue;
//             int theClass=classBuffer[icol];
//             int processClass=0;
//             bool valid=false;
//             if(class_opt.empty()){
//               valid=true;//process every class
//               processClass=theClass;
//             }
//             else{
//               for(int iclass=0;iclass<class_opt.size();++iclass){
//                 if(classBuffer[icol]==class_opt[iclass]){
//                   processClass=iclass;
//                   theClass=class_opt[iclass];
//                   valid=true;//process this class
//                   break;
//                 }
//               }
//             }
//             classReader.image2geo(icol,irow,x,y);
//             sample[0]=x;
//             sample[1]=y;
//             if(verbose_opt[0]>2){
//               std::cout.precision(12);
//               std::cout << theClass << " " << x << " " << y << std::endl;
//             }
//             // find col in img
//             this->geo2image(x,y,iimg,jimg,ref2img);
//             // this->geo2image(ivector[irow/down_opt[0]*classReader.nrOfCol()/down_opt[0]+icol/down_opt[0]],jvector[irow/down_opt[0]*classReader.nrOfCol()/down_opt[0]+icol/down_opt[0]],iimg,jimg);
//             // iimg=ivector[irow/down_opt[0]*classReader.nrOfCol()/down_opt[0]+icol/down_opt[0]];
//             //nearest neighbour
//             iimg=static_cast<int>(iimg);
//             if(static_cast<int>(iimg)<0||static_cast<int>(iimg)>=this->nrOfCol())
//               continue;

//             for(int iband=0;iband<nband&&valid;++iband){
//               int theBand=(band_opt.size()) ? band_opt[iband] : iband;
//               if(srcnodata_opt.size()&&theBand==bndnodata_opt[0]){
//                 // vector<int>::const_iterator bndit=bndnodata_opt.begin();
//                 for(int inodata=0;inodata<srcnodata_opt.size()&&valid;++inodata){
//                   if(imgBuffer[iband][iimg]==srcnodata_opt[inodata])
//                     valid=false;
//                 }
//               }
//             }
//             // oldimgrow=jimg;

//             if(valid){
//               for(int iband=0;iband<imgBuffer.size();++iband){
//                 sample[iband+2]=imgBuffer[iband][iimg];
//               }
//               float theThreshold=(threshold_opt.size()>1)?threshold_opt[processClass]:threshold_opt[0];
//               if(theThreshold>0){//percentual value
//                 double p=static_cast<double>(rand())/(RAND_MAX);
//                 p*=100.0;
//                 if(p>theThreshold)
//                   continue;//do not select for now, go to next column
//               }
//               // else if(nvalid.size()>processClass){//absolute value
//               //   if(nvalid[processClass]>=-theThreshold)
//               //     continue;//do not select any more pixels for this class, go to next column to search for other classes
//               // }
//               writeBuffer.push_back(sample);
//               writeBufferClass.push_back(theClass);
//               ++ntotalvalid;
//               if(nvalid.count(theClass))
//                 nvalid[theClass]+=1;
//               else
//                 nvalid[theClass]=1;
//             }
//             else{
//               ++ntotalinvalid;
//               if(ninvalid.count(theClass))
//                 ninvalid[theClass]+=1;
//               else
//                 ninvalid[theClass]=1;
//             }
//           }
//           progress=static_cast<float>(irow+1.0)/classReader.nrOfRow();
//           MyProgressFunc(progress,pszMessage,pProgressArg);
//         }//irow
//         progress=100;
//         MyProgressFunc(progress,pszMessage,pProgressArg);
//         if(writeBuffer.size()>0){
//           assert(ntotalvalid==writeBuffer.size());
//           if(verbose_opt[0]>0){
//             map<int,unsigned long int>::const_iterator mapit=nvalid.begin();
//             for(mapit=nvalid.begin();mapit!=nvalid.end();++mapit)
//               std::cout << "nvalid for class " << mapit->first << ": " << mapit->second << std::endl;
//             std::cout << "creating image sample writer " << output_opt[0] << " with " << writeBuffer.size() << " samples (" << ntotalinvalid << " invalid)" << std::endl;
//           }
//           // if(ogrWriter.open(output_opt[0],layer_opt,ogrformat_opt[0], wkbPoint, this->getProjectionRef(),papszOptions)!=OGRERR_NONE){
//           if(ogrWriter.open(output_opt[0],ogrformat_opt[0])!=OGRERR_NONE){
//             ostringstream fs;
//             fs << "open ogrWriter failed ";
//             fs << "output name: " << output_opt[0] << ", ";
//             fs << "format: "<< ogrformat_opt[0] << std::endl;
//             throw(fs.str());
//           }
//           if(ogrWriter.pushLayer(layer_opt[0], this->getProjection(),wkbPoint,papszOptions)!=OGRERR_NONE){
//             ostringstream fs;
//             fs << "push layer to ogrWriter with points failed ";
//             fs << "layer name: "<< layer_opt[0] << std::endl;
//             throw(fs.str());
//           }
//           // ogrWriter.open(sample_opt[0],layer_opt[0]);
//           // ostringstream slayer;
//           // slayer << "training data";
//           // std::string layername=slayer.str();
//           // ogrWriter.createLayer(layername, this->getProjection(), wkbPoint, papszOptions);
//           // std::string fieldname="fid";//number of the point
//           // ogrWriter.createField(fieldname,OFTInteger);
//           map<std::string,double> pointAttributes;
//           ogrWriter.createField(label_opt[0],labelType);
//           if(fid_opt.size())
//             ogrWriter.createField(fid_opt[0],OFTInteger64);
//           for(int iband=0;iband<nband;++iband){
//             ogrWriter.createField(bandNames_opt[iband],fieldType);
//           }
//           progress=0;
//           MyProgressFunc(progress,pszMessage,pProgressArg);

//           map<int,short> classDone;
//           Vector2d<double> writeBufferTmp;
//           vector<int> writeBufferClassTmp;

//           if(threshold_opt[0]<0){//absolute threshold
//             map<int,unsigned long int>::iterator mapit;
//             map<int,unsigned long int> ncopied;
//             for(mapit=nvalid.begin();mapit!=nvalid.end();++mapit)
//               ncopied[mapit->first]=0;
//             while(classDone.size()<nvalid.size()){
//               unsigned int index=rand()%writeBufferClass.size();
//               int theClass=writeBufferClass[index];
//               float theThreshold=threshold_opt[0];
//               if(threshold_opt.size()>1&&class_opt.size())
//                 theThreshold=threshold_opt[classmap[theClass]];
//               theThreshold=-theThreshold;
//               if(ncopied[theClass]<theThreshold){
//                 writeBufferClassTmp.push_back(*(writeBufferClass.begin()+index));
//                 writeBufferTmp.push_back(*(writeBuffer.begin()+index));
//                 writeBufferClass.erase(writeBufferClass.begin()+index);
//                 writeBuffer.erase(writeBuffer.begin()+index);
//                 ++(ncopied[theClass]);
//               }
//               else
//                 classDone[theClass]=1;
//               if(ncopied[theClass]>=nvalid[theClass]){
//                 classDone[theClass]=1;
//               }
//             }
//             writeBuffer=writeBufferTmp;
//             writeBufferClass=writeBufferClassTmp;

//             //   while(classDone.size()<nvalid.size()){
//             //     unsigned int index=rand()%writeBufferClass.size();
//             //     int theClass=writeBufferClass[index];
//             //     float theThreshold=threshold_opt[0];
//             //     if(threshold_opt.size()>1&&class_opt.size())
//             //       theThreshold=threshold_opt[classmap[theClass]];
//             //     theThreshold=-theThreshold;
//             //     if(nvalid[theClass]>theThreshold){
//             //       writeBufferClass.erase(writeBufferClass.begin()+index);
//             //       writeBuffer.erase(writeBuffer.begin()+index);
//             //       --(nvalid[theClass]);
//             //     }
//             //     else
//             //       classDone[theClass]=1;
//             //   }
//           }
//           for(unsigned int isample=0;isample<writeBuffer.size();++isample){
//             if(verbose_opt[0]>1)
//               std::cout << "writing sample " << isample << std::endl;
//             pointAttributes[label_opt[0]]=writeBufferClass[isample];
//             if(fid_opt.size())
//               pointAttributes[fid_opt[0]]=isample;
//             for(int iband=0;iband<writeBuffer[0].size()-2;++iband){
//               pointAttributes[bandNames_opt[iband]]=writeBuffer[isample][iband+2];
//             }
//             if(verbose_opt[0]>1)
//               std::cout << "all bands written" << std::endl;
//             // ogrWriter.addPoint(writeBuffer[isample][0],writeBuffer[isample][1],pointAttributes,fieldname,isample);
//             ogrWriter.addPoint(writeBuffer[isample][0],writeBuffer[isample][1],pointAttributes);
//             progress=static_cast<float>(isample+1.0)/writeBuffer.size();
//             MyProgressFunc(progress,pszMessage,pProgressArg);
//           }
//         }
//         else{
//           std::cout << "No data found for any class " << std::endl;
//         }
//         // classReader.close();
//         nsample=writeBuffer.size();
//         if(verbose_opt[0])
//           std::cout << "total number of samples written: " << nsample << std::endl;
//       }
//       else{//class_opt.size()!=0
//         assert(class_opt[0]);
//         //   if(class_opt[0]){
//         assert(threshold_opt.size()==1||threshold_opt.size()==class_opt.size());
//         // Jim classReader;
//         // ImgWriterOgr ogrWriter;
//         // VectorOgr ogrWriter;
//         if(verbose_opt[0]>1){
//           std::cout << "reading position from sample dataset " << std::endl;
//           std::cout << "class thresholds: " << std::endl;
//           for(int iclass=0;iclass<class_opt.size();++iclass){
//             if(threshold_opt.size()>1)
//               std::cout << class_opt[iclass] << ": " << threshold_opt[iclass] << std::endl;
//             else
//               std::cout << class_opt[iclass] << ": " << threshold_opt[0] << std::endl;
//           }
//         }
//         if(verbose_opt[0]>1)
//           std::cout << "opening sample" << std::endl;
//         // classReader.open(sample_opt[0],memory_opt[0]);
//         vector<int> classBuffer(classReader.nrOfCol());
//         // vector<double> classBuffer(classReader.nrOfCol());
//         Vector2d<double> imgBuffer(nband,this->nrOfCol());//[band][col]
//         // vector<double> imgBuffer(nband);//[band]
//         vector<double> sample(2+nband);//x,y,band values
//         Vector2d<double> writeBuffer;
//         vector<int> writeBufferClass;
//         vector<int> selectedClass;
//         Vector2d<double> selectedBuffer;
//         int irow=0;
//         int icol=0;
//         if(verbose_opt[0]>1)
//           std::cout << "extracting sample from image..." << std::endl;

//         progress=0;
//         MyProgressFunc(progress,pszMessage,pProgressArg);
//         for(irow=0;irow<classReader.nrOfRow();++irow){
//           if(irow%down_opt[0])
//             continue;
//           classReader.readData(classBuffer,irow);
//           double x=0;//geo x coordinate
//           double y=0;//geo y coordinate
//           double iimg=0;//image x-coordinate in img image
//           double jimg=0;//image y-coordinate in img image

//           //find row in img
//           classReader.image2geo(icol,irow,x,y);
//           this->geo2image(x,y,iimg,jimg,ref2img);
//           //   ostringstream fs;
//           //   fs << "Error: geo2image failed to find row in image";
//           //   throw(fs.str());
//           // }
//           // this->geo2image(ivector[irow/down_opt[0]*classReader.nrOfCol()/down_opt[0]+icol/down_opt[0]],jvector[irow/down_opt[0]*classReader.nrOfCol()/down_opt[0]+icol/down_opt[0]],iimg,jimg);
//           // iimg=ivector[irow/down_opt[0]*classReader.nrOfCol()/down_opt[0]+icol/down_opt[0]];
//           // jimg=jvector[irow/down_opt[0]*classReader.nrOfCol()/down_opt[0]+icol/down_opt[0]];
//           //nearest neighbour
//           if(static_cast<int>(jimg)<0||static_cast<int>(jimg)>=this->nrOfRow())
//             continue;
//           for(int iband=0;iband<nband;++iband){
//             int theBand=(band_opt.size()) ? band_opt[iband] : iband;
//             this->readData(imgBuffer[iband],static_cast<int>(jimg),theBand);
//           }

//           for(icol=0;icol<classReader.nrOfCol();++icol){
//             if(icol%down_opt[0])
//               continue;
//             int theClass=0;
//             // double theClass=0;
//             int processClass=-1;
//             if(class_opt.empty()){//process every class
//               if(classBuffer[icol]){
//                 processClass=0;
//                 theClass=classBuffer[icol];
//               }
//             }
//             else{
//               for(int iclass=0;iclass<class_opt.size();++iclass){
//                 if(classBuffer[icol]==class_opt[iclass]){
//                   processClass=iclass;
//                   theClass=class_opt[iclass];
//                 }
//               }
//             }
//             if(processClass>=0){
//               classReader.image2geo(icol,irow,x,y);
//               sample[0]=x;
//               sample[1]=y;
//               if(verbose_opt[0]>1){
//                 std::cout.precision(12);
//                 std::cout << theClass << " " << x << " " << y << std::endl;
//               }
//               //find col in img
//               this->geo2image(x,y,iimg,jimg,ref2img);
//               //   ostringstream fs;
//               //   fs << "Error: geo2image failed to find col in img";
//               //   throw(fs.str());
//               // }
//               // this->geo2image(ivector[irow/down_opt[0]*classReader.nrOfCol()/down_opt[0]+icol/down_opt[0]],jvector[irow/down_opt[0]*classReader.nrOfCol()/down_opt[0]+icol/down_opt[0]],iimg,jimg);
//               // //debug
//               // if(!irow%100&&!icol%100)
//               //   std::cout << "x,y,iimg,jimg: " << x << ", " << y << ", " << iimg << ", " << jimg << std::endl;
//               // iimg=ivector[irow/down_opt[0]*classReader.nrOfCol()/down_opt[0]+icol/down_opt[0]];
//               //nearest neighbour
//               iimg=static_cast<int>(iimg);
//               if(static_cast<int>(iimg)<0||static_cast<int>(iimg)>=this->nrOfCol())
//                 continue;
//               bool valid=true;

//               for(int iband=0;iband<nband;++iband){
//                 int theBand=(band_opt.size()) ? band_opt[iband] : iband;
//                 if(srcnodata_opt.size()&&theBand==bndnodata_opt[0]){
//                   // vector<int>::const_iterator bndit=bndnodata_opt.begin();
//                   for(int inodata=0;inodata<srcnodata_opt.size()&&valid;++inodata){
//                     if(imgBuffer[iband][iimg]==srcnodata_opt[inodata])
//                       valid=false;
//                   }
//                 }
//               }
//               if(valid){
//                 for(int iband=0;iband<imgBuffer.size();++iband){
//                   sample[iband+2]=imgBuffer[iband][iimg];
//                 }
//                 float theThreshold=(threshold_opt.size()>1)?threshold_opt[processClass]:threshold_opt[0];
//                 if(theThreshold>0){//percentual value
//                   double p=static_cast<double>(rand())/(RAND_MAX);
//                   p*=100.0;
//                   if(p>theThreshold)
//                     continue;//do not select for now, go to next column
//                 }
//                 // else if(nvalid.size()>processClass){//absolute value
//                 //   if(nvalid[processClass]>=-theThreshold)
//                 //     continue;//do not select any more pixels for this class, go to next column to search for other classes
//                 // }
//                 writeBuffer.push_back(sample);
//                 writeBufferClass.push_back(theClass);
//                 ++ntotalvalid;
//                 if(nvalid.count(theClass))
//                   nvalid[theClass]+=1;
//                 else
//                   nvalid[theClass]=1;
//               }
//               else{
//                 ++ntotalinvalid;
//                 if(ninvalid.count(theClass))
//                   ninvalid[theClass]+=1;
//                 else
//                   ninvalid[theClass]=1;
//               }
//             }//processClass
//           }//icol
//           progress=static_cast<float>(irow+1.0)/classReader.nrOfRow();
//           MyProgressFunc(progress,pszMessage,pProgressArg);
//         }//irow
//         if(writeBuffer.size()>0){
//           if(verbose_opt[0]>0){
//             map<int,unsigned long int>::const_iterator mapit=nvalid.begin();
//             for(mapit=nvalid.begin();mapit!=nvalid.end();++mapit)
//               std::cout << "nvalid for class " << mapit->first << ": " << mapit->second << std::endl;
//             std::cout << "creating image sample writer " << output_opt[0] << " with " << writeBuffer.size() << " samples (" << ntotalinvalid << " invalid)" << std::endl;
//           }
//           assert(ntotalvalid==writeBuffer.size());
//           if(verbose_opt[0]>0)
//             std::cout << "creating image sample writer " << output_opt[0] << " collecting from " << writeBuffer.size() << " samples (" << ntotalinvalid << " invalid)" << std::endl;
//           // ogrWriter.open(output_opt[0],ogrformat_opt[0]);
//           if(ogrWriter.open(output_opt[0],ogrformat_opt[0])!=OGRERR_NONE){
//             ostringstream fs;
//             fs << "open ogrWriter failed ";
//             fs << "output name: " << output_opt[0] << ", ";
//             fs << "format: "<< ogrformat_opt[0] << std::endl;
//             throw(fs.str());
//           }
//           if(ogrWriter.pushLayer(layer_opt[0], this->getProjectionRef(),wkbPoint,papszOptions)!=OGRERR_NONE){
//             ostringstream fs;
//             fs << "push layer to ogrWriter with points failed, ";
//             fs << "layer name: "<< layer_opt[0] << std::endl;
//             throw(fs.str());
//           }
//           // if(ogrWriter.open(output_opt[0],layer_opt,ogrformat_opt[0], wkbPoint, this->getProjectionRef(),papszOptions)!=OGRERR_NONE){
//           //   ostringstream fs;
//           //   fs << "open ogrWriter with points failed ";
//           //   fs << "output name: " << output_opt[0] << ", ";
//           //   fs << "layer name: "<< layer_opt[0] << ", ";
//           //   fs << "format: "<< ogrformat_opt[0] << std::endl;
//           //   throw(fs.str());
//           // }
//           // ostringstream slayer;
//           // slayer << "training data";
//           // std::string layername=slayer.str();
//           // ogrWriter.createLayer(layername, this->getProjection(), wkbPoint, papszOptions);
//           // std::string fieldname="fid";//number of the point
//           // ogrWriter.createField(fieldname,OFTInteger);
//           map<std::string,double> pointAttributes;
//           ogrWriter.createField(label_opt[0],labelType);
//           if(fid_opt.size())
//             ogrWriter.createField(fid_opt[0],OFTInteger64);
//           for(int iband=0;iband<nband;++iband){
//             ogrWriter.createField(bandNames_opt[iband],fieldType);
//           }
//           MyProgressFunc(progress,pszMessage,pProgressArg);
//           progress=0;
//           MyProgressFunc(progress,pszMessage,pProgressArg);

//           map<int,short> classDone;
//           Vector2d<double> writeBufferTmp;
//           vector<int> writeBufferClassTmp;

//           if(threshold_opt[0]<0){//absolute threshold
//             map<int,unsigned long int>::iterator mapit;
//             map<int,unsigned long int> ncopied;
//             for(mapit=nvalid.begin();mapit!=nvalid.end();++mapit){
//               ncopied[mapit->first]=0;
//               if(!mapit->second)
//                 classDone[mapit->first]=1;
//             }
//             while(classDone.size()<nvalid.size()){
//               unsigned int index=rand()%writeBufferClass.size();
//               int theClass=writeBufferClass[index];
//               float theThreshold=threshold_opt[0];
//               if(threshold_opt.size()>1&&class_opt.size())
//                 theThreshold=threshold_opt[classmap[theClass]];
//               theThreshold=-theThreshold;
//               if(ncopied[theClass]<theThreshold){
//                 writeBufferClassTmp.push_back(*(writeBufferClass.begin()+index));
//                 writeBufferTmp.push_back(*(writeBuffer.begin()+index));
//                 writeBufferClass.erase(writeBufferClass.begin()+index);
//                 writeBuffer.erase(writeBuffer.begin()+index);
//                 ++(ncopied[theClass]);
//               }
//               else
//                 classDone[theClass]=1;
//               if(ncopied[theClass]>=nvalid[theClass]){
//                 classDone[theClass]=1;
//               }
//             }
//             writeBuffer=writeBufferTmp;
//             writeBufferClass=writeBufferClassTmp;
//             // while(classDone.size()<nvalid.size()){
//             //   unsigned int index=rand()%writeBufferClass.size();
//             //   int theClass=writeBufferClass[index];
//             //   float theThreshold=threshold_opt[0];
//             //   if(threshold_opt.size()>1&&class_opt.size())
//             //     theThreshold=threshold_opt[classmap[theClass]];
//             //   theThreshold=-theThreshold;
//             //   if(nvalid[theClass]>theThreshold){
//             //     writeBufferClass.erase(writeBufferClass.begin()+index);
//             //     writeBuffer.erase(writeBuffer.begin()+index);
//             //     --(nvalid[theClass]);
//             //   }
//             //   else
//             //     classDone[theClass]=1;
//             // }
//           }

//           for(unsigned int isample=0;isample<writeBuffer.size();++isample){
//             pointAttributes[label_opt[0]]=writeBufferClass[isample];
//             if(fid_opt.size())
//               pointAttributes[fid_opt[0]]=isample;
//             for(int iband=0;iband<writeBuffer[0].size()-2;++iband){
//               pointAttributes[bandNames_opt[iband]]=writeBuffer[isample][iband+2];
//             }
//             // ogrWriter.addPoint(writeBuffer[isample][0],writeBuffer[isample][1],pointAttributes,fid_opt[0],isample);
//             ogrWriter.addPoint(writeBuffer[isample][0],writeBuffer[isample][1],pointAttributes);
//             progress=static_cast<float>(isample+1.0)/writeBuffer.size();
//             MyProgressFunc(progress,pszMessage,pProgressArg);
//           }
//         }
//         else{
//           std::cout << "No data found for any class " << std::endl;
//         }
//         // classReader.close();
//         nsample=writeBuffer.size();
//         if(verbose_opt[0]){
//           std::cout << "total number of samples written: " << nsample << std::endl;
//           std::cout << "total number of valid samples: " << ntotalvalid << std::endl;
//           std::cout << "total number of invalid samples: " << ntotalinvalid << std::endl;
//           if(nvalid.size()==class_opt.size()){
//             for(int iclass=0;iclass<class_opt.size();++iclass)
//               std::cout << "class " << class_opt[iclass] << " has " << nvalid[class_opt[iclass]] << " samples" << std::endl;
//           }
//         }
//       }
//     }
//     else{//vector dataset
//       cerr << "Error: vector sample not supported, consider using pkextractogr" << endl;
//     }//else (vector)
//     progress=1.0;
//     MyProgressFunc(progress,pszMessage,pProgressArg);
//     // this->close();
//   }
//   catch(string predefinedString){
//     std::cout << predefinedString << std::endl;
//     throw;
//   }
// }
