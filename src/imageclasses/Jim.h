/**********************************************************************
Jim.h: class to read/write raster files using GDAL API library
Author(s): Pieter.Kempeneers@ec.europa.eu
Copyright (c) 2016-2019 European Union (Joint Research Centre)
License EUPLv1.2

This file is part of jiplib
***********************************************************************/
/*
   Changes copied by Pieter from version 2.6.8:
   2017-05-04  Kris Vanhoof (vhoofk): Fix rounding issues in bilinear interpolation
   2017-05-08  Kris Vanhoof (vhoofk): Fix rounding issues in bilinear interpolation
   2017-05-08  Kris Vanhoof (vhoofk): Handle nodata values in bilinear interpolation
*/
#ifndef _JIM_H_
#define _JIM_H_

#include <cfloat>
#include <typeinfo>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <list>
#include <algorithm>
#include <utility>
#include <memory>
#include <assert.h>
#include "gdal_priv.h"
#include "base/Vector2d.h"
#include "JimList.h"
#include "apps/AppFactory.h"
#include "config_jiplib.h"

#if MIALIB == 1
extern "C" {
#include "mialib/mialib_swig.h"
#include "mialib/mialib_convolve.h"
#include "mialib/mialib_dem.h"
#include "mialib/mialib_dist.h"
#include "mialib/mialib_erodil.h"
#include "mialib/mialib_format.h"
#include "mialib/mialib_geodesy.h"
#include "mialib/mialib_geometry.h"
#include "mialib/mialib_hmt.h"
#include "mialib/mialib_imem.h"
#include "mialib/mialib_io.h"
#include "mialib/mialib_label.h"
#include "mialib/mialib_miscel.h"
#include "mialib/mialib_opclo.h"
#include "mialib/mialib_pointop.h"
#include "mialib/mialib_proj.h"
#include "mialib/mialib_segment.h"
#include "mialib/mialib_stats.h"
#include "mialib/op.h"
}
#endif

#include <Python.h>
/* #include "numpy/arrayobject.h" */

namespace app{
  class AppFactory;
}

#if MIALIB
enum JIPLIBDataType {JDT_Unknown=0, JDT_Int64=14, JDT_UInt64=15, JDT_Word=16};
#endif

namespace cover{
  enum COVER_TYPE {ALL_TOUCHED=0, ALL_COVERED=1, ALL_CENTER=2};
}

enum DATA_ACCESS { READ_ONLY = 0, UPDATE = 1, WRITE = 3};
enum RESAMPLE { NEAR = 0, BILINEAR = 1, BICUBIC = 2 };
/* enum   GDALDataType { */
/*   GDT_Unknown = 0, GDT_Byte = 1, GDT_UInt16 = 2, GDT_Int16 = 3, */
/*   GDT_UInt32 = 4, GDT_Int32 = 5, GDT_Float32 = 6, GDT_Float64 = 7, */
/*   GDT_CInt16 = 8, GDT_CInt32 = 9, GDT_CFloat32 = 10, GDT_CFloat64 = 11, */
/*   GDT_TypeCount = 12 */
/* } */

/**
 * @param C++ data type to be converted to GDAL data type
 * @return the GDAL data type that corresponds to the given C++ data type
 **/
//template<typename T1> GDALDataType getGDALDataType(){
template<typename T1> GDALDataType type2GDAL(){
  if (typeid(T1) == typeid(char))
    return GDT_Byte;
  else if (typeid(T1) == typeid(unsigned char))
    return GDT_Byte;
  else if (typeid(T1) == typeid(unsigned short))
    return GDT_UInt16;
  else if (typeid(T1) == typeid(short))
    return GDT_Int16;
  else if (typeid(T1) == typeid(int))
    return GDT_Int32;
  else if (typeid(T1) == typeid(unsigned int))
    return GDT_UInt32;
  else if (typeid(T1) == typeid(long))
    return GDT_Int32;
  else if (typeid(T1) == typeid(unsigned long))
    return GDT_UInt32;
  else if (typeid(T1) == typeid(float))
    return GDT_Float32;
  else if (typeid(T1) == typeid(double))
    return GDT_Float64;
  else
    return GDT_Unknown;
};

static GDALDataType string2GDAL(const std::string &typeString){
  std::map<std::string,GDALDataType> typeMap;
  typeMap["Byte"]=GDT_Byte;
  typeMap["UInt16"]=GDT_UInt16;
  typeMap["Int16"]=GDT_Int16;
  typeMap["UInt32"]=GDT_UInt32;
  typeMap["Int32"]=GDT_Int32;
  typeMap["Float32"]=GDT_Float32;
  typeMap["Float64"]=GDT_Float64;
  typeMap["GDT_Byte"]=GDT_Byte;
  typeMap["GDT_UInt16"]=GDT_UInt16;
  typeMap["GDT_Int16"]=GDT_Int16;
  typeMap["GDT_UInt32"]=GDT_UInt32;
  typeMap["GDT_Int32"]=GDT_Int32;
  typeMap["GDT_Float32"]=GDT_Float32;
  typeMap["GDT_Float64"]=GDT_Float64;
  typeMap["int8"]=GDT_Byte;
  typeMap["uint8"]=GDT_Byte;
  typeMap["int16"]=GDT_Int16;
  typeMap["uint16"]=GDT_UInt16;
  typeMap["uint32"]=GDT_UInt32;
  typeMap["int32"]=GDT_Int32;
  typeMap["float32"]=GDT_Float32;
  typeMap["float64"]=GDT_Float64;
  typeMap["Int8"]=GDT_Byte;
  typeMap["UInt8"]=GDT_Byte;
  typeMap["UInt8"]=GDT_Byte;
  typeMap["UInt16"]=GDT_UInt16;
  typeMap["Int16"]=GDT_Int16;
  typeMap["UInt32"]=GDT_UInt32;
  typeMap["Int32"]=GDT_Int32;
  typeMap["Float32"]=GDT_Float32;
  typeMap["Float64"]=GDT_Float64;
  if(typeMap.count(typeString))
    return(typeMap[typeString]);
  else
    return(GDT_Unknown);
}

static JIPLIBDataType string2JDT(const std::string &typeString){
  std::map<std::string,JIPLIBDataType> typeMap;
  typeMap["UInt64"]=JDT_UInt64;
  typeMap["Int64"]=JDT_Int64;
  typeMap["JDT_UInt64"]=JDT_UInt64;
  typeMap["JDT_Int64"]=JDT_Int64;
  typeMap["JDT_Word"]=JDT_Word;
  if(typeMap.count(typeString))
    return(typeMap[typeString]);
  else
    return(JDT_Unknown);
}

static int getDataType(const std::string &typeString){
  //initialize selMap
  /* std::map<std::string,GDALDataType> typeMap; */
  std::map<std::string,int> typeMap;
  typeMap["Byte"]=GDT_Byte;
  typeMap["UInt16"]=GDT_UInt16;
  typeMap["Int16"]=GDT_Int16;
  typeMap["UInt32"]=GDT_UInt32;
  typeMap["Int32"]=GDT_Int32;
  typeMap["Float32"]=GDT_Float32;
  typeMap["Float64"]=GDT_Float64;
  typeMap["GDT_Byte"]=GDT_Byte;
  typeMap["GDT_UInt16"]=GDT_UInt16;
  typeMap["GDT_Int16"]=GDT_Int16;
  typeMap["GDT_UInt32"]=GDT_UInt32;
  typeMap["GDT_Int32"]=GDT_Int32;
  typeMap["GDT_Float32"]=GDT_Float32;
  typeMap["GDT_Float64"]=GDT_Float64;
  typeMap["Int64"]=JDT_Int64;
  typeMap["UInt64"]=JDT_UInt64;
  typeMap["JDT_Int64"]=JDT_Int64;
  typeMap["JDT_UInt64"]=JDT_UInt64;
  typeMap["JDT_Word"]=JDT_Word;
  typeMap["int8"]=GDT_Byte;
  typeMap["uint8"]=GDT_Byte;
  typeMap["int16"]=GDT_Int16;
  typeMap["uint16"]=GDT_UInt16;
  typeMap["uint32"]=GDT_UInt32;
  typeMap["int32"]=GDT_Int32;
  typeMap["float32"]=GDT_Float32;
  typeMap["float64"]=GDT_Float64;
  typeMap["Int8"]=GDT_Byte;
  typeMap["UInt8"]=GDT_Byte;
  typeMap["UInt8"]=GDT_Byte;
  typeMap["UInt16"]=GDT_UInt16;
  typeMap["Int16"]=GDT_Int16;
  typeMap["UInt32"]=GDT_UInt32;
  typeMap["Int32"]=GDT_Int32;
  typeMap["Float32"]=GDT_Float32;
  typeMap["Float64"]=GDT_Float64;
  if(typeMap.count(typeString))
    return(typeMap[typeString]);
  else
    return(GDT_Unknown);
}

static std::size_t getDataTypeSizeBytes(const std::string &typeString){
  int typeInt=getDataType(typeString);
  switch (typeInt){
  case JDT_UInt64:
  case JDT_Int64:
    return(static_cast<std::size_t>(8));
  case JDT_Word:
    return(static_cast<std::size_t>(1));
  default:{
    if(typeInt==GDT_Unknown){
      std::string errorString="Error: data type not supported";
      throw(errorString);
    }
    return(static_cast<std::size_t>(GDALGetDataTypeSize(static_cast<GDALDataType>(typeInt))>>3));
  }
  }
}

static GDALRIOResampleAlg getGDALResample(const std::string &resampleString){
  //initialize selMap
  std::map<std::string,GDALRIOResampleAlg> resampleMap;
  resampleMap["NearestNeighbour"]=GRIORA_NearestNeighbour;
  resampleMap["Bilinear"] = GRIORA_Bilinear;
  resampleMap["Cubic"]=GRIORA_Cubic;
  resampleMap["CubicSpline"]=GRIORA_CubicSpline;
  resampleMap["Lanczos"]=GRIORA_Lanczos;
  resampleMap["Average"]=GRIORA_Average;
  resampleMap["Mode"]=GRIORA_Mode;
  resampleMap["Gauss"]=GRIORA_Gauss;
  resampleMap["GRIORA_NearestNeighbour"]=GRIORA_NearestNeighbour;
  resampleMap["GRIORA_Bilinear"] = GRIORA_Bilinear;
  resampleMap["GRIORA_Cubic"]=GRIORA_Cubic;
  resampleMap["GRIORA_CubicSpline"]=GRIORA_CubicSpline;
  resampleMap["GRIORA_Lanczos"]=GRIORA_Lanczos;
  resampleMap["GRIORA_Average"]=GRIORA_Average;
  resampleMap["GRIORA_Mode"]=GRIORA_Mode;
  resampleMap["GRIORA_Gauss"]=GRIORA_Gauss;
  if(resampleMap.count(resampleString))
    return(resampleMap[resampleString]);
  else
    return(GRIORA_NearestNeighbour);
}

class JimList;
class VectorOgr;

//todo: create name space jiplib?
/**
   Base class for raster dataset (read and write) in a format supported by GDAL. This general raster class is used to store e.g., filename, number of columns, rows and bands of the dataset.
**/
class Jim : public std::enable_shared_from_this<Jim>
{
 public:
    ///default constructor
  Jim();
  /* Jim(PyObject* npArray, bool copyData=false); */
    ///constructor opening an image in memory using an external data pointer
  Jim(void* dataPointer, int ncol, int nrow, const GDALDataType& dataType);
  Jim(std::vector<void*> dataPointers, int ncol, int nrow, const GDALDataType& dataType);
  Jim(void* dataPointer, int ncol, int nrow, int nplane, const GDALDataType& dataType);
  Jim(std::vector<void*> dataPointers, int ncol, int nrow, int nplane, const GDALDataType& dataType);
    ///constructor input image
#if MIALIB == 1
  Jim(IMAGE *mia);
#endif
    ///constructor input image
  Jim(const std::string& filename, bool readData=true, unsigned int memory=0);
    ///constructor output image
  Jim(const std::string& filename, const Jim& imgSrc, unsigned int memory=0, const std::vector<std::string>& options=std::vector<std::string>());
    ///constructor input image
    /* Jim(std::shared_ptr<Jim> imgSrc, bool copyData=true) : m_nplane(1), Jim(imgSrc, copyData){}; */
    ///constructor input image
  Jim(Jim& imgSrc, bool copyData=true);
    ///constructor output image
  Jim(const std::string& filename, int ncol, int nrow, int nband, const GDALDataType& dataType, const std::string& imageType, unsigned int memory=0, const std::vector<std::string>& options=std::vector<std::string>());
  ///constructor output image with nplane
  Jim(int ncol, int nrow, int nband, int nplane, const GDALDataType& dataType);
    ///constructor output image
  Jim(int ncol, int nrow, int nband, const GDALDataType& dataType);
  ///constructor from app
  Jim(app::AppFactory &theApp);

  //from Reader
 /* Jim(const std::string& filename, unsigned int memory=0){ */
 /*    reset(); */
 /*    open(filename, memory); */
 /*  }; */
  // Jim(const std::string& filename, const GDALAccess& readMode=GA_ReadOnly, unsigned int memory=0) : m_writeMode(false) {open(filename, readMode, memory);};
  //from Writer
 /*  ///constructor opening an image for writing, copying image attributes from a source image. Caching is supported when memory>0 */
 /* Jim(const std::string& filename, const Jim& imgSrc, unsigned int memory=0, const std::vector<std::string>& options=std::vector<std::string>()) {open(filename, imgSrc, memory, options); */
 /*  }; */
 /*  ///copy constructor opening an image for writing in memory, copying image attributes from a source image. */
 /* Jim(Jim& imgSrc, bool copyData=true){ */
 /*   reset(); */
 /*   open(imgSrc,copyData); */
 /*  }; */
  ///copy constructor opening an image for writing in memory, copying image attributes from a source image.
  /* Jim(std::shared_ptr<Jim> imgSrc, bool copyData=true){ */
  /*   reset(); */
  /*   open(imgSrc,copyData); */
  /* }; */
  ///constructor opening an image for writing, defining all image attributes. Caching is supported when memory>0
 /* Jim(const std::string& filename, int ncol, int nrow, int nband, const GDALDataType& dataType, const std::string& imageType, unsigned int memory=0, const std::vector<std::string>& options=std::vector<std::string>()) { */
 /*   reset(); */
 /*   open(filename, ncol, nrow, nband, dataType, imageType, memory, options); */
 /*  }; */
  ///constructor opening an image for writing in memory, defining all image attributes
 /* Jim(int ncol, int nrow, int nband, const GDALDataType& dataType) { */
 /*   reset(); */
 /*   open(ncol, nrow, nband, dataType); */
 /*  }; */
 /*  ///constructor opening an image for reading or writing using application arguments */
 /* Jim(app::AppFactory& app){ */
 /*   reset(); */
 /*   open(app); */
 /* }; */

  ///destructor
 /* virtual ~Jim(void){freeMem();}; */
 virtual ~Jim(void);

  ///Create new shared pointer to Jim object
  /**
   *
   * @return shared pointer to new Jim object
   */
 static std::shared_ptr<Jim> createImg();
 /* { */
 /*   return(std::make_shared<Jim>()); */
 /* }; */
  ///create shared pointer to Jim
 static std::shared_ptr<Jim> createImg(const std::shared_ptr<Jim> pSrc, bool copyData=true);
 /* { */
 /*   std::shared_ptr<Jim> pRaster=std::make_shared<Jim>(*pSrc,copyData); */
 /*   return(pRaster); */
 /* }; */
  ///create shared pointer to Jim with random values only for in memory
 static std::shared_ptr<Jim> createImg(app::AppFactory &theApp);
 static std::shared_ptr<Jim> createImg(const std::string filename, bool readData=true, unsigned int memory=0);
  ///get write mode
  bool writeMode(){return(m_access==WRITE);};
  ///get access mode
  DATA_ACCESS getAccess(){return m_access;};
  ///set access mode
  CPLErr setAccess(DATA_ACCESS theAccess){m_access=theAccess;return CE_None;};
  ///set access mode using a string argument
  CPLErr setAccess(std::string accessString){
    if(accessString=="READ_ONLY"){
      m_access=READ_ONLY;
      return CE_None;
    }
    if(accessString=="UPDATE"){
      m_access=UPDATE;
      return CE_None;
    }
    if(accessString=="WRITE"){
      m_access=WRITE;
      return CE_None;
    }
    else
      return CE_Failure;
  }
  ///check if data pointer has been initialized
  bool isInit(){return(m_data.size()>0);};
  ///Set scale for a specific band when writing the raster data values. The scaling and offset are applied on a per band basis. You need to set the scale for each band. If the image data are cached (class was created with memory>0), the scaling is applied on the cached memory.
  CPLErr setScale(double theScale, int band=0){
    if(m_scale.size()!=nrOfBand()){//initialize
      m_scale.resize(nrOfBand());
      for(int iband=0;iband<nrOfBand();++iband)
        m_scale[iband]=1.0;
    }
    m_scale[band]=theScale;
  };
  ///Set offset for a specific band when writing the raster data values. The scaling and offset are applied on a per band basis. You need to set the offset for each band. If the image data are cached (class was created with memory>0), the offset is applied on the cached memory.
  CPLErr setOffset(double theOffset, int band=0){
    if(m_offset.size()!=nrOfBand()){
      m_offset.resize(nrOfBand());
      for(int iband=0;iband<nrOfBand();++iband)
        m_offset[iband]=0.0;
    }
    m_offset[band]=theOffset;
  };
  ///set externalData
  CPLErr setExternalData(bool flag){m_externalData=flag;};
  bool getExternalData() const {return(m_externalData);};
  ///Open image from allocated memory instead of from file. This will allow in place image processing in memory (streaming). Notice that an extra call must be made to set the geotranform and projection. This function has not been tested yet!
  //void open(void* dataPointer, int ncol, int nrow, int nband, const GDALDataType& dataType);
  ///Close the image.
  CPLErr close();
  ///write to file previously set (eg., with setFile). Specialization of the writeData member function of Jim, avoiding reset of the memory.
  CPLErr write();
  ///write to file Specialization of the writeData member function of Jim, avoiding reset of the memory.
  /**
   * @param output (type: std::string) Output image file
   * @param oformat (type: std::string) (default: GTiff) Output image format (see also gdal_translate).
   * @param co (type: std::string) Creation option for output file. Multiple options can be specified.
   * @return CE_None if successful, CE_Failure if not.
   * @param nodata Nodata value to put in image.
   **/
  CPLErr write(app::AppFactory &theApp);
  ///Get the filename of this dataset
  std::string getFileName() const {return m_filename;};
  ///Get the number of columns of this dataset
  int nrOfCol() const { return m_ncol;};
  ///Get the number of rows of this dataset
  int nrOfRow() const { return m_nrow;};
  ///Get the number of bands of this dataset
  int nrOfBand() const { return m_nband;};
#ifdef SWIG
  %pythonprepend nrOfPlane()  "\"\"\"HELP.METHOD.Jim.nrOfPlane()\"\"\""
#endif
     ///Get the number of planes of this dataset
     int nrOfPlane() const { return m_nplane;};
  ///Is this dataset georeferenced (pixel size in y must be negative) ?
  bool isGeoRef() const {std::vector<double> gt(6);getGeoTransform(gt);if(gt.size()!=6) return false;else if(gt[5]<0) return true;else return false;};
  ///Get the projection string (deprecated, use getProjectionRef instead)
  std::string getProjection() const;
  ///Get the projection reference
  std::string getProjectionRef() const;
  ///Get the geotransform data for this dataset as a list of doubles
  void getGeoTransform(double& gt0, double& gt1, double& gt2, double& gt3, double& gt4, double& gt5) const{std::vector<double> gt(6); getGeoTransform(gt);if (gt.size()==6){gt0=gt[0];gt1=gt[1];gt2=gt[2];gt3=gt[3];gt4=gt[4];gt5=gt[5];}};
  /* std::string getGeoTransform() const; */
  ///Get the geotransform data for this dataset
  void getGeoTransform(std::vector<double>& gt) const;
  void getGeoTransform(double* gt) const;
  ///Set the geotransform data for this dataset
  CPLErr setGeoTransform(const std::vector<double>& gt);
  CPLErr setGeoTransform(double* gt);
  ///Copy geotransform information from another georeferenced image
  CPLErr copyGeoTransform(const Jim& imgSrc);
  ///Set the projection for this dataset in well known text (wkt) format
  void setProjection(const std::string& projection){setProjectionProj4(projection);};
  ///Set the projection for this dataset from user input (supports epsg:<number> format)
  void setProjectionProj4(const std::string& projection);
  ///Get the bounding box of this dataset in georeferenced coordinates with coordinate transform
  void getBoundingBox(double& ulx, double& uly, double& lrx, double& lry, OGRCoordinateTransformation *poCT=0) const;
  ///Get the bounding box of this dataset in georeferenced coordinates with coordinate transform
  void getBoundingBox(std::vector<double> &bbvector, OGRCoordinateTransformation *poCT=0) const;
  ///Get the bounding box of this dataset in georeferenced coordinates with coordinate transform
  void getBoundingBox(OGRPolygon *bbPolygon, OGRCoordinateTransformation *poCT=0) const;
  /* ///Get the bounding box of this dataset in georeferenced coordinates with coordinate transform based on EPSG code */
  /* bool getBoundingBox(std::vector<double> &bbvector, int targetEPSG) const; */
  ///Get the center position of this dataset in georeferenced coordinates
  void getCenterPos(double& centerX, double& centerY) const;
  ///Get the upper left corner x (georeferenced) coordinate of this dataset
  double getUlx() const {double ulx, uly, lrx,lry;getBoundingBox(ulx,uly,lrx,lry);return(ulx);};
  ///Get the upper left corner y (georeferenced) coordinate of this dataset
  double getUly() const {double ulx, uly, lrx,lry;getBoundingBox(ulx,uly,lrx,lry);return(uly);};
  ///Get the lower right corner x (georeferenced) coordinate of this dataset
  double getLrx() const {double ulx, uly, lrx,lry;getBoundingBox(ulx,uly,lrx,lry);return(lrx);};
  ///Get the lower right corner y (georeferenced) coordinate of this dataset
  double getLry() const {double ulx, uly, lrx,lry;getBoundingBox(ulx,uly,lrx,lry);return(lry);};
  ///Get the scale for specific band
  double getScale(std::size_t band=0){
    if(m_scale.size()<=band)
      return(1.0);
    else
      return(m_scale[band]);
  };
  ///Get the scale as a standard template library (stl) vector
  void getScale(std::vector<double>& scale) const {scale=m_scale;};
  ///Get the scale for specific band
  double getOffset(std::size_t band=0){
    if(m_offset.size()<=band)
      return(0.0);
    else
      return(m_offset[band]);
  };
  ///Get the offset as a standard template library (stl) vector
  void getOffset(std::vector<double>& offset) const {offset=m_offset;};
  ///Get the no data values of this dataset as a standard template library (stl) vector
  /* std::vector<double> getNoDataValues() const{std::cout << "calling with out" << std::endl; return m_noDataValues;}; */
  ///Get the no data values of this dataset as a standard template library (stl) vector
  void getNoDataValues(std::vector<double>& noDataValues) const;
  ///Print the no data values of this dataset as a standard template library (stl) vector
  CPLErr printNoDataValues() const{
    if(m_noDataValues.size()){
      for(int i=0;i<m_noDataValues.size();++i)
        std::cout << m_noDataValues[i] << " ";
      std::cout << std::endl;
      return(CE_None);
    }
    else
      return(CE_Warning);
  };
  ///Check if value is nodata in this dataset
  bool isNoData(double value) const{if(m_noDataValues.empty()) return false;else return find(m_noDataValues.begin(),m_noDataValues.end(),value)!=m_noDataValues.end();};
  ///Push a no data value for this dataset
  CPLErr pushNoDataValue(double noDataValue);
  ///Set the single no data values of this dataset
  CPLErr setNoDataValue(double nodata){clearNoData(); pushNoDataValue(nodata);if(m_noDataValues.empty()) return(CE_Failure);else return(CE_None);};
  ///Set the no data values of this dataset using a standard template library (stl) vector as input
  CPLErr setNoData(const std::vector<double>& nodata){m_noDataValues=nodata; if(nodata.size() != m_noDataValues.size()) return(CE_Failure);else return(CE_None);};
  ///Set the no data values of this dataset using a standard template library (stl) vector as input
  CPLErr setNoData(const std::list<double>& nodata){
    clearNoData();
    std::list<double>::const_iterator lit=nodata.begin();
    while(lit!=nodata.end())
      pushNoDataValue(*(lit++));
    if(nodata.size() != m_noDataValues.size())
      return(CE_Failure);
    else return(CE_None);
  };
  void setData(double value);
  void setData(double value, int band);
  void setData(double value, double ulx, double uly, double lrx, double lry, int band=0, double dx=0, double dy=0, bool nogeo=false);
  ///Clear all no data values, including the one in GDAL dataset if it is set
  CPLErr clearNoData(int band=0){m_noDataValues.clear();if(m_access!=READ_ONLY&&getRasterBand(band)) getRasterBand(band)->DeleteNoDataValue();return(CE_None);}
  ///Set the GDAL (internal) no data value for this data set. Only a single no data value per band is supported.
  CPLErr GDALSetNoDataValue(double noDataValue, int band=0) {if(getRasterBand(band)) return getRasterBand(band)->SetNoDataValue(noDataValue);else return(CE_Warning);};
  ///Check if a geolocation is covered by this dataset. Only the bounding box is checked, irrespective of no data values.
  bool covers(double x, double y, OGRCoordinateTransformation *poCT=NULL) const;
  ///Check if a region of interest is (partially or all if all is set) covered by this dataset. Only the bounding box is checked, irrespective of no data values.
  bool covers(double ulx, double  uly, double lrx, double lry, std::string coverType="ALL_TOUCHED", OGRCoordinateTransformation *poCT=NULL) const;
  ///Check if an image is (partially or all if all is set) covered by this dataset. Only the bounding box is checked, irrespective of no data values.
  bool covers(const std::shared_ptr<Jim> imgRaster, std::string coverType="ALL_TOUCHED") const{
    OGRSpatialReference thisSpatialRef(getProjectionRef().c_str());
    OGRSpatialReference thatSpatialRef(imgRaster->getProjectionRef().c_str());
    OGRCoordinateTransformation *that2this = OGRCreateCoordinateTransformation(&thatSpatialRef, &thisSpatialRef);
    if(thisSpatialRef.IsSame(&thatSpatialRef)){
      that2this=0;
    }
    else{
      if(!that2this){
        std::ostringstream errorStream;
        errorStream << "Error: cannot create OGRCoordinateTransformation that to this" << std::endl;
        throw(errorStream.str());
      }
    }
    //image bounding box in SRS of the this
    double img_ulx,img_uly,img_lrx,img_lry;
    imgRaster->getBoundingBox(img_ulx,img_uly,img_lrx,img_lry,that2this);
    return covers(img_ulx,img_uly,img_lrx,img_lry,coverType);
  };
  ///Convert georeferenced coordinates (x and y) to image coordinates (column and row)
  bool geo2image(double x, double y, double& i, double& j, OGRCoordinateTransformation *poCT=NULL) const;
  ///Convert image coordinates (column and row) to georeferenced coordinates (x and y)
  bool image2geo(double i, double j, double& x, double& y, OGRCoordinateTransformation *poCT=NULL) const;
  ///Get the pixel cell spacing in x
  double getDeltaX() const {std::vector<double> gt(6);getGeoTransform(gt);return gt[1];};
  ///Get the pixel cell spacing in y
  double getDeltaY() const {std::vector<double> gt(6);getGeoTransform(gt);return -gt[5];};
  ///Get the GDAL datatype for this dataset
  GDALDataType getDataTypeDS(int band=0) const;
  ///Get the internal datatype for this dataset
  int getDataType(int band=0) const;
  GDALDataType getGDALDataType(int band=0) const;
  std::size_t getDataTypeSizeBytes(int band=0) const;
  //GDALDataType getDataType(int band=0) const;
  ///Get the datapointer
  void* getDataPointer(int band=0){return(m_data[band]);};
  ///free memory os data pointer
  void freeMem();
  ///Copy data
  CPLErr copyData(void* data, int band=0);
  ///Copy data
  // void copyData(Jim& imgRaster, int band=0);
  //todo: introduce smart pointer instead of void*
  // std::unique_ptr<void> getDataPointer(int band=0){return(m_data[band]);};
  ///Get the GDAL rasterband for this dataset
  GDALRasterBand* getRasterBand(int band=0) const;
  ///Get the GDAL color table for this dataset as an instance of the GDALColorTable class
  GDALColorTable* getColorTable(int band=0) const;
  ///Get the GDAL driver description of this dataset
  std::string getDriverDescription() const;
  ///Get the image type (implemented as the driver description)
  std::string getImageType() const{return getDriverDescription();};
  ///Get the band coding (interleave)
  std::string getInterleave() const;
  ///Get the compression from the metadata of this dataset
  std::string getCompression() const;
  //Get a pointer to the GDAL dataset
  GDALDataset* getDataset(){return m_gds;};
  ///Get the metadata of this dataset
  char** getMetadata() const;
  // Get the metadata of this dataset in the form of a list of strings (const version)
  void getMetadata(std::list<std::string>& metadata) const;
  ///Get the image description from the driver of this dataset
  std::string getDescription() const;
  ///Get metadata item of this dataset
  std::string getMetadataItem() const;
  ///Get the image description from the metadata of this dataset
  std::string getImageDescription() const;
  int getBlockSize() const{return m_blockSize;};
  int getBlockSizeX(int band=0)
  {
    int blockSizeX=0;
    int blockSizeY=0;
    if(getRasterBand(band))
      getRasterBand(band)->GetBlockSize( &blockSizeX, &blockSizeY );
    return blockSizeX;
  }
  int getBlockSizeY(int band=0)
  {
    int blockSizeX=0;
    int blockSizeY=0;
    if(getRasterBand(band))
      getRasterBand(band)->GetBlockSize( &blockSizeX, &blockSizeY );
    return blockSizeY;
  }
  int nrOfBlockX(int band=0)
  {
    int blockSizeX=0;
    int blockSizeY=0;
    int nXBlocks=0;
    if(getRasterBand(band)){
      getRasterBand(band)->GetBlockSize( &blockSizeX, &blockSizeY );
      nXBlocks = (nrOfCol() + blockSizeX - 1) / blockSizeX;
    }
    return nXBlocks;
  }
  int nrOfBlockY(int band=0)
  {
    int blockSizeX=0;
    int blockSizeY=0;
    int nYBlocks=0;
    if(getRasterBand(band)){
      getRasterBand(band)->GetBlockSize( &blockSizeX, &blockSizeY );
      nYBlocks = (nrOfRow() + blockSizeY - 1) / blockSizeY;
    }
    return nYBlocks;
  }

  ///Create a JSON string from a Jim image
  std::string jim2json();
  ///Clone as new shared pointer to Jim object
  /* std::shared_ptr<Jim> clone() { */
  /*   return(cloneImpl()); */
  /* }; */
  std::shared_ptr<Jim> clone(bool copyData=true);
  std::shared_ptr<Jim> getShared(){return(std::dynamic_pointer_cast<Jim>(shared_from_this()));};
  ///Read all pixels from image in memory for specific dataset band
  CPLErr readDataDS(int band, int ds_band);
  ///Read all pixels from image in memory for specific band
  CPLErr readData(int band){readDataDS(band,band);};
  ///Read all pixels from image in memory
  CPLErr readData();
  ///Read data using the arguments from AppFactory
  /* CPLErr readData(app::AppFactory &app); */
  ///Read a single pixel cell value at a specific column and row for a specific band (all indices start counting from 0)
  template<typename T> void readData(T& value, int col, int row, int band=0);
  template<typename T> void readData3D(T& value, std::size_t col, std::size_t row, std::size_t plane, std::size_t band=0);
  template<typename T> void readData3D(std::vector<T>& buffer, std::size_t minCol, std::size_t maxCol, std::size_t row, std::size_t plane, std::size_t band=0);
  ///Return a single pixel cell value at a specific column and row for a specific band (all indices start counting from 0)
  double readData(int col, int row, int band=0){
    double value;
    readData(value, col, row, band);
    return(value);
  };
  double readData3D(std::size_t col, std::size_t row, std::size_t plane, std::size_t band=0){
    double value;
    readData3D(value, col, row, plane, band);
    return(value);
  };
  ///Read pixel cell values for a range of columns for a specific row and band (all indices start counting from 0)
  template<typename T> CPLErr readData(std::vector<T>& buffer, int minCol, int maxCol, int row, int band=0);
  ///Read pixel cell values for a range of columns for a specific row and band (all indices start counting from 0). The row counter can be floating, in which case a resampling is applied at the row level. You still must apply the resampling at column level. This function will be deprecated, as the GDAL API now supports rasterIO resampling (see http://www.gdal.org/structGDALRasterIOExtraArg.html)
  template<typename T> CPLErr readData(std::vector<T>& buffer, int minCol, int maxCol, double row, int band, RESAMPLE resample);
  ///Read pixel cell values for a range of columns and rows for a specific band (all indices start counting from 0). The buffer is a two dimensional vector (stl vector of stl vector) representing [row][col].
  template<typename T> CPLErr readDataBlock(Vector2d<T>& buffer2d, int minCol, int maxCol, int minRow, int maxRow, int band=0);
  template<typename T> void readDataBlock3D(Vector2d<T>& buffer2d, std::size_t minCol, std::size_t maxCol, std::size_t minRow, std::size_t maxRow, std::size_t plane, std::size_t band=0);
  ///Read pixel cell values for a range of columns and rows for a specific band (all indices start counting from 0). The buffer is a one dimensional stl vector representing all pixel values read starting from upper left to lower right.
  template<typename T> CPLErr readDataBlock(std::vector<T>& buffer , int minCol, int maxCol, int minRow, int maxRow, int band=0);
  template<typename T> void readDataBlock3D(std::vector<T>& buffer , std::size_t minCol, std::size_t maxCol, std::size_t minRow, std::size_t maxRow, std::size_t plane, std::size_t band=0);
  ///Read pixel cell values for a range of columns, rows and bands for a specific band (all indices start counting from 0). The buffer is a one dimensional stl vector representing all pixel values read starting from upper left to lower right, band interleaved.
  template<typename T> CPLErr readData(std::vector<T>& buffer, int row, int band=0);
  ///Read pixel cell values for an entire row for a specific band (all indices start counting from 0). The row counter can be floating, in which case a resampling is applied at the row level. You still must apply the resampling at column level. This function will be deprecated, as the GDAL API now supports rasterIO resampling (see http://www.gdal.org/structGDALRasterIOExtraArg.html)
  template<typename T> CPLErr readData(std::vector<T>& buffer, double row, int band, RESAMPLE resample);
  ///Get the minimum and maximum cell values for a specific band in a region of interest defined by startCol, endCol, startRow and endRow (all indices start counting from 0).
  void getMinMax(int startCol, int endCol, int startRow, int endRow, int band, double& minValue, double& maxValue);
  ///Get the minimum and maximum cell values for a specific band (all indices start counting from 0).
  void getMinMax(double& minValue, double& maxValue, int band=0);
  ///Get the minimum cell values for a specific band and report the column and row in which the minimum value was found (all indices start counting from 0).
  double getMin(int& col, int& row, int band=0);
  ///Get the minimum cell values for a specific band.
  double getMin(int band=0){int theCol=0;int theRow=0;return(getMin(theCol,theRow,band));};
  ///Get the maximum cell values for a specific band and report the column and row in which the maximum value was found (all indices start counting from 0).
  double getMax(int& col, int& row, int band=0);
  ///Get the maximum cell values for a specific band.
  double getMax(int band=0){int theCol=0;int theRow=0;return(getMax(theCol,theRow,band));};
  ///Calculate the image histogram for a specific band using a defined number of bins and constrained   by a minimum and maximum value. A kernel density function can also be applied (default is false).
  double getHistogram(std::vector<double>& histvector, double& min, double& max, int& nbin, int theBand=0, bool kde=false);
  ///Calculate the reference pixel as the centre of gravity pixel (weighted average of all values not taking into account no data values) for a specific band (start counting from 0).
  void getRefPix(double& refX, double &refY, int band=0);
  ///Calculate the range of cell values in the image for a specific band (start counting from 0).
  void getRange(std::vector<short>& range, int band=0);
  ///Calculate the number of valid pixels (with a value not defined as no data).
  unsigned long int getNvalid(int band=0);
  ///Calculate the number of invalid pixels (with a value defined as no data).
  unsigned long int getNinvalid(int band=0);

  //From Reader
  ///Open dataset
  CPLErr open(const std::string& filename, bool readData=true, unsigned int memory=0);
  // void open(const std::string& filename, const GDALAccess& readMode=GA_ReadOnly, unsigned int memory=0);
  //From Writer
  ///Open an image for writing, copying image attributes from a source image. Image is directly written to file. Use the constructor with memory>0 to support caching
  CPLErr open(const std::string& filename, const Jim& imgSrc, const std::vector<std::string>& options=std::vector<std::string>());
  ///Open an image for writing, copying image attributes from a source image. Caching is supported when memory>0
  CPLErr open(const std::string& filename, const Jim& imgSrc, unsigned int memory, const std::vector<std::string>& options=std::vector<std::string>());
  ///Open an image for writing, defining all image attributes. Image is directly written to file. Use the constructor with memory>0 to support caching
  // void open(const std::string& filename, int ncol, int nrow, int nband, const GDALDataType& dataType, const std::string& imageType, const std::vector<std::string>& options=std::vector<std::string>());
  ///Open an image for writing, defining all image attributes. Caching is supported when memory>0
  CPLErr open(const std::string& filename, int ncol, int nrow, int nband, const GDALDataType& dataType, const std::string& imageType, unsigned int memory=0, const std::vector<std::string>& options=std::vector<std::string>());
  ///Open an image for writing in memory, defining image attributes.
  CPLErr open(int ncol, int nrow, int nband, const GDALDataType& dataType);
  ///Open an image for writing in memory with nplane, defining image attributes.
  CPLErr open(int ncol, int nrow, int nband, int nplane, const GDALDataType& dataType);
  ///Open an image for writing in memory, copying image attributes from a source image.
  CPLErr open(Jim& imgSrc,  bool copyData=true);
  ///Open an image for writing in memory, copying image attributes from a source image.
  /* CPLErr open(std::shared_ptr<Jim> imgSrc,  bool copyData=true); */
  ///Open an image for writing using an external data pointer (not tested yet)
  CPLErr open(void* dataPointer, int ncol, int nrow, const GDALDataType& dataType);
  ///Open an image for writing using an external data pointer
  CPLErr open(void* dataPointer, int ncol, int nrow, int nplane, const GDALDataType& dataType, bool  copyData=true);
  ///Open a multiband image for writing using a external data pointers
  CPLErr open(std::vector<void*> dataPointers, int ncol, int nrow, int nplane, const GDALDataType& dataType, bool  copyData=true);
  ///Open an image for writing using an external data pointer (not tested yet)
  CPLErr open(std::vector<void*> dataPointers, int ncol, int nrow, const GDALDataType& dataType);
  ///Open an image for reading or writing using application arguments
  CPLErr open(app::AppFactory &app);

  ///Set the image description (only for GeoTiff format: TIFFTAG_IMAGEDESCRIPTION)
  CPLErr setImageDescription(const std::string& imageDescription){m_gds->SetMetadataItem( "TIFFTAG_IMAGEDESCRIPTION",imageDescription.c_str());};

  ///Write a single pixel cell value at a specific column and row for a specific band (all indices start counting from 0)
  template<typename T> CPLErr writeData(const T& value, int col, int row, int band=0);
  ///Write pixel cell values for a range of columns for a specific row and band (all indices start counting from 0)
  template<typename T> CPLErr writeData(std::vector<T>& buffer, int minCol, int maxCol, int row, int band=0);
  ///Write pixel cell values for an entire row for a specific band (all indices start counting from 0)
  template<typename T> CPLErr writeData(std::vector<T>& buffer, int row, int band=0);
  /* template<typename T> void writeData3D(std::vector<T>& buffer, std::size_t minCol, std::size_t maxCol, std::size_t row, std::size_t plane, std::size_t band=0); */
  /* template<typename T> void writeData3D(std::vector<T>& buffer, std::size_t row, std::size_t plane, std::size_t band=0){ */
  /*   writeData3D(buffer, row, 0, nrOfCol()-1, plane, band); */
  /* }; */
  // deprecated? Write an entire image from memory to file
  // CPLErr writeData(void* pdata, const GDALDataType& dataType, int band=0);
  ///Write pixel cell values for a range of columns and rows for a specific band (all indices start counting from 0). The buffer is a two dimensional vector (stl vector of stl vector) representing [row][col].
  template<typename T> CPLErr writeDataBlock(Vector2d<T>& buffer2d, int minCol, int maxCol, int minRow, int maxRow, int band=0);
  template<typename T> CPLErr writeDataBlock(T value, int minCol, int maxCol, int minRow, int maxRow, int band=0);
  ///Prepare image writer to write to file
  CPLErr setFile(const std::string& filename, const std::string& imageType, unsigned int memory=0, const std::vector<std::string>& options=std::vector<std::string>());
  ///Prepare image writer to write to file
  // void setFile(const std::string& filename, const Jim& imgSrc, unsigned int memory=0, const std::vector<std::string>& options=std::vector<std::string>());
  ///Set the color table using an (ASCII) file with 5 columns (value R G B alpha)
  CPLErr setColorTable(const std::string& filename, int band=0);
  ///Set the color table using the GDAL class GDALColorTable
  CPLErr setColorTable(GDALColorTable* colorTable, int band=0);
  ///Set specific metadata (driver specific)
  CPLErr setMetadata(char** metadata);
  CPLErr rasterizeBuf(const std::string& ogrFilename);
  CPLErr rasterizeBuf(VectorOgr& ogrReader, app::AppFactory &app);
  CPLErr rasterizeBuf(VectorOgr& ogrReader, double burnValue, const std::vector<std::string>& eoption=std::vector<std::string>(), const std::vector<std::string>& layernames=std::vector<std::string>());
  void d_rasterizeBuf(VectorOgr& ogrReader, app::AppFactory &app);
  void d_rasterizeBuf(VectorOgr& ogrReader, double burnValue, const std::vector<std::string>& eoption=std::vector<std::string>(), const std::vector<std::string>& layernames=std::vector<std::string>());

  /* CPLErr rasterizeLayersBuf(std::vector<OGRLayer*>& layers, double burnValue=1.0); */
  /* CPLErr rasterizeLayersBuf(OGRLayer* layer, double burnValue=1.0){std::vector<OGRLayer*> layers;layers.push_back(layer);rasterizeLayersBuf(layers,burnValue);}; */
  ///Apply thresholds: set to no data if not within thresholds t1 and t2
  void setThreshold(Jim& imgWriter, double t1, double t2);
  void setThresholdMin(Jim& imgWriter, double minThreshold);
  void setThresholdMax(Jim& imgWriter, double maxThreshold);
  ///Apply absolute thresholds: set to no data if not within thresholds t1 and t2
  void setAbsThreshold(Jim& imgWriter, double t1, double t2);
  void setAbsThresholdMin(Jim& imgWriter, double minThreshold);
  void setAbsThresholdMax(Jim& imgWriter, double maxThreshold);
  ///Apply thresholds for in memory: set to no data if not within thresholds t1 and t2
  std::shared_ptr<Jim> setThreshold(double t1, double t2);
  ///Apply absolute thresholds for in memory: set to no data if not within thresholds t1 and t2
  std::shared_ptr<Jim> setAbsThreshold(double t1, double t2);
  ///Apply thresholds: set to no data if not within thresholds t1 and t2, else set to value
  void setThreshold(Jim& imgWriter, double t1, double t2, double value);
  void setThresholdMin(Jim& imgWriter, double minThreshold, double value);
  void setThresholdMax(Jim& imgWriter, double maxThreshold, double value);
  ///Apply absolute thresholds: set to no data if not within thresholds t1 and t2, else set to value
  void setAbsThreshold(Jim& imgWriter, double t1, double t2, double value);
  void setAbsThresholdMin(Jim& imgWriter, double minThreshold, double value);
  void setAbsThresholdMax(Jim& imgWriter, double maxThreshold, double value);
  ///Apply thresholds for in memory: set to no data if not within thresholds t1 and t2, else set to value
  std::shared_ptr<Jim> setThreshold(double t1, double t2, double value);
  ///Apply absolute thresholds for in memory: set to no data if not within thresholds t1 and t2, else set to value
  std::shared_ptr<Jim> setAbsThreshold(double t1, double t2, double value);
  ///Apply thresholds with theApp
  void setThreshold(Jim& imgWriter, app::AppFactory &theApp);
  ///Apply thresholds for in memory with theApp
  std::shared_ptr<Jim> setThreshold(app::AppFactory &theApp);

  ///assignment operator
  /* Jim& operator=(Jim& imgSrc); */
#if MIALIB == 1
  bool isEqual(std::shared_ptr<Jim> refImg);
#endif
  ///equal operator
  std::shared_ptr<Jim> eq(std::shared_ptr<Jim> refJim);
  std::shared_ptr<Jim> eq(double value);
  void eq(Jim& refJim, Jim& imgWriter);
  void eq(double value, Jim& imgWriter);
  ///not equal operator
  std::shared_ptr<Jim> ne(std::shared_ptr<Jim> refJim);
  std::shared_ptr<Jim> ne(double value);
  void ne(Jim& refJim, Jim& imgWriter);
  void ne(double value, Jim& imgWriter);
  ///less than operator
  std::shared_ptr<Jim> lt(std::shared_ptr<Jim> refJim);
  std::shared_ptr<Jim> lt(double value);
  void lt(Jim& refJim, Jim& imgWriter);
  void lt(double value, Jim& imgWriter);
  template<typename T> void lt_t(Jim& other, Jim& imgWriter);
  template<typename T> void lt_t(double value, Jim& imgWriter);
  ///less than or equal operator
  std::shared_ptr<Jim> le(std::shared_ptr<Jim> refJim);
  std::shared_ptr<Jim> le(double value);
  void le(Jim& refJim, Jim& imgWriter);
  void le(double value, Jim& imgWriter);
  ///greater than operator
  std::shared_ptr<Jim> gt(std::shared_ptr<Jim> refJim);
  std::shared_ptr<Jim> gt(double value);
  void gt(Jim& refJim, Jim& imgWriter);
  void gt(double value, Jim& imgWriter);
  ///greater than or equal operator
  std::shared_ptr<Jim> ge(std::shared_ptr<Jim> refJim);
  std::shared_ptr<Jim> ge(double value);
  void ge(Jim& refJim, Jim& imgWriter);
  void ge(double value, Jim& imgWriter);
  //lib functions
  ///create color table
  std::shared_ptr<Jim> createct(app::AppFactory& app);
  ///create color table
  void createct(Jim& imgWriter, app::AppFactory& app);
  ///convert image
  void convert(Jim& imgWriter, app::AppFactory& app);
  ///crop image
  void crop(Jim& imgWriter, app::AppFactory& app);
  ///crop image
  void cropOgr(VectorOgr& sampleReader, Jim& imgWriter, app::AppFactory& app);
  ///crop image if it has not been read yet (typically used when Jim has been opened with argument noRead true)
  void cropDS(Jim& imgWriter, app::AppFactory& app);
  ///crop image
  /* CPLErr crop(Jim& imgWriter, double ulx, double uly, double lrx, double lry); */
  /* CPLErr crop(Jim& imgWriter, double ulx, double uly, double lrx, double lry, double dx=0, double dy=0, bool geo=true); */
  ///stack band(s) from another Jim
  std::shared_ptr<Jim> stackBand(Jim& imgSrc, app::AppFactory& app);
  std::shared_ptr<Jim> stackBand(Jim& imgSrc){app::AppFactory theApp;return stackBand(imgSrc,theApp);};
  ///stack band(s) from another Jim
  void stackBand(Jim& imgSrc, Jim& imgWriter, app::AppFactory& app);
  void stackBand(Jim& imgSrc, Jim& imgWriter){app::AppFactory theApp;return stackBand(imgSrc,imgWriter,theApp);};
  ///destructive version of stack band(s) from another Jim
  void d_stackBand(Jim& imgSrc, app::AppFactory& app);
  void d_stackBand(Jim& imgSrc){app::AppFactory theApp; d_stackBand(imgSrc,theApp);};
  ///destructive version of stack plane(s) from another Jim
  void d_stackPlane(Jim& imgSrc, app::AppFactory& app);
  void d_stackPlane(Jim& imgSrc){app::AppFactory theApp; d_stackPlane(imgSrc,theApp);};
  ///crop band(s)
  std::shared_ptr<Jim> cropBand(app::AppFactory& app);
  std::shared_ptr<Jim> cropBand(){app::AppFactory theApp;return cropBand(theApp);};
  ///crop plane(s)
  std::shared_ptr<Jim> cropPlane(app::AppFactory& app);
  std::shared_ptr<Jim> cropPlane(){app::AppFactory theApp;return cropPlane(theApp);};
  void cropBand(Jim& imgWriter, app::AppFactory& app);
  void cropBand(Jim& imgWriter){app::AppFactory theApp;return cropBand(imgWriter,theApp);};
  ///destructive version of cropBand
  void d_cropBand(app::AppFactory& app);
  void d_cropBand(){app::AppFactory theApp;d_cropBand(theApp);};
  void cropPlane(Jim& imgWriter, app::AppFactory& app);
  void cropPlane(Jim& imgWriter){app::AppFactory theApp;return cropPlane(imgWriter,theApp);};
  ///destructive version of cropPlane
  void d_cropPlane(app::AppFactory& app);
  void d_cropPlane(){app::AppFactory theApp;d_cropPlane(theApp);};
  ///convert image only for in memory
  std::shared_ptr<Jim> convert(app::AppFactory& app);
  ///crop image only for in memory
  std::shared_ptr<Jim> crop(app::AppFactory& app);
  ///crop image only for in memory
  /* std::shared_ptr<Jim> crop(double ulx, double uly, double lrx, double lry, double dx=0, double dy=0, bool geo=true); */
  ///crop Jim image in memory based on VectorOgr returning Jim image
  std::shared_ptr<Jim> cropOgr(VectorOgr& sampleReader, app::AppFactory& app);
  ///warp Jim image in memory
  std::shared_ptr<Jim> warp(app::AppFactory& theApp);
  ///warp image
  void warp(Jim& imgWriter, app::AppFactory &theApp);
  ///polygonize image
  std::shared_ptr<VectorOgr> polygonize(app::AppFactory& app, std::shared_ptr<Jim>mask=nullptr);
  ///polygonize image
  void polygonize(VectorOgr& ogrWriter, app::AppFactory& app, std::shared_ptr<Jim> mask=nullptr);
  ///extract pixel values from raster image from a vector sample
  CPLErr extractOgr(VectorOgr& sampleReader, VectorOgr& ogrWriter, app::AppFactory& app);
  ///extract pixel values from raster image from a vector sample
  std::shared_ptr<VectorOgr> extractOgr(VectorOgr& sampleReader, app::AppFactory& app);
  ///extract pixel values from raster image with random or grid sampling
  CPLErr extractSample(VectorOgr& ogrWriter, app::AppFactory& app);
  ///extract pixel values from raster image with random or grid sampling
  std::shared_ptr<VectorOgr> extractSample(app::AppFactory& app);
  ///extract pixel values from raster image from a raster sample
  CPLErr extractImg(Jim& classReader, VectorOgr& ogrWriter, app::AppFactory& app);
  ///extract pixel values from raster image from a raster sample
  std::shared_ptr<VectorOgr> extractImg(Jim& classReader, app::AppFactory& app);
  ///calculate statistics profile based on multiband raster dataset
  void statProfile(Jim& imgWriter, app::AppFactory& app);
  ///calculate statistics profile based on multiband raster dataset only for in memory
  std::shared_ptr<Jim> statProfile(app::AppFactory& app);
  ///filter raster dataset
  /* CPLErr filter(Jim& imgWriter, app::AppFactory& app); */
  ///filter raster dataset in spectral/temporal domain
  void filter1d(Jim& imgWriter, app::AppFactory& app);
  ///filter raster dataset in spatial domain
  void filter2d(Jim& imgWriter, const app::AppFactory& app);
  ///filter raster dataset only for in memory
  /* std::shared_ptr<Jim> filter(app::AppFactory& app); */
  ///filter raster dataset in spectral/temporal domain only for in memory
  std::shared_ptr<Jim> filter1d(app::AppFactory& app);
  ///filter raster dataset in spatial domain only for in memory
  std::shared_ptr<Jim> filter2d(const app::AppFactory& app);
  ///filter raster dataset in spectral/temporal domain
  std::shared_ptr<Jim> firfilter1d(app::AppFactory& app);
  ///filter raster dataset in spectral/temporal domain
  void firfilter1d(Jim& imgWriter, app::AppFactory& app);
  ///filter raster dataset in spectral/temporal domain
  std::shared_ptr<Jim> savgolay(app::AppFactory& app);
  ///filter raster dataset in spectral/temporal domain
  void savgolay(Jim& imgWriter, app::AppFactory& app);
  ///filter raster dataset in spectral/temporal domain
  template<typename T> void firfilter1d_t(Jim& imgWriter, app::AppFactory& app);
  ///forward wavelet transform in spectral/temporal domain
  void d_dwt1d(app::AppFactory& app);
  std::shared_ptr<Jim> dwt1d(app::AppFactory& app);
  ///inverse wavelet transform in spectral/temporal domain
  void d_dwti1d(app::AppFactory& app);
  std::shared_ptr<Jim> dwti1d(app::AppFactory& app);
  ///smooth no data in raster dataset in spectral/temporal domain
  void smoothNoData1d(Jim& imgWriter, app::AppFactory& app);
  template<typename T> void smoothNoData1d_t(Jim& imgWriter, app::AppFactory& app);
  std::shared_ptr<Jim> smoothNoData1d(app::AppFactory& app);
  ///create dataset with statistics based on 3D raster dataset in spectral/temporal domain
  void stats1d(Jim& imgWriter, app::AppFactory& app);
  template<typename T> void stats1d_t(Jim& imgWriter, app::AppFactory& app);
  std::shared_ptr<Jim> stats1d(app::AppFactory& app);
  ///filter raster dataset in spatial domain
  std::shared_ptr<Jim> firfilter2d(app::AppFactory& app);
  ///filter raster dataset in spatial domain
  void firfilter2d(Jim& imgWriter, app::AppFactory& app);
  template<typename T> void firfilter2d_t(Jim& imgWriter, app::AppFactory& app);
  ///forward wavelet transform in spatial domain
  void d_dwt2d(app::AppFactory& app);
  std::shared_ptr<Jim> dwt2d(app::AppFactory& app);
  ///inverse wavelet transform in spatial domain
  void d_dwti2d(app::AppFactory& app);
  std::shared_ptr<Jim> dwti2d(app::AppFactory& app);
  ///check the difference between two images (validate in case of classification image)
  CPLErr diff(Jim& imgReference, app::AppFactory& app);
  ///check the difference between two images (validate in case of classification image)
  CPLErr validate(app::AppFactory& app);
  ///train raster dataset
  /* void train(JimList& referenceReader, app::AppFactory& app); */
  /* void train2d(JimList& referenceReader, app::AppFactory& app); */
  ///classify raster dataset
  void classify(Jim& imgWriter, app::AppFactory& app);
  ///classify raster dataset only for in memory
  std::shared_ptr<Jim> classify(app::AppFactory& app);
  ///svm raster dataset
  void classifySVM(Jim& imgWriter, app::AppFactory& app);
  ///svm raster dataset only for in memory
  std::shared_ptr<Jim> classifySVM(app::AppFactory& app);
  ///svm raster dataset
  /* CPLErr svm(Jim& imgWriter, app::AppFactory& app); */
  ///svm raster dataset only for in memory
  /* std::shared_ptr<Jim> svm(app::AppFactory& app); */
  ///artificial neural network raster dataset
  void classifyANN(Jim& imgWriter, app::AppFactory& app);
  ///artificial neural network raster dataset only for in memory
  std::shared_ptr<Jim> classifyANN(app::AppFactory& app);
  ///artificial neural network raster dataset
  //CPLErr ann(Jim& imgWriter, app::AppFactory& app);
  ///ann raster dataset only for in memory
  //std::shared_ptr<Jim> ann(app::AppFactory& app);
  ///sml train and classify
  void classifySML(Jim& imgWriter, JimList& referenceReader, app::AppFactory& app);
  void classifySML(Jim& imgWriter, app::AppFactory& app);
  template<typename T> void classifySML_t(Jim& imgWriter, JimList& referenceReader, app::AppFactory& app);
  std::shared_ptr<Jim> classifySML(JimList& referenceReader, app::AppFactory& app);
  std::shared_ptr<Jim> classifySML(app::AppFactory& app);
  ///train sml raster dataset
  void trainSML(JimList& referenceReader, app::AppFactory& app);
  void trainSML2d(JimList& referenceReader, app::AppFactory& app);
  template<typename T> void trainSML_t(JimList& referenceReader, app::AppFactory& app);
  template<typename T> std::string trainSML2d_t(JimList& referenceReader, app::AppFactory& app);
  ///train sml raster dataset in memory
  /* std::string trainMem(JimList& referenceReader, app::AppFactory& app); */
  ///classify raster dataset using SML
  template<typename T> void classifySML_t(Jim& imgWriter, app::AppFactory& app);
  ///sml raster dataset only for in memory
  template<typename T> std::shared_ptr<Jim> classifySML_t(app::AppFactory& app);
  ///sml raster dataset
  /* template<typename T> CPLErr classifySML(Jim& imgWriter, app::AppFactory& app); */
  ///stretch raster dataset
  void stretch(Jim& imgWriter, app::AppFactory& app);
  ///stretch raster dataset only for in memory
  std::shared_ptr<Jim> stretch(app::AppFactory& app);
  ///reclass raster dataset
  void reclass(Jim& imgWriter, app::AppFactory& app);
  ///reclass raster dataset only for in memory
  std::shared_ptr<Jim> reclass(app::AppFactory& app);
  //simplified destructive version of reclass
  void d_reclass(app::AppFactory& app);
  ///set mask to raster dataset
  //todo: create template function and make it work for 3D
  void setMask(VectorOgr& ogrReader, Jim& imgWriter, app::AppFactory& app);
  ///set mask to raster dataset
  //todo: create template function and make it work for 3D
  void setMask(JimList& maskReader, Jim& imgWriter, app::AppFactory& app);
  /* CPLErr setMask(Jim& imgWriter, app::AppFactory& app); */
  ///setmask raster dataset only for in memory
  std::shared_ptr<Jim> setMask(VectorOgr& ogrReader, app::AppFactory& app);
  ///setmask raster dataset only for in memory
  std::shared_ptr<Jim> setMask(JimList& maskReader, app::AppFactory& app);
  /* std::shared_ptr<Jim> setMask(app::AppFactory& app); */
  ///setMask destructive version
  /* template<typename T> void d_setMask(Jim& mask, T value); */
  void d_setMask2D(Jim& mask, double value);
  void d_setMask(Jim& mask, double value);
  template<typename T> void d_setMask_t(Jim& mask, double value);
  void d_setMask2D(Jim& mask, Jim& other);
  void d_setMask(Jim& mask, Jim& other);
  template<typename T> void d_setMask_t(Jim& mask, Jim& other);
  ///get mask to raster dataset
  void getMask(Jim& imgWriter, app::AppFactory& app);
  ///getmask raster dataset only for in memory
  std::shared_ptr<Jim> getMask(app::AppFactory& app);
  ///dump raster dataset
  CPLErr dumpImg(app::AppFactory& app);
  ///get statistics
  std::multimap<std::string,std::string> getStats(app::AppFactory& app);
  ///get unique pixels
  /* template<typename T> std::map<std::vector<T>,std::vector<std::pair<unsigned short,unsigned short> > > getUniquePixels(unsigned short start, unsigned short end); */

  /// convert single plane multiband image to single band image with multiple planes
  void d_band2plane();
  ///read data bands into planes
  CPLErr readDataPlanes(std::vector<int> bands);
  ///write data bands from planes
  CPLErr writeDataPlanes();
  /// convert single band multiple plane image to single plane multiband image
  /* void d_plane2band();//not implemented yet */
#if MIALIB == 1
  ///get MIA representation for a particular band
  IMAGE* getMIA(int band=0);
  ///set memory from internal MIA representation for particular band
  CPLErr setMIA(int band=0);
  // ///set memory from MIA representation for particular band
  CPLErr setMIA(IMAGE* mia, int band=0);
  ///convert a GDAL data type to MIA data type
  int getMIADataType();
  ///convert a JIPLIB to MIA data type
  int JIPLIB2MIADataType(int aJIPLIBDataType);
  ///convert a GDAL to MIA data type
  int GDAL2MIADataType(GDALDataType aGDALDataType);
  ///convert a MIA data type to GDAL data type
  int MIA2JIPLIBDataType(int aMIADataType);
#endif

  //test for MIA functions
  template<typename T> void d_jlframebox_t(std::vector<int> box, T val, std::size_t band);
  void d_jlframebox(std::vector<int> box, double val, std::size_t band);
  template<typename T> void d_jldistanceGeodesic_t(Jim& reference, std::size_t graph, std::size_t band);
  void d_jldistanceGeodesic(Jim& reference, std::size_t, std::size_t band);
  std::shared_ptr<Jim> jldistanceGeodesic(Jim& reference, std::size_t graph, std::size_t band);
  void testFunction(){std::cout << "Hello testFunction" << std::endl;};

  //start insert from fun2method_imagetype
  //end insert from fun2method_imagetype
  //start insert from fun2method_imagetype_multi
  //end insert from fun2method_imagetype_multi
  //start insert from fun2method_imagelisttype
  //end insert from fun2method_imagelisttype
  //start insert from fun2method_errortype
  //end insert from fun2method_errortype
  //start insert from fun2method_errortype_d
  //end insert from fun2method_errortype_d
  //start insert from fun2method_errortype_nd
  //end insert from fun2method_errortype_nd
  //start insert from fun2method_errortype_nm
  //end insert from fun2method_errortype_nm
  //start insert from fun2method_errortype_d_nm
  //end insert from fun2method_errortype_d_nm
 protected:
  ///reset all member variables
  void reset();
  //convert data to PyArrayObject
  ///Initialize the memory for read/write image in cache
  CPLErr initMem(unsigned int memory);
  ///filename of this dataset
  std::string m_filename;
  ///instance of the GDAL dataset of this dataset
  GDALDataset *m_gds;
  ///number of columns in this dataset
  int m_ncol;
  ///number of rows in this dataset
  int m_nrow;
  ///number of bands in this dataset
  int m_nband;
  ///number of planes in this dataset
  int m_nplane;
  ///GDAL data type for this dataset
  int m_dataType;
  //GDALDataType m_dataType;
  ///image type for this dataset
  std::string m_imageType;
  ///geotransform information of this dataset
  std::vector<double> m_gt;
  //projection string in wkt format
  std::string m_projection;
  ///no data values for this dataset
  std::vector<double> m_noDataValues;
  ///Vector containing the scale factor to be applied (one scale value for each band)
  std::vector<double> m_scale;
  ///Vector containing the offset factor to be applied (one offset value for each band)
  std::vector<double> m_offset;
  ///Flag for external data pointer, do not delete data pointer m_data when flag is set
  bool m_externalData;

  ///Block size to cache pixel cell values in memory (calculated from user provided memory size in MB)
  unsigned int m_blockSize;
  ///The cached pixel cell values for a certain block: a vector of void pointers (one void pointer for each band)
  std::vector<void*> m_data;
  //todo: use smart pointer
  //std::vector<std::unique_ptr<void[]> > m_data;
  ///first line that has been read in cache for a specific band
  std::vector<int> m_begin;
  ///beyond last line read in cache for a specific band
  std::vector<int> m_end;

  ///register driver for GDAl
  CPLErr registerDriver();
  ///Create options
  std::vector<std::string> m_coptions;
  ///Open options
  std::vector<std::string> m_ooptions;
  ///We are writing a physical file
  /* bool m_writeMode; */
  ///access mode (ReadOnly or GA_Update)
  DATA_ACCESS m_access;
  ///resampling algorithm: GRIORA_NearestNeighbour|GRIORA_Bilinear|GRIORA_Cubic|GRIORA_CubicSpline|GRIORA_Lanczos|GRIORA_Average|GRIORA_Average|GRIORA_Gauss (check http://www.gdal.org/gdal_8h.html#a640ada511cbddeefac67c548e009d5a)"
  GDALRIOResampleAlg m_resample;

  ///Read new block for a specific band in the dataset in specified band of cache (defined by m_begin and m_end) in band ()
  CPLErr readNewBlockDS(int row, int band, int ds_band);
  ///Read new block in cache (defined by m_begin and m_end)
  CPLErr readNewBlock(int row, int band){readNewBlockDS(row,band,band);};
  ///Write new block from cache (defined by m_begin and m_end)
  CPLErr writeNewBlock(int row, int band);

 private:
  std::shared_ptr<Jim> cloneImpl(bool copyData) {
    return(std::make_shared<Jim>(*this,copyData));
  };
#if MIALIB == 1
  std::vector<IMAGE*> m_mia;
#endif
};

template<typename T> void Jim::readData(T& value, int col, int row, int band)
{
  try{
    if(nrOfBand()<=band){
      std::string errorString="Error: band number exceeds number of bands in input image";
      throw(errorString);
    }
    if(nrOfCol()<=col){
      std::string errorString="Error: col number exceeds number of cols in input image";
      throw(errorString);
    }
    if(nrOfRow()<=row){
      std::string errorString="Error: row number exceeds number of rows in input image";
      throw(errorString);
    }
    double dvalue=0;
    double theScale=1;
    double theOffset=0;
    if(m_scale.size()>band||m_offset.size()>band){
      if(m_scale.size()>band)
        theScale=m_scale[band];
      if(m_offset.size()>band)
        theOffset=m_offset[band];
    }
    if(m_data.size()){
      //only support random access reading if entire image is in memory for performance reasons
      if(m_blockSize!=nrOfRow()){
        std::ostringstream s;
        s << "Error: increase memory to support random access reading (now at " << 100.0*m_blockSize/nrOfRow() << "%)";
        throw(s.str());
      }
      if(row<m_begin[band]||row>=m_end[band]){
        if(m_filename.size())
          readNewBlock(row,band);
      }
      int index=(row-m_begin[band])*nrOfCol()+col;
      switch(getDataType()){
      case(GDT_Byte):
        dvalue=theScale*(static_cast<unsigned char*>(m_data[band])[index])+theOffset;
        break;
      case(GDT_Int16):
        dvalue=theScale*(static_cast<short*>(m_data[band])[index])+theOffset;
        break;
      case(GDT_UInt16):
        dvalue=theScale*(static_cast<unsigned short*>(m_data[band])[index])+theOffset;
        break;
      case(GDT_Int32):
        dvalue=theScale*(static_cast<int*>(m_data[band])[index])+theOffset;
        break;
      case(GDT_UInt32):
        dvalue=theScale*(static_cast<unsigned int*>(m_data[band])[index])+theOffset;
        break;
      case(GDT_Float32):
        dvalue=theScale*(static_cast<float*>(m_data[band])[index])+theOffset;
        break;
      case(GDT_Float64):
        dvalue=theScale*(static_cast<double*>(m_data[band])[index])+theOffset;
        break;
      default:
        std::string errorString="Error: data type not supported";
        throw(errorString);
        break;
      }
      value=static_cast<T>(dvalue);
    }
    else{
      //fetch raster band
      GDALRasterBand  *poBand;
      poBand = m_gds->GetRasterBand(band+1);//GDAL uses 1 based index
      poBand->RasterIO(GF_Read,col,row,1,1,&value,1,1,type2GDAL<T>(),0,0);
      dvalue=theScale*value+theOffset;
      value=static_cast<T>(dvalue);
    }
  }
  catch(std::string errorString){
    std::cerr << errorString << std::endl;
    throw;
  }
  catch(...){
    throw;
  }
}

template<typename T> void Jim::readData3D(T& value, std::size_t col, std::size_t row, std::size_t plane, std::size_t band)
{
  try{
    if(nrOfBand()<=band){
      std::string errorString="Error: band number exceeds number of bands in input image";
      throw(errorString);
    }
    if(nrOfCol()<=col){
      std::string errorString="Error: col number exceeds number of cols in input image";
      throw(errorString);
    }
    if(nrOfRow()<=row){
      std::string errorString="Error: row number exceeds number of rows in input image";
      throw(errorString);
    }
    if(nrOfPlane()<=plane){
      std::string errorString="Error: plane number exceeds number of planes in input image";
      throw(errorString);
    }
    double dvalue=0;
    double theScale=1;
    double theOffset=0;
    if(m_scale.size()>band||m_offset.size()>band){
      if(m_scale.size()>band)
        theScale=m_scale[band];
      if(m_offset.size()>band)
        theOffset=m_offset[band];
    }
    if(m_data.size()){
      //only support random access reading if entire image is in memory for performance reasons
      if(m_blockSize!=nrOfRow()){
        std::ostringstream s;
        s << "Error: increase memory to support random access reading (now at " << 100.0*m_blockSize/nrOfRow() << "%)";
        throw(s.str());
      }
      if(row<m_begin[band]||row>=m_end[band]){
        if(m_filename.size())
          readNewBlock(row,band);
      }
      int index=(plane*nrOfRow()*nrOfCol())+(row-m_begin[band])*nrOfCol()+col;
      switch(getDataType()){
      case(GDT_Byte):
        dvalue=theScale*(static_cast<unsigned char*>(m_data[band])[index])+theOffset;
        break;
      case(GDT_Int16):
        dvalue=theScale*(static_cast<short*>(m_data[band])[index])+theOffset;
        break;
      case(GDT_UInt16):
        dvalue=theScale*(static_cast<unsigned short*>(m_data[band])[index])+theOffset;
        break;
      case(GDT_Int32):
        dvalue=theScale*(static_cast<int*>(m_data[band])[index])+theOffset;
        break;
      case(GDT_UInt32):
        dvalue=theScale*(static_cast<unsigned int*>(m_data[band])[index])+theOffset;
        break;
      case(GDT_Float32):
        dvalue=theScale*(static_cast<float*>(m_data[band])[index])+theOffset;
        break;
      case(GDT_Float64):
        dvalue=theScale*(static_cast<double*>(m_data[band])[index])+theOffset;
        break;
      default:
        std::string errorString="Error: data type not supported";
        throw(errorString);
        break;
      }
      value=static_cast<T>(dvalue);
    }
    else{
      std::ostringstream s;
      s << "Error: read 3D image only supported for in memory images";
      throw(s.str());
    }
  }
  catch(std::string errorString){
    std::cerr << errorString << std::endl;
    throw;
  }
  catch(...){
    throw;
  }
}

template<typename T> void Jim::readData3D(std::vector<T>& buffer, std::size_t minCol, std::size_t maxCol, std::size_t row, std::size_t plane, std::size_t band)
{
  try{
    if(nrOfBand()<=band){
      std::string errorString="Error: band number exceeds number of bands in input image";
      throw(errorString);
    }
    if(nrOfRow()<=row){
      std::string errorString="Error: row number exceeds number of rows in input image";
      throw(errorString);
    }
    if(nrOfPlane()<=plane){
      std::string errorString="Error: plane number exceeds number of planes in input image";
      throw(errorString);
    }
    double theScale=1;
    double theOffset=0;
    if(m_scale.size()>band||m_offset.size()>band){
      if(m_scale.size()>band)
        theScale=m_scale[band];
      if(m_offset.size()>band)
        theOffset=m_offset[band];
    }
    if(m_data.size()){
      if(row<m_begin[band]||row>=m_end[band]){
        if(m_filename.size())
          readNewBlock(row,band);
      }
      if(buffer.size()!=maxCol-minCol+1)
        buffer.resize(maxCol-minCol+1);
      int index=plane*nrOfRow()*nrOfCol()+(row-m_begin[band])*nrOfCol();
      int minindex=(index+minCol);
      int maxindex=(index+maxCol);
      /* if(type2GDAL<T>()==getDataType()){//no conversion needed */
      /*   buffer.assign(static_cast<T*>(m_data[band])+minindex,static_cast<T*>(m_data[band])+maxindex); */
      /*   typename std::vector<T>::iterator bufit=buffer.begin(); */
      /*   while(bufit!=buffer.end()){ */
      /*     double dvalue=theScale*(*bufit)+theOffset; */
      /*     *(bufit++)=static_cast<T>(dvalue); */
      /*   } */
      /* } */
      /* else{ */
      typename std::vector<T>::iterator bufit=buffer.begin();
      for(index=minindex;index<=maxindex;++index,++bufit){
        double dvalue=0;
        switch(getDataType()){
        case(GDT_Byte):
          dvalue=theScale*(static_cast<unsigned char*>(m_data[band])[index])+theOffset;
          break;
        case(GDT_Int16):
          dvalue=theScale*(static_cast<short*>(m_data[band])[index])+theOffset;
          break;
        case(GDT_UInt16):
          dvalue=theScale*(static_cast<unsigned short*>(m_data[band])[index])+theOffset;
          break;
        case(GDT_Int32):
          dvalue=theScale*(static_cast<int*>(m_data[band])[index])+theOffset;
          break;
        case(GDT_UInt32):
          dvalue=theScale*(static_cast<unsigned int*>(m_data[band])[index])+theOffset;
          break;
        case(GDT_Float32):
          dvalue=theScale*(static_cast<float*>(m_data[band])[index])+theOffset;
          break;
        case(GDT_Float64):
          dvalue=theScale*(static_cast<double*>(m_data[band])[index])+theOffset;
          break;
        default:
          std::string errorString="Error: data type not supported";
          throw(errorString);
          break;
        }
        // double dvalue=theScale*(*(static_cast<double*>(m_data[band])+index))+theOffset;
        *(bufit)=static_cast<T>(dvalue);
      }
    }
    else{
      std::ostringstream s;
      s << "Error: read 3D image only supported for in memory images";
      throw(s.str());
    }
  }
  catch(std::string errorString){
    std::cerr << errorString << std::endl;
    throw;
  }
  catch(...){
    throw;
  }
}

/**
 * @param[out] buffer The vector with all cell values that were read
 * @param[in] minCol First column from where to start reading (counting starts from 0)
 * @param[in] maxCol Last column that must be read (counting starts from 0)
 * @param[in] row The row number to read (counting starts from 0)
 * @param[in] band The band number to read (counting starts from 0)
 **/
template<typename T> CPLErr Jim::readData(std::vector<T>& buffer, int minCol, int maxCol, int row, int band)
{
  try{
    if(nrOfBand()<=band){
      std::string errorString="Error: band number exceeds number of bands in input image";
      throw(errorString);
    }
    if(maxCol<minCol){
      std::string errorString="Error: maxCol must be larger or equal to minCol";
      throw(errorString);
    }
    if(minCol<0){
      std::string errorString="Error: col number must be positive";
      throw(errorString);
    }
    if(maxCol<0){
      std::string errorString="Error: col number must be positive";
      throw(errorString);
    }
    if(row<0){
      std::string errorString="Error: row number must be positive";
      throw(errorString);
    }
    if(nrOfCol()<=minCol){
      std::string errorString="Error: col number exceeds number of cols in input image";
      throw(errorString);
    }
    if(nrOfCol()<=maxCol){
      std::string errorString="Error: col number exceeds number of cols in input image";
      throw(errorString);
    }
    if(nrOfRow()<=row){
      std::string errorString="Error: row number exceeds number of rows in input image";
      throw(errorString);
    }
    CPLErr returnValue=CE_None;
    double theScale=1;
    double theOffset=0;
    if(m_scale.size()>band||m_offset.size()>band){
      if(m_scale.size()>band)
        theScale=m_scale[band];
      if(m_offset.size()>band)
        theOffset=m_offset[band];
    }
    if(m_data.size()){
      if(row<m_begin[band]||row>=m_end[band]){
        if(m_filename.size())
          returnValue=readNewBlock(row,band);
      }
      if(buffer.size()!=maxCol-minCol+1)
        buffer.resize(maxCol-minCol+1);
      int index=(row-m_begin[band])*nrOfCol();
      int minindex=(index+minCol);
      int maxindex=(index+maxCol);
      /* if(type2GDAL<T>()==getDataType()){//no conversion needed */
      /*   buffer.assign(static_cast<T*>(m_data[band])+minindex,static_cast<T*>(m_data[band])+maxindex); */
      /*   typename std::vector<T>::iterator bufit=buffer.begin(); */
      /*   while(bufit!=buffer.end()){ */
      /*     double dvalue=theScale*(*bufit)+theOffset; */
      /*     *(bufit++)=static_cast<T>(dvalue); */
      /*   } */
      /* } */
      /* else{ */
      typename std::vector<T>::iterator bufit=buffer.begin();
      for(index=minindex;index<=maxindex;++index,++bufit){
        double dvalue=0;
        switch(getDataType()){
        case(GDT_Byte):
          dvalue=theScale*(static_cast<unsigned char*>(m_data[band])[index])+theOffset;
          break;
        case(GDT_Int16):
          dvalue=theScale*(static_cast<short*>(m_data[band])[index])+theOffset;
          break;
        case(GDT_UInt16):
          dvalue=theScale*(static_cast<unsigned short*>(m_data[band])[index])+theOffset;
          break;
        case(GDT_Int32):
          dvalue=theScale*(static_cast<int*>(m_data[band])[index])+theOffset;
          break;
        case(GDT_UInt32):
          dvalue=theScale*(static_cast<unsigned int*>(m_data[band])[index])+theOffset;
          break;
        case(GDT_Float32):
          dvalue=theScale*(static_cast<float*>(m_data[band])[index])+theOffset;
          break;
        case(GDT_Float64):
          dvalue=theScale*(static_cast<double*>(m_data[band])[index])+theOffset;
          break;
        default:
          std::string errorString="Error: data type not supported";
          throw(errorString);
          break;
        }
        // double dvalue=theScale*(*(static_cast<double*>(m_data[band])+index))+theOffset;
        *(bufit)=static_cast<T>(dvalue);
      }
    }
    else{
      //fetch raster band
      GDALRasterBand  *poBand;
      poBand = m_gds->GetRasterBand(band+1);//GDAL uses 1 based index
      if(buffer.size()!=maxCol-minCol+1)
        buffer.resize(maxCol-minCol+1);
      returnValue=poBand->RasterIO(GF_Read,minCol,row,buffer.size(),1,&(buffer[0]),buffer.size(),1,type2GDAL<T>(),0,0);
      if(m_scale.size()>band||m_offset.size()>band){
        for(int index=0;index<buffer.size();++index)
          buffer[index]=theScale*static_cast<double>(buffer[index])+theOffset;
      }
    }
    return(returnValue);
  }
  catch(std::string errorString){
    std::cerr << errorString << std::endl;
    throw;
  }
  catch(...){
    throw;
  }
}

/**
 * @param[out] buffer The vector with all cell values that were read
 * @param[in] minCol First column from where to start reading (counting starts from 0)
 * @param[in] maxCol Last column that must be read (counting starts from 0)
 * @param[in] row The row number to read (counting starts from 0)
 * @param[in] band The band number to read (counting starts from 0)
 * @param[in] resample The resampling method (currently only BILINEAR and NEAR are supported)
 **/
template<typename T> CPLErr Jim::readData(std::vector<T>& buffer, int minCol, int maxCol, double row, int band, RESAMPLE resample)
{
  CPLErr returnValue=CE_None;
  std::vector<T> readBuffer_upper;
  std::vector<T> readBuffer_lower;
  if(buffer.size()!=maxCol-minCol+1)
    buffer.resize(maxCol-minCol+1);
  double upperRow=row-0.5;
  upperRow=static_cast<int>(upperRow+FLT_EPSILON);
  double lowerRow=row+0.5;
  lowerRow=static_cast<int>(lowerRow+FLT_EPSILON);
  switch(resample){
  case(BILINEAR):
    if(lowerRow>=nrOfRow())
      lowerRow=nrOfRow()-1;
    if(upperRow<0)
      upperRow=0;
    returnValue=readData(readBuffer_upper,minCol,maxCol,static_cast<int>(upperRow),band);
    returnValue=readData(readBuffer_lower,minCol,maxCol,static_cast<int>(lowerRow),band);
    //do interpolation in y
    for(int icol=0;icol<maxCol-minCol+1;++icol){
      //buffer[icol]=(lowerRow-row+0.5)*readBuffer_upper[icol]+(1-lowerRow+row-0.5)*readBuffer_lower[icol];
      double upperVal = readBuffer_upper[icol];
      double lowerVal = readBuffer_lower[icol];
      if (!isNoData(upperVal)) {
        if (!isNoData(lowerVal))
          buffer[icol]=(lowerRow-row+0.5)*upperVal+(1-lowerRow+row-0.5)*lowerVal;
        else
          buffer[icol] = upperVal;
      }
      else
        buffer[icol] = lowerVal;
    }
    break;
  default:
    //returnValue=readData(buffer,minCol,maxCol,static_cast<int>(row),band);
    returnValue=readData(buffer,minCol,maxCol,static_cast<int>(row+FLT_EPSILON),band);
    break;
  }
  return(returnValue);
}

/**
 * @param[out] buffer2d Two dimensional vector of type Vector2d (stl vector of stl vector) representing [row][col]. This vector contains all cell values that were read
 * @param[in] minCol First column from where to start reading (counting starts from 0)
 * @param[in] maxCol Last column that must be read (counting starts from 0)
 * @param[in] minRow First row from where to start reading (counting starts from 0)
 * @param[in] maxRow Last row that must be read (counting starts from 0)
 * @param[in] band The band number to read (counting starts from 0)
 **/
template<typename T> CPLErr Jim::readDataBlock(Vector2d<T>& buffer2d, int minCol, int maxCol, int minRow, int maxRow, int band)
{
  CPLErr returnValue=CE_None;
  buffer2d.resize(maxRow-minRow+1,maxCol-minCol+1);
  typename std::vector<T> buffer;
  returnValue=readDataBlock(buffer,minCol,maxCol,minRow,maxRow,band);
  typename std::vector<T>::const_iterator startit=buffer.begin();
  typename std::vector<T>::const_iterator endit=startit;
  for(int irow=minRow;irow<=maxRow;++irow){
    //buffer2d[irow-minRow].resize(maxCol-minCol+1);
    endit+=maxCol-minCol+1;
    buffer2d[irow-minRow].assign(startit,endit);
    startit+=maxCol-minCol+1;
  }
  return(returnValue);
}

template<typename T> void Jim::readDataBlock3D(Vector2d<T>& buffer2d, std::size_t minCol, std::size_t maxCol, std::size_t minRow, std::size_t maxRow, std::size_t plane, std::size_t band){
  buffer2d.resize(maxRow-minRow+1,maxCol-minCol+1);
  typename std::vector<T> buffer;
  readDataBlock3D(buffer,minCol,maxCol,minRow,maxRow,plane,band);
  typename std::vector<T>::const_iterator startit=buffer.begin();
  typename std::vector<T>::const_iterator endit=startit;
  for(int irow=minRow;irow<=maxRow;++irow){
    //buffer2d[irow-minRow].resize(maxCol-minCol+1);
    endit+=maxCol-minCol+1;
    buffer2d[irow-minRow].assign(startit,endit);
    startit+=maxCol-minCol+1;
  }
}

/**
 * @param[out] buffer One dimensional vector representing all pixel values read starting from upper left to lower right.
 * @param[in] minCol First column from where to start reading (counting starts from 0)
 * @param[in] maxCol Last column that must be read (counting starts from 0)
 * @param[in] minRow First row from where to start reading (counting starts from 0)
 * @param[in] maxRow Last row that must be read (counting starts from 0)
 * @param[in] band The band number to read (counting starts from 0)
 **/
template<typename T> CPLErr Jim::readDataBlock(std::vector<T>& buffer, int minCol, int maxCol, int minRow, int maxRow, int band)
{
  try{
    CPLErr returnValue=CE_None;
    double theScale=1;
    double theOffset=0;
    if(m_scale.size()>band)
      theScale=m_scale[band];
    if(m_offset.size()>band)
      theOffset=m_offset[band];
    if(minCol>=nrOfCol() ||
       (minCol<0) ||
       (maxCol>=nrOfCol()) ||
       (minCol>maxCol) ||
       (minRow>=nrOfRow()) ||
       (minRow<0) ||
       (maxRow>=nrOfRow()) ||
       (minRow>maxRow)){
      std::string errorString="block not within image boundaries";
      throw(errorString);
    }
    if(buffer.size()!=(maxRow-minRow+1)*(maxCol-minCol+1))
      buffer.resize((maxRow-minRow+1)*(maxCol-minCol+1));
    if(m_data.size()){
      typename std::vector<T>::iterator bufit=buffer.begin();
      for(int irow=minRow;irow<=maxRow;++irow){
        if(irow<m_begin[band]||irow>=m_end[band]){
          if(m_filename.size())
            returnValue=readNewBlock(irow,band);
        }
        int index=(irow-m_begin[band])*nrOfCol();
        int minindex=(index+minCol);//*(GDALGetDataTypeSize(getDataType())>>3);
        int maxindex=(index+maxCol);//*(GDALGetDataTypeSize(getDataType())>>3);

        for(index=minindex;index<=maxindex;++index,++bufit){
          double dvalue=0;
          switch(getDataType()){
          case(GDT_Byte):
            dvalue=theScale*(static_cast<unsigned char*>(m_data[band])[index])+theOffset;
            break;
          case(GDT_Int16):
            dvalue=theScale*(static_cast<short*>(m_data[band])[index])+theOffset;
            break;
          case(GDT_UInt16):
            dvalue=theScale*(static_cast<unsigned short*>(m_data[band])[index])+theOffset;
            break;
          case(GDT_Int32):
            dvalue=theScale*(static_cast<int*>(m_data[band])[index])+theOffset;
            break;
          case(GDT_UInt32):
            dvalue=theScale*(static_cast<unsigned int*>(m_data[band])[index])+theOffset;
            break;
          case(GDT_Float32):
            dvalue=theScale*(static_cast<float*>(m_data[band])[index])+theOffset;
            break;
          case(GDT_Float64):
            dvalue=theScale*(static_cast<double*>(m_data[band])[index])+theOffset;
            break;
          default:
            std::string errorString="Error: data type not supported";
            throw(errorString);
            break;
          }
          *(bufit)=static_cast<T>(dvalue);
        }//for index
      }
    }
    else{
      //fetch raster band
      GDALRasterBand  *poBand;
      if(nrOfBand()<=band){
        std::string errorString="Error: band number exceeds number of bands in input image";
        throw(errorString);
      }
      poBand = m_gds->GetRasterBand(band+1);//GDAL uses 1 based index
      returnValue=poBand->RasterIO(GF_Read,minCol,minRow,maxCol-minCol+1,maxRow-minRow+1,&(buffer[0]),(maxCol-minCol+1),(maxRow-minRow+1),type2GDAL<T>(),0,0);
      if(m_scale.size()>band||m_offset.size()>band){
        for(int index=0;index<buffer.size();++index)
          buffer[index]=theScale*buffer[index]+theOffset;
      }
    }
    return(returnValue);
  }
  catch(std::string errorString){
    std::cerr << errorString << std::endl;
    throw;
  }
  catch(...){
    throw;
  }
}

template<typename T> void Jim::readDataBlock3D(std::vector<T>& buffer, std::size_t minCol, std::size_t maxCol, std::size_t minRow, std::size_t maxRow, std::size_t plane, std::size_t band)
{
  try{
    double theScale=1;
    double theOffset=0;
    if(m_scale.size()>band)
      theScale=m_scale[band];
    if(m_offset.size()>band)
      theOffset=m_offset[band];
    if(minCol>=nrOfCol() ||
       (minCol<0) ||
       (maxCol>=nrOfCol()) ||
       (minCol>maxCol) ||
       (minRow>=nrOfRow()) ||
       (minRow<0) ||
       (maxRow>=nrOfRow()) ||
       (minRow>maxRow) ||
       (plane>=nrOfPlane()) ||
       (plane<0)){
      std::string errorString="block not within image boundaries";
      throw(errorString);
    }
    if(buffer.size()!=(maxRow-minRow+1)*(maxCol-minCol+1))
      buffer.resize((maxRow-minRow+1)*(maxCol-minCol+1));
    if(m_data.size()){
      typename std::vector<T>::iterator bufit=buffer.begin();
      for(int irow=minRow;irow<=maxRow;++irow){
        if(irow<m_begin[band]||irow>=m_end[band]){
          if(m_filename.size())
            readNewBlock(irow,band);
        }
        /* int index=(irow-m_begin[band])*nrOfCol(); */
        std::size_t index=(plane*nrOfRow()*nrOfCol())+(irow-m_begin[band])*nrOfCol();
        std::size_t minindex=(index+minCol);//*(GDALGetDataTypeSize(getDataType())>>3);
        std::size_t maxindex=(index+maxCol);//*(GDALGetDataTypeSize(getDataType())>>3);

        for(index=minindex;index<=maxindex;++index,++bufit){
          double dvalue=0;
          switch(getDataType()){
          case(GDT_Byte):
            dvalue=theScale*(static_cast<unsigned char*>(m_data[band])[index])+theOffset;
            break;
          case(GDT_Int16):
            dvalue=theScale*(static_cast<short*>(m_data[band])[index])+theOffset;
            break;
          case(GDT_UInt16):
            dvalue=theScale*(static_cast<unsigned short*>(m_data[band])[index])+theOffset;
            break;
          case(GDT_Int32):
            dvalue=theScale*(static_cast<int*>(m_data[band])[index])+theOffset;
            break;
          case(GDT_UInt32):
            dvalue=theScale*(static_cast<unsigned int*>(m_data[band])[index])+theOffset;
            break;
          case(GDT_Float32):
            dvalue=theScale*(static_cast<float*>(m_data[band])[index])+theOffset;
            break;
          case(GDT_Float64):
            dvalue=theScale*(static_cast<double*>(m_data[band])[index])+theOffset;
            break;
          default:
            std::string errorString="Error: data type not supported";
            throw(errorString);
            break;
          }
          *(bufit)=static_cast<T>(dvalue);
        }//for index
      }
    }
  }
  catch(std::string errorString){
    std::cerr << errorString << std::endl;
    throw;
  }
  catch(...){
    throw;
  }
}

/**
 * @param[out] buffer The vector with all cell values that were read
 * @param[in] row The row number to read (counting starts from 0)
 * @param[in] band The band number to read (counting starts from 0)
 **/
template<typename T> CPLErr Jim::readData(std::vector<T>& buffer, int row, int band)
{
  return(readData(buffer,0,nrOfCol()-1,row,band));
}

/**
 * @param[out] buffer The vector with all cell values that were read
 * @param[in] row The row number to read (counting starts from 0)
 * @param[in] band The band number to read (counting starts from 0)
 * @param[in] resample The resampling method (currently only BILINEAR and NEAR are supported).
 **/
template<typename T> CPLErr Jim::readData(std::vector<T>& buffer, double row, int band, RESAMPLE resample)
{
  return(readData(buffer,0,nrOfCol()-1,row,band,resample));
}

//From Writer
/**
 * @param[in] value The cell value to write
 * @param[in] col The column number to write (counting starts from 0)
 * @param[in] row The row number to write (counting starts from 0)
 * @param[in] band The band number to write (counting starts from 0)
 * @return true if write successful
 **/
template<typename T> CPLErr Jim::writeData(const T& value, int col, int row, int band)
{
  CPLErr returnValue=CE_None;
  if(band>=nrOfBand()+1){
    std::ostringstream s;
    s << "Error: band (" << band << ") exceeds nrOfBand (" << nrOfBand() << ")";
    throw(s.str());
  }
  if(col>=nrOfCol()){
    std::ostringstream s;
    s << "Error: col (" << col << ") exceeds nrOfCol (" << nrOfCol() << ")";
    throw(s.str());
  }
  if(col<0){
    std::ostringstream s;
    s << "Error: col (" << col << ") is negative";
    throw(s.str());
  }
  if(row>=nrOfRow()){
    std::ostringstream s;
    s << "Error: row (" << row << ") exceeds nrOfRow (" << nrOfRow() << ")";
    throw(s.str());
  }
  if(row<0){
    std::ostringstream s;
    s << "Error: row (" << row << ") is negative";
    throw(s.str());
  }
  double theScale=1;
  double theOffset=0;
  if(m_scale.size()>band||m_offset.size()>band){
    if(m_scale.size()>band)
      theScale=m_scale[band];
    if(m_offset.size()>band)
      theOffset=m_offset[band];
  }
  if(m_data.size()){
    //only support random access writing if entire image is in memory
    if(m_blockSize!=nrOfRow()){
      std::ostringstream s;
      s << "Error: increase memory to support random access writing (now at " << 100.0*m_blockSize/nrOfRow() << "%)";
      throw(s.str());
    }
    int index=(row-m_begin[band])*nrOfCol()+col;
    double dvalue=theScale*value+theOffset;
    switch(getDataType()){
    case(GDT_Byte):
      static_cast<unsigned char*>(m_data[band])[index]=static_cast<unsigned char>(dvalue);
      break;
    case(GDT_Int16):
      static_cast<short*>(m_data[band])[index]=static_cast<short>(dvalue);
      break;
    case(GDT_UInt16):
      static_cast<unsigned short*>(m_data[band])[index]=static_cast<unsigned short>(dvalue);
      break;
    case(GDT_Int32):
      static_cast<int*>(m_data[band])[index]=static_cast<int>(dvalue);
      break;
    case(GDT_UInt32):
      static_cast<unsigned int*>(m_data[band])[index]=static_cast<unsigned int>(dvalue);
      break;
    case(GDT_Float32):
      static_cast<float*>(m_data[band])[index]=static_cast<float>(dvalue);
      break;
    case(GDT_Float64):
      static_cast<double*>(m_data[band])[index]=static_cast<double>(dvalue);
      break;
    default:
      std::string errorString="Error: data type not supported";
      throw(errorString);
      break;
    }
  }
  else{
    //fetch raster band
    GDALRasterBand  *poBand;
    T dvalue=theScale*value+theOffset;
    returnValue=poBand->RasterIO(GF_Write,col,row,1,1,&dvalue,1,1,type2GDAL<T>(),0,0);
  }
  return(returnValue);
}

template<typename T> CPLErr Jim::writeData(std::vector<T>& buffer, int minCol, int maxCol, int row, int band)
{
  CPLErr returnValue=CE_None;
  if(buffer.size()!=maxCol-minCol+1){
    std::string errorstring="invalid size of buffer";
    throw(errorstring);
  }
  if(minCol>=nrOfCol()){
    std::ostringstream s;
    s << "minCol (" << minCol << ") exceeds nrOfCol (" << nrOfCol() << ")";
    throw(s.str());
  }
  if(minCol<0){
    std::ostringstream s;
    s << "mincol (" << minCol << ") is negative";
    throw(s.str());
  }
  if(maxCol>=nrOfCol()){
    std::ostringstream s;
    s << "maxCol (" << maxCol << ") exceeds nrOfCol (" << nrOfCol() << ")";
    throw(s.str());
  }
  if(maxCol<minCol){
    std::ostringstream s;
    s << "maxCol (" << maxCol << ") is less than minCol (" << minCol << ")";
    throw(s.str());
  }
  if(row>=nrOfRow()){
    std::ostringstream s;
    s << "row (" << row << ") exceeds nrOfRow (" << nrOfRow() << ")";
    throw(s.str());
  }
  if(row<0){
    std::ostringstream s;
    s << "row (" << row << ") is negative";
    throw(s.str());
  }
  if(m_data.size()){
    if(minCol>0){
      std::ostringstream s;
      s << "Error: increase memory to support random access writing (now at " << 100.0*m_blockSize/nrOfRow() << "%)";
      throw(s.str());
    }
    if(row>=m_end[band]){
      if(row>=m_end[band]+m_blockSize){
        std::ostringstream s;
        s << "Error: increase memory to support random access writing (now at " << 100.0*m_blockSize/nrOfRow() << "%)";
        throw(s.str());
      }
      else if(m_filename.size())
        returnValue=writeNewBlock(row,band);
    }
    int index=(row-m_begin[band])*nrOfCol();
    int minindex=(index+minCol);
    int maxindex=(index+maxCol);
    typename std::vector<T>::const_iterator bufit=buffer.begin();
    double theScale=1;
    double theOffset=0;
    if(m_scale.size()>band)
      theScale=m_scale[band];
    if(m_offset.size()>band)
      theOffset=m_offset[band];
    for(index=minindex;index<=maxindex;++index,++bufit){
      double dvalue=theScale*(*(bufit))+theOffset;
      switch(getDataType()){
      case(GDT_Byte):
        static_cast<unsigned char*>(m_data[band])[index]=static_cast<unsigned char>(dvalue);
        break;
      case(GDT_Int16):
        static_cast<short*>(m_data[band])[index]=static_cast<short>(dvalue);
        break;
      case(GDT_UInt16):
        static_cast<unsigned short*>(m_data[band])[index]=static_cast<unsigned short>(dvalue);
        break;
      case(GDT_Int32):
        static_cast<int*>(m_data[band])[index]=static_cast<int>(dvalue);
        break;
      case(GDT_UInt32):
        static_cast<unsigned int*>(m_data[band])[index]=static_cast<unsigned int>(dvalue);
        break;
      case(GDT_Float32):
        static_cast<float*>(m_data[band])[index]=static_cast<float>(dvalue);
        break;
      case(GDT_Float64):
        static_cast<double*>(m_data[band])[index]=static_cast<double>(dvalue);
        break;
      default:
        std::string errorString="Error: data type not supported";
        throw(errorString);
        break;
      }
    }
  }
  else{
    //todo: scaling and offset!
    //fetch raster band
    GDALRasterBand  *poBand;
    if(band>=nrOfBand()+1){
      std::ostringstream s;
      s << "band (" << band << ") exceeds nrOfBand (" << nrOfBand() << ")";
      throw(s.str());
    }
    poBand = m_gds->GetRasterBand(band+1);//GDAL uses 1 based index
    returnValue=poBand->RasterIO(GF_Write,minCol,row,buffer.size(),1,&(buffer[0]),buffer.size(),1,type2GDAL<T>(),0,0);
  }
  return(returnValue);
}

/* template<typename T> void Jim::writeData3D(std::vector<T>& buffer, std::size_t minCol, std::size_t maxCol, std::size_t row, std::size_t plane, std::size_t band) */
/* { */
/*   if(buffer.size()!=maxCol-minCol+1){ */
/*     std::string errorstring="invalid size of buffer"; */
/*     throw(errorstring); */
/*   } */
/*   if(minCol>=nrOfCol()){ */
/*     std::ostringstream s; */
/*     s << "minCol (" << minCol << ") exceeds nrOfCol (" << nrOfCol() << ")"; */
/*     throw(s.str()); */
/*   } */
/*   if(minCol<0){ */
/*     std::ostringstream s; */
/*     s << "mincol (" << minCol << ") is negative"; */
/*     throw(s.str()); */
/*   } */
/*   if(maxCol>=nrOfCol()){ */
/*     std::ostringstream s; */
/*     s << "maxCol (" << maxCol << ") exceeds nrOfCol (" << nrOfCol() << ")"; */
/*     throw(s.str()); */
/*   } */
/*   if(maxCol<minCol){ */
/*     std::ostringstream s; */
/*     s << "maxCol (" << maxCol << ") is less than minCol (" << minCol << ")"; */
/*     throw(s.str()); */
/*   } */
/*   if(row>=nrOfRow()){ */
/*     std::ostringstream s; */
/*     s << "row (" << row << ") exceeds nrOfRow (" << nrOfRow() << ")"; */
/*     throw(s.str()); */
/*   } */
/*   if(row<0){ */
/*     std::ostringstream s; */
/*     s << "row (" << row << ") is negative"; */
/*     throw(s.str()); */
/*   } */
/*   if(plane>=nrOfPlane()){ */
/*     std::ostringstream s; */
/*     s << "plane (" << plane << ") exceeds nrOfPlane (" << nrOfPlane() << ")"; */
/*     throw(s.str()); */
/*   } */
/*   if(plane<0){ */
/*     std::ostringstream s; */
/*     s << "plane (" << plane << ") is negative"; */
/*     throw(s.str()); */
/*   } */
/*   if(m_data.size()){ */
/*     if(minCol>0){ */
/*       std::ostringstream s; */
/*       s << "Error: increase memory to support random access writing (now at " << 100.0*m_blockSize/nrOfRow() << "%)"; */
/*       throw(s.str()); */
/*     } */
/*     if(row>=m_end[band]){ */
/*       if(row>=m_end[band]+m_blockSize){ */
/*         std::ostringstream s; */
/*         s << "Error: increase memory to support random access writing (now at " << 100.0*m_blockSize/nrOfRow() << "%)"; */
/*         throw(s.str()); */
/*       } */
/*       else if(m_filename.size()) */
/*         writeNewBlock(row,band); */
/*     } */
/*     int index=plane*nrOfRow()*nrOfCol()+(row-m_begin[band])*nrOfCol(); */
/*     int minindex=(index+minCol); */
/*     int maxindex=(index+maxCol); */
/*     typename std::vector<T>::const_iterator bufit=buffer.begin(); */
/*     double theScale=1; */
/*     double theOffset=0; */
/*     if(m_scale.size()>band) */
/*       theScale=m_scale[band]; */
/*     if(m_offset.size()>band) */
/*       theOffset=m_offset[band]; */
/*     for(index=minindex;index<=maxindex;++index,++bufit){ */
/*       double dvalue=theScale*(*(bufit))+theOffset; */
/*       switch(getDataType()){ */
/*       case(GDT_Byte): */
/*         static_cast<unsigned char*>(m_data[band])[index]=static_cast<unsigned char>(dvalue); */
/*         break; */
/*       case(GDT_Int16): */
/*         static_cast<short*>(m_data[band])[index]=static_cast<short>(dvalue); */
/*         break; */
/*       case(GDT_UInt16): */
/*         static_cast<unsigned short*>(m_data[band])[index]=static_cast<unsigned short>(dvalue); */
/*         break; */
/*       case(GDT_Int32): */
/*         static_cast<int*>(m_data[band])[index]=static_cast<int>(dvalue); */
/*         break; */
/*       case(GDT_UInt32): */
/*         static_cast<unsigned int*>(m_data[band])[index]=static_cast<unsigned int>(dvalue); */
/*         break; */
/*       case(GDT_Float32): */
/*         static_cast<float*>(m_data[band])[index]=static_cast<float>(dvalue); */
/*         break; */
/*       case(GDT_Float64): */
/*         static_cast<double*>(m_data[band])[index]=static_cast<double>(dvalue); */
/*         break; */
/*       default: */
/*         std::string errorString="Error: data type not supported"; */
/*         throw(errorString); */
/*         break; */
/*       } */
/*     } */
/*   } */
/*   else{ */
/*       std::ostringstream s; */
/*       s << "writeData3D only supported for data in memory"; */
/*       throw(s.str()); */
/*   } */
/* } */

/**
 * @param[in] buffer The vector with all cell values to write
 * @param[in] row The row number to write (counting starts from 0)
 * @param[in] band The band number to write (counting starts from 0)
 * @return true if write successful
 **/
template<typename T> CPLErr Jim::writeData(std::vector<T>& buffer, int row, int band)
{
  return writeData(buffer,0,nrOfCol()-1,row,band);
}

/**
 * @param[in] buffer2d Two dimensional vector of type Vector2d (stl vector of stl vector) representing [row][col]. This vector contains all cell values that must be written
 * @param[in] minCol First column from where to start writing (counting starts from 0)
 * @param[in] maxCol Last column that must be written (counting starts from 0)
 * @param[in] row The row number to write (counting starts from 0)
 * @param[in] band The band number to write (counting starts from 0)
 * @return true if write successful
 **/
template<typename T> CPLErr Jim::writeDataBlock(Vector2d<T>& buffer2d, int minCol, int maxCol, int minRow, int maxRow, int band)
{
  CPLErr returnValue=CE_None;
  double theScale=1;
  double theOffset=0;
  if(m_scale.size()>band)
    theScale=m_scale[band];
  if(m_offset.size()>band)
    theOffset=m_offset[band];
  if(buffer2d.size()!=maxRow-minRow+1){
    std::string errorstring="invalid buffer size";
    throw(errorstring);
  }
  if(band>=nrOfBand()+1){
    std::ostringstream s;
    s << "band (" << band << ") exceeds nrOfBand (" << nrOfBand() << ")";
    throw(s.str());
  }
  if(minCol>=nrOfCol()){
    std::ostringstream s;
    s << "minCol (" << minCol << ") exceeds nrOfCol (" << nrOfCol() << ")";
    throw(s.str());
  }
  if(minCol<0){
    std::ostringstream s;
    s << "mincol (" << minCol << ") is negative";
    throw(s.str());
  }
  if(minCol>0){
    std::ostringstream s;
    s << "Error: increase memory to support random access writing (now at " << 100.0*m_blockSize/nrOfRow() << "%)";
    throw(s.str());
  }
  if(maxCol>=nrOfCol()){
    std::ostringstream s;
    s << "maxCol (" << maxCol << ") exceeds nrOfCol (" << nrOfCol() << ")";
    throw(s.str());
  }
  if(maxCol<minCol){
    std::ostringstream s;
    s << "maxCol (" << maxCol << ") is less than minCol (" << minCol << ")";
    throw(s.str());
  }
  if(m_data.size()){
    for(int irow=minRow;irow<=maxRow;++irow){
      if(irow>=nrOfRow()){
        std::ostringstream s;
        s << "row (" << irow << ") exceeds nrOfRow (" << nrOfRow() << ")";
        throw(s.str());
      }
      if(irow<0){
        std::ostringstream s;
        s << "row (" << irow << ") is negative";
        throw(s.str());
      }
      if(irow<m_begin[band]){
        std::ostringstream s;
        s << "Error: increase memory to support random access writing (now at " << 100.0*m_blockSize/nrOfRow() << "%)";
        throw(s.str());
      }
      if(irow>=m_end[band]){
        if(irow>=m_end[band]+m_blockSize){
          std::ostringstream s;
          s << "Error: increase memory to support random access writing (now at " << 100.0*m_blockSize/nrOfRow() << "%)";
          throw(s.str());
        }
        else if(m_filename.size())
          returnValue=writeNewBlock(irow,band);
      }
      int index=(irow-m_begin[band])*nrOfCol();
      int minindex=index+minCol;
      int maxindex=index+maxCol;
      typename std::vector<T>::iterator bufit=buffer2d[irow-minRow].begin();
      for(index=minindex;index<=maxindex;++index,++bufit){
        double dvalue=theScale*(*(bufit))+theOffset;
        switch(getDataType()){
        case(GDT_Byte):
          static_cast<unsigned char*>(m_data[band])[index]=static_cast<unsigned char>(dvalue);
          break;
        case(GDT_Int16):
          static_cast<short*>(m_data[band])[index]=static_cast<short>(dvalue);
          break;
        case(GDT_UInt16):
          static_cast<unsigned short*>(m_data[band])[index]=static_cast<unsigned short>(dvalue);
          break;
        case(GDT_Int32):
          static_cast<int*>(m_data[band])[index]=static_cast<int>(dvalue);
          break;
        case(GDT_UInt32):
          static_cast<unsigned int*>(m_data[band])[index]=static_cast<unsigned int>(dvalue);
          break;
        case(GDT_Float32):
          static_cast<float*>(m_data[band])[index]=static_cast<float>(dvalue);
          break;
        case(GDT_Float64):
          static_cast<double*>(m_data[band])[index]=static_cast<double>(dvalue);
          break;
        default:
          std::string errorString="Error: data type not supported";
          throw(errorString);
          break;
        }
      }
    }
  }
  else{
    //todo: apply scaling and offset!
    typename std::vector<T> buffer((maxRow-minRow+1)*(maxCol-minCol+1));
    //fetch raster band
    GDALRasterBand  *poBand;
    // typename std::vector<T>::iterator startit=buffer.begin();
    for(int irow=minRow;irow<=maxRow;++irow){
      buffer.insert(buffer.begin()+(maxCol-minCol+1)*(irow-minRow),buffer2d[irow-minRow].begin(),buffer2d[irow-minRow].end());
    }
    poBand = m_gds->GetRasterBand(band+1);//GDAL uses 1 based index
    returnValue=poBand->RasterIO(GF_Write,minCol,minRow,maxCol-minCol+1,maxRow-minRow+1,&(buffer[0]),(maxCol-minCol+1),(maxRow-minRow+1),type2GDAL<T>(),0,0);
  }
  return(returnValue);
}

template<typename T> CPLErr Jim::writeDataBlock(T value, int minCol, int maxCol, int minRow, int maxRow, int band){
  CPLErr returnValue=CE_None;
  double theScale=1;
  double theOffset=0;
  if(m_scale.size()>band)
    theScale=m_scale[band];
  if(m_offset.size()>band)
    theOffset=m_offset[band];
  if(band>=nrOfBand()+1){
    std::ostringstream s;
    s << "band (" << band << ") exceeds nrOfBand (" << nrOfBand() << ")";
    throw(s.str());
  }
  if(minCol>=nrOfCol()){
    std::ostringstream s;
    s << "minCol (" << minCol << ") exceeds nrOfCol (" << nrOfCol() << ")";
    throw(s.str());
  }
  if(minCol<0){
    std::ostringstream s;
    s << "mincol (" << minCol << ") is negative";
    throw(s.str());
  }
  if(minCol>0){
    std::ostringstream s;
    s << "Error: increase memory to support random access writing (now at " << 100.0*m_blockSize/nrOfRow() << "%)";
    throw(s.str());
  }
  if(maxCol>=nrOfCol()){
    std::ostringstream s;
    s << "maxCol (" << maxCol << ") exceeds nrOfCol (" << nrOfCol() << ")";
    throw(s.str());
  }
  if(maxCol<minCol){
    std::ostringstream s;
    s << "maxCol (" << maxCol << ") is less than minCol (" << minCol << ")";
    throw(s.str());
  }
  if(m_data.size()){
    for(int irow=minRow;irow<=maxRow;++irow){
      if(irow>=nrOfRow()){
        std::ostringstream s;
        s << "row (" << irow << ") exceeds nrOfRow (" << nrOfRow() << ")";
        throw(s.str());
      }
      if(irow<0){
        std::ostringstream s;
        s << "row (" << irow << ") is negative";
        throw(s.str());
      }
      if(irow<m_begin[band]){
        std::ostringstream s;
        s << "Error: increase memory to support random access writing (now at " << 100.0*m_blockSize/nrOfRow() << "%)";
        throw(s.str());
      }
      if(irow>=m_end[band]){
        if(irow>=m_end[band]+m_blockSize){
          std::ostringstream s;
          s << "Error: increase memory to support random access writing (now at " << 100.0*m_blockSize/nrOfRow() << "%)";
          throw(s.str());
        }
        else if(m_filename.size())
          returnValue=writeNewBlock(irow,band);
      }
      int index=(irow-m_begin[band])*nrOfCol();
      int minindex=index+minCol;
      int maxindex=index+maxCol;
      for(index=minindex;index<=maxindex;++index){
        double dvalue=value;
        switch(getDataType()){
        case(GDT_Byte):
          static_cast<unsigned char*>(m_data[band])[index]=static_cast<unsigned char>(dvalue);
          break;
        case(GDT_Int16):
          static_cast<short*>(m_data[band])[index]=static_cast<short>(dvalue);
          break;
        case(GDT_UInt16):
          static_cast<unsigned short*>(m_data[band])[index]=static_cast<unsigned short>(dvalue);
          break;
        case(GDT_Int32):
          static_cast<int*>(m_data[band])[index]=static_cast<int>(dvalue);
          break;
        case(GDT_UInt32):
          static_cast<unsigned int*>(m_data[band])[index]=static_cast<unsigned int>(dvalue);
          break;
        case(GDT_Float32):
          static_cast<float*>(m_data[band])[index]=static_cast<float>(dvalue);
          break;
        case(GDT_Float64):
          static_cast<double*>(m_data[band])[index]=static_cast<double>(dvalue);
          break;
        default:
          std::string errorString="Error: data type not supported";
          throw(errorString);
          break;
        }
      }
    }
  }
  else{
    //todo: apply scaling and offset!
    typename std::vector<T> buffer((maxRow-minRow+1)*(maxCol-minCol+1),value);
    //fetch raster band
    GDALRasterBand  *poBand;
    poBand = m_gds->GetRasterBand(band+1);//GDAL uses 1 based index
    returnValue=poBand->RasterIO(GF_Write,minCol,minRow,maxCol-minCol+1,maxRow-minRow+1,&(buffer[0]),(maxCol-minCol+1),(maxRow-minRow+1),type2GDAL<T>(),0,0);
  }
  return(returnValue);
}

#endif // _JIM_H_
