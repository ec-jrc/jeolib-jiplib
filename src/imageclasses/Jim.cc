/**********************************************************************
Jim.cc: class to read raster files using GDAL API library
Copyright (C) 2008-2018 Pieter Kempeneers

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
/*
  Changes copied by Pieter from version 2.6.8:
  2017-05-03  Kris Vanhoof (vhoofk): Fix rounding issues in image size calculation
*/
#include <iostream>
#include "ogr_spatialref.h"
extern "C" {
#include "gdal_alg.h"
#include "gdalwarper.h"
}
#include <config_jiplib.h>
#include "Jim.h"
#include "VectorOgr.h"
#include "base/Optionjl.h"
#include "algorithms/StatFactory.h"

using namespace std;

Jim::Jim(){
  reset();
}

size_t Jim::getDataTypeSizeBytes(int band) const {
#if MIALIB == 1
  switch (getDataType()){
  case JDT_UInt64:
  case JDT_Int64:
    return(static_cast<size_t>(8));
  default:{
    GDALDataType theType=getGDALDataType(band);
    if(theType==GDT_Unknown){
      std::string errorString="Error: data type not supported";
      throw(errorString);
    }
    return(static_cast<size_t>(GDALGetDataTypeSize(getGDALDataType(band))>>3));
  }
  }
#else
    GDALDataType theType=getGDALDataType(band);
    if(theType==GDT_Unknown){
      std::string errorString="Error: data type not supported";
      throw(errorString);
    }
    return(static_cast<size_t>(GDALGetDataTypeSize(getGDALDataType(band))>>3));
#endif
}

/// convert single plane multiband image to single band image with multiple planes
CPLErr Jim::band2plane(){
  //temporary buffer
  m_data.resize(nrOfBand()+1);
  m_data[nrOfBand()]=(void *) calloc(static_cast<size_t>(nrOfCol()*m_blockSize),getDataTypeSizeBytes());
  //copy first band
  memcpy(m_data[nrOfBand()],m_data[0],getDataTypeSizeBytes()*nrOfCol()*m_blockSize);
  //delete temporary buffer
  free(m_data[nrOfBand()]);
  //erase m_data buffer
  m_data.erase(m_data.begin()+nrOfBand());
  //allocate memory
  m_data[0]=(void *) calloc(static_cast<size_t>(nrOfBand()*nrOfCol()*m_blockSize),getDataTypeSizeBytes());
  //copy rest of the bands
  for(size_t iband=1;iband<nrOfBand();++iband){
    //memcp
    memcpy(static_cast<char*>(m_data[0])+iband*nrOfCol()*nrOfRow(),static_cast<char*>(m_data[iband]),getDataTypeSizeBytes()*nrOfCol()*m_blockSize);
    // memcpy(m_data[0]+iband*nrOfCol()*nrOfRow(),m_data[iband],getDataTypeSizeBytes()*nrOfCol()*m_blockSize);
    free(m_data[iband]);
    m_data.erase(m_data.begin()+iband);
  }
  m_nplane=nrOfBand();
  m_nband=1;
}

#if MIALIB == 1
IMAGE* Jim::getMIA(int band){
  if(getBlockSize()!=nrOfRow()){
    std::ostringstream s;
    s << "Error: increase memory to support MIA library functions (now at " << 100.0*getBlockSize()/nrOfRow() << "%)";
    throw(s.str());
  }
  if(m_mia.size()<band+1)
    m_mia.resize(band+1);
  if(m_mia[band])
    delete(m_mia[band]);
  m_mia[band]=new(IMAGE);
  m_mia[band]->p_im=m_data[band];/* Pointer to image data */
  m_mia[band]->DataType=getMIADataType();
  m_mia[band]->nx=nrOfCol();
  m_mia[band]->ny=nrOfRow();
  m_mia[band]->nz=nrOfPlane();
  m_mia[band]->NByte=m_mia[band]->nx * m_mia[band]->ny * m_mia[band]->nz * getDataTypeSizeBytes();//assumes image data type is not of bit type!!!
  //todo: remove m_mia[band]->vol and only rely on the getVolume function
  m_mia[band]->vol=0;//use getVolume() function
  m_mia[band]->lut=0;
  //USHORT *lut;   /* Pointer to colour map */
  //mia->g=getgetDataType();//not used
  return m_mia[band];
}

/**
 * set jim data pointer to the m_mia data pointer
 *
 * @param band the band for which the MIA image pointer needs to be set
 *
 * @return CE_None if successful
 */
CPLErr Jim::setMIA(int band){
  try{
    // if(m_mia->nz>1){
    //   std::string errorString="Error: MIA image with nz>1 not supported";
    //   throw(errorString);
    // }
    if(m_mia.size()<band+1){
      std::ostringstream s;
      s << "Error: illegal band number when setting MIA in Jim";
      throw(s.str());
    }
    if(m_ncol!=m_mia[band]->nx){
      std::ostringstream s;
      s << "Error: x dimension of image (" << m_ncol << ") does not match MIA (" << m_mia[band]->nx << ")";
      throw(s.str());
    }
    if(m_nrow!=m_mia[band]->ny){
      std::ostringstream s;
      s << "Error: y dimension of image (" << m_nrow << ") does not match MIA (" << m_mia[band]->ny << ")";
      throw(s.str());
    }
    if(m_nband<=band){
      std::ostringstream s;
      std::string errorString="Error: band exceeds number of bands in target image";
      throw(errorString);
    }
    // if(m_nband>1&&m_dataType!=MIA2GDALDataType(m_mia[band]->DataType)){
    if( (m_dataType!=MIA2JIPLIBDataType(m_mia[band]->DataType)) && nrOfBand() > 1){
      std::cout << "Warning: changing data type of multiband image, make sure to set all bands" << std::endl;
    }
    m_dataType=MIA2JIPLIBDataType(m_mia[band]->DataType);
    m_data[band]=(void *)m_mia[band]->p_im;
    // m_data[band]=(unsigned char *)m_mia[band]->p_im + band * nrOfRow() * nrOfCol() * (GDALGetDataTypeSize(getDataType())>>3);
    m_begin[band]=0;
    m_end[band]=m_begin[band]+getBlockSize();
  }
  catch(std::string errorString){
    std::cerr << errorString << std::endl;
    throw;
  }
  catch(...){
    throw;
  }
  return(CE_None);
}

/**
 * set Jim attributes from external MIA image
 *
 * @param mia the MIA image pointer to be set
 * @param band the band for which the MIA image pointer needs to be set
 *
 * @return C_None if successful
 */
CPLErr Jim::setMIA(IMAGE* mia, int band){
  try{
    if(nrOfBand()){
      if(m_ncol!=mia->nx){
        std::ostringstream s;
        s << "Error: x dimension of image (" << m_ncol << ") does not match MIA (" << mia->nx << ")";
        throw(s.str());
      }
      if(m_nrow!=mia->ny){
        std::ostringstream s;
        s << "Error: y dimension of image (" << m_nrow << ") does not match MIA (" << mia->ny << ")";
        throw(s.str());
      }
      if(m_nplane!=mia->nz){
        std::string errorString="Error: number of planes of images do not match";
        throw(errorString);
      }
      if(m_dataType!=MIA2JIPLIBDataType(m_mia[band]->DataType)){
        std::string errorString="Error: inconsistent data types for multiband image";
        throw(errorString);
      }
    }
    if(m_mia.size()<band+1){
      m_mia.resize(band+1);
      m_nband=band+1;
    }
    if(m_data.size()<band+1){
      m_data.resize(band+1);
      m_begin.resize(band+1);
      m_end.resize(band+1);
    }
    m_nplane=mia->nz;
    m_ncol=mia->nx;
    m_nrow=mia->ny;
    m_blockSize=m_nrow;
    m_mia[band]=mia;
    // setExternalData(true);//todo: need to fix memory leak when setMIA used for single band only! (either create vector<bool> m_externalData or only allow for setMIA all bands)
    this->setMIA(band);
  }
  catch(std::string errorString){
    std::cerr << errorString << std::endl;
    throw;
  }
  catch(...){
    throw;
  }
  return(CE_None);
}

 int Jim::getMIADataType(){
   switch (getDataType()){
   case GDT_Byte:
     return t_UCHAR;
   case GDT_UInt16:
     return t_USHORT;
   case GDT_Int16:
     return t_SHORT;
   case GDT_UInt32:
     return t_UINT32;
   case GDT_Int32:
     return t_INT32;
   case GDT_Float32:
     return t_FLOAT;
   case GDT_Float64:
     return t_DOUBLE;
   case JDT_UInt64:
     return t_UINT64;
   case JDT_Int64:
     return t_INT64;
   case t_UNSUPPORTED:
     return GDT_Unknown;
   default:
     return GDT_Unknown;
   }
 }

 int Jim::JIPLIB2MIADataType(int aJIPLIBDataType){
   //function exists, but introduced for naming consistency
   if(aJIPLIBDataType==JDT_UInt64)
     return(t_UINT64);
   else if(aJIPLIBDataType==JDT_Int64)
     return(t_INT64);
   else
     return(GDAL2MIALDataType(aJIPLIBDataType));
 }

 ///convert a GDAL to MIA data type
 int Jim::GDAL2MIADataType(GDALDataType aGDALDataType){
   //function exists, but introduced for naming consistency
   return(GDAL2MIALDataType(aGDALDataType));
 }

 ///convert a MIA data type to GDAL data type
 int Jim::MIA2JIPLIBDataType(int aMIADataType){
   switch (aMIADataType){
   case t_UCHAR:
     return GDT_Byte;
   case t_USHORT:
     return GDT_UInt16;
   case t_SHORT:
     return GDT_Int16;
   case t_UINT32:
     return GDT_UInt32;
   case t_INT32:
     return GDT_Int32;
   case t_FLOAT:
     return GDT_Float32;
   case t_DOUBLE:
     return GDT_Float64;
   case t_UINT64:
     return JDT_UInt64;
   case t_INT64:
     return JDT_Int64;
   case t_UNSUPPORTED:
     return GDT_Unknown;
   default:
     return GDT_Unknown;
   }
 }
#endif

void Jim::reset()
{
  m_gds=0;
  m_ncol=0;
  m_nrow=0;
  m_nband=0;
  m_nplane=1;//todo: check if better set to 0 or 1?
  m_dataType=GDT_Unknown;
  m_gt.resize(6);
  for(int index=0;index<6;++index){m_gt[index]=0;};
  m_noDataValues.clear();
  m_projection=std::string();
  m_scale.clear();
  m_offset.clear();
  m_blockSize=0;
  m_begin.clear();
  m_end.clear();
  m_options.clear();
  // m_writeMode=false;
  m_access=READ_ONLY;
  m_resample=GRIORA_NearestNeighbour;
  m_filename.clear();
  freeMem();
  // m_data.clear();
  //is there any reason why we cannot set external data to false?
  m_externalData=false;
#if MIALIB == 1
  for(int iband=0;iband<m_mia.size();++iband)
    delete(m_mia[iband]);
  m_mia.clear();
#endif
}

/**
 * @param dataPointers External pointers to which the image data should be written in memory
 * @param ncol The number of columns in the image
 * @param nrow The number of rows in the image
 * @param dataType The data type of the image (one of the GDAL supported datatypes: GDT_Byte, GDT_[U]Int[16|32], GDT_Float[32|64])
 **/
Jim::Jim(void* dataPointer, int ncol, int nrow, const GDALDataType& dataType){
  reset();
  open(dataPointer,ncol,nrow,dataType);
}

/**
 * @param dataPointers External pointers to which the image data should be written in memory
 * @param ncol The number of columns in the image
 * @param nrow The number of rows in the image
 * @param dataType The data type of the image (one of the GDAL supported datatypes: GDT_Byte, GDT_[U]Int[16|32], GDT_Float[32|64])
 **/
Jim::Jim(std::vector<void*> dataPointers, int ncol, int nrow, const GDALDataType& dataType){
  reset();
  open(dataPointers,ncol,nrow,dataType);
}

///constructor opening an image in memory using an external data pointer
Jim::Jim(void* dataPointer, int ncol, int nrow, int nplane, const GDALDataType& dataType){
  reset();
  open(dataPointer,ncol,nrow,nplane,dataType);
}
///constructor opening a multiband image in memory using an external data pointer
Jim::Jim(std::vector<void*> dataPointers, int ncol, int nrow, int nplane, const GDALDataType& dataType){
  reset();
  open(dataPointers,ncol,nrow,nplane,dataType);
}

Jim::Jim(const std::string& filename, const Jim& imgSrc, unsigned int memory, const std::vector<std::string>& options){
  reset();
  open(filename,imgSrc,memory,options);
}

#if MIALIB == 1
///constructor input image
Jim::Jim(IMAGE *mia) : m_nplane(1){
  reset();
  setMIA(mia,0);
}
#endif

///constructor output image
Jim::Jim(const std::string& filename, bool readData, unsigned int memory){
  reset();
  open(filename,readData,memory);
}

///constructor output image
Jim::Jim(Jim& imgSrc, bool copyData){
  reset();
  open(imgSrc,copyData);
}
///constructor output image
Jim::Jim(const std::string& filename, int ncol, int nrow, int nband, const GDALDataType& dataType, const std::string& imageType, unsigned int memory, const std::vector<std::string>& options){
  reset();
  open(filename, ncol, nrow, nband, dataType, imageType, memory, options);
}
///constructor output image with nplane
Jim::Jim(int ncol, int nrow, int nband, int nplane, const GDALDataType& dataType){
  reset();
  open(ncol,nrow,nband,nplane,dataType);
}
///constructor output image
Jim::Jim(int ncol, int nrow, int nband, const GDALDataType& dataType){
  reset();
  open(ncol, nrow, nband, dataType);
}
//test
// Jim::Jim(app::AppFactory &theApp): m_nplane(1), Jim(theApp){};
Jim::Jim(app::AppFactory &theApp){
  reset();
  open(theApp);
}


///destructor
Jim::~Jim(void){
#if MIALIB == 1
  if(m_mia.size()){
    for(int iband=0;iband<m_mia.size();++iband)
      if(m_mia[iband])
        delete(m_mia[iband]);
    m_mia.clear();
  }
#endif
  // Jim::reset();
  close();
}
/**
 * @param memory Available memory to cache image raster data (in MB)
 **/
CPLErr Jim::initMem(unsigned int memory)
{
  if(memory<=0)
    m_blockSize=nrOfRow();
  else{
    m_blockSize=static_cast<unsigned int>(memory*1000000/nrOfBand()/nrOfCol()/getDataTypeSizeBytes());
    if(getBlockSizeY(0))
      m_blockSize-=m_blockSize%getBlockSizeY(0);
  }
  if(m_blockSize<1)
    m_blockSize=1;
  if(m_blockSize>nrOfRow())
    m_blockSize=nrOfRow();
  m_begin.resize(nrOfBand());
  m_end.resize(nrOfBand());
  freeMem();
  m_data.resize(nrOfBand());
  for(int iband=0;iband<nrOfBand();++iband){
    m_data[iband]=(void *) calloc(static_cast<size_t>(nrOfPlane()*nrOfCol()*m_blockSize),getDataTypeSizeBytes());
    if(!(m_data[iband])){
      std::string errorString="Error: could not allocate memory in initMem";
      throw(errorString);
    }
  }
  return(CE_None);
}

/**
 * @param memory Available memory to cache image raster data (in MB)
 **/
void Jim::freeMem()
{
//  if(m_data.size()&&m_filename.size()){
  for(int iband=0;iband<m_data.size();++iband){
    if(m_data.size()>iband){
      if(m_externalData)
        m_data[iband]=0;
      else
        free(m_data[iband]);
        // delete(m_data[iband]);
      // CPLFree(m_data[iband]);
    }
  }
  m_data.clear();
}

// Jim& Jim::operator=(Jim& imgSrc)
// {
//   bool copyData=true;
//   //check for assignment to self (of the form v=v)
//   if(this==&imgSrc)
//     return *this;
//   else{
//     Jim::open(imgSrc,copyData);
//     return *this;
//   }
// }

///relational == operator
/**
 * @param refImg Use this as the reference image
 * @return true if image is equal to reference image
 **/
#if MIALIB == 1
bool Jim::isEqual(std::shared_ptr<Jim> refImg){
  bool isEqual=true;
  if(nrOfBand()!=refImg->nrOfBand())
    return(false);
  if(nrOfRow()!=refImg->nrOfRow())
    return(false);
  if(nrOfCol()!=refImg->nrOfCol())
    return(false);

  for(int iband=0;iband<nrOfBand();++iband){
    if(getDataType(iband)!=refImg->getDataType(iband)){
      isEqual=false;
      break;
    }
    IMAGE* refMIA=refImg->getMIA(iband);
    IMAGE* thisMIA=this->getMIA(iband);
    if(::imequalp(thisMIA,refMIA)){
      isEqual=false;
      break;
    }
  }
  return(isEqual);
}
#endif
/**
 * @param imgSrc Use this source image as a template to copy image attributes
 **/
// Jim& Jim::operator=(Jim& imgSrc)
// {
//   bool copyData=true;
//   //check for assignment to self (of the form v=v)
//   if(this==&imgSrc)
//      return *this;
//   else{
//     open(imgSrc,copyData);
//     return *this;
//   }
// }

// ///assignment operator for in memory image
// std::shared_ptr<Jim> Jim::operator=(std::shared_ptr<Jim> imgSrc){
//   std::shared_ptr<Jim> imgWriter=std::make_shared<Jim>();
//   *imgWriter=*imgSrc;
//   return(imgWriter);
// }

/**
 * @param dataPointer External pointer to which the image data should be written in memory
 * @param ncol The number of columns in the image
 * @param nrow The number of rows in the image
 * @param dataType The data type of the image (one of the GDAL supported datatypes: GDT_Byte, GDT_[U]Int[16|32], GDT_Float[32|64])
 **/
CPLErr Jim::open(void* dataPointer, int ncol, int nrow, const GDALDataType& dataType){
  m_ncol=ncol;
  m_nrow=nrow;
  m_nband=1;
  m_dataType=dataType;
  m_data.resize(m_nband);
  m_begin.resize(m_nband);
  m_end.resize(m_nband);
  m_blockSize=nrow;//memory contains entire image and has been read already
  if(dataPointer){
    for(int iband=0;iband<m_nband;++iband){
      // m_data[iband]=dataPointer+iband*ncol*nrow*getDataTypeSizeBytes();
      m_data[iband]=(uint8_t*)dataPointer;
      m_begin[iband]=0;
      m_end[iband]=m_begin[iband]+m_blockSize;
    }
    // m_externalData=true;
    return(CE_None);
  }
  else
    return(CE_Failure);
}

/**
 * @param dataPointers External pointers to which the image data should be written in memory
 * @param ncol The number of columns in the image
 * @param nrow The number of rows in the image
 * @param dataType The data type of the image (one of the GDAL supported datatypes: GDT_Byte, GDT_[U]Int[16|32], GDT_Float[32|64])
 **/
CPLErr Jim::open(std::vector<void*> dataPointers, int ncol, int nrow, const GDALDataType& dataType){
  m_ncol=ncol;
  m_nrow=nrow;
  m_nband=dataPointers.size();
  m_dataType=dataType;
  m_data.resize(m_nband);
  m_begin.resize(m_nband);
  m_end.resize(m_nband);
  m_blockSize=nrow;//memory contains entire image and has been read already
  if(dataPointers.size()){
    for(int iband=0;iband<m_nband;++iband){
      if(dataPointers[iband]){
        // m_data[iband]=dataPointer+iband*ncol*nrow*getDataTypeSizeBytes();
        m_data[iband]=(uint8_t*)dataPointers[iband];
        // m_data[iband]=(uint8_t*)dataPointers[iband]+iband*ncol*nrow*getDataTypeSizeBytes();
        m_begin[iband]=0;
        m_end[iband]=m_begin[iband]+m_blockSize;
      }
    }
    // m_externalData=true;
    return(CE_None);
  }
  else
    return(CE_Failure);
}

///Open an image for writing, copying data from dataPointer
CPLErr Jim::open(void* dataPointer, int ncol, int nrow, int nplane, const GDALDataType& dataType, bool copyData){
  m_ncol=ncol;
  m_nrow=nrow;
  m_nplane=nplane;
  m_nband=1;
  m_dataType=dataType;
  m_data.resize(m_nband);
  m_begin.resize(m_nband);
  m_end.resize(m_nband);
  m_blockSize=nrow;//memory contains entire image and has been read already
  if(dataPointer){
    if(copyData){
      initMem(0);
      memcpy(m_data[0],dataPointer,getDataTypeSizeBytes()*nrOfCol()*m_blockSize*nrOfPlane());
    }
    else{
      m_data[0]=(uint8_t*)dataPointer;
      m_begin[0]=0;
      m_end[0]=m_begin[0]+m_blockSize;
    }
    /*
    for(int iband=0;iband<m_nband;++iband){
      m_data[iband]=(char *)dataPointer+iband*ncol*nrow*nplane*getDataTypeSizeBytes();
      m_begin[iband]=0;
      m_end[iband]=m_begin[iband]+m_blockSize;
    }
*/
    // m_externalData=true;
    return(CE_None);
  }
  else
    return(CE_Failure);
}

///Open a multiband image for writing, based on an existing image object
CPLErr Jim::open(std::vector<void*> dataPointers, int ncol, int nrow, int nplane, const GDALDataType& dataType, bool copyData){
  m_ncol=ncol;
  m_nrow=nrow;
  m_nplane=nplane;
  m_nband=dataPointers.size();
  m_dataType=dataType;
  m_data.resize(m_nband);
  m_begin.resize(m_nband);
  m_end.resize(m_nband);
  m_blockSize=nrow;//memory contains entire image and has been read already
  if(dataPointers.size()){
    if(copyData){
      initMem(0);
      for(int iband=0;iband<m_nband;++iband){
        if(dataPointers[iband]){
          memcpy(m_data[iband],dataPointers[iband],getDataTypeSizeBytes()*nrOfCol()*m_blockSize*nrOfPlane());
          m_begin[iband]=0;
          m_end[iband]=m_begin[iband]+m_blockSize;
        }
      }
    }
    else{
      for(int iband=0;iband<m_nband;++iband){
        if(dataPointers[iband]){
          m_data[iband]=(uint8_t*)dataPointers[iband];
          m_begin[iband]=0;
          m_end[iband]=m_begin[iband]+m_blockSize;
        }
      }
    }
    // m_externalData=true;
    return(CE_None);
  }
  else
    return(CE_Failure);
}
CPLErr Jim::close()
{
  if(m_gds){
    GDALClose(m_gds);
  }
  reset();
}

/**
 * @return the projection of this data set in string format
 **/
std::string Jim::getProjection() const
{
  // if(m_gds)
  //   return(m_gds->GetProjectionRef());
  // else
  return(m_projection);
}

/**
 * @return the projection of this data set in string format
 **/
std::string Jim::getProjectionRef() const
{
  // if(m_gds)
  //   return(m_gds->GetProjectionRef());
  // else
  return(m_projection);
}

/**
 * @param projection projection string to be used for this dataset
 * @return the projection of this data set in string format
 **/
CPLErr Jim::setProjectionProj4(const std::string& projection)
{
  if(projection.size()){
    OGRSpatialReference theRef;
    theRef.SetFromUserInput(projection.c_str());
    char *wktString;
    theRef.exportToWkt(&wktString);
    m_projection=wktString;
    //todo: must not be limited to access==WRITE?
    if(m_gds&&m_access==WRITE)
      return(m_gds->SetProjection(wktString));
    else
      return(CE_Warning);
  }
  else
    return(CE_Failure);
}

/**
 * @param projection projection string to be used for this dataset
 **/
//deprecated?
// CPLErr Jim::setProjection(const std::string& projection)
// {
//   m_projection=projection;
//   if(m_gds){
//     return(m_gds->SetProjection(projection.c_str()));
//   }
//   else{
//     return(CE_Warning);
//   }
// }

/**
 * @param band get data type for this band (start counting from 0)
 * @return the GDAL data type of this data set for the selected band
 **/
GDALDataType Jim::getDataTypeDS(int band) const
{
  if(nrOfBand()<=band){
    std::string errorString="Error: band number exceeds available bands in getDataType";
    throw(errorString);
  }
  if(getRasterBand(band))
    return((getRasterBand(band)->GetRasterDataType()));
  else
    return(GDT_Unknown);
}

/**
 * @param band get data type for this band (start counting from 0)
 * @return the data type of this data set for the selected band
 **/
int Jim::getDataType(int band) const
//GDALDataType Jim::getDataType(int band) const
{
  if(nrOfBand()<=band){
    std::string errorString="Error: band number exceeds available bands in getDataType";
    throw(errorString);
  }
  // if(getRasterBand(band))
  //   return((getRasterBand(band)->GetRasterDataType()));
  // else
  return(m_dataType);
}

/**
 * @param band get data type for this band (start counting from 0)
 * @return the GDAL data type of this data set for the selected band
 **/
GDALDataType Jim::getGDALDataType(int band) const
{
  if(nrOfBand()<=band){
    std::string errorString="Error: band number exceeds available bands in getDataType";
    throw(errorString);
  }
  // if(getRasterBand(band))
  //   return((getRasterBand(band)->GetRasterDataType()));
  // else
  switch(getDataType()){
  case(GDT_Byte):
    return(GDT_Byte);
  case(GDT_Int16):
    return(GDT_Int16);
  case(GDT_UInt16):
    return(GDT_UInt16);
  case(GDT_Int32):
    return(GDT_Int32);
  case(GDT_UInt32):
    return(GDT_UInt32);
  case(GDT_Float32):
    return(GDT_Float32);
  case(GDT_Float64):
    return(GDT_Float64);
  default:
    return(GDT_Unknown);
  }
}

/**
 * @param band get GDAL raster band for this band (start counting from 0)
 * @return the GDAL raster band of this data set for the selected band
 **/
GDALRasterBand* Jim::getRasterBand(int band) const
{
  if(nrOfBand()<=band){
    std::string errorString="Error: band number exceeds available bands in getRasterBand";
    throw(errorString);
  }
  if(m_gds)
    return((m_gds->GetRasterBand(band+1)));
  else
    return(0);
}

/**
 * @param band get GDAL color table for this band (start counting from 0)
 * @return the GDAL color table of this data set for the selected band
 **/
GDALColorTable* Jim::getColorTable(int band) const
{
  if(nrOfBand()<=band){
    std::string errorString="Error: band number exceeds available bands in getColorTable";
    throw(errorString);
  }
  GDALRasterBand* theRasterBand=getRasterBand(band);
  if(theRasterBand)
    return(theRasterBand->GetColorTable());
  else
    return(0);
}

/**
 * @return the driver description of this data set in string format
 **/
std::string Jim::getDriverDescription() const
{
  std::string driverDescription;
  if(m_gds)
    driverDescription=m_gds->GetDriver()->GetDescription();
  return(driverDescription);
}

/**
 * @param gt pointer to the six geotransform parameters:
 * @param adfGeoTransform[0] top left x
 * @param GeoTransform[1] w-e pixel resolution
 * @param GeoTransform[2] rotation, 0 if image is "north up"
 * @param GeoTransform[3] top left y
 * @param GeoTransform[4] rotation, 0 if image is "north up"
 * @param GeoTransform[5] n-s pixel resolution
 * return CE_None if geotransform could be set for GDAL dataset
 **/
CPLErr Jim::setGeoTransform(double* gt){
  m_gt.resize(6);
  // m_isGeoRef=true;
  m_gt[0]=gt[0];
  m_gt[1]=gt[1];
  m_gt[2]=gt[2];
  m_gt[3]=gt[3];
  m_gt[4]=gt[4];
  m_gt[5]=gt[5];
  //todo: must not be limited to access==WRITE?
  if(m_gds&&m_access==WRITE)
    return(m_gds->SetGeoTransform(&m_gt[0]));
  else
    return(CE_Warning);
}

CPLErr Jim::setGeoTransform(const std::vector<double>& gt){
  if(gt.size()==6){
    m_gt.resize(6);
    // m_isGeoRef=true;
    m_gt[0]=gt[0];
    m_gt[1]=gt[1];
    m_gt[2]=gt[2];
    m_gt[3]=gt[3];
    m_gt[4]=gt[4];
    m_gt[5]=gt[5];
    //todo: must not be limited to access==WRITE?
    if(m_gds&&m_access==WRITE)
      return(m_gds->SetGeoTransform(&m_gt[0]));
  }
  else
    return(CE_Warning);
}

/**
 * @param imgSrc Use this source image as a template to copy geotranform information
 * return CE_None if geotransform could be copied for GDAL dataset
 **/
CPLErr Jim::copyGeoTransform(const Jim& imgSrc)
{
  vector<double> gt;
  imgSrc.getGeoTransform(gt);
  return(setGeoTransform(gt));
}

/**
 * @param imgSrc Use this pointer to source image as a template to copy geotranform information
 **/
// void Jim::copyGeoTransform(const std::shared_ptr<Jim>& imgSrc)
// {
//   double gt[6];
//   imgSrc->getGeoTransform(gt);
//   setGeoTransform(gt);
// }

/**
 * @param gt pointer to the six geotransform parameters:
 * @param adfGeoTransform[0] top left x
 * @param GeoTransform[1] w-e pixel resolution
 * @param GeoTransform[2] rotation, 0 if image is "north up"
 * @param GeoTransform[3] top left y
 * @param GeoTransform[4] rotation, 0 if image is "north up"
 * @param GeoTransform[5] n-s pixel resolution
 **/
void Jim::getGeoTransform(double* gt) const{
  if(m_gt.size()==6){
    gt[0]=m_gt[0];
    gt[1]=m_gt[1];
    gt[2]=m_gt[2];
    gt[3]=m_gt[3];
    gt[4]=m_gt[4];
    gt[5]=m_gt[5];
  }
  else
    gt=0;
}

void Jim::getGeoTransform(vector<double>& gt) const{
  gt=m_gt;
}

/**
 * @return the geotransform of this data set in string format
 **/
// std::string Jim::getGeoTransform() const
// {
//   std::string gtString;
//   double gt[6];// { 444720, 30, 0, 3751320, 0, -30 };
//   getGeoTransform(gt);
//   std::ostringstream s;
//   s << "[" << gt[0] << "," << gt[1] << "," << gt[2] << "," << gt[3] << "," << gt[4] << "," << gt[5] << "]";
//   gtString=s.str();
//   return(s.str());
// }

/**
 * @return the metadata of this data set in C style string format (const version)
 **/
char** Jim::getMetadata() const
{
  if(m_gds){
    if(m_gds->GetMetadata()!=NULL)
      return(m_gds->GetMetadata());
    else
      return(0);
  }
  else
    return(0);
    // return (char**)"";
}

/**
 * @return the metadata of this data set in standard template library (stl) string format
 **/
CPLErr Jim::getMetadata(std::list<std::string>& metadata) const
{
  if(m_gds){
    char** cmetadata=m_gds->GetMetadata();
    while(*cmetadata!=NULL){
      metadata.push_back(*(cmetadata));
      ++cmetadata;
    }
    return(CE_None);
  }
  else
    return(CE_Warning);
}

/**
 * @return the description of this data set in string format
 **/
std::string Jim::getDescription() const
{
  if(m_gds){
    if(m_gds->GetDriver()->GetDescription()!=NULL)
      return m_gds->GetDriver()->GetDescription();
    else
      return(std::string());
  }
  else
    return(std::string());
}

/**
 * @return the meta data item of this data set in string format
 **/
std::string Jim::getMetadataItem() const
{
  if(m_gds){
    if(m_gds->GetDriver()->GetMetadataItem( GDAL_DMD_LONGNAME )!=NULL)
      return m_gds->GetDriver()->GetMetadataItem( GDAL_DMD_LONGNAME );
    return(std::string());
  }
  else
    return(std::string());
}

/**
 * @return the image description (TIFFTAG) of this data set in string format
 **/
std::string Jim::getImageDescription() const
{
  if(m_gds){
    if(m_gds->GetDriver()->GetMetadataItem("TIFFTAG_IMAGEDESCRIPTION")!=NULL)
      return m_gds->GetDriver()->GetMetadataItem("TIFFTAG_IMAGEDESCRIPTION");
    return(std::string());
  }
  else
    return(std::string());
}

/**
 * @return the band coding interleave of this data set in string format
 **/
std::string Jim::getInterleave() const
{
  if(m_gds){
    if(m_gds->GetMetadataItem( "INTERLEAVE", "IMAGE_STRUCTURE"))
      return m_gds->GetMetadataItem( "INTERLEAVE", "IMAGE_STRUCTURE");
    else
      return("BAND");
  }
  else
    return(std::string());
}

/**
 * @return the compression meta data of this data set in string format
 **/
std::string Jim::getCompression() const
{
  if(m_gds){
    if(m_gds->GetMetadataItem( "COMPRESSION", "IMAGE_STRUCTURE"))
      return m_gds->GetMetadataItem( "COMPRESSION", "IMAGE_STRUCTURE");
    return("NONE");
  }
  else
    return("NONE");
}

/**
 * assuming
 * adfGeotransform[0]: ULX (upper left X coordinate)
 * adfGeotransform[1]: $cos(\alpha)\cdot\textrm{Xres}$
 * adfGeotransform[2]: $-sin(\alpha)\cdot\textrm{Xres}$
 * adfGeotransform[3]: ULY (upper left Y coordinate)
 * adfGeotransform[4]: $-sin(\alpha)\cdot\textrm{Yres}$
 * adfGeotransform[5]: $-cos(\alpha)\cdot\textrm{Yres}$

 * @param ulx upper left coordinate in x
 * @param uly upper left coordinate in y
 * @param lrx lower left coordinate in x
 * @param lry lower left coordinate in y
 * @return true if successful and if image is georeferenced
 **/
void Jim::getBoundingBox(double& ulx, double& uly, double& lrx, double& lry, OGRCoordinateTransformation *poCT) const
{
  try{
    vector<double> gt(6);
    getGeoTransform(gt);

    ulx=gt[0];
    uly=gt[3];
    lrx=gt[0]+nrOfCol()*gt[1]+nrOfRow()*gt[2];
    lry=gt[3]+nrOfCol()*gt[4]+nrOfRow()*gt[5];
    if(poCT){
      std::vector<double> xvector(4);//ulx,urx,llx,lrx
      std::vector<double> yvector(4);//uly,ury,lly,lry
      xvector[0]=ulx;
      xvector[1]=lrx;
      xvector[2]=ulx;
      xvector[3]=lrx;
      yvector[0]=uly;
      yvector[1]=uly;
      yvector[2]=lry;
      yvector[3]=lry;
      if(!poCT->Transform(xvector.size(),&xvector[0],&yvector[0])){
        std::ostringstream errorStream;
        errorStream << "Error: cannot apply OGRCoordinateTransformation in Jim::getBoundingBox" << std::endl;
        throw(errorStream.str());
      }
      ulx=std::min(xvector[0],xvector[2]);
      lrx=std::max(xvector[1],xvector[3]);
      uly=std::max(yvector[0],yvector[1]);
      lry=std::min(yvector[2],yvector[3]);
    }
  }
  catch(std::string errorString){
    std::cerr << errorString << std::endl;
    throw;
  }
}

/**
 * assuming
 * adfGeotransform[0]: ULX (upper left X coordinate)
 * adfGeotransform[1]: $cos(\alpha)\cdot\textrm{Xres}$
 * adfGeotransform[2]: $-sin(\alpha)\cdot\textrm{Xres}$
 * adfGeotransform[3]: ULY (upper left Y coordinate)
 * adfGeotransform[4]: $-sin(\alpha)\cdot\textrm{Yres}$
 * adfGeotransform[5]: $-cos(\alpha)\cdot\textrm{Yres}$

 * @param bbvector[0] upper left coordinate in x
 * @param bbvector[1] upper left coordinate in y
 * @param bbvector[2] lower left coordinate in x
 * @param bbvector[3] lower left coordinate in y
 * @return true if successful and if image is georeferenced
 **/
void Jim::getBoundingBox(std::vector<double> &bbvector, OGRCoordinateTransformation *poCT) const
{
  bbvector.resize(4);
  getBoundingBox(bbvector[0],bbvector[1],bbvector[2],bbvector[3],poCT);
}

void Jim::getBoundingBox(OGRPolygon *bbPolygon, OGRCoordinateTransformation *poCT) const{
  OGRLinearRing bbRing;
  std::vector<double> bbvector;
  getBoundingBox(bbvector,poCT);
  OGRPoint ul;
  OGRPoint ur;
  OGRPoint lr;
  OGRPoint ll;
  ul.setX(bbvector[0]);
  ul.setY(bbvector[1]);
  ur.setX(bbvector[2]);
  ur.setY(bbvector[1]);
  lr.setX(bbvector[2]);
  lr.setY(bbvector[3]);
  ll.setX(bbvector[0]);
  ll.setY(bbvector[3]);
  bbRing.addPoint(&ul);
  bbRing.addPoint(&ur);
  bbRing.addPoint(&lr);
  bbRing.addPoint(&ll);
  bbRing.addPoint(&ul);
  bbPolygon->addRing(&bbRing);
  bbPolygon->closeRings();
}

// ///get bounding box with coordinate transform based on EPSG code
// bool Jim::getBoundingBox(std::vector<double> &bbvector, int targetEPSG) const
// {
//   bbvector.resize(4);
//   OGRSpatialReference targetSRS;
//   if( targetSRS.importFromEPSG(targetEPSG) != OGRERR_NONE ){
//     std::ostringstream errorStream;
//     errorStream << "Error: cannot import SRS from EPSG code: " << targetEPSG << std::endl;
//     throw(errorStream.str());
//   }
//   return getBoundingBox(bbvector,&targetSRS);
// }

/**
 * assuming
 * adfGeotransform[0]: ULX (upper left X coordinate)
 * adfGeotransform[1]: $cos(\alpha)\cdot\textrm{Xres}$
 * adfGeotransform[2]: $-sin(\alpha)\cdot\textrm{Xres}$
 * adfGeotransform[3]: ULY (upper left Y coordinate)
 * adfGeotransform[4]: $-sin(\alpha)\cdot\textrm{Yres}$
 * adfGeotransform[5]: $-cos(\alpha)\cdot\textrm{Yres}$
 * @param x, y centre coordinates in x and y
 * @return true if image is georeferenced
 **/
void Jim::getCenterPos(double& centerX, double& centerY) const
{
  vector<double> gt(6);
  getGeoTransform(gt);

  centerX=gt[0]+(nrOfCol()/2.0)*gt[1]+(nrOfRow()/2.0)*gt[2];
  centerY=gt[3]+(nrOfCol()/2.0)*gt[4]+(nrOfRow()/2.0)*gt[5];
  // if(isGeoRef())
  //   return(CE_None);
  // else
  //   return(CE_Warning);
}

/**
 * assuming
 * adfGeotransform[0]: ULX (upper left X coordinate)
 * adfGeotransform[1]: $cos(\alpha)\cdot\textrm{Xres}$
 * adfGeotransform[2]: $-sin(\alpha)\cdot\textrm{Xres}$
 * adfGeotransform[3]: ULY (upper left Y coordinate)
 * adfGeotransform[4]: $-sin(\alpha)\cdot\textrm{Yres}$
 * adfGeotransform[5]: $-cos(\alpha)\cdot\textrm{Yres}$
 * @param x,y georeferenced coordinates in x and y
 * @param i,j image coordinates (can be fraction of pixels)
 * @return true if image is georeferenced
 **/
bool Jim::geo2image(double x, double y, double& i, double& j, OGRCoordinateTransformation *poCT) const
{
  vector<double> gt(6);
  getGeoTransform(gt);

  double thisX=x;
  double thisY=y;
  try{
    if(poCT){
      if(!poCT->Transform(1,&thisX,&thisY)){
        std::ostringstream errorStream;
        errorStream << "Error: cannot apply OGRCoordinateTransformation in Jim::geo2image" << std::endl;
        throw(errorStream.str());
      }
    }
    double denom=(gt[1]-gt[2]*gt[4]/gt[5]);
    double eps=0.00001;
    if(fabs(denom)>eps){
      i=(thisX-gt[0]-gt[2]/gt[5]*(thisY-gt[3]))/denom;
      j=(thisY-gt[3]-gt[4]*(thisX-gt[0]-gt[2]/gt[5]*(thisY-gt[3]))/denom)/gt[5];
    }
    if(isGeoRef())
      return true;
    else
      return false;
  }
  catch(std::string errorstring){
    std::cerr << errorstring << std::endl;
    throw;
  }
}

/**
 * assuming
 * adfGeotransform[0]: ULX (upper left X coordinate)
 * adfGeotransform[1]: $cos(\alpha)\cdot\textrm{Xres}$
 * adfGeotransform[2]: $-sin(\alpha)\cdot\textrm{Xres}$
 * adfGeotransform[3]: ULY (upper left Y coordinate)
 * adfGeotransform[4]: $-sin(\alpha)\cdot\textrm{Yres}$
 * adfGeotransform[5]: $-cos(\alpha)\cdot\textrm{Yres}$
 * @param i,j image coordinates (can be fraction of pixels)
 * @param x,y georeferenced coordinates in x and y (can be fraction of pixels)
 * @return true if image is georeferenced
 **/
bool Jim::image2geo(double i, double j, double& x, double& y, OGRCoordinateTransformation *poCT) const
{
  try{
    vector<double> gt(6);
    getGeoTransform(gt);

    x=gt[0]+(0.5+i)*gt[1]+(0.5+j)*gt[2];
    y=gt[3]+(0.5+i)*gt[4]+(0.5+j)*gt[5];
    if(poCT){
      if(!poCT->Transform(1,&x,&y)){
        std::ostringstream errorStream;
        errorStream << "Error: cannot apply OGRCoordinateTransformation in Jim::image2geo" << std::endl;
        throw(errorStream.str());
      }
    }
    if(isGeoRef())
      return true;
    else
      return false;
  }
  catch(std::string errorstring){
    std::cerr << errorstring << std::endl;
    throw;
  }

}

/**
 * assuming
 * adfGeotransform[0]: ULX (upper left X coordinate)
 * adfGeotransform[1]: $cos(\alpha)\cdot\textrm{Xres}$
 * adfGeotransform[2]: $-sin(\alpha)\cdot\textrm{Xres}$
 * adfGeotransform[3]: ULY (upper left Y coordinate)
 * adfGeotransform[4]: $-sin(\alpha)\cdot\textrm{Yres}$
 * adfGeotransform[5]: $-cos(\alpha)\cdot\textrm{Yres}$
 * @param x,y georeferenced coordinates in x and y
 * @return true if image covers the georeferenced location
 **/
bool Jim::covers(double x, double  y, OGRCoordinateTransformation *poCT) const
{
  double theULX, theULY, theLRX, theLRY;
  getBoundingBox(theULX,theULY,theLRX,theLRY);
  double ximg=x;
  double yimg=y;
  if(poCT){
    if(!poCT->Transform(1,&ximg,&yimg)){
      std::ostringstream errorStream;
      errorStream << "Error: cannot apply OGRCoordinateTransformation in Jim::covers (1)" << std::endl;
      throw(errorStream.str());
    }
  }
  return((ximg > theULX)&&
         (ximg < theLRX)&&
         (yimg < theULY)&&
         (yimg >theLRY));
}

/**
 * assuming
 * adfGeotransform[0]: ULX (upper left X coordinate)
 * adfGeotransform[1]: $cos(\alpha)\cdot\textrm{Xres}$
 * adfGeotransform[2]: $-sin(\alpha)\cdot\textrm{Xres}$
 * adfGeotransform[3]: ULY (upper left Y coordinate)
 * adfGeotransform[4]: $-sin(\alpha)\cdot\textrm{Yres}$
 * adfGeotransform[5]: $-cos(\alpha)\cdot\textrm{Yres}$
 * @param ulx upper left coordinate in x
 * @param uly upper left coordinate in y
 * @param lrx lower left coordinate in x
 * @param lry lower left coordinate in y
 * @return true if image (partially or all if all is set) covers the bounding box
 **/
bool Jim::covers(double ulx, double  uly, double lrx, double lry, bool all, OGRCoordinateTransformation *poCT) const
{
  double theULX, theULY, theLRX, theLRY;
  getBoundingBox(theULX,theULY,theLRX,theLRY);
  double ulximg=ulx;
  double ulyimg=uly;
  double lrximg=lrx;
  double lryimg=lry;
  if(poCT){
    std::vector<double> xvector(4);//ulx,urx,llx,lrx
    std::vector<double> yvector(4);//uly,ury,lly,lry
    xvector[0]=ulximg;
    xvector[1]=lrximg;
    xvector[2]=ulximg;
    xvector[3]=lrximg;
    yvector[0]=ulyimg;
    yvector[1]=ulyimg;
    yvector[2]=lryimg;
    yvector[3]=lryimg;
    if(!poCT->Transform(xvector.size(),&xvector[0],&yvector[0])){
      std::ostringstream errorStream;
      errorStream << "Error: cannot apply OGRCoordinateTransformation in Jim::covers (2)" << std::endl;
      throw(errorStream.str());
    }
    ulximg=std::min(xvector[0],xvector[2]);
    lrximg=std::max(xvector[1],xvector[3]);
    ulyimg=std::max(yvector[0],yvector[1]);
    lryimg=std::min(yvector[2],yvector[3]);
  }
  if(all)
    return((theULX<ulximg)&&(theULY>ulyimg)&&(theLRX>lrximg)&&(theLRY<lryimg));
  else
    return((ulximg < theLRX)&&(lrximg > theULX)&&(lryimg < theULY)&&(ulyimg > theLRY));
}

/**
 * @param noDataValues standard template library (stl) vector containing no data values
 * @return number of no data values in this dataset
 **/
CPLErr Jim::getNoDataValues(std::vector<double>& noDataValues) const
{
  if(m_noDataValues.size()){
    noDataValues=m_noDataValues;
    return(CE_None);
  }
  else
    return(CE_Warning);
}

/**
 * @param noDataValue no data value to be pushed for this dataset
 * @return CE_None if successful
 **/
CPLErr Jim::pushNoDataValue(double noDataValue)
{
  if(find(m_noDataValues.begin(),m_noDataValues.end(),noDataValue)==m_noDataValues.end())
    m_noDataValues.push_back(noDataValue);
  if(m_noDataValues.size())
    return(CE_None);
  else
    return(CE_Warning);
}

CPLErr Jim::open(const std::string& filename, bool readData, unsigned int memory){
  // reset();
  // m_nplane=1;
  m_access=READ_ONLY;
  m_filename = filename;
  registerDriver();
  initMem(memory);
  for(int iband=0;iband<m_nband;++iband){
    m_begin[iband]=0;
    m_end[iband]=0;
  }
  if(readData){
    for(int iband=0;iband<nrOfBand();++iband)
      readDataDS(iband,iband);
  }
  return(CE_None);
}

CPLErr Jim::registerDriver()
{
  GDALAllRegister();
  if(writeMode()){
    GDALDriver *poDriver;
    poDriver = GetGDALDriverManager()->GetDriverByName(m_imageType.c_str());
    if( poDriver == NULL ){
      std::ostringstream s;
      s << "FileOpenError (" << m_imageType << ")";
      throw(s.str());
    }
    char **papszMetadata;
    papszMetadata = poDriver->GetMetadata();
    //todo: try and catch if CREATE is not supported (as in PNG)
    if( ! CSLFetchBoolean( papszMetadata, GDAL_DCAP_CREATE, FALSE )){
      std::ostringstream s;
      s << "Error: image type " << m_imageType << " not supported";
      throw(s.str());
    }
    char **papszOptions=NULL;
    for(std::vector<std::string>::const_iterator optionIt=m_options.begin();optionIt!=m_options.end();++optionIt)
      papszOptions=CSLAddString(papszOptions,optionIt->c_str());

    m_gds=poDriver->Create(m_filename.c_str(),nrOfCol(),nrOfRow(),nrOfBand(),getGDALDataType(),papszOptions);
    // m_gds=poDriver->Create(m_filename.c_str(),m_ncol,m_nrow,m_nband,m_dataType,papszOptions);
    vector<double> gt(6);
    getGeoTransform(gt);
    if(setGeoTransform(gt)!=CE_None)
      std::cerr << "Warning: could not write geotransform information in " << m_filename << std::endl;
    if(setProjection(m_projection)!=CE_None)
      std::cerr << "Warning: could not write projection information in " << m_filename << std::endl;


    if(m_noDataValues.size()){
      for(int iband=0;iband<nrOfBand();++iband)
        GDALSetNoDataValue(m_noDataValues[0],iband);
    }

    // m_gds->SetMetadataItem( "TIFFTAG_DOCUMENTNAME", m_filename.c_str());
    // std::string versionString="pktools ";
    // versionString+=JIPLIB_VERSION;
    // versionString+=" by Pieter Kempeneers";
    // m_gds->SetMetadataItem( "TIFFTAG_SOFTWARE", versionString.c_str());
    time_t rawtime;
    time ( &rawtime );

    time_t tim=time(NULL);
    tm *now=localtime(&tim);
    std::ostringstream datestream;
    //date std::string must be 20 characters long...
    datestream << now->tm_year+1900;
    if(now->tm_mon+1<10)
      datestream << ":0" << now->tm_mon+1;
    else
      datestream << ":" << now->tm_mon+1;
    if(now->tm_mday<10)
      datestream << ":0" << now->tm_mday;
    else
      datestream << ":" << now->tm_mday;
    if(now->tm_hour<10)
      datestream << " 0" << now->tm_hour;
    else
      datestream << " " << now->tm_hour;
    if(now->tm_min<10)
      datestream << ":0" << now->tm_min;
    else
      datestream << ":" << now->tm_min;
    if(now->tm_sec<10)
      datestream << ":0" << now->tm_sec;
    else
      datestream << ":" << now->tm_sec;
    m_gds->SetMetadataItem( "TIFFTAG_DATETIME", datestream.str().c_str());
  }
  else{
#if GDAL_VERSION_MAJOR < 2
    if(m_access==UPDATE)
      m_gds = (GDALDataset *) GDALOpen(m_filename.c_str(), GA_Update);
    else
      m_gds = (GDALDataset *) GDALOpen(m_filename.c_str(), GA_ReadOnly );
#else
    if(m_access==UPDATE)
      m_gds = (GDALDataset*) GDALOpenEx(m_filename.c_str(), GDAL_OF_UPDATE|GDAL_OF_RASTER, NULL, NULL, NULL);
    else
      m_gds = (GDALDataset*) GDALOpenEx(m_filename.c_str(), GDAL_OF_READONLY|GDAL_OF_RASTER, NULL, NULL, NULL);
#endif

    if(m_gds == NULL){
      std::string errorString="FileOpenError";
      throw(errorString);
    }
    m_ncol=m_gds->GetRasterXSize();
    m_nrow=m_gds->GetRasterYSize();
    m_nband=m_gds->GetRasterCount();
    m_dataType=getDataTypeDS();
    m_imageType=getImageType();
    vector<double> adfGeoTransform(6);
    m_gds->GetGeoTransform( &adfGeoTransform[0] );
    m_gt[0]=adfGeoTransform[0];
    m_gt[1]=adfGeoTransform[1];
    m_gt[2]=adfGeoTransform[2];
    m_gt[3]=adfGeoTransform[3];
    m_gt[4]=adfGeoTransform[4];
    m_gt[5]=adfGeoTransform[5];
    m_projection=m_gds->GetProjectionRef();
  }
  return(CE_None);
}

//Open raster dataset for reading or writing
CPLErr Jim::open(app::AppFactory &app){
  //input
  Optionjl<std::string> input_opt("fn", "filename", "filename");
  Optionjl<std::string> resample_opt("r", "resample", "resample: GRIORA_NearestNeighbour|GRIORA_Bilinear|GRIORA_Cubic|GRIORA_CubicSpline|GRIORA_Lanczos|GRIORA_Average|GRIORA_Average|GRIORA_Gauss (check http://www.gdal.org/gdal_8h.html#a640ada511cbddeefac67c548e009d5a)","GRIORA_NearestNeighbour");
  // Optionjl<std::string> extra_opt("extra", "extra", "RGDALRasterIOExtraArg (check http://www.gdal.org/structGDALRasterIOExtraArg.html)");
  // Optionjl<std::string> targetSRS_opt("t_srs", "t_srs", "Target spatial reference system in EPSG format (e.g., epsg:3035)");//todo
  //output
  Optionjl<double> nodata_opt("nodata", "nodata", "Nodata value to put in image.");
  Optionjl<int> nsample_opt("ns", "ncol", "Number of columns");
  Optionjl<int> nline_opt("nl", "nrow", "Number of rows");
  Optionjl<int> nband_opt("nb", "nband", "Number of bands",1);
  Optionjl<std::string> otype_opt("ot", "otype", "Data type for output image ({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64/Int64/UInt64})","Byte");
  Optionjl<std::string>  oformat_opt("of", "oformat", "Output image format (see also gdal_translate).","GTiff");
  Optionjl<std::string> option_opt("co", "co", "Creation option for output file. Multiple options can be specified.");
  Optionjl<unsigned long int> seed_opt("seed", "seed", "seed value for random generator",0);
  Optionjl<double> mean_opt("mean", "mean", "Mean value for random generator",0);
  Optionjl<double> stdev_opt("stdev", "stdev", "Standard deviation for Gaussian random generator",0);
  Optionjl<double> uniform_opt("uniform", "uniform", "start and end values for random value with uniform distribution",0);
  Optionjl<std::string> assignSRS_opt("a_srs", "a_srs", "Assign the spatial reference for the output file, e.g., epsg:3035 to use European projection and force to European grid");
  Optionjl<std::string> sourceSRS_opt("s_srs", "s_srs", "Source spatial reference for the input file, e.g., epsg:3035 to use European projection and force to European grid");
  Optionjl<std::string> targetSRS_opt("t_srs", "t_srs", "Target spatial reference for the output file, e.g., epsg:3035 to use European projection and force to European grid");
  // Optionjl<std::string> description_opt("d", "description", "Set image description");
  //input and output
  Optionjl<int> band_opt("b", "band", "Bands to open, index starts from 0");
  Optionjl<std::string> extent_opt("e", "extent", "get boundary from extent from polygons in vector file");
  Optionjl<double> ulx_opt("ulx", "ulx", "Upper left x value bounding box");
  Optionjl<double> uly_opt("uly", "uly", "Upper left y value bounding box");
  Optionjl<double> lrx_opt("lrx", "lrx", "Lower right x value bounding box");
  Optionjl<double> lry_opt("lry", "lry", "Lower right y value bounding box");
  Optionjl<double> dx_opt("dx", "dx", "Resolution in x");
  Optionjl<double> dy_opt("dy", "dy", "Resolution in y");
  Optionjl<bool> align_opt("align", "align", "Align output bounding box to input image",false);
  Optionjl<std::string> access_opt("access", "access", "access (READ_ONLY, UPDATE)","READ_ONLY",2);
  Optionjl<bool> noread_opt("noread", "noread", "do not read data when opening",false);
  Optionjl<bool> band2plane_opt("band2plane", "band2plane", "read bands as planes",false);
  Optionjl<unsigned long int>  memory_opt("mem", "mem", "Buffer size (in MB) to read image data blocks in memory",0,1);
  Optionjl<short> verbose_opt("verbose", "verbose", "verbose output",0,2);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=input_opt.retrieveOption(app);
    nodata_opt.retrieveOption(app);
    band_opt.retrieveOption(app);
    ulx_opt.retrieveOption(app);
    uly_opt.retrieveOption(app);
    lrx_opt.retrieveOption(app);
    lry_opt.retrieveOption(app);
    dx_opt.retrieveOption(app);
    dy_opt.retrieveOption(app);
    align_opt.retrieveOption(app);
    resample_opt.retrieveOption(app);
    extent_opt.retrieveOption(app);
    // extra_opt.retrieveOption(app);
    // targetSRS_opt.retrieveOption(app);
    nsample_opt.retrieveOption(app);
    nline_opt.retrieveOption(app);
    nband_opt.retrieveOption(app);
    otype_opt.retrieveOption(app);
    oformat_opt.retrieveOption(app);
    option_opt.retrieveOption(app);
    seed_opt.retrieveOption(app);
    mean_opt.retrieveOption(app);
    stdev_opt.retrieveOption(app);
    uniform_opt.retrieveOption(app);
    assignSRS_opt.retrieveOption(app);
    sourceSRS_opt.retrieveOption(app);
    targetSRS_opt.retrieveOption(app);
    access_opt.retrieveOption(app);
    noread_opt.retrieveOption(app);
    band2plane_opt.retrieveOption(app);
    memory_opt.retrieveOption(app);
    verbose_opt.retrieveOption(app);
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
  statfactory::StatFactory stat;

  OGRCoordinateTransformation *target2gds=0;
  OGRCoordinateTransformation *gds2target=0;

  //get bounding box from extentReader if defined
  VectorOgr extentReader;

  if(input_opt.empty()){
    OGRSpatialReference assignSpatialRef;
    if(assignSRS_opt.size())
      assignSpatialRef.SetFromUserInput(assignSRS_opt[0].c_str());

    //get bounding box from extentReader if defined
    if(extent_opt.size()){
      double e_ulx;
      double e_uly;
      double e_lrx;
      double e_lry;
      for(int iextent=0;iextent<extent_opt.size();++iextent){
        std::vector<std::string> layernames;
        layernames.clear();
        extentReader.open(extent_opt[iextent],layernames,true);

        OGRSpatialReference *vectorSpatialRef=extentReader.getLayer(0)->GetSpatialRef();
        OGRCoordinateTransformation *vector2assign=0;
        vector2assign = OGRCreateCoordinateTransformation(vectorSpatialRef, &assignSpatialRef);
        if(assignSpatialRef.IsSame(vectorSpatialRef)){
          vector2assign=0;
        }
        else{
          if(!vector2assign){
            std::ostringstream errorStream;
            errorStream << "Error: cannot create OGRCoordinateTransformation vector to assignSRS" << std::endl;
            throw(errorStream.str());
          }
        }
        extentReader.getExtent(e_ulx,e_uly,e_lrx,e_lry,vector2assign);
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
    }

    if(dx_opt.size()||dy_opt.size()){
      if(dx_opt.empty()){
        std::ostringstream errorStream;
        errorStream << "Warning: cell size in x not defined (use option --dx)." << std::endl;
        // Jim();
        throw(errorStream.str());
      }
      if(dy_opt.empty()){
        std::ostringstream errorStream;
        errorStream << "Warning: cell size in y not defined (use option --dy)." << std::endl;
        // Jim();
        throw(errorStream.str());
      }
      if(ulx_opt.empty()||uly_opt.empty()||lrx_opt.empty()||lry_opt.empty()){
        std::ostringstream errorStream;
        errorStream << "Warning: bounding box not defined (use options --ulx --uly --lrx --lry)." << std::endl;
        // Jim();
        throw(errorStream.str());
      }
      nsample_opt.clear();
      nsample_opt.push_back((lrx_opt[0]-ulx_opt[0])/dx_opt[0]);
      nline_opt.clear();
      nline_opt.push_back((uly_opt[0]-lry_opt[0])/dy_opt[0]);
    }
    else if(nsample_opt.size()||nline_opt.size()){
      if(nsample_opt.empty()){
        std::ostringstream errorStream;
        errorStream << "Warning: no number of columns (use option --ncol)." << std::endl;
        // Jim();
        throw(errorStream.str());
      }
      if(nline_opt.empty()){
        std::ostringstream errorStream;
        errorStream << "Warning: no number of rows (use option --nrow)." << std::endl;
        // Jim();
        throw(errorStream.str());
      }
    }
    // GDALDataType theType=string2GDAL(otype_opt[0]);
    int theType=0;
    theType=string2GDAL(otype_opt[0]);
    if(theType==GDT_Unknown)
      theType=string2JDT(otype_opt[0]);
    // open(nsample_opt[0],nline_opt[0],nband_opt[0],theType);
    m_ncol = nsample_opt[0];
    m_nrow = nline_opt[0];
    m_nband = nband_opt[0];
    m_nplane = 1;//todo: support planes
    m_dataType = theType;
    initMem(0);
    for(int iband=0;iband<m_nband;++iband){
      m_begin[iband]=0;
      m_end[iband]=m_begin[iband]+m_blockSize;
    }
    if(m_filename!=""){
      // m_writeMode=true;
      m_access=WRITE;
      registerDriver();
    }
    if(ulx_opt.size()&&uly_opt.size()&&lrx_opt.size()&&lry_opt.size()){
      double gt[6];
      if(ulx_opt[0]<lrx_opt[0])
        gt[0]=ulx_opt[0];
      else
        gt[0]=0;
      if(dx_opt.size())
        gt[1]=dx_opt[0];
      else if(lrx_opt[0]-ulx_opt[0]>0){
        gt[1]=lrx_opt[0]-ulx_opt[0];
        gt[1]/=nrOfCol();
      }
      else
        gt[1]=1;
      gt[2]=0;
      if(uly_opt[0]>lry_opt[0])
        gt[3]=uly_opt[0];
      else
        gt[3]=0;
      gt[4]=0;
      if(dy_opt.size())
        gt[5]=-dy_opt[0];
      else if(uly_opt[0]-lry_opt[0]>0){
        gt[5]=lry_opt[0]-uly_opt[0];
        gt[5]/=nrOfRow();
      }
      else
        gt[5]=-1;
      setGeoTransform(gt);
      if(assignSRS_opt.size())
        setProjectionProj4(assignSRS_opt[0]);
    }
    else{
      //test
      double gt[6];
      gt[0]=0;
      gt[1]=1;
      gt[2]=0;
      gt[3]=0;
      gt[4]=0;
      gt[5]=-1;
      setGeoTransform(gt);
    }
    gsl_rng* rndgen=stat.getRandomGenerator(seed_opt[0]);
    double value=mean_opt[0];
    std::vector<double> lineBuffer(nrOfCol(),value);
    double a=0;
    double b=1;
    std::string distribution="none";
    if(uniform_opt.size()>1){
      distribution="uniform";
      a=uniform_opt[0];
      b=uniform_opt[1];
    }
    else if(stdev_opt[0]>0){
      distribution="gaussian";
      a=mean_opt[0];
      b=stdev_opt[0];
    }
    else
      distribution="none";
#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
    for(unsigned int iband=0;iband<nrOfBand();++iband){
      std::vector<double> lineBuffer(nrOfCol(),mean_opt[0]);
      for(unsigned int irow=0;irow<nrOfRow();++irow){
        for(unsigned int icol=0;icol<nrOfCol();++icol){
          if(stat.getDistributionType(distribution)==statfactory::StatFactory::none)
            break;
          else
            value=stat.getRandomValue(rndgen,distribution,a,b);
          lineBuffer[icol]=value;
        }
        writeData(lineBuffer,irow,iband);
      }
    }
    stat.freeRandomGenerator(rndgen);
  }
  else if(input_opt.size()){
    setAccess(access_opt[0]);
    m_filename=input_opt[0];
    //set class member variables based on GDAL dataset
    registerDriver();
    if(assignSRS_opt.size())
      setProjectionProj4(assignSRS_opt[0]);
    if(sourceSRS_opt.size())
      setProjectionProj4(sourceSRS_opt[0]);
    // OGRSpatialReference gdsSpatialRef(m_gds->GetProjectionRef());
    OGRSpatialReference gdsSpatialRef(getProjectionRef().c_str());
    if(extent_opt.size()){
      double e_ulx;
      double e_uly;
      double e_lrx;
      double e_lry;
      for(int iextent=0;iextent<extent_opt.size();++iextent){
        std::vector<std::string> layernames;
        layernames.clear();
        extentReader.open(extent_opt[iextent],layernames,true);

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
    }
    else{
      if(targetSRS_opt.size()){
        OGRSpatialReference targetSpatialRef;
        targetSpatialRef.SetFromUserInput(targetSRS_opt[0].c_str());

        char *wktString;
        targetSpatialRef.exportToWkt(&wktString);
        // m_projection=wktString;
        gds2target = OGRCreateCoordinateTransformation(&gdsSpatialRef, &targetSpatialRef);
        target2gds = OGRCreateCoordinateTransformation(&targetSpatialRef, &gdsSpatialRef);
        if(targetSpatialRef.IsSame(&gdsSpatialRef)){
          target2gds=0;
          gds2target=0;
        }
        else{
          if(!target2gds){
            std::ostringstream errorStream;
            errorStream << "Error: cannot create OGRCoordinateTransformation target to GDAL dataset" << std::endl;
            throw(errorStream.str());
          }
          if(!gds2target){
            std::ostringstream errorStream;
            errorStream << "Error: cannot create OGRCoordinateTransformation GDAL dataset to target" << std::endl;
            throw(errorStream.str());
          }
        }
        // if(dx_opt.size()){//convert to number of samples
        //   if(ulx_opt.size()&&lrx_opt.size()){
        //     nsample_opt.clear();
        //     nsample_opt.push_back((lrx_opt[0]-ulx_opt[0])/dx_opt[0]);
        //     dx_opt.clear();
        //   }
        //   else{
        //     dx_opt.clear();
        //   }
        // }
        // if(dy_opt.size()){//convert to number of lines
        //   if(uly_opt.size()&&lry_opt.size()){
        //     nline_opt.clear();
        //     nline_opt.push_back((uly_opt[0]-lry_opt[0])/dy_opt[0]);
        //     dy_opt.clear();
        //   }
        //   else{
        //     dy_opt.clear();
        //   }
        // }
        //both ulx and uly need to be set
        if(ulx_opt.size() && uly_opt.size()){
          if(target2gds){
            if(!target2gds->Transform(ulx_opt.size(),&ulx_opt[0],&uly_opt[0])){
              std::ostringstream errorStream;
              std::cerr << "Error: cannot apply OGRCoordinateTransformation" << std::endl;
              errorStream << "Error: cannot apply OGRCoordinateTransformation" << std::endl;
              throw(errorStream.str());
            }
          }
        }
        else{
          ulx_opt.clear();
          uly_opt.clear();
        }
        if(lrx_opt.size() && lry_opt.size()){
          if(target2gds){
            if(!target2gds->Transform(lrx_opt.size(),&lrx_opt[0],&lry_opt[0])){
              std::ostringstream errorStream;
              errorStream << "Error: cannot apply OGRCoordinateTransformation" << std::endl;
              throw(errorStream.str());
            }
          }
        }
        else{
          lrx_opt.clear();
          lry_opt.clear();
        }
      }
    }

    std::vector<double> gds_gt(6);
    m_gds->GetGeoTransform(&gds_gt[0]);
    int gds_ncol=m_gds->GetRasterXSize();
    int gds_nrow=m_gds->GetRasterYSize();
    double gds_ulx=gds_gt[0];
    double gds_uly=gds_gt[3];
    double gds_lrx=gds_gt[0]+gds_ncol*gds_gt[1]+gds_nrow*gds_gt[2];
    double gds_lry=gds_gt[3]+gds_ncol*gds_gt[4]+gds_nrow*gds_gt[5];
    double gds_dx=gds_gt[1];
    double gds_dy=-gds_gt[5];

    if(ulx_opt.empty())
      ulx_opt.push_back(gds_ulx);
    else if(align_opt[0]){
      if(ulx_opt[0]>this->getUlx())
        ulx_opt[0]-=fmod(ulx_opt[0]-this->getUlx(),getDeltaX());
      else if(ulx_opt[0]<this->getUlx())
        ulx_opt[0]+=fmod(this->getUlx()-ulx_opt[0],getDeltaX())-getDeltaX();
    }
    if(uly_opt.empty())
      uly_opt.push_back(gds_uly);
    else if(align_opt[0]){
      if(uly_opt[0]<this->getUly())
        uly_opt[0]+=fmod(this->getUly()-uly_opt[0],getDeltaY());
      else if(uly_opt[0]>this->getUly())
        uly_opt[0]-=fmod(uly_opt[0]-this->getUly(),getDeltaY())+getDeltaY();
    }
    if(lrx_opt.empty())
      lrx_opt.push_back(gds_lrx);
    else if(align_opt[0]){
      if(lrx_opt[0]<this->getLrx())
        lrx_opt[0]+=fmod(this->getLrx()-lrx_opt[0],getDeltaX());
      else if(lrx_opt[0]>this->getLrx())
        lrx_opt[0]-=fmod(lrx_opt[0]-this->getLrx(),getDeltaX())+getDeltaX();
    }
    if(lry_opt.empty())
      lry_opt.push_back(gds_lry);
    else if(align_opt[0]){
      if(lry_opt[0]>this->getLry())
        lry_opt[0]-=fmod(lry_opt[0]-this->getLry(),getDeltaY());
      else if(lry_opt[0]<this->getLry())
        lry_opt[0]+=fmod(this->getLry()-lry_opt[0],getDeltaY())-getDeltaY();
    }
    //now ulx_opt, uly_opt, lrx_opt and lry_opt are in GDS SRS coordinates
    if(band_opt.empty()){
      while(band_opt.size()<nrOfBand())
        band_opt.push_back(band_opt.size());
    }
    m_nband=band_opt.size();

    GDALRasterBand *poBand;//we will fetch the first band to obtain the gds metadata
    poBand = m_gds->GetRasterBand(band_opt[0]+1);//GDAL uses 1 based index

    std::vector<double> gds_bb;
    getBoundingBox(gds_bb);
    if(dx_opt.empty()){
      if(nsample_opt.size()){
        if(verbose_opt[0])
          std::cout << "nsample_opt[0]: " << nsample_opt[0] << std::endl;
        dx_opt.push_back((lrx_opt[0]-ulx_opt[0])/nsample_opt[0]);
        // dx_opt.push_back((gds_lrx-gds_ulx)/nsample_opt[0]);
      }
      else
        dx_opt.push_back(gds_dx);
    }
    if(dy_opt.empty()){
      if(nline_opt.size()){
        if(verbose_opt[0])
          std::cout << "nline_opt[0]: " << nline_opt[0] << std::endl;
        dy_opt.push_back((uly_opt[0]-lry_opt[0])/nline_opt[0]);
        // dy_opt.push_back((gds_bb[1]-gds_bb[3])/nline_opt[0]);
      }
      else
        dy_opt.push_back(getDeltaY());
    }
    if(verbose_opt[0]){
      std::cout << "dx_opt[0]: " << dx_opt[0] << std::endl;
      std::cout << "dy_opt[0]: " << dy_opt[0] << std::endl;
    }
    //force bounding box to be within dataset
    if(ulx_opt[0]<gds_ulx)
      ulx_opt[0]=gds_ulx;
    if(uly_opt[0]>gds_uly)
      uly_opt[0]=gds_uly;
    if(lrx_opt[0]>gds_lrx)
      lrx_opt[0]=gds_lrx;
    if(lry_opt[0]<gds_lry)
      lry_opt[0]=gds_lry;
    std::vector<double> gt(6);
    gt[0]=ulx_opt[0];
    gt[3]=uly_opt[0];
    gt[1]=dx_opt[0];//todo: adfGeotransform[1]: $cos(\alpha)\cdot\textrm{Xres}$
    gt[2]=0;//todo: $-sin(\alpha)\cdot\textrm{Xres}$
    gt[4]=0;//todo: $-sin(\alpha)\cdot\textrm{Yres}$
    gt[5]=-dy_opt[0];//todo: a$-cos(\alpha)\cdot\textrm{Yres}
    setGeoTransform(gt);
    m_resample=getGDALResample(resample_opt[0]);
    int nBufXSize=static_cast<unsigned int>(ceil((lrx_opt[0]-ulx_opt[0])/dx_opt[0]-FLT_EPSILON));
    int nBufYSize=static_cast<unsigned int>(ceil((uly_opt[0]-lry_opt[0])/dy_opt[0]-FLT_EPSILON));
    m_ncol=nBufXSize;
    m_nrow=nBufYSize;

    //we initialize memory using class member variables instead of those read from GDAL dataset
    if(band2plane_opt[0]){
#if MIALIB == 1
      m_nplane=m_nband;
      m_nband=1;
      initMem(memory_opt[0]);
      m_begin[0]=0;
      m_end[0]=m_begin[0]+m_blockSize;
      if(!noread_opt[0]){
        readDataPlanes(band_opt);
      }
#else
      std::ostringstream errorStream;
      errorStream << "Warning: planes not supported, please compile with MIALIB" << std::endl;
      throw(errorStream.str());
#endif
    }
    else{
      if(!noread_opt[0]){
        if(!covers(ulx_opt[0],uly_opt[0],lrx_opt[0],lry_opt[0])){
          std::ostringstream errorStream;
          errorStream << "Warning: raster dataset does not cover required bounding box" << std::endl;
          throw(errorStream.str());
        }
        //we initialize memory using class member variables instead of those read from GDAL dataset
        initMem(memory_opt[0]);
        for(int iband=0;iband<nrOfBand();++iband){
          m_begin[iband]=0;
          m_end[iband]=m_begin[iband]+m_blockSize;
          if(!noread_opt[0]){
            //we can not use readData(iband) because sequence of band_opt might not correspond bands in GDAL dataset
            readDataDS(iband,band_opt[iband]);
          }
        }
      }
      else{
        if(memory_opt[0]<=0)
          m_blockSize=nrOfRow();
        else{
          m_blockSize=static_cast<unsigned int>(memory_opt[0]*1000000/nrOfBand()/nrOfCol()/getDataTypeSizeBytes());
          if(getBlockSizeY(0))
            m_blockSize-=m_blockSize%getBlockSizeY(0);
          if(m_blockSize<1)
            m_blockSize=1;
          if(m_blockSize>nrOfRow())
            m_blockSize=nrOfRow();
        }
      }
    }
  }
  else{
    std::ostringstream errorStream;
    errorStream << "Warning: no number of rows or columns provided, nor input filename." << std::endl;
    // Jim();
    throw(errorStream.str());
  }
  setNoData(nodata_opt);
  return(CE_None);
}

/**
 * @param band Band that must be read to cache
 * @return true if block was read
 **/
CPLErr Jim::readDataDS(int band, int ds_band)
{
  CPLErr returnValue=CE_None;
  if(m_blockSize<nrOfRow()){
    std::ostringstream s;
    s << "Error: increase memory to read all pixels in memory (now at " << 100.0*m_blockSize/nrOfRow() << "%)";
    throw(s.str());
  }
  if(m_gds == NULL){
    std::string errorString="Error in readData";
    throw(errorString);
  }
  //should have been set already...
  m_begin[band]=0;
  m_end[band]=m_begin[band]+m_blockSize;
  return(readNewBlockDS(0,band,ds_band));
}

// CPLErr Jim::readData(int band)
// {
//   CPLErr returnValue=CE_None;
//   if(m_blockSize<nrOfRow()){
//     std::ostringstream s;
//     s << "Error: increase memory to read all pixels in memory (now at " << 100.0*m_blockSize/nrOfRow() << "%)";
//     throw(s.str());
//   }
//   if(m_gds == NULL){
//     std::string errorString="Error in readData";
//     throw(errorString);
//   }
//   m_begin[band]=0;
//   m_end[band]=nrOfRow();
//   GDALRasterBand  *poBand;
//   if(nrOfBand()<=iband){
//     std::string errorString="Error: band number exceeds available bands in readData";
//     throw(errorString);
//   }
//   poBand = m_gds->GetRasterBand(band+1);//GDAL uses 1 based index
//   returnValue=poBand->RasterIO(GF_Read,0,m_begin[band],nrOfCol(),m_end[band]-m_begin[band],m_data[band],nrOfCol(),m_end[band]-m_begin[band],getDataType(),0,0);
//   return(returnValue);//new block was read
// }

/**
 * @return true if block was read
 **/
CPLErr Jim::readData()
{
  CPLErr returnValue=CE_None;
  for(int iband=0;iband<nrOfBand();++iband){
    if(readData(iband)!=CE_None)
      returnValue=CE_Failure;
  }
  return(returnValue);//new block was read
}

/**
 * @param row Read a new block for caching this row (if needed)
 * @param band Band that must be read to cache
 * @return true if block was read
 **/
CPLErr Jim::readNewBlockDS(int row, int iband, int ds_band){
  CPLErr returnValue=CE_None;
  if(m_gds == NULL){
    std::string errorString="Error in readNewBlock";
    throw(errorString);
  }
  if(m_end[iband]<m_blockSize)//first time
    m_end[iband]=m_blockSize;
  while(row>=m_end[iband]&&m_begin[iband]<nrOfRow()){
    m_begin[iband]+=m_blockSize;
    m_end[iband]=m_begin[iband]+m_blockSize;
  }
  if(m_end[iband]>nrOfRow())
    m_end[iband]=nrOfRow();

  int gds_ncol=m_gds->GetRasterXSize();
  int gds_nrow=m_gds->GetRasterYSize();
  int gds_nband=m_gds->GetRasterCount();
  vector<double> gds_gt(6);
  m_gds->GetGeoTransform(&gds_gt[0]);
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
  // double dfYSize=(getUly()-getLry())/gds_dy;//y-size in pixels of region to read in original image
  // double dfYOff=diffYm/gds_dy;
  // int nYOff=static_cast<int>(dfYOff);
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
  // for(int iband=0;iband<m_nband;++iband){
  //fetch raster band
  GDALRasterBand  *poBand;
  if(nrOfBand()<=iband){
    std::string errorString="Error: band number exceeds available bands in readNewBlock";
    throw(errorString);
  }
  poBand = m_gds->GetRasterBand(ds_band+1);//GDAL uses 1 based index

  dfYSize=(m_end[iband]-m_begin[iband])*getDeltaY()/gds_dy;//y-size in pixels of region to read in original image
  nYSize=static_cast<unsigned int>(ceil((m_end[iband]-m_begin[iband])*getDeltaY()/gds_dy));//y-size in pixels of region to read in original image
  if(nYSize>gds_nrow)
    nYSize=gds_nrow;
  dfYOff=(gds_uly-getUly())/gds_dy+m_begin[iband]*getDeltaY()/gds_dy;
  nYOff=static_cast<int>(dfYOff);
  if(poBand->GetOverviewCount()){
    //calculate number of desired samples in overview
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
    nXSize=static_cast<unsigned int>(ceil((getLrx()-getUlx())/ods_dx));//x-size in pixels of region to read in overview image
    if(nXSize>ods_ncol)
      nXSize=ods_ncol;
    dfXOff=diffXm/ods_dx;
    nXOff=static_cast<int>(dfXOff);
    dfYSize=(m_end[iband]-m_begin[iband])*getDeltaY()/ods_dy;//y-size in pixels of region to read in overview image
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

  returnValue=poBand->RasterIO(GF_Read,nXOff,nYOff+m_begin[iband],nXSize,nYSize,m_data[iband],nrOfCol(),m_end[iband]-m_begin[iband],getGDALDataType(),0,0,&sExtraArg);
  //   returnValue=poBand->RasterIO(GF_Read,0,m_begin[iband],nrOfCol(),m_end[iband]-m_begin[iband],m_data[iband],nrOfCol(),m_end[iband]-m_begin[iband],getDataType(),0,0);
  // }
  return(returnValue);//new block was read
}

// CPLErr Jim::readNewBlock(int row, int band)
// {
//   CPLErr returnValue=CE_None;
//   if(m_gds == NULL){
//     std::string errorString="Error in readNewBlock";
//     throw(errorString);
//   }
//   if(m_end[band]<m_blockSize)//first time
//     m_end[band]=m_blockSize;
//   while(row>=m_end[band]&&m_begin[band]<nrOfRow()){
//     m_begin[band]+=m_blockSize;
//     m_end[band]=m_begin[band]+m_blockSize;
//   }
//   if(m_end[band]>nrOfRow())
//     m_end[band]=nrOfRow();
//   for(int iband=0;iband<m_nband;++iband){
//     //fetch raster band
//     GDALRasterBand  *poBand;
//     //todo: replace assert with exception
//     assert(iband<nrOfBand()+1);
//     poBand = m_gds->GetRasterBand(iband+1);//GDAL uses 1 based index
//     returnValue=poBand->RasterIO(GF_Read,0,m_begin[iband],nrOfCol(),m_end[iband]-m_begin[iband],m_data[iband],nrOfCol(),m_end[iband]-m_begin[iband],getDataType(),0,0);
//   }
//   return(returnValue);//new block was read
// }

#if MIALIB == 1
CPLErr Jim::readDataPlanes(std::vector<int> bands){
  CPLErr returnValue=CE_None;
  if(m_gds == NULL){
    std::string errorString="Error in readNewBlock";
    throw(errorString);
  }
  if(m_end[0]<m_blockSize)//first time
    m_end[0]=m_blockSize;
  if(m_end[0]>nrOfRow())
    m_end[0]=nrOfRow();

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
  dfYSize=(m_end[0]-m_begin[0])*getDeltaY()/gds_dy;//y-size in pixels of region to read in original image
  nYSize=static_cast<unsigned int>(ceil((m_end[0]-m_begin[0])*getDeltaY()/gds_dy));//y-size in pixels of region to read in original image
  if(nYSize>gds_nrow)
    nYSize=gds_nrow;
  dfYOff=(gds_uly-getUly())/gds_dy+m_begin[0]*getDeltaY()/gds_dy;
  nYOff=static_cast<int>(dfYOff);
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
  std::vector<int> gdalbands=bands;
  for(int iband=0;iband<bands.size();++iband)
    gdalbands[iband]=bands[iband]+1;

// eRWFlag	Either GF_Read to read a region of data, or GF_Write to write a region of data.
// nXOff	The pixel offset to the top left corner of the region of the band to be accessed. This would be zero to start from the left side.
// nYOff	The line offset to the top left corner of the region of the band to be accessed. This would be zero to start from the top.
// nXSize	The width of the region of the band to be accessed in pixels.
// nYSize	The height of the region of the band to be accessed in lines.
// pData	The buffer into which the data should be read, or from which it should be written. This buffer must contain at least nBufXSize * nBufYSize * nBandCount words of type eBufType. It is organized in left to right,top to bottom pixel order. Spacing is controlled by the nPixelSpace, and nLineSpace parameters.
// nBufXSize	the width of the buffer image into which the desired region is to be read, or from which it is to be written.
// nBufYSize	the height of the buffer image into which the desired region is to be read, or from which it is to be written.
// eBufType	the type of the pixel values in the pData data buffer. The pixel values will automatically be translated to/from the GDALRasterBand data type as needed.
// nBandCount	the number of bands being read or written.
// panBandMap	the list of nBandCount band numbers being read/written. Note band numbers are 1 based. This may be NULL to select the first nBandCount bands.
// nPixelSpace	The byte offset from the start of one pixel value in pData to the start of the next pixel value within a scanline. If defaulted (0) the size of the datatype eBufType is used.
// nLineSpace	The byte offset from the start of one scanline in pData to the start of the next. If defaulted (0) the size of the datatype eBufType * nBufXSize is used.
// nBandSpace	the byte offset from the start of one bands data to the start of the next. If defaulted (0) the value will be nLineSpace * nBufYSize implying band sequential organization of the data buffer.
// psExtraArg	(new in GDAL 2.0) pointer to a GDALRasterIOExtraArg structure with additional arguments to specify resampling and progress callback, or NULL for default behaviour. The GDAL_RASTERIO_RESAMPLING configuration option can also be defined to override the default resampling to one of BILINEAR, CUBIC, CUBICSPLINE, LANCZOS, AVERAGE or MODE.
  returnValue=getDataset()->RasterIO(GF_Read,nXOff,nYOff+m_begin[0],nXSize,nYSize,m_data[0],nrOfCol(),m_end[0]-m_begin[0],getGDALDataType(),gdalbands.size(),&gdalbands[0],0,0,0,&sExtraArg);
  return(returnValue);//new block was read
}
#endif
/**
 * @param x Reported column where minimum value in image was found (start counting from 0)
 * @param y Reported row where minimum value in image was found (start counting from 0)
 * @param band Search mininum value in image for this band
 * @return minimum value in image for the selected band
 **/
double Jim::getMin(int& x, int& y, int band){
  double minValue=0;
  std::vector<double> lineBuffer(nrOfCol());
  bool isValid=false;
  for(int irow=0;irow<nrOfRow();++irow){
    readData(lineBuffer,irow,band);
    for(int icol=0;icol<nrOfCol();++icol){
      if(isNoData(lineBuffer[icol]))
        continue;
      if(isValid){
        if(lineBuffer[icol]<minValue){
          y=irow;
          x=icol;
          minValue=lineBuffer[icol];
        }
      }
      else{
        y=irow;
        x=icol;
        minValue=lineBuffer[icol];
        isValid=true;
      }
    }
  }
  if(isValid)
    return minValue;
  else
    throw(static_cast<std::string>("Warning: not initialized"));
}

/**
 * @param x Reported column where maximum value in image was found (start counting from 0)
 * @param y Reported row where maximum value in image was found (start counting from 0)
 * @param band Search mininum value in image for this band
 * @return maximum value in image for the selected band
 **/
double Jim::getMax(int& x, int& y, int band){
  double maxValue=0;
  std::vector<double> lineBuffer(nrOfCol());
  bool isValid=false;
  for(int irow=0;irow<nrOfRow();++irow){
    readData(lineBuffer,irow,band);
    for(int icol=0;icol<nrOfCol();++icol){
      if(isNoData(lineBuffer[icol]))
        continue;
      if(isValid){
        if(lineBuffer[icol]>maxValue){
          y=irow;
          x=icol;
          maxValue=lineBuffer[icol];
        }
      }
      else{
        y=irow;
        x=icol;
        maxValue=lineBuffer[icol];
        isValid=true;
      }
    }
  }
  if(isValid)
    return maxValue;
  else
    throw(static_cast<std::string>("Warning: no valid pixels found"));
}

/**
 * @param startCol, endCol, startRow, endRow Search extreme value in this region of interest (all indices start counting from 0)
 * @param band Search extreme value in image for this band
 * @param minValue Reported minimum value within searched region
 * @param maxValue Reported maximum value within searched region
 **/
CPLErr Jim::getMinMax(int startCol, int endCol, int startRow, int endRow, int band, double& minValue, double& maxValue)
{
  bool isConstraint=(maxValue>minValue);
  double minConstraint=minValue;
  double maxConstraint=maxValue;
  std::vector<double> lineBuffer(endCol-startCol+1);
  bool isValid=false;
  //todo: replace assert with exception
  assert(endRow<nrOfRow());
  for(int irow=startCol;irow<endRow+1;++irow){
    readData(lineBuffer,startCol,endCol,irow,band);
    for(int icol=0;icol<lineBuffer.size();++icol){
      if(isNoData(lineBuffer[icol]))
        continue;
      if(isValid){
        if(isConstraint){
          if(lineBuffer[icol]<minConstraint)
            continue;
          if(lineBuffer[icol]>maxConstraint)
            continue;
        }
        if(lineBuffer[icol]<minValue)
          minValue=lineBuffer[icol];
        if(lineBuffer[icol]>maxValue)
          maxValue=lineBuffer[icol];
      }
      else{
        if(isConstraint){
          if(lineBuffer[icol]<minConstraint)
            continue;
          if(lineBuffer[icol]>maxConstraint)
            continue;
        }
        minValue=lineBuffer[icol];
        maxValue=lineBuffer[icol];
        isValid=true;
      }
    }
  }
  if(!isValid)
    throw(static_cast<std::string>("Warning: not initialized"));
  return(CE_None);
}

/**
 * @param minValue Reported minimum value in image
 * @param maxValue Reported maximum value in image
 * @param band Search extreme value in image for this band
 **/
CPLErr Jim::getMinMax(double& minValue, double& maxValue, int band)
{
  bool isConstraint=(maxValue>minValue);
  double minConstraint=minValue;
  double maxConstraint=maxValue;
  std::vector<double> lineBuffer(nrOfCol());
  bool isValid=false;
  for(int irow=0;irow<nrOfRow();++irow){
    readData(lineBuffer,irow,band);
    for(int icol=0;icol<nrOfCol();++icol){
      if(isNoData(lineBuffer[icol]))
        continue;
      if(isValid){
        if(isConstraint){
          if(lineBuffer[icol]<minConstraint)
            continue;
          if(lineBuffer[icol]>maxConstraint)
            continue;
        }
        if(lineBuffer[icol]<minValue)
          minValue=lineBuffer[icol];
        if(lineBuffer[icol]>maxValue)
          maxValue=lineBuffer[icol];
      }
      else{
        if(isConstraint){
          if(lineBuffer[icol]<minConstraint)
            continue;
          if(lineBuffer[icol]>maxConstraint)
            continue;
        }
        minValue=lineBuffer[icol];
        maxValue=lineBuffer[icol];
        isValid=true;
      }
    }
  }
  if(!isValid)
    throw(static_cast<std::string>("Warning: not initialized"));
  return(CE_None);
}


/**
 * @param histvector The reported histogram with counts per bin
 * @param min, max Only calculate histogram for values between min and max. If min>=max, calculate min and max from the image
 * @param nbin Number of bins used for calculating the histogram. If nbin is 0, the number of bins is  automatically calculated from min and max
 * @param theBand The band for which to calculate the histogram (start counting from 0)
 * @param kde Apply kernel density function for a Gaussian basis function
 * @return number of valid pixels in this dataset for the the selected band
 **/
double Jim::getHistogram(std::vector<double>& histvector, double& min, double& max, int& nbin, int theBand, bool kde){
  double minValue=0;
  double maxValue=0;

  if(min>=max)
    getMinMax(minValue,maxValue,theBand);
  else{
    minValue=min;
    maxValue=max;
  }
  if(min<max&&min>minValue)
    minValue=min;
  if(min<max&&max<maxValue)
    maxValue=max;
  min=minValue;
  max=maxValue;

  double sigma=0;
  if(kde){
    double meanValue=0;
    double stdDev=0;
    GDALProgressFunc pfnProgress;
    void* pProgressData;
    GDALRasterBand* rasterBand;
    rasterBand=getRasterBand(theBand);
    rasterBand->ComputeStatistics(0,&minValue,&maxValue,&meanValue,&stdDev,pfnProgress,pProgressData);
    //rest minvalue and MaxValue as ComputeStatistics does not account for nodata, scale and offset
    minValue=min;
    maxValue=max;

    if(m_scale.size()>theBand){
      stdDev*=m_scale[theBand];
    }
    sigma=1.06*stdDev*pow(getNvalid(theBand),-0.2);
  }

  double scale=0;
  if(maxValue>minValue){
    if(nbin==0)
      nbin=maxValue-minValue+1;
    scale=static_cast<double>(nbin-1)/(maxValue-minValue);
  }
  else
    nbin=1;
  //todo: replace assert with exception
  assert(nbin>0);
  if(histvector.size()!=nbin){
    histvector.resize(nbin);
    for(int i=0;i<nbin;histvector[i++]=0);
  }
  double nvalid=0;
  unsigned long int ninvalid=0;
  std::vector<double> lineBuffer(nrOfCol());
  for(int irow=0;irow<nrOfRow();++irow){
    readData(lineBuffer,irow,theBand);
    for(int icol=0;icol<nrOfCol();++icol){
      if(isNoData(lineBuffer[icol]))
        ++ninvalid;
      else if(lineBuffer[icol]>maxValue)
        ++ninvalid;
      else if(lineBuffer[icol]<minValue)
        ++ninvalid;
      else if(nbin==1)
        ++histvector[0];
      else{//scale to [0:nbin]
        if(sigma>0){
          //create kde for Gaussian basis function
          //todo: speed up by calculating first and last bin with non-zero contriubtion...
          //todo: calculate real surface below pdf by using gsl_cdf_gaussian_P(x-mean+binsize,sigma)-gsl_cdf_gaussian_P(x-mean,sigma)
          for(int ibin=0;ibin<nbin;++ibin){
            double icenter=minValue+static_cast<double>(maxValue-minValue)*(ibin+0.5)/nbin;
            double thePdf=gsl_ran_gaussian_pdf(lineBuffer[icol]-icenter, sigma);
            histvector[ibin]+=thePdf;
            nvalid+=thePdf;
          }
        }
        else{
          int theBin=static_cast<unsigned long int>(scale*(lineBuffer[icol]-minValue));
          //todo: replace assert with exception
          assert(theBin>=0);
          assert(theBin<nbin);
          ++histvector[theBin];
          ++nvalid;
        }
        // else if(lineBuffer[icol]==maxValue)
        //   ++histvector[nbin-1];
        // else
        //   ++histvector[static_cast<int>(static_cast<double>(lineBuffer[icol]-minValue)/(maxValue-minValue)*(nbin-1))];
      }
    }
  }
  // unsigned long int nvalid=nrOfCol()*nrOfRow()-ninvalid;
  return nvalid;
}

/**
 * @param range Sorted vector containing the range of image values
 * @param band The band for which to calculate the range
 **/
CPLErr Jim::getRange(std::vector<short>& range, int band)
{
  std::vector<short> lineBuffer(nrOfCol());
  range.clear();
  for(int irow=0;irow<nrOfRow();++irow){
    readData(lineBuffer,irow,band);
    for(int icol=0;icol<nrOfCol();++icol){
      if(find(range.begin(),range.end(),lineBuffer[icol])==range.end())
        range.push_back(lineBuffer[icol]);
    }
  }
  sort(range.begin(),range.end());
  return(CE_None);
}

/**
 * @param band The band for which to calculate the number of valid pixels
 * @return number of valid pixels in this dataset for the the selected band
 **/
unsigned long int Jim::getNvalid(int band)
{
  unsigned long int nvalid=0;
  if(m_noDataValues.size()){
    std::vector<double> lineBuffer(nrOfCol());
    for(int irow=0;irow<nrOfRow();++irow){
      readData(lineBuffer,irow,band);
      for(int icol=0;icol<nrOfCol();++icol){
        if(isNoData(lineBuffer[icol]))
          continue;
        else
          ++nvalid;
      }
    }
    return nvalid;
  }
  else
    return(nrOfCol()*nrOfRow());
}

/**
 * @param band The band for which to calculate the number of valid pixels
 * @return number of invalid pixels in this dataset for the the selected band
 **/
unsigned long int Jim::getNinvalid(int band)
{
  unsigned long int nvalid=0;
  if(m_noDataValues.size()){
    std::vector<double> lineBuffer(nrOfCol());
    for(int irow=0;irow<nrOfRow();++irow){
      readData(lineBuffer,irow,band);
      for(int icol=0;icol<nrOfCol();++icol){
        if(isNoData(lineBuffer[icol]))
          continue;
        else
          ++nvalid;
      }
    }
    return (nrOfCol()*nrOfRow())-nvalid;
  }
  else
    return(0);
}

/**
 * @param refX, refY Calculated reference pixel position in geo-refererenced coordinates
 * @param band The band for which to calculate the number of valid pixels
 **/

void Jim::getRefPix(double& refX, double &refY, int band)
{
  std::vector<double> lineBuffer(nrOfCol());
  double validCol=0;
  double validRow=0;
  int nvalidCol=0;
  int nvalidRow=0;
  for(int irow=0;irow<nrOfRow();++irow){
    readData(lineBuffer,irow,band);
    for(int icol=0;icol<nrOfCol();++icol){
      // bool valid=(find(m_noDataValues.begin(),m_noDataValues.end(),lineBuffer[icol])==m_noDataValues.end());
      // if(valid){
      if(!isNoData(lineBuffer[icol])){
        validCol+=icol+1;
        ++nvalidCol;
        validRow+=irow+1;
        ++nvalidRow;
      }
    }
  }
  if(isGeoRef()){
    //reference coordinate is lower left corner of pixel in center of gravity
    //we need geo coordinates for exactly this location: validCol(Row)/nvalidCol(Row)-0.5
    double cgravi=validCol/nvalidCol-0.5;
    double cgravj=validRow/nvalidRow-0.5;
    double refpixeli=floor(cgravi);
    double refpixelj=ceil(cgravj-1);
    //but image2geo provides location at center of pixel (shifted half pixel right down)
    image2geo(refpixeli,refpixelj,refX,refY);
    //refX and refY now refer to center of gravity pixel
    refX-=0.5*getDeltaX();//shift to left corner
    refY-=0.5*getDeltaY();//shift to lower left corner
  }
  else{
    refX=floor(validCol/nvalidCol-0.5);//left corner
    refY=floor(validRow/nvalidRow-0.5);//upper corner
    //shift to lower left corner of pixel
    refY+=1;
  }
}

//from Writer
/**
 * @param row Write a new block for caching this row (if needed)
 * @param band Band that must be written in cache
 * @return true if write was successful
 **/
CPLErr Jim::writeNewBlock(int row, int band)
{
  CPLErr returnValue=CE_None;
  if(m_gds == NULL){
    std::string errorString="Error in writeNewBlock";
    throw(errorString);
  }
  //todo: replace assert with exception
  //assert(row==m_end)
  if(m_end[band]>nrOfRow())
    m_end[band]=nrOfRow();
  //fetch raster band
  GDALRasterBand  *poBand;
  if(nrOfBand()<=band){
    std::string errorString="Error: band number exceeds available bands in writeNewBlock";
    throw(errorString);
  }
  poBand = m_gds->GetRasterBand(band+1);//GDAL uses 1 based index
  returnValue=poBand->RasterIO(GF_Write,0,m_begin[band],nrOfCol(),m_end[band]-m_begin[band],m_data[band],nrOfCol(),m_end[band]-m_begin[band],getGDALDataType(),0,0);
  if(m_begin[band]+m_blockSize<nrOfRow()){
    m_begin[band]+=m_blockSize;//m_begin points to first line in block that will be written next
    m_end[band]=m_begin[band]+m_blockSize;//m_end points to last line in block that will be written next
  }
  if(m_end[band]>nrOfRow())
    m_end[band]=nrOfRow();
  return(returnValue);//new block was written
}


CPLErr Jim::write(){
  //write, but do not reset
  if(m_data.size()&&m_filename.size()){
    for(int iband=0;iband<nrOfBand();++iband)
      writeNewBlock(nrOfRow(),iband);
  }
  char **papszOptions=NULL;
  for(std::vector<std::string>::const_iterator optionIt=m_options.begin();optionIt!=m_options.end();++optionIt)
    papszOptions=CSLAddString(papszOptions,optionIt->c_str());
  if(papszOptions)
    CSLDestroy(papszOptions);
  if(m_gds)
    GDALClose(m_gds);
  m_gds=0;
  // reset();
}

///write to file without reset (keep data in memory)
/**
 * @param filename Filename of the output raster dataset
 * @param oformat Image type. Currently only those formats where the drivers support the Create method can be written
 * @param co Creation option for output file. Multiple options can be specified.
 * @param nodata Nodata value to put in image.
 * @return CE_None if successful, else CE_Failure
 **/
CPLErr Jim::write(app::AppFactory &app){
  setFile(app);
  write();
}

///Create a JSON string from a Jim image
std::string Jim::jim2json(){
  Json::Value custom;
  custom["size"]=static_cast<int>(1);
  int iimg=0;
  Json::Value image;
  image["path"]=getFileName();
  std::string wktString=getProjectionRef();
  std::string key("EPSG");
  std::size_t foundEPSG=wktString.rfind(key);
  std::string fromEPSG=wktString.substr(foundEPSG);//EPSG","32633"]]'
  std::size_t foundFirstDigit=fromEPSG.find_first_of("0123456789");
  std::size_t foundLastDigit=fromEPSG.find_last_of("0123456789");
  std::string epsgString=fromEPSG.substr(foundFirstDigit,foundLastDigit-foundFirstDigit+1);
  image["epsg"]=atoi(epsgString.c_str());
  std::ostringstream os;
  os << iimg++;
  custom["0"]=image;
  Json::FastWriter fastWriter;
  return(fastWriter.write(custom));
}

std::shared_ptr<Jim> Jim::clone(bool copyData) {
  std::shared_ptr<Jim> pJim=std::dynamic_pointer_cast<Jim>(cloneImpl(copyData));
  if(pJim){
    return(pJim);
  }
  else{
    std::cerr << "Warning: static pointer cast may slice object" << std::endl;
    return(std::static_pointer_cast<Jim>(cloneImpl(copyData)));
  }
}

// /**
//  * @param filename Open a raster dataset with this filename
//  * @param imgSrc Use this source image as a template to copy image attributes
//  * @param options Creation options
//  **/
// void Jim::open(const std::string& filename, const ImgReaderGdal& imgSrc, const std::vector<std::string>& options)
// {
//   m_ncol=imgSrc.nrOfCol();
//   m_nrow=imgSrc.nrOfRow();
//   m_nband=imgSrc.nrOfBand();
//   m_dataType=imgSrc.getDataType();
//   setFile(filename,imgSrc,options);
//   // m_filename=filename;
//   // m_options=options;
//   // setDriver(imgSrc);
// }

/**
 * @param filename Open a raster dataset with this filename
 * @param imgSrc Use this source image as a template to copy image attributes
 * @param options Creation options
 **/
CPLErr Jim::open(const std::string& filename, const Jim& imgSrc, const std::vector<std::string>& options)
{
  return(open(filename,imgSrc,0,options));
}

/**
 * @param filename Open a raster dataset with this filename
 * @param imgSrc Use this source image as a template to copy image attributes
 * @param memory Available memory to cache image raster data (in MB)
 * @param options Creation options
 **/
CPLErr Jim::open(const std::string& filename, const Jim& imgSrc, unsigned int memory, const std::vector<std::string>& options)
{
  m_ncol=imgSrc.nrOfCol();
  m_nrow=imgSrc.nrOfRow();
  m_nband=imgSrc.nrOfBand();
  m_nplane=imgSrc.nrOfPlane();
  m_dataType=imgSrc.getDataType();
  setProjection(imgSrc.getProjection());
  copyGeoTransform(imgSrc);
  if(setFile(filename,imgSrc.getImageType(),memory,options)!=CE_None)
    return(CE_Failure);
  m_gds->SetMetadata(imgSrc.getMetadata());
  if(imgSrc.getColorTable()!=NULL)
    setColorTable(imgSrc.getColorTable());
  return(CE_None);
}

/**
 * @param imgSrc Use this source image as a template to copy image attributes
 * @param copyData Copy data from source image when true
 **/
CPLErr Jim::open(Jim& imgSrc, bool copyData)
{
  m_ncol=imgSrc.nrOfCol();
  m_nrow=imgSrc.nrOfRow();
  m_nband=imgSrc.nrOfBand();
  m_nplane=imgSrc.nrOfPlane();
  m_dataType=imgSrc.getDataType();
  setProjection(imgSrc.getProjection());
  copyGeoTransform(imgSrc);
  imgSrc.getNoDataValues(m_noDataValues);
  imgSrc.getScale(m_scale);
  imgSrc.getOffset(m_offset);
  initMem(0);
  for(int iband=0;iband<m_nband;++iband){
    m_begin[iband]=0;
    m_end[iband]=m_begin[iband]+m_blockSize;
    if(copyData)
      imgSrc.copyData(m_data[iband],iband);
  }
  //todo: check if filename needs to be set, but as is it is used for writing, I don't think so.
  // if(imgSrc.getFileName()!=""){
  //   m_filename=imgSrc.getFileName();
  // std::cerr << "Warning: filename not set, dataset not defined yet" << std::endl;
  // }
  return(CE_None);
}

/**
 * @param imgSrc Use this source image as a template to copy image attributes
 * @param copyData Copy data from source image when true
 **/
// CPLErr Jim::open(std::shared_ptr<Jim> imgSrc, bool copyData)
// {
//   m_ncol=imgSrc->nrOfCol();
//   m_nrow=imgSrc->nrOfRow();
//   m_nband=imgSrc->nrOfBand();
//   m_dataType=imgSrc->getDataType();
//   setProjection(imgSrc->getProjection());
//   copyGeoTransform(imgSrc);
//   imgSrc->getNoDataValues(m_noDataValues);
//   imgSrc->getScale(m_scale);
//   imgSrc->getOffset(m_offset);
//   if(m_filename!=""){
//     m_writeMode=true;
//     registerDriver();
//   }
//   else
//     m_writeMode=false;
//   initMem(0);
//   for(int iband=0;iband<m_nband;++iband){
//     m_begin[iband]=0;
//     m_end[iband]=m_begin[iband]+m_blockSize;
//     if(copyData){
//       std::vector<double> lineInput(nrOfCol());
//       for(int iband=0;iband<nrOfBand();++iband){
//         for(int irow=0;irow<nrOfRow();++irow){
//           imgSrc->readData(lineInput,irow,iband,NEAR);
//           writeData(lineInput,irow,iband);
//         }
//       }
//       // imgSrc->copyData(m_data[iband],iband);
//     }
//   }
//   //todo: check if filename needs to be set, but as is it is used for writing, I don't think so.
//   // if(imgSrc->getFileName()!=""){
//   //   m_filename=imgSrc->getFileName();
//     // std::cerr << "Warning: filename not set, dataset not defined yet" << std::endl;
//   // }
//   return(CE_None);
// }

// /**
//  * @param filename Open a raster dataset with this filename
//  * @param ncol Number of columns in image
//  * @param nrow Number of rows in image
//  * @param nband Number of bands in image
//  * @param dataType The data type of the image (one of the GDAL supported datatypes: GDT_Byte, GDT_[U]Int[16|32], GDT_Float[32|64])
//  * @param imageType Image type. Currently only those formats where the drivers support the Create method can be written
//  * @param options Creation options
//  **/
// void Jim::open(const std::string& filename, int ncol, int nrow, unsigned int nband, const GDALDataType& dataType, const std::string& imageType, const std::vector<std::string>& options)
// {
//   m_ncol = ncol;
//   m_nrow = nrow;
//   m_nband = nband;
//   m_dataType = dataType;
//   setFile(filename,imageType,options);
//   // m_filename = filename;
//   // m_options=options;
//   // setDriver(imageType);
// }

/**
 * @param filename Open a raster dataset with this filename
 * @param ncol Number of columns in image
 * @param nrow Number of rows in image
 * @param nband Number of bands in image
 * @param dataType The data type of the image (one of the GDAL supported datatypes: GDT_Byte, GDT_[U]Int[16|32], GDT_Float[32|64])
 * @param imageType Image type. Currently only those formats where the drivers support the Create method can be written
 * @param memory Available memory to cache image raster data (in MB)
 * @param options Creation options
 **/
CPLErr Jim::open(const std::string& filename, int ncol, int nrow, int nband, const GDALDataType& dataType, const std::string& imageType, unsigned int memory, const std::vector<std::string>& options)
{
  m_ncol = ncol;
  m_nrow = nrow;
  m_nband = nband;
  m_nplane = 1;
  m_dataType = dataType;
  return(setFile(filename,imageType,memory,options));
}

/**
 * @param ncol Number of columns in image
 * @param nrow Number of rows in image
 * @param nband Number of bands in image
 * @param dataType The data type of the image (one of the GDAL supported datatypes: GDT_Byte, GDT_[U]Int[16|32], GDT_Float[32|64])
 **/
CPLErr Jim::open(int ncol, int nrow, int nband, const GDALDataType& dataType)
{
  m_ncol = ncol;
  m_nrow = nrow;
  m_nplane=1;
  m_nband = nband;
  m_dataType = dataType;
  initMem(0);
  for(int iband=0;iband<m_nband;++iband){
    m_begin[iband]=0;
    m_end[iband]=m_begin[iband]+m_blockSize;
  }
  if(m_filename!=""){
    // m_writeMode=true;
    m_access=WRITE;
    registerDriver();
  }
  return(CE_None);
}

///Open an image for writing
 CPLErr Jim::open(int ncol, int nrow, int nband, int nplane, const GDALDataType& dataType){
   m_ncol=ncol;
   m_nrow=nrow;
   m_nplane=nplane;
   m_nband=nband;
   m_dataType=dataType;
   m_data.resize(m_nband);
   m_begin.resize(m_nband);
   m_end.resize(m_nband);
   m_blockSize=nrow;//memory contains entire image and has been read already
   initMem(0);
   for(int iband=0;iband<m_nband;++iband){
     m_begin[iband]=0;
     m_end[iband]=m_begin[iband]+m_blockSize;
   }
   if(m_filename!=""){
     // m_writeMode=true;
     m_access=WRITE;
     registerDriver();
   }
   return(CE_None);
 }

///set file attributes for writing
/**
 * @param filename Open a raster dataset with this filename
 * @param imageType Image type. Currently only those formats where the drivers support the Create method can be written
 **/
CPLErr Jim::setFile(const std::string& filename, const std::string& imageType, unsigned int memory, const std::vector<std::string>& options)
{
  CPLErr returnValue=CE_None;
  m_access=WRITE;
  // m_writeMode=true;
  m_filename=filename;
  m_options=options;
  m_imageType=imageType;
  if(nrOfCol()&&nrOfRow()&&nrOfBand()){
    registerDriver();
    if(m_data.empty())
      initMem(memory);
    for(int iband=0;iband<nrOfBand();++iband){
      m_begin[iband]=0;
      m_end[iband]=m_begin[iband]+m_blockSize;
    }
    if(m_noDataValues.size()){
      for(int iband=0;iband<nrOfBand();++iband){
        returnValue=GDALSetNoDataValue(m_noDataValues[0],iband);
        if(returnValue!=CE_None)
          break;
      }
    }
  }
  return(returnValue);
}

///set file attributes for writing
/**
 * @param filename Filename of the output raster dataset
 * @param oformat Image type. Currently only those formats where the drivers support the Create method can be written
 * @param co Creation option for output file. Multiple options can be specified.
 * @param nodata Nodata value to put in image.
 **/
CPLErr Jim::setFile(app::AppFactory &app){
  Optionjl<std::string> input_opt("fn", "filename", "filename");
  Optionjl<std::string>  oformat_opt("of", "oformat", "Output image format (see also gdal_translate).","GTiff");
  Optionjl<std::string> option_opt("co", "co", "Creation option for output file. Multiple options can be specified.");
  Optionjl<double> nodata_opt("nodata", "nodata", "Nodata value to put in image.");
  Optionjl<string> colorTable_opt("ct", "ct", "color table (file with 5 columns: id R G B ALFA (0: transparent, 255: solid)");
  Optionjl<string> description_opt("d", "description", "Set image description");
  Optionjl<unsigned long int>  memory_opt("mem", "mem", "Buffer size (in MB) to read image data blocks in memory",0,1);

  option_opt.setHide(1);
  memory_opt.setHide(1);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  doProcess=input_opt.retrieveOption(app);
  oformat_opt.retrieveOption(app);
  option_opt.retrieveOption(app);
  nodata_opt.retrieveOption(app);
  colorTable_opt.retrieveOption(app);
  description_opt.retrieveOption(app);
  memory_opt.retrieveOption(app);
  if(!doProcess){
    std::cout << std::endl;
    std::ostringstream helpStream;
    helpStream << "help info: ";
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
  if(nodata_opt.size())
    setNoData(nodata_opt);
  CPLErr result=setFile(input_opt[0],oformat_opt[0],memory_opt[0],option_opt);
  if(colorTable_opt.size()){
    if(colorTable_opt[0]!="none"||colorTable_opt[0]!="None")
      setColorTable(colorTable_opt[0]);
  }
  if(description_opt.size()){
    if(description_opt[0]!="none"||description_opt[0]!="None")
      setImageDescription(description_opt[0]);
  }
  return(result);
}

///Copy data
CPLErr Jim::copyData(void* data, int band){
  memcpy(data,m_data[band],getDataTypeSizeBytes()*nrOfCol()*m_blockSize*nrOfPlane());
  // memcpy(data,m_data[band],(GDALGetDataTypeSize(getDataType())>>3)*nrOfCol()*m_blockSize);
  return(CE_None);
};

 std::shared_ptr<Jim> Jim::createImg() {
   return(std::make_shared<Jim>());
 };

 std::shared_ptr<Jim> Jim::createImg(app::AppFactory &theApp){
   std::shared_ptr<Jim> pJim=std::make_shared<Jim>(theApp);
   return(pJim);
 }

 std::shared_ptr<Jim> Jim::createImg(const std::shared_ptr<Jim> pSrc, bool copyData){
   std::shared_ptr<Jim> pJim=std::make_shared<Jim>(*pSrc,copyData);
   return(pJim);
 }

 /* ///Create new shared pointer to Jim object */
 /**
  * @param input (type: std::string) input filename
  * @return shared pointer to new Jim object
  **/
 std::shared_ptr<Jim> Jim::createImg(const std::string filename, bool readData, unsigned int memory){
   std::shared_ptr<Jim> pJim=std::make_shared<Jim>(filename,readData,memory);
   // std::shared_ptr<Jim> pJim=std::make_shared<Jim>(filename,memory);
   return(pJim);
 }


/**
 * @param metadata Set this metadata when writing the image (if supported byt the driver)
 **/
CPLErr Jim::setMetadata(char** metadata)
{
  if(m_gds){
    m_gds->SetMetadata(metadata);
    return(CE_None);
  }
  else
    return(CE_Warning);
}

//default projection: ETSR-LAEA
// std::string Jim::setProjection(void)
// {
//   std::string theProjection;
//   OGRSpatialReference oSRS;
//   char *pszSRS_WKT = NULL;
//   //// oSRS.importFromEPSG(3035);
//   oSRS.SetGeogCS("ETRS89","European_Terrestrial_Reference_System_1989","GRS 1980",6378137,298.2572221010042,"Greenwich",0,"degree",0.0174532925199433);
//   // cout << setprecision(16) << "major axis: " << oSRS.GetSemiMajor(NULL) << endl;//notice that major axis can be set to a different value than the default to the well known standard corresponding to the name (European_Terrestrial_Reference_System_1989), but that new value, while recognized by GetSemiMajor, will not be written in the geotiff tag!
//   oSRS.SetProjCS( "ETRS89 / ETRS-LAEA" );
//   oSRS.SetLAEA(52,10,4321000,3210000);
//   oSRS.exportToWkt( &pszSRS_WKT );
//   theProjection=pszSRS_WKT;
//   CPLFree( pszSRS_WKT );
//   assert(m_gds);
//   m_gds->SetProjection(theProjection.c_str());
//   return(theProjection);
// }


/**
 * @param filename ASCII file containing 5 columns: index R G B ALFA (0:transparent, 255:solid)
 * @param band band number to set color table (starting counting from 0)
 **/
CPLErr Jim::setColorTable(const std::string& filename, int band)
{
  //todo: fool proof table in file (no checking currently done...)
  std::ifstream ftable(filename.c_str(),std::ios::in);
  std::string line;
//   poCT=new GDALColorTable();
  GDALColorTable colorTable;
  short nline=0;
  while(getline(ftable,line)){
    ++nline;
    std::istringstream ist(line);
    GDALColorEntry sEntry;
    short id;
    ist >> id >> sEntry.c1 >> sEntry.c2 >> sEntry.c3 >> sEntry.c4;
    colorTable.SetColorEntry(id,&sEntry);
  }
  if(m_gds){
    (m_gds->GetRasterBand(band+1))->SetColorTable(&colorTable);
    return(CE_None);
  }
  else
    return(CE_Warning);
}

/**
 * @param colorTable Instance of the GDAL class GDALColorTable
 * @param band band number to set color table (starting counting from 0)
 **/
CPLErr Jim::setColorTable(GDALColorTable* colorTable, int band)
{
  if(m_gds){
    (m_gds->GetRasterBand(band+1))->SetColorTable(colorTable);
    return(CE_None);
  }
  else
    return(CE_Warning);
}

// //write an entire image from memory to file
// bool Jim::writeData(void* pdata, const GDALDataType& dataType, int band){
//   //fetch raster band
//   GDALRasterBand  *poBand;
//   if(band>=nrOfBand()+1){
//     std::ostringstream s;
//     s << "band (" << band << ") exceeds nrOfBand (" << nrOfBand() << ")";
//     throw(s.str());
//   }
//   poBand = m_gds->GetRasterBand(band+1);//GDAL uses 1 based index
//   poBand->RasterIO(GF_Write,0,0,nrOfCol(),nrOfRow(),pdata,nrOfCol(),nrOfRow(),dataType,0,0);
//   return true;
// }

/**
 * @param ogrReader Vector dataset as an instance of the ImgReaderOgr that must be rasterized
 * @param burnValues Values to burn into raster cells (one value for each band)
 * @param eoption special options controlling rasterization (ATTRIBUTE|CHUNKYSIZE|ALL_TOUCHED|BURN_VALUE_FROM|MERGE_ALG)
 * "ATTRIBUTE":
 * Identifies an attribute field on the features to be used for a burn in value. The value will be burned into all output bands. If specified, padfLayerBurnValues will not be used and can be a NULL pointer.
 * "CHUNKYSIZE":
 * The height in lines of the chunk to operate on. The larger the chunk size the less times we need to make a pass through all the shapes. If it is not set or set to zero the default chunk size will be used. Default size will be estimated based on the GDAL cache buffer size using formula: cache_size_bytes/scanline_size_bytes, so the chunk will not exceed the cache.
 * "ALL_TOUCHED":
 * May be set to TRUE to set all pixels touched by the line or polygons, not just those whose center is within the polygon or that are selected by brezenhams line algorithm. Defaults to FALSE.
 "BURN_VALUE_
 * May be set to "Z" to use the Z values of the geometries. The value from padfLayerBurnValues or the attribute field value is added to this before burning. In default case dfBurnValue is burned as it is. This is implemented properly only for points and lines for now. Polygons will be burned using the Z value from the first point. The M value may be supported in the future.
 * "MERGE_ALG":
 * May be REPLACE (the default) or ADD. REPLACE results in overwriting of value, while ADD adds the new value to the existing raster, suitable for heatmaps for instance.
 * @param layernames Names of the vector dataset layers to process. Leave empty to process all layers
 **/
// CPLErr Jim::rasterizeOgr(ImgReaderOgr& ogrReader, const std::vector<double>& burnValues, const std::vector<std::string>& eoption, const std::vector<std::string>& layernames ){
//   std::vector<int> bands;
//   if(burnValues.empty()&&eoption.empty()){
//     std::string errorString="Error: either burn values or control options must be provided";
//     throw(errorString);
//   }
//   for(int iband=0;iband<nrOfBand();++iband)
//     bands.push_back(iband+1);
//   std::vector<OGRLayerH> layers;
//   int nlayer=0;

//   std::vector<double> burnBands;//burn values for all bands in a single layer
//   std::vector<double> burnLayers;//burn values for all bands and all layers
//   if(burnValues.size()){
//     burnBands=burnValues;
//     while(burnBands.size()<nrOfBand())
//       burnBands.push_back(burnValues[0]);
//   }
//   for(int ilayer=0;ilayer<ogrReader.getLayerCount();++ilayer){
//     std::string currentLayername=ogrReader.getLayer(ilayer)->GetName();
//     if(layernames.size())
//       if(find(layernames.begin(),layernames.end(),currentLayername)==layernames.end())
//         continue;
//     std::cout << "processing layer " << currentLayername << std::endl;
//     layers.push_back((OGRLayerH)ogrReader.getLayer(ilayer));
//     ++nlayer;
//     if(burnValues.size()){
//       for(int iband=0;iband<nrOfBand();++iband)
//         burnLayers.insert(burnLayers.end(),burnBands.begin(),burnBands.end());
//     }
//   }
//   void* pTransformArg=NULL;
//   GDALProgressFunc pfnProgress=NULL;
//   void* pProgressArg=NULL;

//   char **coptions=NULL;
//   for(std::vector<std::string>::const_iterator optionIt=eoption.begin();optionIt!=eoption.end();++optionIt)
//     coptions=CSLAddString(coptions,optionIt->c_str());

//   if(eoption.size()){
//     if(GDALRasterizeLayers( (GDALDatasetH)m_gds,nrOfBand(),&(bands[0]),layers.size(),&(layers[0]),NULL,pTransformArg,NULL,coptions,pfnProgress,pProgressArg)!=CE_None){
//       std::string errorString(CPLGetLastErrorMsg());
//       throw(errorString);
//     }
//   }
//   else if(burnValues.size()){
//     if(GDALRasterizeLayers( (GDALDatasetH)m_gds,nrOfBand(),&(bands[0]),layers.size(),&(layers[0]),NULL,pTransformArg,&(burnLayers[0]),NULL,pfnProgress,pProgressArg)!=CE_None){
//       std::string errorString(CPLGetLastErrorMsg());
//       throw(errorString);
//     }
//   }
//   else{
//     std::string errorString="Error: either attribute fieldname or burn values must be set to rasterize vector dataset";
//     throw(errorString);
//   }
//   //do not overwrite m_gds with what is in m_data
//   m_access=READ_ONLY;
//   // m_writeMode=false;
//   return(CE_None);
// }

// /**
//  * @param ogrReader Vector dataset as an instance of the ImgReaderOgr that must be rasterized
//  * @param burnValues Values to burn into raster cells (one value for each band)
//  * @param layernames Names of the vector dataset layers to process. Leave empty to process all layers
//  **/
// CPLErr Jim::rasterizeBuf(ImgReaderOgr& ogrReader, double burnValue, const std::vector<std::string>& layernames ){
//   if(m_blockSize<nrOfRow()){
//     std::ostringstream s;
//     s << "Error: increase memory to perform rasterize in entirely in buffer (now at " << 100.0*m_blockSize/nrOfRow() << "%)";
//     throw(s.str());
//   }
//   std::vector<OGRLayerH> layers;
//   int nlayer=0;

//   std::vector<double> burnBands;//burn values for all bands in a single layer
//   while(burnBands.size()<nrOfBand())
//     burnBands.push_back(burnValue);
//   for(int ilayer=0;ilayer<ogrReader.getLayerCount();++ilayer){
//     std::string currentLayername=ogrReader.getLayer(ilayer)->GetName();
//     if(layernames.size())
//       if(find(layernames.begin(),layernames.end(),currentLayername)==layernames.end())
//         continue;
//     std::cout << "processing layer " << currentLayername << std::endl;
//     layers.push_back((OGRLayerH)ogrReader.getLayer(ilayer));
//     ++nlayer;
//   }
//   void* pTransformArg=NULL;
//   GDALProgressFunc pfnProgress=NULL;
//   void* pProgressArg=NULL;

//   for(int iband=0;iband<nrOfBand();++iband){
//     Vector2d<double> initBlock(nrOfRow(),nrOfCol());
//     writeDataBlock(initBlock,0,nrOfCol()-1,0,nrOfRow()-1,iband);
//     double gt[6];
//     getGeoTransform(gt);
//     if(GDALRasterizeLayersBuf(m_data[iband],nrOfCol(),nrOfRow(),getGDALDataType(),getDataTypeSizeBytes(),0,layers.size(),&(layers[0]), getProjectionRef().c_str(),gt,NULL, pTransformArg, burnBands[iband],NULL,pfnProgress,pProgressArg)!=CE_None){
//       std::string errorString(CPLGetLastErrorMsg());
//       throw(errorString);
//     }
//   }
//   return(CE_None);
// }

CPLErr Jim::rasterizeBuf(const std::string& ogrFilename){
  VectorOgr ogrReader;
  std::vector<std::string> layernames;
  layernames.clear();
  ogrReader.open(ogrFilename,layernames,true);
  CPLErr retValue=CE_None;
  retValue=rasterizeBuf(ogrReader,1);
  ogrReader.close();
  return(retValue);
}

/**
 * @param ogrReader Vector dataset as an instance of the ImgReaderOgr that must be rasterized
 * @param burnValue Value to burn into raster cells
 * @param eoption special options controlling rasterization (ATTRIBUTE|CHUNKYSIZE|ALL_TOUCHED|BURN_VALUE_FROM|MERGE_ALG)
 * "ATTRIBUTE":
 * Identifies an attribute field on the features to be used for a burn in value. The value will be burned into all output bands. If specified, padfLayerBurnValues will not be used and can be a NULL pointer.
 * "ALL_TOUCHED":
 * May be set to TRUE to set all pixels touched by the line or polygons, not just those whose center is within the polygon or that are selected by brezenhams line algorithm. Defaults to FALSE.
 "BURN_VALUE_FROM":
 * May be set to "Z" to use the Z values of the geometries. The value from padfLayerBurnValues or the attribute field value is added to this before burning. In default case dfBurnValue is burned as it is. This is implemented properly only for points and lines for now. Polygons will be burned using the Z value from the first point. The M value may be supported in the future.
 * "MERGE_ALG":
 * May be REPLACE (the default) or ADD. REPLACE results in overwriting of value, while ADD adds the new value to the existing raster, suitable for heatmaps for instance.
 * @param layernames Names of the vector dataset layers to process. Leave empty to process all layers
 **/
// CPLErr Jim::rasterizeBuf(ImgReaderOgr& ogrReader, double burnValue, const std::vector<std::string>& eoption, const std::vector<std::string>& layernames ){
//   if(m_blockSize<nrOfRow()){
//     std::ostringstream s;
//     s << "Error: increase memory to perform rasterize in entirely in buffer (now at " << 100.0*m_blockSize/nrOfRow() << "%)";
//     throw(s.str());
//   }
//   std::vector<OGRLayerH> layers;
//   int nlayer=0;

//   for(int ilayer=0;ilayer<ogrReader.getLayerCount();++ilayer){
//     std::string currentLayername=ogrReader.getLayer(ilayer)->GetName();
//     if(layernames.size())
//       if(find(layernames.begin(),layernames.end(),currentLayername)==layernames.end())
//         continue;
//     std::cout << "processing layer " << currentLayername << std::endl;
//     layers.push_back((OGRLayerH)ogrReader.getLayer(ilayer));
//     ++nlayer;
//   }
//   void* pTransformArg=NULL;
//   GDALProgressFunc pfnProgress=NULL;
//   void* pProgressArg=NULL;

//   char **coptions=NULL;
//   for(std::vector<std::string>::const_iterator optionIt=eoption.begin();optionIt!=eoption.end();++optionIt)
//     coptions=CSLAddString(coptions,optionIt->c_str());

//   for(int iband=0;iband<nrOfBand();++iband){
//     Vector2d<double> initBlock(nrOfRow(),nrOfCol());
//     writeDataBlock(initBlock,0,nrOfCol()-1,0,nrOfRow()-1,iband);
//     double gt[6];
//     getGeoTransform(gt);
//     if(GDALRasterizeLayersBuf(m_data[iband],nrOfCol(),nrOfRow(),getGDALDataType(),getDataTypeSizeBytes(),0,layers.size(),&(layers[0]), getProjectionRef().c_str(),gt,NULL, pTransformArg, burnValue, coptions,pfnProgress,pProgressArg)!=CE_None){
//       std::string errorString(CPLGetLastErrorMsg());
//       throw(errorString);
//     }
//   }
//   return(CE_None);
// }

//todo: support transform
///Warning: this function cannot be exported via SWIG to Python as such, as it is destructive
CPLErr Jim::rasterizeBuf(VectorOgr& ogrReader, double burnValue, const std::vector<std::string>& eoption, const std::vector<std::string>& layernames ){
  if(m_blockSize<nrOfRow()){
    std::ostringstream s;
    s << "Error: increase memory to perform rasterize in entirely in buffer (now at " << 100.0*m_blockSize/nrOfRow() << "%)";
    throw(s.str());
  }
  std::vector<OGRLayerH> layers;
  if(layernames.size()){
    for(std::vector<std::string>::const_iterator nit=layernames.begin();nit!=layernames.end();++nit)
      layers.push_back((OGRLayerH)ogrReader.getLayer(*nit));
  }
  else{
    for(size_t ilayer=0;ilayer<ogrReader.getLayerCount();++ilayer)
      layers.push_back((OGRLayerH)ogrReader.getLayer(ilayer));
  }
  void* pTransformArg=NULL;
  GDALProgressFunc pfnProgress=NULL;
  void* pProgressArg=NULL;

  char **coptions=NULL;
  for(std::vector<std::string>::const_iterator optionIt=eoption.begin();optionIt!=eoption.end();++optionIt)
    coptions=CSLAddString(coptions,optionIt->c_str());

#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
  for(int iband=0;iband<nrOfBand();++iband){
    Vector2d<double> initBlock(nrOfRow(),nrOfCol());
    writeDataBlock(initBlock,0,nrOfCol()-1,0,nrOfRow()-1,iband);
    vector<double> gt(6);
    getGeoTransform(gt);
    if(GDALRasterizeLayersBuf(m_data[iband],nrOfCol(),nrOfRow(),getGDALDataType(),getDataTypeSizeBytes(),0,layers.size(),&(layers[0]), getProjectionRef().c_str(),&gt[0],NULL, pTransformArg, burnValue, coptions,pfnProgress,pProgressArg)!=CE_None){
      std::string errorString(CPLGetLastErrorMsg());
      throw(errorString);
    }
  }
  return(CE_None);
}

/**
 *
 *
 * @param t1 minimum threshold
 * @param t2 maximum threshold
 *
 * @return output image
 */
std::shared_ptr<Jim> Jim::setThreshold(double t1, double t2){
  std::shared_ptr<Jim> imgWriter=Jim::createImg();
  setThreshold(*imgWriter, t1, t2);
  return(imgWriter);
}

/**
 *
 *
 * @param t1 minimum threshold
 * @param t2 maximum threshold
 *
 * @return output image
 */
std::shared_ptr<Jim> Jim::setAbsThreshold(double t1, double t2){
  std::shared_ptr<Jim> imgWriter=Jim::createImg();
  setAbsThreshold(*imgWriter, t1, t2);
  return(imgWriter);
}

/**
 *
 *
 * @param t1 minimum threshold
 * @param t2 maximum threshold
 * @param bg value if outside thresholds
 *
 * @return CE_None if success, CE_Failure if failed
 */
CPLErr Jim::setThreshold(Jim& imgWriter, double t1, double t2){
  try{
    imgWriter.open(*this,false);
    if(m_noDataValues.empty()){
      std::string errorString="Error: no data value not set";
      throw(errorString);
    }
#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
    for(int iband=0;iband<nrOfBand();++iband){
      std::vector<double> lineInput(nrOfCol());
      for(int irow=0;irow<nrOfRow();++irow){
        readData(lineInput,irow,iband);
        for(int icol=0;icol<nrOfCol();++icol){
          if(lineInput[icol]>=t1&&lineInput[icol]<=t2)
            continue;
          else
            lineInput[icol]=m_noDataValues[0];
        }
        imgWriter.writeData(lineInput,irow,iband);
      }
    }
  }
  catch(std::string errorstring){
    std::cerr << errorstring << std::endl;
    throw;
  }
  catch(...){
    throw;
  }
  return(CE_None);
}

/**
 *
 *
 * @param t1 minimum threshold
 * @param t2 maximum threshold
 * @param bg value if outside thresholds
 *
 * @return CE_None if success, CE_Failure if failed
 */
CPLErr Jim::setAbsThreshold(Jim& imgWriter, double t1, double t2){
  try{
    imgWriter.open(*this,false);
    if(m_noDataValues.empty()){
      std::string errorString="Error: no data value not set";
      throw(errorString);
    }
#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
    for(int iband=0;iband<nrOfBand();++iband){
      std::vector<double> lineInput(nrOfCol());
      for(int irow=0;irow<nrOfRow();++irow){
        readData(lineInput,irow,iband);
        for(int icol=0;icol<nrOfCol();++icol){
          if(fabs(lineInput[icol])>=t1&&fabs(lineInput[icol])<=t2)
            continue;
          else
            lineInput[icol]=m_noDataValues[0];
        }
        imgWriter.writeData(lineInput,irow,iband);
      }
    }
  }
  catch(std::string errorstring){
    std::cerr << errorstring << std::endl;
    throw;
  }
  catch(...){
    throw;
  }
  return(CE_None);
}

/**
 *
 *
 * @param t1 minimum threshold
 * @param t2 maximum threshold
 * @param value if within thresholds (else set to no data value)
 *
 * @return output image
 */
std::shared_ptr<Jim> Jim::setThreshold(double t1, double t2, double value){
  std::shared_ptr<Jim> imgWriter=Jim::createImg();
  setThreshold(*imgWriter, t1, t2, value);
  return(imgWriter);
}

/**
 *
 *
 * @param t1 minimum threshold
 * @param t2 maximum threshold
 * @param value if within thresholds (else set to no data value)
 *
 * @return output image
 */
std::shared_ptr<Jim> Jim::setAbsThreshold(double t1, double t2, double value){
  std::shared_ptr<Jim> imgWriter=Jim::createImg();
  setAbsThreshold(*imgWriter, t1, t2, value);
  return(imgWriter);
}

std::shared_ptr<Jim> Jim::setThreshold(app::AppFactory &theApp){
  std::shared_ptr<Jim> imgWriter=Jim::createImg();
  setThreshold(*imgWriter, theApp);
  return(imgWriter);
}
/**
 *
 *
 * @param t1 minimum threshold
 * @param t2 maximum threshold
 * @param value if within thresholds (else set to no data value)
 *
 * @return CE_None if success, CE_Failure if failed
 */
CPLErr Jim::setThreshold(Jim& imgWriter, double t1, double t2, double value){
  try{
    imgWriter.open(*this,false);
    if(m_noDataValues.empty()){
      std::string errorString="Error: no data value not set";
      throw(errorString);
    }
#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
    for(int iband=0;iband<nrOfBand();++iband){
      std::vector<double> lineInput(nrOfCol());
      for(int irow=0;irow<nrOfRow();++irow){
        readData(lineInput,irow,iband);
        for(int icol=0;icol<nrOfCol();++icol){
          if((lineInput[icol]>=t1)&&(lineInput[icol]<=t2))
            lineInput[icol]=value;
          else
            lineInput[icol]=m_noDataValues[0];
        }
        imgWriter.writeData(lineInput,irow,iband);
      }
    }
  }
  catch(std::string errorstring){
    std::cerr << errorstring << std::endl;
    throw;
  }
  catch(...){
    throw;
  }
  return(CE_None);
}

/**
 *
 *
 * @param t1 minimum threshold
 * @param t2 maximum threshold
 * @param value if within thresholds (else set to no data value)
 *
 * @return CE_None if success, CE_Failure if failed
 */
CPLErr Jim::setAbsThreshold(Jim& imgWriter, double t1, double t2, double value){
  try{
    imgWriter.open(*this,false);
    if(m_noDataValues.empty()){
      std::string errorString="Error: no data value not set";
      throw(errorString);
    }
#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
    for(int iband=0;iband<nrOfBand();++iband){
      std::vector<double> lineInput(nrOfCol());
      for(int irow=0;irow<nrOfRow();++irow){
        readData(lineInput,irow,iband);
        for(int icol=0;icol<nrOfCol();++icol){
          if((fabs(lineInput[icol])>=t1)&&(fabs(lineInput[icol])<=t2))
            lineInput[icol]=value;
          else
            lineInput[icol]=m_noDataValues[0];
        }
        imgWriter.writeData(lineInput,irow,iband);
      }
    }
  }
  catch(std::string errorstring){
    std::cerr << errorstring << std::endl;
    throw;
  }
  catch(...){
    throw;
  }
  return(CE_None);
}

CPLErr Jim::setThreshold(Jim& imgWriter, app::AppFactory &theApp){
  Optionjl<double> min_opt("min", "min", "minimum value to be valid");
  Optionjl<double> max_opt("max", "max", "maximum value to be valid");
  Optionjl<double> value_opt("value", "value", "value to be set if within min and max (if not set, valid pixels will remain their input value)");
  Optionjl<bool> abs_opt("abs", "abs", "check for absolute values",false);
  Optionjl<double> nodata_opt("nodata", "nodata", "Nodata value to put in image.");

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  doProcess=min_opt.retrieveOption(theApp);
  max_opt.retrieveOption(theApp);
  value_opt.retrieveOption(theApp);
  abs_opt.retrieveOption(theApp);
  nodata_opt.retrieveOption(theApp);
  if(!doProcess){
    std::cout << std::endl;
    std::ostringstream helpStream;
    helpStream << "help info: ";
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
  if(nodata_opt.size())
    setNoData(nodata_opt);
  if(abs_opt[0]){
    if(value_opt.size())
      setAbsThreshold(imgWriter,min_opt[0],max_opt[0],value_opt[0]);
    else
      setAbsThreshold(imgWriter,min_opt[0],max_opt[0]);
  }
  else{
    if(value_opt.size())
      setThreshold(imgWriter,min_opt[0],max_opt[0],value_opt[0]);
    else
      setThreshold(imgWriter,min_opt[0],max_opt[0]);
  }
}

