/**********************************************************************
jim.cc: class to read raster files
History
2016/06/24 - Created by Pieter Kempeneers
Change log
***********************************************************************/
#include "jim.h"

using namespace jiplib;

IMAGE Jim::getMIA(unsigned int band){
  if(getBlockSize()!=nrOfRow()){
    std::ostringstream s;
    s << "Error: increase memory to support MIA library functions (now at " << 100.0*getBlockSize()/nrOfRow() << "%)";
    throw(s.str());
  }    
  IMAGE mia;
  mia.p_im=m_data[band];/* Pointer to image data */
  mia.DataType=GDAL2LIIARDataType(getDataType());
  mia.nx=nrOfCol();
  mia.ny=nrOfRow();
  mia.nz=1;
  mia.NByte=mia.nx*mia.ny*mia.nz*GDALGetDataTypeSize(getDataType())>>3;//assumes image data type is not of bit type!!!
  //todo: remove mia.vol and only rely on the getVolume function
  mia.vol=0;//not used.
  mia.lut=0;
  //USHORT *lut;   /* Pointer to colour map */
  //mia->g=getgetDataType();//not used
  return mia;
}

// IMAGE Jim::getMIA(){
//   if(getBlockSize()!=nrOfRow()){
//     std::ostringstream s;
//     s << "Error: increase memory to support MIA library functions (now at " << 100.0*getBlockSize()/nrOfRow() << "%)";
//     throw(s.str());
//   }
//   IMAGE* mia_out=create_image(GDAL2LIIARDataType(getDataType()),nrOfCol(),nrOfRow(),nrOfBand());
//   for(int iband=0;iband<nrOfBand();++iband){
//     IMAGE mia_in=getMIA(iband);
//     //imputop(&mia_in,mia_out,0,0,iband,OR_op);
//     imputop(&mia_in,mia_out,0,0,iband,11);
//   }
//   return *mia_out;
// }

// CPLErr Jim::setMIA(IMAGE& mia){
//   m_gds=0;
//   m_ncol = mia.nx;
//   m_nrow = mia.ny;
//   m_nband=mia.nz;
//   m_dataType = LIIAR2GDALDataType(mia.DataType);
//   initMem(0);
//   for(unsigned int iband=0;iband<m_nband;++iband){
//     m_data[iband]=mia.p_im+iband*nrOfRow()*nrOfCol()*(GDALGetDataTypeSize(getDataType())>>3);
//     m_begin[iband]=0;
//     m_end[iband]=m_begin[iband]+getBlockSize();
//   }
//   return(CE_None);
// }

CPLErr Jim::setMIA(IMAGE& mia, unsigned int band){
  if(mia.nz>1){
    std::string errorString="Error: MIA image with nz>1 not supported";
    throw(errorString);
  }
  if(m_ncol!=mia.nx){
    std::string errorString="Error: dimensions of images in do not match";
    throw(errorString);
  }
  if(m_ncol!=mia.ny){
    std::string errorString="Error: dimensions of images do not match";
    throw(errorString);
  }
  if(m_nband<=band){
    std::string errorString="Error: band exceeds number of bands in target image";
    throw(errorString);
  }
  if(m_dataType!=LIIAR2GDALDataType(mia.DataType)){
    std::string errorString="Error: data types of images do not match";
    throw(errorString);
  }
  m_data[band]=mia.p_im+band*nrOfRow()*nrOfCol()*(GDALGetDataTypeSize(getDataType())>>3);
  m_begin[band]=0;
  m_end[band]=m_begin[band]+getBlockSize();
  return(CE_None);
}

CPLErr Jim::arith(Jim& imgRaster, int theOperation){
  if(imgRaster.nrOfCol()!=this->nrOfCol()){
    std::string errorString="Error: dimensions of images do not match";
    throw(errorString);
  }
  if(imgRaster.nrOfRow()!=this->nrOfRow()){
    std::string errorString="Error: dimensions of images do not match";
    throw(errorString);
  }
  if(imgRaster.nrOfBand()!=this->nrOfBand()){
    std::string errorString="Error: dimensions of images do not match";
    throw(errorString);
  }
  for(int iband=0;iband<nrOfBand();++iband){
    IMAGE mia1=this->getMIA(iband);
    IMAGE mia2=imgRaster.getMIA(iband);
    if(::arith(&mia1, &mia2, theOperation)!=NO_ERROR){
      return(CE_Failure);
    }
  }
  return(CE_None);
}

CPLErr Jim::rdil(Jim& mask, int graph, int flag){
  IMAGE maskMIA=mask.getMIA(0);
  IMAGE markMIA=this->getMIA(0);
  ::rdil(&markMIA,&maskMIA,graph,flag);
  if(flag)//rdil reallocates data pointer
    mask.setMIA(maskMIA,0);
}

CPLErr Jim::imequalp(Jim& ref){
  CPLErr result=CE_None;
  if(ref.nrOfBand()!=this->nrOfBand())
    return(CE_Failure);
  for(int iband=0;iband<nrOfBand();++iband){
    IMAGE refMIA=ref.getMIA(iband);
    IMAGE thisMIA=this->getMIA(iband);
    if(::imequalp(&thisMIA,&refMIA))
      return(CE_Failure);
  }
  return(result);
}


/**
 * @param imgSrc Use this source image as the reference image
 **/
bool Jim::operator==(Jim& refImg)
{
  bool isEqual=true;
  if(nrOfBand()!=refImg.nrOfBand())
    return(false);
  if(nrOfRow()!=refImg.nrOfRow())
    return(false);
  if(nrOfCol()!=refImg.nrOfCol())
    return(false);

  for(int iband=0;iband<nrOfBand();++iband){
    if(getDataType(iband)!=refImg.getDataType(iband)){
      isEqual=false;
      break;
    }
    IMAGE refMIA=refImg.getMIA(iband);
    IMAGE thisMIA=this->getMIA(iband);
    if(::imequalp(&thisMIA,&refMIA)){
      isEqual=false;
      break;
    }
  }
  return(isEqual);
}


