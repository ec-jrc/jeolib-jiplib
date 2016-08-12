/**********************************************************************
jim.cc: class to read raster files
History
2016/06/24 - Created by Pieter Kempeneers
Change log
***********************************************************************/
#include "jim.h"

using namespace jiplib;

/** 
 * 
 * 
 * @param band the band to get the MIA image representation for
 * 
 * @return a MIA image representation
 */
IMAGE Jim::getMIA(unsigned int band){
  if(getBlockSize()!=nrOfRow()){
    std::ostringstream s;
    s << "Error: increase memory to support MIA library functions (now at " << 100.0*getBlockSize()/nrOfRow() << "%)";
    throw(s.str());
  }    
  IMAGE mia;
  mia.p_im=m_data[band];/* Pointer to image data */
  mia.DataType=GDAL2MIADataType(getDataType());
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

/** 
 * 
 * 
 * @param mia the MIA image pointer to be set
 * @param band the band for which the MIA image pointer needs to be set
 * 
 * @return C_None if successful
 */
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
  if(m_dataType!=MIA2GDALDataType(mia.DataType)){
    std::ostringstream s;
    s << "Error: data types of images do not match: ";
    s << m_dataType << ", " << MIA2GDALDataType(mia.DataType);
    throw(s.str());
  }
  m_data[band]=mia.p_im+band*nrOfRow()*nrOfCol()*(GDALGetDataTypeSize(getDataType())>>3);
  m_begin[band]=0;
  m_end[band]=m_begin[band]+getBlockSize();
  return(CE_None);
}

/** 
 * 
 * 
 * @param imgRaster is operand
 * @param theOperation the operation to be performed
 * @param iband is the band for which the function needs to be performed (default 0 is first band)
 * 
 * @return CE_None if successful
 */
CPLErr Jim::arith(Jim& imgRaster, int theOperation, unsigned int iband){
  // try{
  //   if(imgRaster.nrOfBand()<iband){
  //     std::string errorString="Error: band number exceeds number of bands in input image";
  //     throw(errorString);
  //   }
  //   if(nrOfBand()<iband){
  //     std::string errorString="Error: band number exceeds number of bands in input image";
  //     throw(errorString);
  //   }
  //   IMAGE mia1=this->getMIA(iband);
  //   IMAGE mia2=imgRaster.getMIA(iband);
  //   ::arith(&mia1, &mia2, theOperation);
  //   setMIA(mia1,iband);
  //   imgRaster.setMIA(mia2,iband);
  //   return(CE_None);
  // }
  // catch(std::string errorString){
  //   std::cerr << errorString << std::endl;
  //   return(CE_Failure);
  // }
  //test
  return(CE_None);
}

/** 
 * 
 * 
 * @param mask
 * @param graph
 * @param flag
 * @param iband is the band for which the function needs to be performed (default is 0: first band)
 * 
 * @return CE_None if successful
 */
CPLErr Jim::rdil(Jim& mask, int graph, int flag, unsigned int iband){
  IMAGE markMIA=this->getMIA(iband);
  IMAGE maskMIA=mask.getMIA(iband);
  CPLErr success=CE_None;
  ::rdil(&markMIA,&maskMIA,graph,flag);
  setMIA(markMIA,iband);
  mask.setMIA(maskMIA,iband);
  return(success);
}

/** 
 * 
 * 
 * @param mask
 * @param grap
 * @param flag
 * @param iband is the band for which the arithmetic operation needs to be performed (default is 0: first band)
 * 
 * @return CE_None if successful
 */
CPLErr Jim::rero(Jim& mask, int graph, int flag, unsigned int iband){
  IMAGE markMIA=this->getMIA(iband);
  IMAGE maskMIA=mask.getMIA(iband);
  CPLErr success=CE_None;
  ::rero(&markMIA,&maskMIA,graph,flag);
  setMIA(markMIA,iband);
  mask.setMIA(maskMIA,iband);
  return(success);
}

/**
 * @param imgSrc Use this source image as a template to copy image attributes
 **/
Jim& Jim::operator=(Jim& imgSrc)
{
  bool copyData=true;
  //check for assignment to self (of the form v=v)
  if(this==&imgSrc)
     return *this;
  else{
    open(imgSrc,copyData);
    return *this;
  }
}


/** 
 * 
 * @param refImg Use this as the reference image
 * 
 * @return true if image is equal to reference image
 */
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

std::string Jim::f4(Jim& imgRaster, unsigned int band)
{
  try{
    IMAGE markMIA=this->getMIA(band);
    IMAGE maskMIA=imgRaster.getMIA(band);
    // ::rero(&markMIA,&maskMIA,8,1);
    // setMIA(markMIA,band);
    // imgRaster.setMIA(maskMIA,band);
  }
  catch(std::string errorString){
    return(errorString);
  }
}

std::string Jim::f5(Jim& imgRaster, unsigned int band)
{
  try{
    IMAGE markMIA=this->getMIA(band);
    IMAGE maskMIA=imgRaster.getMIA(band);
    ::rero(&markMIA,&maskMIA,8,1);
    setMIA(markMIA,band);
    imgRaster.setMIA(maskMIA,band);
  }
  catch(std::string errorString){
    return(errorString);
  }
}
