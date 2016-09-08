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
 * @return pointer to MIA image representation
 */
IMAGE* Jim::getMIA(unsigned int band){
  if(getBlockSize()!=nrOfRow()){
    std::ostringstream s;
    s << "Error: increase memory to support MIA library functions (now at " << 100.0*getBlockSize()/nrOfRow() << "%)";
    throw(s.str());
  }
  m_mia=new(IMAGE);
  m_mia->p_im=m_data[band];/* Pointer to image data */
  m_mia->DataType=GDAL2MIADataType(getDataType());
  m_mia->nx=nrOfCol();
  m_mia->ny=nrOfRow();
  m_mia->nz=1;
  m_mia->NByte=m_mia->nx*m_mia->ny*m_mia->nz*GDALGetDataTypeSize(getDataType())>>3;//assumes image data type is not of bit type!!!
  //todo: remove m_mia->vol and only rely on the getVolume function
  m_mia->vol=0;//not used.
  m_mia->lut=0;
  //USHORT *lut;   /* Pointer to colour map */
  //mia->g=getgetDataType();//not used
  return m_mia;
}

/**
 *
 *
 * @param band the band for which the MIA image pointer needs to be set
 *
 * @return C_None if successful
 */
CPLErr Jim::setMIA(unsigned int band){
  if(m_mia->nz>1){
    std::string errorString="Error: MIA image with nz>1 not supported";
    throw(errorString);
  }
  if(m_ncol!=m_mia->nx){
    std::string errorString="Error: dimensions of images in do not match";
    throw(errorString);
  }
  if(m_ncol!=m_mia->ny){
    std::string errorString="Error: dimensions of images do not match";
    throw(errorString);
  }
  if(m_nband<=band){
    std::string errorString="Error: band exceeds number of bands in target image";
    throw(errorString);
  }
  if(m_dataType!=MIA2GDALDataType(m_mia->DataType)){
    std::ostringstream s;
    s << "Error: data types of images do not match: ";
    s << m_dataType << ", " << MIA2GDALDataType(m_mia->DataType);
    throw(s.str());
  }
  m_data[band]=m_mia->p_im+band*nrOfRow()*nrOfCol()*(GDALGetDataTypeSize(getDataType())>>3);
  m_begin[band]=0;
  m_end[band]=m_begin[band]+getBlockSize();
  return(CE_None);
}

// /**
//  *
//  *
//  * @param mia the MIA image pointer to be set
//  * @param band the band for which the MIA image pointer needs to be set
//  *
//  * @return C_None if successful
//  */
// CPLErr Jim::setMIA(IMAGE* mia, unsigned int band){
//   if(mia.nz>1){
//     std::string errorString="Error: MIA image with nz>1 not supported";
//     throw(errorString);
//   }
//   if(m_ncol!=mia.nx){
//     std::string errorString="Error: dimensions of images in do not match";
//     throw(errorString);
//   }
//   if(m_ncol!=mia.ny){
//     std::string errorString="Error: dimensions of images do not match";
//     throw(errorString);
//   }
//   if(m_nband<=band){
//     std::string errorString="Error: band exceeds number of bands in target image";
//     throw(errorString);
//   }
//   if(m_dataType!=MIA2GDALDataType(mia.DataType)){
//     std::ostringstream s;
//     s << "Error: data types of images do not match: ";
//     s << m_dataType << ", " << MIA2GDALDataType(mia.DataType);
//     throw(s.str());
//   }
//   m_data[band]=mia.p_im+band*nrOfRow()*nrOfCol()*(GDALGetDataTypeSize(getDataType())>>3);
//   m_begin[band]=0;
//   m_end[band]=m_begin[band]+getBlockSize();
//   return(CE_None);
// }

/**
 *
 *
 * @param imgRaster is operand
 * @param theOperation the operation to be performed
 * @param iband is the band for which the function needs to be performed (default 0 is first band)
 *
 * @return CE_None if successful
 */
CPLErr Jim::arith(std::shared_ptr<Jim> imgRaster, int theOperation, unsigned int iband){
  try{
    if(imgRaster->nrOfBand()<iband){
      std::string errorString="Error: band number exceeds number of bands in input image";
      throw(errorString);
    }
    if(nrOfBand()<iband){
      std::string errorString="Error: band number exceeds number of bands in input image";
      throw(errorString);
    }
    IMAGE* mia1=this->getMIA(iband);
    IMAGE* mia2=imgRaster->getMIA(iband);
    if(::arith(mia1, mia2, theOperation) == NO_ERROR){
      this->setMIA(iband);
      imgRaster->setMIA(iband);
      return(CE_None);
    }
    else{
      this->setMIA(iband);
      imgRaster->setMIA(iband);
      std::string errorString="Error: arith function in MIA failed";
      throw(errorString);
    }
  }
  catch(std::string errorString){
    std::cerr << errorString << std::endl;
    return(CE_Failure);
  }
  catch(...){
    return(CE_Failure);
  }
}

/**
 *
 *
 * @param imgRaster is operand
 * @param theOperation the operation to be performed
 * @param iband is the band for which the function needs to be performed (default 0 is first band)
 *
 * @return shared pointer to resulting image
 */
std::shared_ptr<jiplib::Jim> Jim::getArith(std::shared_ptr<Jim> imgRaster, int theOperation, unsigned int iband){
  try{
    if(imgRaster->nrOfBand()<iband){
      std::string errorString="Error: band number exceeds number of bands in input image";
      throw(errorString);
    }
    if(nrOfBand()<iband){
      std::string errorString="Error: band number exceeds number of bands in input image";
      throw(errorString);
    }
    std::shared_ptr<jiplib::Jim> pJim=std::make_shared<jiplib::Jim>(shared_from_this(), true);
    IMAGE* mia1=pJim->getMIA(iband);
    IMAGE* mia2=imgRaster->getMIA(iband);
    if(::arith(mia1, mia2, theOperation) == NO_ERROR){
      pJim->setMIA(iband);
      imgRaster->setMIA(iband);
      return(pJim);
    }
    else{
      imgRaster->setMIA(iband);
      std::string errorString="Error: arith function in MIA failed";
      throw(errorString);
    }
  }
  catch(std::string errorString){
    std::cerr << errorString << std::endl;
    return(NULL);
  }
  catch(...){
    return(NULL);
  }
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
CPLErr Jim::rdil(std::shared_ptr<Jim> mask, int graph, int flag, unsigned int iband){
  IMAGE* markMIA=this->getMIA(iband);
  IMAGE* maskMIA=mask->getMIA(iband);
  if(::rdil(markMIA,maskMIA,graph,flag) == NO_ERROR){
    this->setMIA(iband);
    mask->setMIA(iband);
    return(CE_None);
  }
  this->setMIA(iband);
  mask->setMIA(iband);
  return(CE_Failure);
}

/**
 *
 *
 * @param mask
 * @param graph
 * @param flag
 * @param iband is the band for which the operation needs to be performed (default is 0: first band)
 *
 * @return shared pointer to resulting image
 */
std::shared_ptr<jiplib::Jim> Jim::getRdil(std::shared_ptr<Jim> mask, int graph, int flag, unsigned int iband){
  std::shared_ptr<jiplib::Jim> pJim=std::make_shared<jiplib::Jim>(shared_from_this(), true);
  IMAGE* markMIA=pJim->getMIA(iband);
  IMAGE* maskMIA=mask->getMIA(iband);
  if (::rdil(markMIA,maskMIA,graph,flag) == NO_ERROR){
    pJim->setMIA(iband);
  }
  else
    pJim=NULL;
  mask->setMIA(iband);
  return(pJim);
}
/**
 *
 *
 * @param mask
 * @param graph
 * @param flag
 * @param iband is the band for which the operation needs to be performed (default is 0: first band)
 *
 * @return CE_None if successful
 */
CPLErr Jim::rero(std::shared_ptr<Jim> mask, int graph, int flag, unsigned int iband){
  IMAGE* markMIA=this->getMIA(iband);
  IMAGE* maskMIA=mask->getMIA(iband);
  if (::rero(markMIA,maskMIA,graph,flag) == NO_ERROR){
    this->setMIA(iband);
    mask->setMIA(iband);
    return(CE_None);
  }
  this->setMIA(iband);
  mask->setMIA(iband);
  return(CE_Failure);
}

/**
 *
 *
 * @param mask
 * @param graph
 * @param flag
 * @param iband is the band for which the operation needs to be performed (default is 0: first band)
 *
 * @return shared pointer to resulting image
 */
std::shared_ptr<jiplib::Jim> Jim::getRero(std::shared_ptr<Jim> mask, int graph, int flag, unsigned int iband){
  std::shared_ptr<jiplib::Jim> pJim=std::make_shared<jiplib::Jim>(shared_from_this(), true);
  IMAGE* markMIA=pJim->getMIA(iband);
  IMAGE* maskMIA=mask->getMIA(iband);
  if (::rero(markMIA,maskMIA,graph,flag) == NO_ERROR){
    pJim->setMIA(iband);
  }
  else
    pJim=NULL;
  mask->setMIA(iband);
  return(pJim);
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
bool Jim::operator==(std::shared_ptr<Jim> refImg)
{
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
    IMAGE* refMIA=refImg.getMIA(iband);
    IMAGE* thisMIA=this->getMIA(iband);
    if(::imequalp(thisMIA,refMIA)){
      isEqual=false;
      break;
    }
  }
  return(isEqual);
}
