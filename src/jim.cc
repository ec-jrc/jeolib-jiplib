/**********************************************************************
jim.cc: class to read raster files
History
2016/06/24 - Created by Pieter Kempeneers
Change log
***********************************************************************/
#include "jim.h"

using namespace jiplib;

// //test
// void Jim::open(int ncol, int nrow, int nband, int dataType){
//   ImgRaster::open(ncol,nrow,nband,static_cast<GDALDataType>(dataType));
// }

/**
 *
 *
 * @param band the band to get the MIA image representation for
 *
 * @return pointer to MIA image representation
 */
IMAGE* Jim::getMIA(int band){
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
  m_mia->NByte=m_mia->nx*m_mia->ny*m_mia->nz*(GDALGetDataTypeSize(getDataType())>>3);//assumes image data type is not of bit type!!!
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
CPLErr Jim::setMIA(int band){
  try{
    if(m_mia->nz>1){
      std::string errorString="Error: MIA image with nz>1 not supported";
      throw(errorString);
    }
    if(m_ncol!=m_mia->nx){
      std::ostringstream s;
      s << "Error: x dimension of image (" << m_ncol << ") does not match MIA (" << m_mia->nx << ")";
      throw(s.str());
    }
    if(m_nrow!=m_mia->ny){
      std::ostringstream s;
      s << "Error: y dimension of image (" << m_nrow << ") does not match MIA (" << m_mia->ny << ")";
      throw(s.str());
    }
    if(m_nband<=band){
      std::ostringstream s;
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
  }
  catch(std::string errorString){
    std::cerr << errorString << std::endl;
    return(CE_Failure);
  }
  catch(...){
    return(CE_Failure);
  }
  return(CE_None);
}

/**
 *
 *
 * @param mia the MIA image pointer to be set
 * @param band the band for which the MIA image pointer needs to be set
 *
 * @return C_None if successful
 */
CPLErr Jim::setMIA(IMAGE* mia, int band){
  try{
    if(nrOfBand()>1){
      if(mia->nz>1){
        std::string errorString="Error: MIA image with nz>1 not supported";
        throw(errorString);
      }
      if(m_ncol!=mia->nx){
        std::string errorString="Error: dimensions of images in do not match";
        throw(errorString);
      }
      if(m_nrow!=mia->ny){
        std::string errorString="Error: dimensions of images do not match";
        throw(errorString);
      }
      if(m_nband<=band){
        std::string errorString="Error: band exceeds number of bands in target image";
        throw(errorString);
      }
      if(m_dataType!=MIA2GDALDataType(mia->DataType)){
        std::ostringstream s;
        s << "Error: data types of images do not match: ";
        s << m_dataType << ", " << MIA2GDALDataType(mia->DataType);
        throw(s.str());
      }
    }
    else{
      reset();
      m_nplane=mia->nz;
      m_ncol=mia->nx;
      m_nrow=mia->ny;
      m_nband=1;
      m_dataType=MIA2GDALDataType(mia->DataType);
      m_data.resize(m_nband);
      m_blockSize=m_nrow;
      m_begin.resize(m_nband);
      m_end.resize(m_nband);
    }
    m_mia=mia;
    setExternalData(true);//todo: need to fix memory leak when setMIA used for single band only! (either create vector<bool> m_externalData or only allow for setMIA all bands)
    this->setMIA(band);
  }
  catch(std::string errorString){
    std::cerr << errorString << std::endl;
    return(CE_Failure);
  }
  catch(...){
    return(CE_Failure);
  }
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
CPLErr Jim::shift(int value, int iband){
  try{
    if(nrOfBand()<=iband){
      std::string errorString="Error: band number exceeds number of bands in input image";
      throw(errorString);
    }
    IMAGE* mia1=this->getMIA(iband);
    if(::shift(mia1, value) == NO_ERROR){
      this->setMIA(iband);
      return(CE_None);
    }
    else{
      this->setMIA(iband);
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
 * @return CE_None if successful
 */
// CPLErr Jim::arith(std::shared_ptr<Jim> imgRaster, int theOperation, int iband){
CPLErr Jim::arith(Jim& imgRaster, int theOperation, int iband){
  try{
    if(imgRaster.nrOfBand()<=iband){
      std::string errorString="Error: band number exceeds number of bands in input image";
      throw(errorString);
    }
    if(nrOfBand()<=iband){
      std::string errorString="Error: band number exceeds number of bands in input image";
      throw(errorString);
    }
    IMAGE* mia1=this->getMIA(iband);
    IMAGE* mia2=imgRaster.getMIA(iband);
    if(::arith(mia1, mia2, theOperation) == NO_ERROR){
      this->setMIA(iband);
      imgRaster.setMIA(iband);
      return(CE_None);
    }
    else{
      this->setMIA(iband);
      imgRaster.setMIA(iband);
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
std::shared_ptr<jiplib::Jim> Jim::getArith(std::shared_ptr<Jim> inputImg, int theOperation, int iband){
  try{
    if(inputImg->nrOfBand()<=iband){
      std::string errorString="Error: band number exceeds number of bands in input image";
      throw(errorString);
    }
    if(nrOfBand()<=iband){
      std::string errorString="Error: band number exceeds number of bands in input image";
      throw(errorString);
    }
    std::shared_ptr<jiplib::Jim> pJim=std::make_shared<jiplib::Jim>(*this, true);
    IMAGE* mia1=pJim->getMIA(iband);
    IMAGE* mia2=inputImg->getMIA(iband);
    if(::arith(mia1, mia2, theOperation) == NO_ERROR){
      pJim->setMIA(iband);
      inputImg->setMIA(iband);
      return(pJim);
    }
    else{
      inputImg->setMIA(iband);
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
CPLErr Jim::rdil(std::shared_ptr<Jim> mask, int graph, int flag, int iband){
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
std::shared_ptr<jiplib::Jim> Jim::getRdil(std::shared_ptr<Jim> mask, int graph, int flag, int iband){
  std::shared_ptr<jiplib::Jim> pJim=std::make_shared<jiplib::Jim>(*this, true);
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
CPLErr Jim::rero(std::shared_ptr<Jim> mask, int graph, int flag, int iband){
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
std::shared_ptr<jiplib::Jim> Jim::getRero(std::shared_ptr<Jim> mask, int graph, int flag, int iband){
  std::shared_ptr<jiplib::Jim> pJim=std::make_shared<jiplib::Jim>(*this, true);
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
    ImgRaster::open(imgSrc,copyData);
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
