/**********************************************************************
jloperators.cc: operators for Jim
Author(s): Pieter.Kempeneers@ec.europa.eu
Copyright (c) 2016-2019 European Union (Joint Research Centre)
License EUPLv1.2

This file is part of jiplib
***********************************************************************/
#include <iostream>
#include "Jim.h"

using namespace std;

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

std::shared_ptr<Jim> Jim::eq(std::shared_ptr<Jim> refJim){
  shared_ptr<Jim> imgWriter=Jim::createImg();
  this->eq(*refJim,*imgWriter);
  return(imgWriter);
}

std::shared_ptr<Jim> Jim::eq(double value){
  shared_ptr<Jim> imgWriter=Jim::createImg();
  this->eq(value,*imgWriter);
  return(imgWriter);
}

void Jim::eq(Jim& other, Jim& imgWriter){
  if(m_data.empty()){
    std::ostringstream s;
    s << "Error: Jim not initialized, m_data is empty";
    std::cerr << s.str() << std::endl;
    throw(s.str());
  }
  if(nrOfBand()!=other.nrOfBand()){
    std::ostringstream s;
    s << "Error: number of bands do not match";
    std::cerr << s.str() << std::endl;
    throw(s.str());
  }
  if(nrOfRow()!=other.nrOfRow()){
    std::ostringstream s;
    s << "Error: number of rows do not match";
    std::cerr << s.str() << std::endl;
    throw(s.str());
  }
  if(nrOfCol()!=other.nrOfCol()){
    std::ostringstream s;
    s << "Error: number of cols do not match";
    std::cerr << s.str() << std::endl;
    throw(s.str());
  }
  imgWriter.open(nrOfCol(),nrOfRow(),nrOfBand(),GDT_Byte);
  imgWriter.copyGeoTransform(*this);
  imgWriter.setProjection(this->getProjection());
  for(size_t iband=0;iband<nrOfBand();++iband){
#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
    for(int irow=0;irow<nrOfRow();++irow){
      for(int icol=0;icol<nrOfCol();++icol){
        if(readData(icol,irow,iband)==other.readData(icol,irow,iband))
          imgWriter.writeData(1,icol,irow,iband);
        else
          imgWriter.writeData(0,icol,irow,iband);
      }
    }
  }
}

void Jim::eq(double value, Jim& imgWriter){
  if(m_data.empty()){
    std::ostringstream s;
    s << "Error: Jim not initialized, m_data is empty";
    std::cerr << s.str() << std::endl;
    throw(s.str());
  }
  imgWriter.open(nrOfCol(),nrOfRow(),nrOfBand(),GDT_Byte);
  imgWriter.copyGeoTransform(*this);
  imgWriter.setProjection(this->getProjection());
  for(size_t iband=0;iband<nrOfBand();++iband){
#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
    for(int irow=0;irow<nrOfRow();++irow){
      for(int icol=0;icol<nrOfCol();++icol){
        if(readData(icol,irow,iband)==value)
          imgWriter.writeData(1,icol,irow,iband);
        else
          imgWriter.writeData(0,icol,irow,iband);
      }
    }
  }
}

std::shared_ptr<Jim> Jim::ne(std::shared_ptr<Jim> refJim){
  shared_ptr<Jim> imgWriter=Jim::createImg();
  this->ne(*refJim,*imgWriter);
  return(imgWriter);
}

std::shared_ptr<Jim> Jim::ne(double value){
  shared_ptr<Jim> imgWriter=Jim::createImg();
  this->ne(value,*imgWriter);
  return(imgWriter);
}

void Jim::ne(Jim& other, Jim& imgWriter){
  if(m_data.empty()){
    std::ostringstream s;
    s << "Error: Jim not initialized, m_data is empty";
    std::cerr << s.str() << std::endl;
    throw(s.str());
  }
  if(nrOfBand()!=other.nrOfBand()){
    std::ostringstream s;
    s << "Error: number of bands do not match";
    std::cerr << s.str() << std::endl;
    throw(s.str());
  }
  if(nrOfRow()!=other.nrOfRow()){
    std::ostringstream s;
    s << "Error: number of rows do not match";
    std::cerr << s.str() << std::endl;
    throw(s.str());
  }
  if(nrOfCol()!=other.nrOfCol()){
    std::ostringstream s;
    s << "Error: number of cols do not match";
    std::cerr << s.str() << std::endl;
    throw(s.str());
  }
  imgWriter.open(nrOfCol(),nrOfRow(),nrOfBand(),GDT_Byte);
  imgWriter.copyGeoTransform(*this);
  imgWriter.setProjection(this->getProjection());
  for(size_t iband=0;iband<nrOfBand();++iband){
#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
    for(int irow=0;irow<nrOfRow();++irow){
      for(int icol=0;icol<nrOfCol();++icol){
        if(readData(icol,irow,iband)!=other.readData(icol,irow,iband))
          imgWriter.writeData(1,icol,irow,iband);
        else
          imgWriter.writeData(0,icol,irow,iband);
      }
    }
  }
}

void Jim::ne(double value, Jim& imgWriter){
  if(m_data.empty()){
    std::ostringstream s;
    s << "Error: Jim not initialized, m_data is empty";
    std::cerr << s.str() << std::endl;
    throw(s.str());
  }
  imgWriter.open(nrOfCol(),nrOfRow(),nrOfBand(),GDT_Byte);
  imgWriter.copyGeoTransform(*this);
  imgWriter.setProjection(this->getProjection());
  for(size_t iband=0;iband<nrOfBand();++iband){
#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
    for(int irow=0;irow<nrOfRow();++irow){
      for(int icol=0;icol<nrOfCol();++icol){
        if(readData(icol,irow,iband)!=value)
          imgWriter.writeData(1,icol,irow,iband);
        else
          imgWriter.writeData(0,icol,irow,iband);
      }
    }
  }
}

std::shared_ptr<Jim> Jim::lt(std::shared_ptr<Jim> refJim){
  shared_ptr<Jim> imgWriter=Jim::createImg();
  this->lt(*refJim,*imgWriter);
  return(imgWriter);
}

std::shared_ptr<Jim> Jim::lt(double value){
  shared_ptr<Jim> imgWriter=Jim::createImg();
  this->lt(value,*imgWriter);
  return(imgWriter);
}

void Jim::lt(Jim& other, Jim& imgWriter){
  if(m_data.empty()){
    std::ostringstream s;
    s << "Error: Jim not initialized, m_data is empty";
    std::cerr << s.str() << std::endl;
    throw(s.str());
  }
  if(nrOfBand()!=other.nrOfBand()){
    std::ostringstream s;
    s << "Error: number of bands do not match";
    std::cerr << s.str() << std::endl;
    throw(s.str());
  }
  if(nrOfRow()!=other.nrOfRow()){
    std::ostringstream s;
    s << "Error: number of rows do not match";
    std::cerr << s.str() << std::endl;
    throw(s.str());
  }
  if(nrOfCol()!=other.nrOfCol()){
    std::ostringstream s;
    s << "Error: number of cols do not match";
    std::cerr << s.str() << std::endl;
    throw(s.str());
  }
  imgWriter.open(nrOfCol(),nrOfRow(),nrOfBand(),GDT_Byte);
  imgWriter.copyGeoTransform(*this);
  imgWriter.setProjection(this->getProjection());
  for(size_t iband=0;iband<nrOfBand();++iband){
#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
    for(int irow=0;irow<nrOfRow();++irow){
      for(int icol=0;icol<nrOfCol();++icol){
        if(readData(icol,irow,iband)<other.readData(icol,irow,iband))
          imgWriter.writeData(1,icol,irow,iband);
        else
          imgWriter.writeData(0,icol,irow,iband);
      }
    }
  }
}

void Jim::lt(double value, Jim& imgWriter){
  if(m_data.empty()){
    std::ostringstream s;
    s << "Error: Jim not initialized, m_data is empty";
    std::cerr << s.str() << std::endl;
    throw(s.str());
  }
  imgWriter.open(nrOfCol(),nrOfRow(),nrOfBand(),GDT_Byte);
  imgWriter.copyGeoTransform(*this);
  imgWriter.setProjection(this->getProjection());
  for(size_t iband=0;iband<nrOfBand();++iband){
#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
    for(int irow=0;irow<nrOfRow();++irow){
      for(int icol=0;icol<nrOfCol();++icol){
        if(readData(icol,irow,iband)<value)
          imgWriter.writeData(1,icol,irow,iband);
        else
          imgWriter.writeData(0,icol,irow,iband);
      }
    }
  }
}

std::shared_ptr<Jim> Jim::le(std::shared_ptr<Jim> refJim){
  shared_ptr<Jim> imgWriter=Jim::createImg();
  this->le(*refJim,*imgWriter);
  return(imgWriter);
}

std::shared_ptr<Jim> Jim::le(double value){
  shared_ptr<Jim> imgWriter=Jim::createImg();
  this->le(value,*imgWriter);
  return(imgWriter);
}

void Jim::le(Jim& other, Jim& imgWriter){
  if(m_data.empty()){
    std::ostringstream s;
    s << "Error: Jim not initialized, m_data is empty";
    std::cerr << s.str() << std::endl;
    throw(s.str());
  }
  if(nrOfBand()!=other.nrOfBand()){
    std::ostringstream s;
    s << "Error: number of bands do not match";
    std::cerr << s.str() << std::endl;
    throw(s.str());
  }
  if(nrOfRow()!=other.nrOfRow()){
    std::ostringstream s;
    s << "Error: number of rows do not match";
    std::cerr << s.str() << std::endl;
    throw(s.str());
  }
  if(nrOfCol()!=other.nrOfCol()){
    std::ostringstream s;
    s << "Error: number of cols do not match";
    std::cerr << s.str() << std::endl;
    throw(s.str());
  }
  imgWriter.open(nrOfCol(),nrOfRow(),nrOfBand(),GDT_Byte);
  imgWriter.copyGeoTransform(*this);
  imgWriter.setProjection(this->getProjection());
  for(size_t iband=0;iband<nrOfBand();++iband){
#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
    for(int irow=0;irow<nrOfRow();++irow){
      for(int icol=0;icol<nrOfCol();++icol){
        if(readData(icol,irow,iband)<=other.readData(icol,irow,iband))
          imgWriter.writeData(1,icol,irow,iband);
        else
          imgWriter.writeData(0,icol,irow,iband);
      }
    }
  }
}

void Jim::le(double value, Jim& imgWriter){
  if(m_data.empty()){
    std::ostringstream s;
    s << "Error: Jim not initialized, m_data is empty";
    std::cerr << s.str() << std::endl;
    throw(s.str());
  }
  imgWriter.open(nrOfCol(),nrOfRow(),nrOfBand(),GDT_Byte);
  imgWriter.copyGeoTransform(*this);
  imgWriter.setProjection(this->getProjection());
  for(size_t iband=0;iband<nrOfBand();++iband){
#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
    for(int irow=0;irow<nrOfRow();++irow){
      for(int icol=0;icol<nrOfCol();++icol){
        if(readData(icol,irow,iband)<=value)
          imgWriter.writeData(1,icol,irow,iband);
        else
          imgWriter.writeData(0,icol,irow,iband);
      }
    }
  }
}

std::shared_ptr<Jim> Jim::gt(std::shared_ptr<Jim> refJim){
  shared_ptr<Jim> imgWriter=Jim::createImg();
  this->gt(*refJim,*imgWriter);
  return(imgWriter);
}

std::shared_ptr<Jim> Jim::gt(double value){
  shared_ptr<Jim> imgWriter=Jim::createImg();
  this->gt(value,*imgWriter);
  return(imgWriter);
}

void Jim::gt(Jim& other, Jim& imgWriter){
  if(m_data.empty()){
    std::ostringstream s;
    s << "Error: Jim not initialized, m_data is empty";
    std::cerr << s.str() << std::endl;
    throw(s.str());
  }
  if(nrOfBand()!=other.nrOfBand()){
    std::ostringstream s;
    s << "Error: number of bands do not match";
    std::cerr << s.str() << std::endl;
    throw(s.str());
  }
  if(nrOfRow()!=other.nrOfRow()){
    std::ostringstream s;
    s << "Error: number of rows do not match";
    std::cerr << s.str() << std::endl;
    throw(s.str());
  }
  if(nrOfCol()!=other.nrOfCol()){
    std::ostringstream s;
    s << "Error: number of cols do not match";
    std::cerr << s.str() << std::endl;
    throw(s.str());
  }
  imgWriter.open(nrOfCol(),nrOfRow(),nrOfBand(),GDT_Byte);
  imgWriter.copyGeoTransform(*this);
  imgWriter.setProjection(this->getProjection());
  for(size_t iband=0;iband<nrOfBand();++iband){
#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
    for(int irow=0;irow<nrOfRow();++irow){
      for(int icol=0;icol<nrOfCol();++icol){
        if(readData(icol,irow,iband)>other.readData(icol,irow,iband))
          imgWriter.writeData(1,icol,irow,iband);
        else
          imgWriter.writeData(0,icol,irow,iband);
      }
    }
  }
}

void Jim::gt(double value, Jim& imgWriter){
  if(m_data.empty()){
    std::ostringstream s;
    s << "Error: Jim not initialized, m_data is empty";
    std::cerr << s.str() << std::endl;
    throw(s.str());
  }
  imgWriter.open(nrOfCol(),nrOfRow(),nrOfBand(),GDT_Byte);
  imgWriter.copyGeoTransform(*this);
  imgWriter.setProjection(this->getProjection());
  for(size_t iband=0;iband<nrOfBand();++iband){
#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
    for(int irow=0;irow<nrOfRow();++irow){
      for(int icol=0;icol<nrOfCol();++icol){
        if(readData(icol,irow,iband)>value)
          imgWriter.writeData(1,icol,irow,iband);
        else
          imgWriter.writeData(0,icol,irow,iband);
      }
    }
  }
}

std::shared_ptr<Jim> Jim::ge(std::shared_ptr<Jim> refJim){
  shared_ptr<Jim> imgWriter=Jim::createImg();
  this->ge(*refJim,*imgWriter);
  return(imgWriter);
}

std::shared_ptr<Jim> Jim::ge(double value){
  shared_ptr<Jim> imgWriter=Jim::createImg();
  this->ge(value,*imgWriter);
  return(imgWriter);
}

void Jim::ge(Jim& other, Jim& imgWriter){
  if(m_data.empty()){
    std::ostringstream s;
    s << "Error: Jim not initialized, m_data is empty";
    std::cerr << s.str() << std::endl;
    throw(s.str());
  }
  if(nrOfBand()!=other.nrOfBand()){
    std::ostringstream s;
    s << "Error: number of bands do not match";
    std::cerr << s.str() << std::endl;
    throw(s.str());
  }
  if(nrOfRow()!=other.nrOfRow()){
    std::ostringstream s;
    s << "Error: number of rows do not match";
    std::cerr << s.str() << std::endl;
    throw(s.str());
  }
  if(nrOfCol()!=other.nrOfCol()){
    std::ostringstream s;
    s << "Error: number of cols do not match";
    std::cerr << s.str() << std::endl;
    throw(s.str());
  }
  imgWriter.open(nrOfCol(),nrOfRow(),nrOfBand(),GDT_Byte);
  imgWriter.copyGeoTransform(*this);
  imgWriter.setProjection(this->getProjection());
  for(size_t iband=0;iband<nrOfBand();++iband){
#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
    for(int irow=0;irow<nrOfRow();++irow){
      for(int icol=0;icol<nrOfCol();++icol){
        if(readData(icol,irow,iband)>=other.readData(icol,irow,iband))
          imgWriter.writeData(1,icol,irow,iband);
        else
          imgWriter.writeData(0,icol,irow,iband);
      }
    }
  }
}

void Jim::ge(double value, Jim& imgWriter){
  if(m_data.empty()){
    std::ostringstream s;
    s << "Error: Jim not initialized, m_data is empty";
    std::cerr << s.str() << std::endl;
    throw(s.str());
  }
  imgWriter.open(nrOfCol(),nrOfRow(),nrOfBand(),GDT_Byte);
  imgWriter.copyGeoTransform(*this);
  imgWriter.setProjection(this->getProjection());
  for(size_t iband=0;iband<nrOfBand();++iband){
#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
    for(int irow=0;irow<nrOfRow();++irow){
      for(int icol=0;icol<nrOfCol();++icol){
        if(readData(icol,irow,iband)>=value)
          imgWriter.writeData(1,icol,irow,iband);
        else
          imgWriter.writeData(0,icol,irow,iband);
      }
    }
  }
}
