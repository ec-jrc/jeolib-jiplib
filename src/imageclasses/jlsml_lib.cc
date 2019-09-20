/**********************************************************************
jlsml_lib.cc: classify raster image using SML
Author(s): Pieter.Kempeneers@ec.europa.eu
Copyright (c) 2016-2019 European Union (Joint Research Centre)
License EUPLv1.2

This file is part of jiplib
***********************************************************************/
#include "jlsml_lib.h"

void Jim::classifySML(Jim& imgWriter, JimList& referenceReader, app::AppFactory& app){
  switch(getDataType()){
  case(GDT_Byte):
    classifySML_t<unsigned char>(imgWriter,referenceReader,app);
    break;
  case(GDT_Int16):
    classifySML_t<short>(imgWriter,referenceReader,app);
    break;
  case(GDT_UInt16):
    classifySML_t<unsigned short>(imgWriter,referenceReader,app);
    break;
  case(GDT_Int32):
    classifySML_t<int>(imgWriter,referenceReader,app);
    break;
  case(GDT_UInt32):
    classifySML_t<unsigned int>(imgWriter,referenceReader,app);
    break;
  case(GDT_Float32):
  case(GDT_Float64):
  default:
    std::string errorString="Error: data type not supported";
    throw(errorString);
    break;
  }
}

std::shared_ptr<Jim> Jim::classifySML(JimList& referenceReader, app::AppFactory& app){
  try{
    std::shared_ptr<Jim> imgWriter=createImg();
    classifySML(*imgWriter, referenceReader, app);
    return(imgWriter);
  }
  catch(std::string helpString){
    std::cerr << helpString << std::endl;
    throw;
  }
}
