/**********************************************************************
jlsml_lib.cc: classify raster image using SML
Author(s): Pieter.Kempeneers@ec.europa.eu
Copyright (c) 2016-2019 European Union (Joint Research Centre)
License EUPLv1.2

This file is part of jiplib
***********************************************************************/
#include "jlsml_lib.h"

void Jim::classifySML(Jim& imgWriter, app::AppFactory& app){
  switch(getDataType()){
  case(GDT_Byte):
    classifySML_t<unsigned char>(imgWriter,app);
    break;
  case(GDT_Int16):
    classifySML_t<short>(imgWriter,app);
    break;
  case(GDT_UInt16):
    classifySML_t<unsigned short>(imgWriter,app);
    break;
  case(GDT_Int32):
    classifySML_t<int>(imgWriter,app);
    break;
  case(GDT_UInt32):
    classifySML_t<unsigned int>(imgWriter,app);
    break;
  case(GDT_Float32):
  case(GDT_Float64):
  default:
    std::string errorString="Error: data type not supported";
  throw(errorString);
  break;
  }
}

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

std::shared_ptr<Jim> Jim::classifySML(app::AppFactory& app){
  try{
    std::shared_ptr<Jim> imgWriter=createImg();
    classifySML(*imgWriter, app);
    return(imgWriter);
  }
  catch(std::string helpString){
    std::cerr << helpString << std::endl;
    throw;
  }
}

void Jim::trainSML(JimList& referenceReader, app::AppFactory& app){
  //--------------------------- command line options ------------------------------------
  // Optionjl<std::string> model_opt("model", "model", "Model filename to save trained classifier.");
  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    // doProcess=model_opt.retrieveOption(app);
    // if(model_opt.empty()){
    //   std::string errorString="Error: filename to save model is emtpy";
    //   throw(errorString);
    // }
    // std::ofstream outputStream(model_opt[0]);
    switch(getDataType()){
    case(GDT_Byte):
      trainSML_t<unsigned char>(referenceReader, app);
      // outputStream << trainSML_t<unsigned char>(referenceReader, app);
      break;
    case(GDT_Int16):
      trainSML_t<unsigned short>(referenceReader, app);
      // outputStream << trainSML_t<unsigned short>(referenceReader, app);
      break;
    case(GDT_UInt16):
      trainSML_t<short>(referenceReader, app);
      // outputStream << trainSML_t<short>(referenceReader, app);
      break;
    case(GDT_Int32):
      trainSML_t<int>(referenceReader, app);
      // outputStream << trainSML_t<int>(referenceReader, app);
      break;
    case(GDT_UInt32):
      trainSML_t<unsigned int>(referenceReader, app);
      // outputStream << trainSML_t<unsigned int>(referenceReader, app);
      break;
    // case(GDT_Float32):
      // trainSML<float>(referenceReader, app);
      // outputStream << trainSML<float>(referenceReader, app);
      // break;
    // case(GDT_Float64):
      // trainSML<double>(referenceReader, app);
      // outputStream << trainSML<double>(referenceReader, app);
      // break;
    default:
      std::ostringstream errorStream;
    errorStream << "Error: data type " << getDataType() << " not supported" << std::endl;
    throw(errorStream.str());//help was invoked, stop processing
    break;
    }
    // outputStream.close();
  }
  catch(std::string predefinedString){
    std::cerr << predefinedString << std::endl;
    throw;
  }
}

void Jim::trainSML2d(JimList& referenceReader, app::AppFactory& app){
  //--------------------------- command line options ------------------------------------
  Optionjl<std::string> method_opt("m", "method", "classification method: 'sml' (symbolic machine learning)");
  Optionjl<std::string> model_opt("model", "model", "Model filename to save trained classifier.");

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=method_opt.retrieveOption(app);
    model_opt.retrieveOption(app);

    if(method_opt.empty()){
      std::cout << std::endl;
      std::ostringstream helpStream;
      helpStream << "Error: no classification method provided" << std::endl;
      throw(helpStream.str());//help was invoked, stop processing
    }
    if(model_opt.empty()){
      std::string errorString="Error: filename to save model is emtpy";
      throw(errorString);
    }
    std::ofstream outputStream(model_opt[0]);
    switch(getDataType()){
    case(GDT_Byte):
      outputStream << trainSML2d_t<unsigned char>(referenceReader, app);
      break;
    case(GDT_Int16):
      outputStream << trainSML2d_t<unsigned short>(referenceReader, app);
      break;
    case(GDT_UInt16):
      outputStream << trainSML2d_t<short>(referenceReader, app);
      break;
    case(GDT_Int32):
      outputStream << trainSML2d_t<int>(referenceReader, app);
      break;
    case(GDT_UInt32):
      outputStream << trainSML2d_t<unsigned int>(referenceReader, app);
      break;
    case(GDT_Float32):
      // outputStream << trainSML<float>(referenceReader, app);
      // break;
    case(GDT_Float64):
      // outputStream << trainSML<double>(referenceReader, app);
      // break;
    default:
      std::ostringstream errorStream;
    errorStream << "Error: data type " << getDataType() << " not supported" << std::endl;
    throw(errorStream.str());//help was invoked, stop processing
    break;
    }
    outputStream.close();
  }
  catch(std::string predefinedString){
    std::cerr << predefinedString << std::endl;
    throw;
  }
}
