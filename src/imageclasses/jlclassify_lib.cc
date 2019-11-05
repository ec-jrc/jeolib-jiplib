/**********************************************************************
jlclassify_lib.cc: classify raster image
Author(s): Pieter.Kempeneers@ec.europa.eu
Copyright (c) 2016-2019 European Union (Joint Research Centre)
License EUPLv1.2

This file is part of jiplib
***********************************************************************/
#include <vector>
#include <map>
#include <sstream>
#include <memory>
#include "jlclassify_lib.h"
#include "Jim.h"
#include "VectorOgr.h"
#include "base/Optionjl.h"
#include "apps/AppFactory.h"
#include "jlsml_lib.h"

using namespace std;
using namespace app;

shared_ptr<Jim> Jim::classify(app::AppFactory& app){
  try{
    shared_ptr<Jim> imgWriter=createImg();
    classify(*imgWriter, app);
    return(imgWriter);
  }
  catch(string helpString){
    cerr << helpString << endl;
    throw;
  }
}

shared_ptr<VectorOgr> VectorOgr::classify(app::AppFactory& app){
  std::shared_ptr<VectorOgr> ogrWriter=VectorOgr::createVector();
  classify(*ogrWriter, app);
  return(ogrWriter);
}

void Jim::classify(Jim& imgWriter, app::AppFactory& app){
  //--------------------------- command line options ------------------------------------
  Optionjl<std::string> method_opt("m", "method", "classification method: 'svm' (support vector machine), 'ann' (artificial neural network), 'sml' (symbolic machine learning)");

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=method_opt.retrieveOption(app);
    if(method_opt.empty()){
      cout << endl;
      std::ostringstream helpStream;
      helpStream << "Error: no classification method provided" << std::endl;
      throw(helpStream.str());//help was invoked, stop processing
    }
    switch(getClassifier(method_opt[0])){
    case(SVM):
      classifySVM(imgWriter,app);
      break;
    case(ANN):
      classifyANN(imgWriter,app);
      break;
    case(SML):
      classifySML(imgWriter,app);
      break;
    default:
      std::ostringstream errorStream;
      errorStream << "Error: classification method " << method_opt[0] << "(" << getClassifier(method_opt[0]) << ")" << " is not supported" << std::endl;
      throw(errorStream.str());//help was invoked, stop processing
    }
  }
  catch(string predefinedString){
    std::cout << predefinedString << std::endl;
    throw;
  }
}


void VectorOgr::train(app::AppFactory& app){
  //--------------------------- command line options ------------------------------------
  Optionjl<std::string> method_opt("m", "method", "classification method: 'svm' (support vector machine), 'ann' (artificial neural network)");
  Optionjl<std::string> model_opt("model", "model", "Model filename to save trained classifier.");

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=method_opt.retrieveOption(app);
    model_opt.retrieveOption(app);

    if(method_opt.empty()){
      cout << endl;
      std::ostringstream helpStream;
      helpStream << "Error: no classification method provided" << std::endl;
      throw(helpStream.str());//help was invoked, stop processing
    }
    if(model_opt.empty()){
      std::string errorString="Error: filename to save model is emtpy";
      throw(errorString);
    }
    switch(getClassifier(method_opt[0])){
    case(SVM):{
      std::ofstream outputStream(model_opt[0]);
      outputStream << trainSVM(app);
      outputStream.close();
      break;
    }
    case(ANN):{
      std::ofstream outputStream(model_opt[0]);
      outputStream << trainANN(app);
      outputStream.close();
      break;
    }
    default:
      std::ostringstream errorStream;
      errorStream << "Error: classification method " << method_opt[0] << " not supported" << std::endl;
      throw(errorStream.str());//help was invoked, stop processing
    }
  }
  catch(string predefinedString){
    std::cerr << predefinedString << std::endl;
    throw;
  }
}

// std::string Jim::trainMem(JimList& referenceReader, app::AppFactory& app){
//   //--------------------------- command line options ------------------------------------
//   Optionjl<string> method_opt("m", "method", "classification method: 'svm' (support vector machine), 'ann' (artificial neural network)");

//   bool doProcess;//stop process when program was invoked with help option (-h --help)
//   try{
//     doProcess=method_opt.retrieveOption(app);
//     if(method_opt.empty()){
//       cout << endl;
//       std::ostringstream helpStream;
//       helpStream << "Error: no classification method provided" << std::endl;
//       throw(helpStream.str());//help was invoked, stop processing
//     }
//     switch(getClassifier(method_opt[0])){
//     case(SML):
//       switch(getDataType()){
//       case(GDT_Byte):
//         return(trainSML_t<unsigned char>(referenceReader, app));
//       case(GDT_Int16):
//         return(trainSML_t<unsigned short>(referenceReader, app));
//       case(GDT_UInt16):
//         return(trainSML_t<short>(referenceReader, app));
//       case(GDT_Int32):
//         return(trainSML_t<int>(referenceReader, app));
//       case(GDT_UInt32):
//         return(trainSML_t<unsigned int>(referenceReader, app));
//       case(GDT_Float32):
//         // return(trainSML<float>(referenceReader, app));
//       case(GDT_Float64):
//         // return(trainSML<double>(referenceReader, app));
//       default:
//         std::ostringstream errorStream;
//         errorStream << "Error: data type " << getDataType() << " not supported" << std::endl;
//         throw(errorStream.str());//help was invoked, stop processing
//       }
//     default:
//       std::ostringstream errorStream;
//       errorStream << "Error: classification method " << method_opt[0] << " not supported" << std::endl;
//       throw(errorStream.str());//help was invoked, stop processing
//     }
//     return(std::string());
//   }
//   catch(string predefinedString){
//     std::cerr << predefinedString << std::endl;
//     throw;
//   }
// }

// std::string VectorOgr::trainMem(app::AppFactory& app){
//   //--------------------------- command line options ------------------------------------
//   Optionjl<string> method_opt("m", "method", "classification method: 'svm' (support vector machine), 'ann' (artificial neural network)");

//   bool doProcess;//stop process when program was invoked with help option (-h --help)
//   try{
//     doProcess=method_opt.retrieveOption(app);
//     if(method_opt.empty()){
//       cout << endl;
//       std::ostringstream helpStream;
//       helpStream << "Error: no classification method provided" << std::endl;
//       throw(helpStream.str());//help was invoked, stop processing
//     }
//     switch(getClassifier(method_opt[0])){
//     case(SVM):
//       return(trainSVM(app));
//       break;
//     case(ANN):
//       return(trainANN(app));
//       break;
//     default:
//       std::ostringstream errorStream;
//       errorStream << "Error: classification method " << method_opt[0] << " not supported" << std::endl;
//       throw(errorStream.str());//help was invoked, stop processing
//     }
//     return(std::string());
//   }
//   catch(string predefinedString){
//     std::cerr << predefinedString << std::endl;
//     throw;
//   }
// }

void VectorOgr::classify(VectorOgr& ogrWriter, app::AppFactory& app){
  //--------------------------- command line options ------------------------------------
  Optionjl<string> method_opt("m", "method", "classification method: 'svm' (support vector machine), 'ann' (artificial neural network)");

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=method_opt.retrieveOption(app);
    if(method_opt.empty()){
      cout << endl;
      std::ostringstream helpStream;
      helpStream << "Error: no classification method provided" << std::endl;
      throw(helpStream.str());//help was invoked, stop processing
    }
    switch(getClassifier(method_opt[0])){
    case(SVM):
      classifySVM(ogrWriter,app);
      break;
    case(ANN):
      classifyANN(ogrWriter,app);
      break;
    default:
      std::ostringstream errorStream;
      errorStream << "Error: classification method " << method_opt[0] << " not supported" << std::endl;
      throw(errorStream.str());//help was invoked, stop processing
    }
  }
  catch(string predefinedString){
    std::cout << predefinedString << std::endl;
    throw;
  }
}
