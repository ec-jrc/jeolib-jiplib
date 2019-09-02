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

/**
 * @param training (type: std::string) Training vector file. A single vector file contains all training features (must be set as: b0, b1, b2,...) for all classes (class numbers identified by label option). Use multiple training files for bootstrap aggregation (alternative to the bag and bsize options, where a random subset is taken from a single training file)
 * @param cv (type: unsigned short) (default: 0) N-fold cross validation mode
 * @param cmf (type: std::string) (default: ascii) Format for confusion matrix (ascii or latex)
 * @param tln (type: std::string) Training layer name(s)
 * @param class (type: std::string) List of class names.
 * @param reclass (type: short) List of class values (use same order as in class opt).
 * @param f (type: std::string) (default: SQLite) Output ogr format for active training sample
 * @param ct (type: std::string) Color table in ASCII format having 5 columns: id R G B ALFA (0: transparent, 255: solid)
 * @param label (type: std::string) (default: label) Attribute name for class label in training vector file.
 * @param prior (type: double) (default: 0) Prior probabilities for each class (e.g., -p 0.3 -p 0.3 -p 0.2 ). Used for input only (ignored for cross validation)
 * @param extent (type: std::string) Only classify within extent from polygons in vector file
 * @param mask (type: std::string) Only classify within specified mask. For raster mask, set nodata values with the option msknodata.
 * @param msknodata (type: short) (default: 0) Mask value(s) not to consider for classification. Values will be taken over in classification image.
 * @param nodata (type: unsigned short) (default: 0) Nodata value to put where image is masked as nodata
 * @param band (type: unsigned int) Band index (starting from 0, either use band option or use start to end)
 * @param startband (type: unsigned int) Start band sequence number
 * @param endband (type: unsigned int) End band sequence number
 * @param balance (type: unsigned int) (default: 0) Balance the input data to this number of samples for each class
 * @param min (type: unsigned int) (default: 0) If number of training pixels is less then min, do not take this class into account (0: consider all classes)
 * @param bag (type: unsigned short) (default: 1) Number of bootstrap aggregations
 * @param bagsize (type: int) (default: 100) Percentage of features used from available training features for each bootstrap aggregation (one size for all classes, or a different size for each class respectively
 * @param comb (type: unsigned short) (default: 0) How to combine bootstrap aggregation classifiers (0: sum rule, 1: product rule, 2: max rule). Also used to aggregate classes with rc option.
 * @param classbag (type: std::string) Output for each individual bootstrap aggregation
 * @param prob (type: std::string) Probability image.
 * @param priorimg (type: std::string) (default: ) Prior probability image (multi-band img with band for each class
 * @param offset (type: double) (default: 0) Offset value for each spectral band input features: refl[band]=(DN[band]-offset[band])/scale[band]
 * @param scale (type: double) (default: 0) Scale value for each spectral band input features: refl=(DN[band]-offset[band])/scale[band] (use 0 if scale min and max in each band to -1.0 and 1.0)
 * @param random (type: bool) (default: 1) Randomize training data for balancing and bagging
 * @return shared pointer to classified image object
 **/
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

/**
 * @param app application specific option arguments
 * @return output Vector
 **/
shared_ptr<VectorOgr> VectorOgr::classify(app::AppFactory& app){
  std::shared_ptr<VectorOgr> ogrWriter=VectorOgr::createVector();
  classify(*ogrWriter, app);
  return(ogrWriter);
}

/**
 * @param training (type: std::string) Training vector file. A single vector file contains all training features (must be set as: b0, b1, b2,...) for all classes (class numbers identified by label option). Use multiple training files for bootstrap aggregation (alternative to the bag and bsize options, where a random subset is taken from a single training file)
 * @param cv (type: unsigned short) (default: 0) N-fold cross validation mode
 * @param cmf (type: std::string) (default: ascii) Format for confusion matrix (ascii or latex)
 * @param tln (type: std::string) Training layer name(s)
 * @param class (type: std::string) List of class names.
 * @param reclass (type: short) List of class values (use same order as in class opt).
 * @param f (type: std::string) (default: SQLite) Output ogr format for active training sample
 * @param ct (type: std::string) Color table in ASCII format having 5 columns: id R G B ALFA (0: transparent, 255: solid)
 * @param label (type: std::string) (default: label) Attribute name for class label in training vector file.
 * @param prior (type: double) (default: 0) Prior probabilities for each class (e.g., -p 0.3 -p 0.3 -p 0.2 ). Used for input only (ignored for cross validation)
 * @param extent (type: std::string) Only classify within extent from polygons in vector file
 * @param mask (type: std::string) Only classify within specified mask. For raster mask, set nodata values with the option msknodata.
 * @param msknodata (type: short) (default: 0) Mask value(s) not to consider for classification. Values will be taken over in classification image.
 * @param nodata (type: unsigned short) (default: 0) Nodata value to put where image is masked as nodata
 * @param band (type: unsigned int) Band index (starting from 0, either use band option or use start to end)
 * @param startband (type: unsigned int) Start band sequence number
 * @param endband (type: unsigned int) End band sequence number
 * @param balance (type: unsigned int) (default: 0) Balance the input data to this number of samples for each class
 * @param min (type: unsigned int) (default: 0) If number of training pixels is less then min, do not take this class into account (0: consider all classes)
 * @param bag (type: unsigned short) (default: 1) Number of bootstrap aggregations
 * @param bagsize (type: int) (default: 100) Percentage of features used from available training features for each bootstrap aggregation (one size for all classes, or a different size for each class respectively
 * @param comb (type: unsigned short) (default: 0) How to combine bootstrap aggregation classifiers (0: sum rule, 1: product rule, 2: max rule). Also used to aggregate classes with rc option.
 * @param classbag (type: std::string) Output for each individual bootstrap aggregation
 * @param prob (type: std::string) Probability image.
 * @param priorimg (type: std::string) (default: ) Prior probability image (multi-band img with band for each class
 * @param offset (type: double) (default: 0) Offset value for each spectral band input features: refl[band]=(DN[band]-offset[band])/scale[band]
 * @param scale (type: double) (default: 0) Scale value for each spectral band input features: refl=(DN[band]-offset[band])/scale[band] (use 0 if scale min and max in each band to -1.0 and 1.0)
 * @param random (type: bool) (default: 1) Randomize training data for balancing and bagging
 * @return CE_None if successful, CE_Failure if failed
 **/
void Jim::classify(Jim& imgWriter, app::AppFactory& app){
  vector<double> priors;

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
      return(classifySVM(imgWriter,app));
    case(ANN):
      return(classifyANN(imgWriter,app));
    case(SML):
      switch(getDataType()){
      case(GDT_Byte):
        return(classifySML<unsigned char>(imgWriter, app));
      case(GDT_Int16):
        return(classifySML<unsigned short>(imgWriter, app));
      case(GDT_UInt16):
        return(classifySML<short>(imgWriter, app));
      case(GDT_Int32):
        return(classifySML<int>(imgWriter, app));
      case(GDT_UInt32):
        return(classifySML<unsigned int>(imgWriter, app));
      case(GDT_Float32):
        // return(classifySML<float>(imgWriter, app));
      case(GDT_Float64):
        // return(classifySML<double>(imgWriter, app));
      default:
        std::ostringstream errorStream;
        errorStream << "Error: data type " << getDataType() << " not supported" << std::endl;
        throw(errorStream.str());//help was invoked, stop processing
      }
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


CPLErr Jim::train(JimList& referenceReader, app::AppFactory& app){
  //--------------------------- command line options ------------------------------------
  Optionjl<string> method_opt("m", "method", "classification method: 'sml' (symbolic machine learning)");
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
    case(SML):{
      std::ofstream outputStream(model_opt[0]);
      switch(getDataType()){
      case(GDT_Byte):
        outputStream << trainSML<unsigned char>(referenceReader, app);
        break;
      case(GDT_Int16):
        outputStream << trainSML<unsigned short>(referenceReader, app);
        break;
      case(GDT_UInt16):
        outputStream << trainSML<short>(referenceReader, app);
        break;
      case(GDT_Int32):
        outputStream << trainSML<int>(referenceReader, app);
        break;
      case(GDT_UInt32):
        outputStream << trainSML<unsigned int>(referenceReader, app);
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
      break;
    }
    default:
      std::ostringstream errorStream;
      errorStream << "Error: classification method " << method_opt[0] << " not supported" << std::endl;
      throw(errorStream.str());//help was invoked, stop processing
    }
    return(CE_None);
  }
  catch(string predefinedString){
    std::cerr << predefinedString << std::endl;
    throw;
  }
}

OGRErr VectorOgr::train(app::AppFactory& app){
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
    return(OGRERR_NONE);
  }
  catch(string predefinedString){
    std::cerr << predefinedString << std::endl;
    throw;
  }
}

std::string Jim::trainMem(JimList& referenceReader, app::AppFactory& app){
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
    case(SML):
      switch(getDataType()){
      case(GDT_Byte):
        return(trainSML<unsigned char>(referenceReader, app));
      case(GDT_Int16):
        return(trainSML<unsigned short>(referenceReader, app));
      case(GDT_UInt16):
        return(trainSML<short>(referenceReader, app));
      case(GDT_Int32):
        return(trainSML<int>(referenceReader, app));
      case(GDT_UInt32):
        return(trainSML<unsigned int>(referenceReader, app));
      case(GDT_Float32):
        // return(trainSML<float>(referenceReader, app));
      case(GDT_Float64):
        // return(trainSML<double>(referenceReader, app));
      default:
        std::ostringstream errorStream;
        errorStream << "Error: data type " << getDataType() << " not supported" << std::endl;
        throw(errorStream.str());//help was invoked, stop processing
      }
    default:
      std::ostringstream errorStream;
      errorStream << "Error: classification method " << method_opt[0] << " not supported" << std::endl;
      throw(errorStream.str());//help was invoked, stop processing
    }
    return(std::string());
  }
  catch(string predefinedString){
    std::cerr << predefinedString << std::endl;
    throw;
  }
}

std::string VectorOgr::trainMem(app::AppFactory& app){
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
      return(trainSVM(app));
      break;
    case(ANN):
      return(trainANN(app));
      break;
    default:
      std::ostringstream errorStream;
      errorStream << "Error: classification method " << method_opt[0] << " not supported" << std::endl;
      throw(errorStream.str());//help was invoked, stop processing
    }
    return(std::string());
  }
  catch(string predefinedString){
    std::cerr << predefinedString << std::endl;
    throw;
  }
}

void VectorOgr::classify(VectorOgr& ogrWriter, app::AppFactory& app){
  vector<double> priors;

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
