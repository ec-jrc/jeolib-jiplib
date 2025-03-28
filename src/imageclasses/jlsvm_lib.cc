/**********************************************************************
jlsvm_lib.cc: classify raster image using Support Vector Machine
Author(s): Pieter.Kempeneers@ec.europa.eu
Copyright (C) 2016-2020 European Union (Joint Research Centre)

This file is part of jiplib.

jiplib is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

jiplib is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with jiplib.  If not, see <https://www.gnu.org/licenses/>.
***********************************************************************/
#include <stdlib.h>
#include <vector>
#include <map>
#include <memory>
#include <algorithm>
#include "Jim.h"
#include "VectorOgr.h"
#include "base/Optionjl.h"
#include "base/PosValue.h"
#include "algorithms/ConfusionMatrix.h"
#include "algorithms/StatFactory.h"
#include "algorithms/svm.h"
#include "apps/AppFactory.h"

namespace svm{
  enum SVM_TYPE {C_SVC=0, nu_SVC=1,one_class=2, epsilon_SVR=3, nu_SVR=4};
  enum KERNEL_TYPE {linear=0,polynomial=1,radial=2,sigmoid=3};
}

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

using namespace std;
using namespace app;

// OGRErr VectorOgr::trainSVM(app::AppFactory& app){
std::string VectorOgr::trainSVM(app::AppFactory& app){
  confusionmatrix::ConfusionMatrix cm;

  //--------------------------- command line options ------------------------------------
  // Optionjl<std::string> model_opt("model", "model", "Model filename to save trained classifier.");
  // Optionjl<std::string> tlayer_opt("tln", "tln", "Sample layer name(s)");
  // Optionjl<std::string> attribute_opt("af", "af", "Attribute filter");
  Optionjl<std::string> label_opt("label", "label", "Attribute name for class label in training vector file.","label");
  Optionjl<unsigned int> balance_opt("bal", "balance", "Balance the input data to this number of samples for each class", 0);
  Optionjl<bool> random_opt("random", "random", "Randomize training data for balancing", true, 2);
  Optionjl<unsigned int> minSize_opt("min", "min", "If number of training pixels is less then min, do not take this class into account (0: consider all classes)", 0);
  // Optionjl<unsigned int> band_opt("b", "band", "Band index (starting from 0, either use band option or use start to end)");
  // Optionjl<unsigned int> bstart_opt("sband", "startband", "Start band sequence number");
  // Optionjl<unsigned int> bend_opt("eband", "endband", "End band sequence number");
  Optionjl<std::string> bandNames_opt("bn", "bandname", "Band name(s) to use. Leave empty to use all bands");
  Optionjl<double> offset_opt("offset", "offset", "Offset value for each spectral band input features: refl[band]=(DN[band]-offset[band])/scale[band]", 0.0);
  Optionjl<double> scale_opt("scale", "scale", "Scale value for each spectral band input features: refl=(DN[band]-offset[band])/scale[band] (use 0 if scale min and max in each band to -1.0 and 1.0)", 0.0);
  Optionjl<unsigned short> cv_opt("cv", "cv", "N-fold cross validation mode",0);
  Optionjl<std::string> cmformat_opt("cmf","cmf","Format for confusion matrix (ascii or latex)","ascii");
  Optionjl<std::string> svm_type_opt("svmt", "svmtype", "Type of SVM (C_SVC, nu_SVC,one_class, epsilon_SVR, nu_SVR)","C_SVC");
  Optionjl<std::string> kernel_type_opt("kt", "kerneltype", "Type of kernel function (linear,polynomial,radial,sigmoid) ","radial");
  Optionjl<unsigned short> kernel_degree_opt("kd", "kd", "Degree in kernel function",3);
  Optionjl<float> gamma_opt("g", "gamma", "Gamma in kernel function",1.0);
  Optionjl<float> coef0_opt("c0", "coef0", "Coef0 in kernel function",0);
  Optionjl<float> ccost_opt("cc", "ccost", "The parameter C of C_SVC, epsilon_SVR, and nu_SVR",1000);
  Optionjl<float> nu_opt("nu", "nu", "The parameter nu of nu_SVC, one_class SVM, and nu_SVR",0.5);
  Optionjl<float> epsilon_loss_opt("eloss", "eloss", "The epsilon in loss function of epsilon_SVR",0.1);
  Optionjl<int> cache_opt("cache", "cache", "Cache memory size in MB",100);
  Optionjl<float> epsilon_tol_opt("etol", "etol", "The tolerance of termination criterion",0.001);
  Optionjl<bool> shrinking_opt("shrink", "shrink", "Whether to use the shrinking heuristics",false);
  Optionjl<bool> prob_est_opt("pe", "probest", "Whether to train a SVC or SVR model for probability estimates",true,2);
  // Optionjl<bool> weight_opt("wi", "wi", "Set the parameter C of class i to weight*C, for C_SVC",true);
  Optionjl<std::string> classname_opt("c", "class", "List of class names.");
  Optionjl<short> classvalue_opt("r", "reclass", "List of class values (use same order as in class opt).");
  Optionjl<short> verbose_opt("v", "verbose", "Verbose level",0,2);

  // band_opt.setHide(1);
  // bstart_opt.setHide(1);
  // bend_opt.setHide(1);
  minSize_opt.setHide(1);
  svm_type_opt.setHide(1);
  kernel_type_opt.setHide(1);
  kernel_degree_opt.setHide(1);
  coef0_opt.setHide(1);
  nu_opt.setHide(1);
  epsilon_loss_opt.setHide(1);
  cache_opt.setHide(1);
  epsilon_tol_opt.setHide(1);
  shrinking_opt.setHide(1);
  prob_est_opt.setHide(1);
  random_opt.setHide(1);
  verbose_opt.setHide(2);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=label_opt.retrieveOption(app);
    // doProcess=model_opt.retrieveOption(app);
    // tlayer_opt.retrieveOption(app);
    // attribute_opt.retrieveOption(app);
    balance_opt.retrieveOption(app);
    random_opt.retrieveOption(app);
    minSize_opt.retrieveOption(app);
    // band_opt.retrieveOption(app);
    // bstart_opt.retrieveOption(app);
    // bend_opt.retrieveOption(app);
    bandNames_opt.retrieveOption(app);
    offset_opt.retrieveOption(app);
    scale_opt.retrieveOption(app);
    cv_opt.retrieveOption(app);
    cmformat_opt.retrieveOption(app);
    svm_type_opt.retrieveOption(app);
    kernel_type_opt.retrieveOption(app);
    kernel_degree_opt.retrieveOption(app);
    gamma_opt.retrieveOption(app);
    coef0_opt.retrieveOption(app);
    ccost_opt.retrieveOption(app);
    nu_opt.retrieveOption(app);
    epsilon_loss_opt.retrieveOption(app);
    cache_opt.retrieveOption(app);
    epsilon_tol_opt.retrieveOption(app);
    shrinking_opt.retrieveOption(app);
    prob_est_opt.retrieveOption(app);
    classname_opt.retrieveOption(app);
    classvalue_opt.retrieveOption(app);
    verbose_opt.retrieveOption(app);
    // memory_opt.retrieveOption(app);

    if(!doProcess){
      cout << endl;
      std::ostringstream helpStream;
      helpStream << "short option -h shows basic options only, use long option --help to show all options" << std::endl;
      throw(helpStream.str());//help was invoked, stop processing
    }

    std::map<std::string, svm::SVM_TYPE> svmMap;

    svmMap["C_SVC"]=svm::C_SVC;
    svmMap["nu_SVC"]=svm::nu_SVC;
    svmMap["one_class"]=svm::one_class;
    svmMap["epsilon_SVR"]=svm::epsilon_SVR;
    svmMap["nu_SVR"]=svm::nu_SVR;

    std::map<std::string, svm::KERNEL_TYPE> kernelMap;

    kernelMap["linear"]=svm::linear;
    kernelMap["polynomial"]=svm::polynomial;
    kernelMap["radial"]=svm::radial;
    kernelMap["sigmoid;"]=svm::sigmoid;

    // if(model_opt.empty()){
    //   std::string errorString="Error: filename to save model is empty";
    //   throw(errorString);
    // }

    unsigned int nclass=0;
    unsigned int nband=0;
    struct svm_model* theModel=NULL;
    svm_parameter param;
    svm_problem prob;
    svm_node * x_space;

    map<string,Vector2d<float> > trainingMap;
    vector< Vector2d<float> > trainingPixels;//[class][sample][band]
    map<string,short> classValueMap;
    vector<std::string> nameVector;
    if(classname_opt.size()){
      assert(classname_opt.size()==classvalue_opt.size());
      for(unsigned int iclass=0;iclass<classname_opt.size();++iclass)
        classValueMap[classname_opt[iclass]]=classvalue_opt[iclass];
    }
    // setAttributeFilter(attribute_opt[0]);
    sortByLabel(trainingMap,label_opt[0],bandNames_opt);

    //convert trainingMap to trainingPixels, removing small classes (< minSize_opt[0])
    map<string,Vector2d<float> >::iterator mapit=trainingMap.begin();
    while(mapit!=trainingMap.end()){
      //delete small classes
      if((mapit->second).size()<minSize_opt[0]){
        trainingMap.erase(mapit);
        continue;
      }
      trainingPixels.push_back(mapit->second);
      if(verbose_opt[0]>1)
        std::cout << mapit->first << ": " << (mapit->second).size() << " samples" << std::endl;
      ++mapit;
    }
    nclass=trainingPixels.size();
    if(classname_opt.size())
      assert(nclass==classname_opt.size());
    nband=trainingPixels[0][0].size();
    if(verbose_opt[0]>0){
      std::cout << "nclass: " << nclass << std::endl;
      std::cout << "nband: " << nband << std::endl;
    }
    //balance training data
    if(balance_opt[0]>0){
      while(balance_opt.size()<nclass)
        balance_opt.push_back(balance_opt.back());
      if(random_opt[0])
        srand(time(NULL));
      int totalSamples=0;
      for(short iclass=0;iclass<nclass;++iclass){
        if(trainingPixels[iclass].size()>balance_opt[iclass]){
          while(trainingPixels[iclass].size()>balance_opt[iclass]){
            int index=rand()%trainingPixels[iclass].size();
            trainingPixels[iclass].erase(trainingPixels[iclass].begin()+index);
          }
        }
        else{
          int oldsize=trainingPixels[iclass].size();
          for(int isample=trainingPixels[iclass].size();isample<balance_opt[iclass];++isample){
            int index = rand()%oldsize;
            trainingPixels[iclass].push_back(trainingPixels[iclass][index]);
          }
        }
        totalSamples+=trainingPixels[iclass].size();
      }
    }

    statfactory::StatFactory stat;
    vector<double> offset;
    vector<double> scale;
    //set scale and offset
    offset.resize(nband);
    scale.resize(nband);
    for(int iband=0;iband<nband;++iband){
      offset[iband]=(offset_opt.size()==1)?offset_opt[0]:offset_opt[iband];
      scale[iband]=(scale_opt.size()==1)?scale_opt[0]:scale_opt[iband];
      //search for min and maximum
      if(scale[iband]<=0){
        float theMin=trainingPixels[0][0][iband];
        float theMax=trainingPixels[0][0][iband];
        for(short iclass=0;iclass<nclass;++iclass){
          for(int isample=0;isample<trainingPixels[iclass].size();++isample){
            if(theMin>trainingPixels[iclass][isample][iband])
              theMin=trainingPixels[iclass][isample][iband];
            if(theMax<trainingPixels[iclass][isample][iband])
              theMax=trainingPixels[iclass][isample][iband];
          }
        }
        offset[iband]=theMin+(theMax-theMin)/2.0;
        scale[iband]=(theMax-theMin)/2.0;
      }
    }
    // map<string,Vector2d<float> >::iterator mapit=trainingMap.begin();

    mapit=trainingMap.begin();
    bool doSort=true;
    while(mapit!=trainingMap.end()){
      nameVector.push_back(mapit->first);
      if(classValueMap.size()){
        //check if name in training is covered by classname_opt (values can not be 0)
        if(classValueMap[mapit->first]>0){
          if(cm.getClassIndex(type2string<short>(classValueMap[mapit->first]))<0){
            cm.pushBackClassName(type2string<short>(classValueMap[mapit->first]),doSort);
          }
        }
        else{
          std::string errorString="Error: names in classname option are not complete, please check names in training vector and make sure classvalue is > 0";
          throw(errorString);
        }
      }
      else
        cm.pushBackClassName(mapit->first,doSort);
      ++mapit;
    }
    if(classname_opt.empty()){
      //std::cerr << "Warning: no class name and value pair provided for all " << nclass << " classes, using string2type<int> instead!" << std::endl;
      for(int iclass=0;iclass<nclass;++iclass){
        if(verbose_opt[0])
          std::cout << iclass << " " << cm.getClass(iclass) << " -> " << string2type<short>(cm.getClass(iclass)) << std::endl;
        classValueMap[cm.getClass(iclass)]=string2type<short>(cm.getClass(iclass));
      }
    }

    //todo: fill x_space directly from mapPixels (not going through trainingPixels)?
    //Calculate features of training set
    vector< Vector2d<float> > trainingFeatures(nclass);
    for(short iclass=0;iclass<nclass;++iclass){
      int nctraining=0;
      if(verbose_opt[0]>=1)
        std::cout << "calculating features for class " << iclass << std::endl;
      if(random_opt[0])
        srand(time(NULL));
      nctraining=trainingPixels[iclass].size();
      if(nctraining<=0)
        nctraining=1;
      assert(nctraining<=trainingPixels[iclass].size());
      if(verbose_opt[0]>1)
        std::cout << "nctraining (class " << iclass << "): " << nctraining << std::endl;
      trainingFeatures[iclass].resize(nctraining);
      for(int isample=0;isample<nctraining;++isample){
        //scale pixel values according to scale and offset!!!
        for(int iband=0;iband<nband;++iband){
          float value=trainingPixels[iclass][isample][iband];
          trainingFeatures[iclass][isample].push_back((value-offset[iband])/scale[iband]);
        }
      }
      assert(trainingFeatures[iclass].size()==nctraining);
    }

    unsigned int nFeatures=trainingFeatures[0][0].size();
    unsigned int ntraining=0;
    for(short iclass=0;iclass<nclass;++iclass)
      ntraining+=trainingFeatures[iclass].size();
    prob.l=ntraining;
    prob.y = Malloc(double,prob.l);
    prob.x = Malloc(struct svm_node *,prob.l);
    x_space = Malloc(struct svm_node,(nFeatures+1)*ntraining);
    unsigned long int spaceIndex=0;
    int lIndex=0;
    for(short iclass=0;iclass<nclass;++iclass){
      for(int isample=0;isample<trainingFeatures[iclass].size();++isample){
        prob.x[lIndex]=&(x_space[spaceIndex]);
        for(int ifeature=0;ifeature<nFeatures;++ifeature){
          x_space[spaceIndex].index=ifeature+1;
          x_space[spaceIndex].value=trainingFeatures[iclass][isample][ifeature];
          ++spaceIndex;
        }
        x_space[spaceIndex++].index=-1;
        prob.y[lIndex]=iclass;
        ++lIndex;
      }
    }

    //set SVM parameters through command line options
    param.svm_type = svmMap[svm_type_opt[0]];
    param.kernel_type = kernelMap[kernel_type_opt[0]];
    param.degree = kernel_degree_opt[0];
    param.gamma = (gamma_opt[0]>0)? gamma_opt[0] : 1.0/nFeatures;
    param.coef0 = coef0_opt[0];
    param.nu = nu_opt[0];
    param.cache_size = cache_opt[0];
    param.C = ccost_opt[0];
    param.eps = epsilon_tol_opt[0];
    param.p = epsilon_loss_opt[0];
    param.shrinking = (shrinking_opt[0])? 1 : 0;
    param.probability = (prob_est_opt[0])? 1 : 0;
    param.nr_weight = 0;//not used: I use priors and balancing
    param.weight_label = NULL;
    param.weight = NULL;
    param.verbose=(verbose_opt[0]>1)? true:false;

    //checking parameters
    svm_check_parameter(&prob,&param);
    //parameters ok, training
    theModel=svm_train(&prob,&param);
    //SVM is now trained
    if(cv_opt[0]>1){
      //Cross validating
      double *target = Malloc(double,prob.l);
      svm_cross_validation(&prob,&param,cv_opt[0],target);
      assert(param.svm_type != EPSILON_SVR&&param.svm_type != NU_SVR);//only for regression
      for(int i=0;i<prob.l;i++){
        string refClassName=nameVector[prob.y[i]];
        string className=nameVector[target[i]];
        if(classValueMap.size())
          cm.incrementResult(type2string<short>(classValueMap[refClassName]),type2string<short>(classValueMap[className]),1.0);
        else
          cm.incrementResult(cm.getClass(prob.y[i]),cm.getClass(target[i]),1.0);
      }
      free(target);
    }
    // *NOTE* Because svm_model contains pointers to svm_problem, you can
    // not free the memory used by svm_problem if you are still using the
    // svm_model produced by svm_train().
    if(cv_opt[0]>1){
      assert(cm.nReference());
      cm.setFormat(cmformat_opt[0]);
      cm.reportSE95(false);
      std::cout << cm << std::endl;
    }

    // svm_save_model(model_opt[0],theModel,offset, scale, classValueMap, nameVector, bandNames_opt);
    std::ostringstream outputStream;
    svm_serialize_model(outputStream,theModel,offset, scale, classValueMap, nameVector, bandNames_opt);
    svm_destroy_param(&param);
    free(prob.y);
    free(prob.x);
    free(x_space);
    svm_free_and_destroy_model(&theModel);
    // return OGRERR_NONE;
    return outputStream.str();
  }
  catch(std::string errorString){
    std::cerr << errorString << std::endl;
    // return(OGRERR_FAILURE);
    throw;
  }
}

/**
 * @param app application specific option arguments
 * @return output classified raster dataset
 **/
shared_ptr<Jim> Jim::classifySVM(app::AppFactory& app){
  try{
    shared_ptr<Jim> imgWriter=createImg();
    classifySVM(*imgWriter, app);
    return(imgWriter);
  }
  catch(string helpString){
    cerr << helpString << endl;
    throw;
  }
}

/**
 * @param app application specific option arguments
 * @return output classified raster dataset
 **/
// shared_ptr<Jim> Jim::svm(app::AppFactory& app){
//   try{
//     shared_ptr<Jim> imgWriter=createImg();
//     svm(*imgWriter, app);
//     return(imgWriter);
//   }
//   catch(string helpString){
//     cerr << helpString << endl;
//     return(0);
//   }
// }


/**
 * @param imgWriter output classified raster dataset
 * @param app application specific option arguments
 * @return CE_None if successful, CE_Failure if failed
 **/
void Jim::classifySVM(Jim& imgWriter, app::AppFactory& app){
  vector<double> priors;

  //--------------------------- command line options ------------------------------------
  Optionjl<std::string> model_opt("model", "model", "Model filename for trained classifier.");
  Optionjl<unsigned int> band_opt("b", "band", "Band index (starting from 0). The band order must correspond to the band names defined in the model. Leave empty to use all bands");
  // Optionjl<std::string> bandNames_opt("bn", "bandname", "Band name(s) to use. Leave empty to use all bands");
  Optionjl<unsigned int> bstart_opt("sband", "startband", "Start band sequence number");
  Optionjl<unsigned int> bend_opt("eband", "endband", "End band sequence number");
  // Optionjl<double> offset_opt("offset", "offset", "Offset value for each spectral band input features: refl[band]=(DN[band]-offset[band])/scale[band]", 0.0);
  // Optionjl<double> scale_opt("scale", "scale", "Scale value for each spectral band input features: refl=(DN[band]-offset[band])/scale[band] (use 0 if scale min and max in each band to -1.0 and 1.0)", 0.0);
  Optionjl<double> priors_opt("prior", "prior", "Prior probabilities for each class (e.g., -p 0.3 -p 0.3 -p 0.2 ). Used for input only", 0.0);
  Optionjl<string> priorimg_opt("pim", "priorimg", "Prior probability image (multi-band img with band for each class","",2);
  Optionjl<string> extent_opt("e", "extent", "Only classify within extent from polygons in vector file");
  Optionjl<string> eoption_opt("eo","eo", "special extent options controlling rasterization: ATTRIBUTE|CHUNKYSIZE|ALL_TOUCHED|BURN_VALUE_FROM|MERGE_ALG, e.g., -eo ATTRIBUTE=fieldname");
  Optionjl<string> mask_opt("m", "mask", "Only classify within specified mask. For raster mask, set nodata values with the option msknodata.");
  Optionjl<short> msknodata_opt("msknodata", "msknodata", "Mask value(s) not to consider for classification. Values will be taken over in classification image.", 0);
  Optionjl<double> srcnodata_opt("srcnodata", "srcnodata", "Nodata value in source",0);
  Optionjl<unsigned short> dstnodata_opt("dstnodata", "dstnodata", "Nodata value to put where image is masked as nodata", 0);
  // Optionjl<unsigned short> nodata_opt("nodata", "nodata", "Nodata value to put where image is masked as nodata", 0);
  Optionjl<string> colorTable_opt("ct", "ct", "Color table in ASCII format having 5 columns: id R G B ALFA (0: transparent, 255: solid)");
  Optionjl<string> prob_opt("prob", "prob", "Probability image.");
  Optionjl<string> entropy_opt("entropy", "entropy", "Entropy image (measure for uncertainty of classifier output","",2);
  // Optionjl<string> classname_opt("c", "class", "List of class names.");
  // Optionjl<short> classvalue_opt("r", "reclass", "List of class values (use same order as in class opt).");
  Optionjl<short> verbose_opt("v", "verbose", "Verbose level",0,2);

  extent_opt.setHide(1);
  eoption_opt.setHide(1);
  band_opt.setHide(1);
  bstart_opt.setHide(1);
  bend_opt.setHide(1);
  prob_opt.setHide(1);
  priorimg_opt.setHide(1);
  // offset_opt.setHide(1);
  // scale_opt.setHide(1);
  verbose_opt.setHide(2);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=model_opt.retrieveOption(app);
    colorTable_opt.retrieveOption(app);
    extent_opt.retrieveOption(app);
    eoption_opt.retrieveOption(app);
    mask_opt.retrieveOption(app);
    msknodata_opt.retrieveOption(app);
    srcnodata_opt.retrieveOption(app);
    dstnodata_opt.retrieveOption(app);
    // nodata_opt.retrieveOption(app);
    // Advanced options
    band_opt.retrieveOption(app);
    // bandNames_opt.retrieveOption(app);
    bstart_opt.retrieveOption(app);
    bend_opt.retrieveOption(app);
    prob_opt.retrieveOption(app);
    priorimg_opt.retrieveOption(app);
    // offset_opt.retrieveOption(app);
    // scale_opt.retrieveOption(app);
    priors_opt.retrieveOption(app);
    entropy_opt.retrieveOption(app);
    // classname_opt.retrieveOption(app);
    // classvalue_opt.retrieveOption(app);
    verbose_opt.retrieveOption(app);
    // memory_opt.retrieveOption(app);

    if(!doProcess){
      cout << endl;
      std::ostringstream helpStream;
      helpStream << "short option -h shows basic options only, use long option --help to show all options" << std::endl;
      throw(helpStream.str());//help was invoked, stop processing
    }

    if(entropy_opt[0]=="")
      entropy_opt.clear();
    if(priorimg_opt[0]=="")
      priorimg_opt.clear();

    if(model_opt.empty()){
      string errorString="Error: no model filename provided";
      throw(errorString);
    }

    VectorOgr extentReader;
    Jim maskReader;

    double ulx=0;
    double uly=0;
    double lrx=0;
    double lry=0;

    if(extent_opt.size()){
      if(mask_opt.size()){
        string errorString="Error: can only either mask or extent, not both";
        throw(errorString);
      }
      std::vector<std::string> layernames;
      layernames.clear();
      extentReader.open(extent_opt[0],layernames,true);
      // readLayer = extentReader.getDataSource()->GetLayer(0);
      extentReader.getExtent(ulx,uly,lrx,lry);
      maskReader.open(this->nrOfCol(),this->nrOfRow(),1,GDT_Float64);
      double gt[6];
      this->getGeoTransform(gt);
      maskReader.setGeoTransform(gt);
      maskReader.setProjection(this->getProjection());
      // vector<double> burnValues(1,1);//burn value is 1 (single band)
      // maskReader.rasterizeBuf(extentReader);
      maskReader.d_rasterizeBuf(extentReader,dstnodata_opt[0],eoption_opt);
      maskReader.GDALSetNoDataValue(dstnodata_opt[0]);
      extentReader.close();
    }

    unsigned int totalSamples=0;
    struct svm_model* svm;
    struct svm_parameter param;

    std::vector<double> offset;
    std::vector<double> scale;
    map<string,short> classValueMap;
    vector<std::string> nameVector;
    vector<std::string> bandNames;


    struct svm_model* theModel=svm_load_model(model_opt[0].c_str(),offset,scale,classValueMap,nameVector,bandNames);

    // std::map<std::string,short>::const_iterator mapit=classValueMap.begin();
    // for(int index=0;index<nameVector.size();++index)
    //   std::cout << nameVector[index] << std::endl;
    unsigned int nclass=theModel->nr_class;
    unsigned int nband=bandNames.size()?bandNames.size():offset.size();//todo: find a more elegant way to define number of bands (in model?)

    vector<unsigned long int> ntotalClass(nclass);

    //normalize priors from command line
    if(priors_opt.size()>1){//priors from argument list
      priors.resize(priors_opt.size());
      double normPrior=0;
      for(unsigned short iclass=0;iclass<priors_opt.size();++iclass){
        priors[iclass]=priors_opt[iclass];
        normPrior+=priors[iclass];
      }
      //normalize
      for(unsigned short iclass=0;iclass<priors_opt.size();++iclass)
        priors[iclass]/=normPrior;
    }
    else{//default: equal priors for each class
      if(priors_opt.size()==1){
        priors.resize(nclass);
        for(short iclass=0;iclass<nclass;++iclass)
          priors[iclass]=1.0/nclass;
      }
    }

    //convert start and end band options to vector of band indexes
    if(bstart_opt.size()){
      if(bend_opt.size()!=bstart_opt.size()){
        string errorstring="Error: options for start and end band indexes must be provided as pairs, missing end band";
        throw(errorstring);
      }
      band_opt.clear();
      for(int ipair=0;ipair<bstart_opt.size();++ipair){
        if(bend_opt[ipair]<=bstart_opt[ipair]){
          std::ostringstream errorStream;
          errorStream << "Error: index for end band (" << bend_opt[ipair] << ") must be smaller then start band (" << bstart_opt[ipair] << ")";
          throw(errorStream.str());
        }
        for(unsigned int iband=bstart_opt[ipair];iband<=bend_opt[ipair];++iband)
          band_opt.push_back(iband);
      }
    }

    //sort bands
    if(band_opt.size()){
      if(nband!=band_opt.size()){
        std::ostringstream errorStream;
        errorStream << "Error: number of bands (" << nband << ") does not correspond to number of bands specified (" << band_opt.size() << ")" << std::endl;
        throw(errorStream.str());
      }
      // std::sort(band_opt.begin(),band_opt.end());
    }
    else{
      unsigned short iband=0;
      while(band_opt.size()<nrOfBand())
        band_opt.push_back(iband++);
    }

    // if(classname_opt.size()){
    //   assert(classname_opt.size()==classvalue_opt.size());
    //   for(unsigned int iclass=0;iclass<classname_opt.size();++iclass)
    //     classValueMap[classname_opt[iclass]]=classvalue_opt[iclass];
    // }

    const char* pszMessage;
    void* pProgressArg=NULL;
    GDALProgressFunc pfnProgress=GDALTermProgress;
    float progress=0;
    if(!verbose_opt[0])
      MyProgressFunc(progress,pszMessage,pProgressArg);
    //-------------------------------- open image file ------------------------------------
    bool inputIsRaster=true;
    int nrow=nrOfRow();
    int ncol=nrOfCol();
    if(this->isInit()){
      if(verbose_opt[0]>=1)
        std::cout << "opening image" << std::endl;
      // imgWriter.open(*this,false);
      imgWriter.open(ncol,nrow,1,GDT_Byte);
      if(verbose_opt[0]>=1)
        std::cout << "set nodata" << std::endl;
      imgWriter.GDALSetNoDataValue(dstnodata_opt[0]);
      if(verbose_opt[0]>=1)
        std::cout << "copyGeoTransform" << std::endl;
      imgWriter.copyGeoTransform(*this);
      if(verbose_opt[0]>=1)
        std::cout << "setProjection" << std::endl;
      imgWriter.setProjection(this->getProjection());
      if(colorTable_opt.size())
        imgWriter.setColorTable(colorTable_opt[0],0);
      Jim priorReader;
      if(priorimg_opt.size()){
        if(verbose_opt[0]>=1)
          std::cout << "opening prior image " << priorimg_opt[0] << std::endl;
        priorReader.open(priorimg_opt[0]);
        assert(priorReader.nrOfCol()==ncol);
        assert(priorReader.nrOfRow()==nrow);
      }

      vector<char> classOut(ncol);//classified line for writing to image file

      Jim probImage;
      Jim entropyImage;
      std::string imageType="GTiff";

      if(prob_opt.size()){
        probImage.open(prob_opt[0],ncol,nrow,nclass,GDT_Byte,imageType);
        probImage.GDALSetNoDataValue(dstnodata_opt[0]);
        probImage.copyGeoTransform(imgWriter);
        probImage.setProjection(this->getProjection());
      }
      if(entropy_opt.size()){
        entropyImage.open(entropy_opt[0],ncol,nrow,1,GDT_Byte,imageType);
        entropyImage.GDALSetNoDataValue(dstnodata_opt[0]);
        entropyImage.copyGeoTransform(imgWriter);
        entropyImage.setProjection(this->getProjection());
      }

      if(mask_opt.size()){
        if(verbose_opt[0]>=1)
          std::cout << "opening mask image file " << mask_opt[0] << std::endl;
        maskReader.open(mask_opt[0]);
      }

      for(unsigned int iline=0;iline<nrow;++iline){
        vector<float> buffer(ncol);
        vector<short> lineMask;
        Vector2d<float> linePrior;
        if(priorimg_opt.size())
          linePrior.resize(nclass,ncol);//prior prob for each class
        Vector2d<float> hpixel(ncol);
        Vector2d<float> probOut(nclass,ncol);//posterior prob for each (internal) class
        vector<float> entropy(ncol);
        if(band_opt.size()){
          for(unsigned int iband=0;iband<band_opt.size();++iband){
            if(verbose_opt[0]>=2)
              std::cout << "reading band " << band_opt[iband] << std::endl;
            assert(band_opt[iband]>=0);
            assert(band_opt[iband]<nrOfBand());
            readData(buffer,iline,band_opt[iband]);
            for(unsigned int icol=0;icol<ncol;++icol)
              hpixel[icol].push_back(buffer[icol]);
          }
        }
        else{
          for(unsigned int iband=0;iband<nband;++iband){
            if(verbose_opt[0]>=2)
              std::cout << "reading band " << iband << std::endl;
            assert(iband>=0);
            assert(iband<nrOfBand());
            readData(buffer,iline,iband);
            for(unsigned int icol=0;icol<ncol;++icol)
              hpixel[icol].push_back(buffer[icol]);
          }
        }
        assert(nband==hpixel[0].size());
        if(verbose_opt[0]>1)
          std::cout << "used bands: " << nband << std::endl;
        //read prior
        if(priorimg_opt.size()){
          for(unsigned int iclass=0;iclass<nclass;++iclass){
            if(verbose_opt.size()>1)
              std::cout << "Reading " << priorimg_opt[0] << " band " << iclass << " line " << iline << std::endl;
            priorReader.readData(linePrior[iclass],iline,iclass);
          }
        }
        double oldRowMask=-1;//keep track of row mask to optimize number of line readings
        //process per pixel
        for(int icol=0;icol<ncol;++icol){
          assert(hpixel[icol].size()==nband);
          bool doClassify=true;
          bool masked=false;
          double geox=0;
          double geoy=0;
          if(maskReader.isInit()){
            //read mask
            double colMask=0;
            double rowMask=0;

            imgWriter.image2geo(icol,iline,geox,geoy);
            maskReader.geo2image(geox,geoy,colMask,rowMask);
            colMask=static_cast<int>(colMask);
            rowMask=static_cast<int>(rowMask);
            if(rowMask>=0&&rowMask<maskReader.nrOfRow()&&colMask>=0&&colMask<maskReader.nrOfCol()){
              if(static_cast<int>(rowMask)!=static_cast<int>(oldRowMask)){
                assert(rowMask>=0&&rowMask<maskReader.nrOfRow());
                // maskReader.readData(lineMask[imask],GDT_Int32,static_cast<int>(rowMask));
                maskReader.readData(lineMask,static_cast<unsigned int>(rowMask));
                oldRowMask=rowMask;
              }
              short theMask=0;
              for(short ivalue=0;ivalue<msknodata_opt.size();++ivalue){
                // if(msknodata_opt[ivalue]>=0){//values set in msknodata_opt are invalid
                if(lineMask[colMask]==msknodata_opt[ivalue]){
                  theMask=lineMask[colMask];
                  masked=true;
                  break;
                }
              }
              if(masked){
                classOut[icol]=theMask;
                continue;
              }
            }
          }
          bool valid=false;
          for(int iband=0;iband<hpixel[icol].size();++iband){
            if(hpixel[icol][iband]!=srcnodata_opt[0]){
              valid=true;
              break;
            }
          }
          if(!valid)
            doClassify=false;
          for(short iclass=0;iclass<nclass;++iclass)
            probOut[iclass][icol]=0;
          if(!doClassify){
            classOut[icol]=dstnodata_opt[0];
            continue;//next column
          }
          //----------------------------------- classification -------------------
          vector<double> result(nclass);
          struct svm_node *x;
          x = (struct svm_node *) malloc((nband+1)*sizeof(struct svm_node));
          for(int iband=0;iband<nband;++iband){
            x[iband].index=iband+1;
            x[iband].value=(hpixel[icol][iband]-offset[iband])/scale[iband];
          }

          x[nband].index=-1;//to end svm feature vector
          double predict_label=0;
          vector<float> prValues(nclass);
          float maxP=0;
          // if(!prob_est_opt[0]){
          if(theModel->param.probability>0){
            predict_label = svm_predict(theModel,x);
            for(short iclass=0;iclass<nclass;++iclass){
              if(iclass==static_cast<short>(predict_label))
                result[iclass]=1;
              else
                result[iclass]=0;
            }
          }
          else{
            if(!svm_check_probability_model(theModel)){
              string errorstring="Error: svm probability model check failed";
              throw(errorstring);
            }
            predict_label = svm_predict_probability(theModel,x,&(result[0]));
          }
          double normPrior=0;
          if(priorimg_opt.size()){
            for(short iclass=0;iclass<nclass;++iclass)
              normPrior+=linePrior[iclass][icol];
          }
          for(short iclass=0;iclass<nclass;++iclass){
            if(priorimg_opt.size())
              priors[iclass]=linePrior[iclass][icol]/normPrior;//todo: check if correct for all cases... (automatic classValueMap and manual input for names and values)
            probOut[iclass][icol]=result[iclass]*priors[iclass];
          }

          //search for max class prob
          float max1=0;//max probability
          float max2=0;//second max probability
          float norm=0;
          for(short iclass=0;iclass<nclass;++iclass){
            if(probOut[iclass][icol]>max1){
              max1=probOut[iclass][icol];
              classOut[icol]=classValueMap[nameVector[iclass]];
            }
            else if(probOut[iclass][icol]>max2)
              max2=probOut[iclass][icol];
            norm+=probOut[iclass][icol];
          }
          //normalize probOut and convert to percentage
          entropy[icol]=0;
          for(short iclass=0;iclass<nclass;++iclass){
            float prv=probOut[iclass][icol];
            prv/=norm;
            entropy[icol]-=prv*log(prv)/log(2.0);
            prv*=100.0;

            probOut[iclass][icol]=static_cast<short>(prv+0.5);
          }
          entropy[icol]/=log(static_cast<double>(nclass))/log(2.0);
          entropy[icol]=static_cast<short>(100*entropy[icol]+0.5);
          free(x);

          if(verbose_opt[0])
            ++ntotalClass[static_cast<short>(classOut[icol])];
        }//icol
        //----------------------------------- write output ------------------------------------------
        if(prob_opt.size()){
          for(short iclass=0;iclass<nclass;++iclass)
            probImage.writeData(probOut[iclass],iline,iclass);
        }
        if(entropy_opt.size()){
          entropyImage.writeData(entropy,iline);
        }

        imgWriter.writeData(classOut,iline);
        if(!verbose_opt[0]){
          progress=static_cast<float>(iline+1.0)/imgWriter.nrOfRow();
          MyProgressFunc(progress,pszMessage,pProgressArg);
        }
      }
      // imgWriter.close();
      if(mask_opt.size())
        maskReader.close();
      if(priorimg_opt.size())
        priorReader.close();
      if(prob_opt.size())
        probImage.close();
      if(entropy_opt.size())
        entropyImage.close();
    }
    // svm_destroy_param(&param);
    // svm_destroy_param(&param);
    // free(prob.y);
    // free(prob.x);
    // free(x_space);
    svm_free_and_destroy_model(&theModel);
    if(verbose_opt[0]){
      for(unsigned short iclass=0;iclass<nclass;++iclass)
        std::cout << iclass << ": " << ntotalClass[iclass];
      std::cout << std::endl;
    }
  }
  catch(string predefinedString){
    std::cout << predefinedString << std::endl;
    throw;
  }
}

/**
 * @param imgWriter output classified raster dataset
 * @param app application specific option arguments
 * @return CE_None if successful, CE_Failure if failed
 **/
// CPLErr Jim::svm(Jim& imgWriter, app::AppFactory& app){
//   vector<double> priors;

//   //--------------------------- command line options ------------------------------------
//   Optionjl<string> training_opt("t", "training", "Training vector file. A single vector file contains all training features (must be set as: b0, b1, b2,...) for all classes (class numbers identified by label option). Use multiple training files for bootstrap aggregation (alternative to the bag and bsize options, where a random subset is taken from a single training file)");
//   Optionjl<string> tlayer_opt("tln", "tln", "Training layer name(s)");
//   Optionjl<string> label_opt("label", "label", "Attribute name for class label in training vector file.","label");
//   Optionjl<unsigned int> balance_opt("bal", "balance", "Balance the input data to this number of samples for each class", 0);
//   Optionjl<bool> random_opt("random", "random", "Randomize training data for balancing and bagging", true, 2);
//   Optionjl<unsigned int> minSize_opt("min", "min", "If number of training pixels is less then min, do not take this class into account (0: consider all classes)", 0);
//   Optionjl<unsigned int> band_opt("b", "band", "Band index (starting from 0, either use band option or use start to end)");
//   Optionjl<unsigned int> bstart_opt("sband", "startband", "Start band sequence number");
//   Optionjl<unsigned int> bend_opt("eband", "endband", "End band sequence number");
//   Optionjl<double> offset_opt("offset", "offset", "Offset value for each spectral band input features: refl[band]=(DN[band]-offset[band])/scale[band]", 0.0);
//   Optionjl<double> scale_opt("scale", "scale", "Scale value for each spectral band input features: refl=(DN[band]-offset[band])/scale[band] (use 0 if scale min and max in each band to -1.0 and 1.0)", 0.0);
//   Optionjl<double> priors_opt("prior", "prior", "Prior probabilities for each class (e.g., -p 0.3 -p 0.3 -p 0.2 ). Used for input only (ignored for cross validation)", 0.0);
//   Optionjl<string> priorimg_opt("pim", "priorimg", "Prior probability image (multi-band img with band for each class","",2);
//   Optionjl<unsigned short> cv_opt("cv", "cv", "N-fold cross validation mode",0);
//   Optionjl<string> cmformat_opt("cmf","cmf","Format for confusion matrix (ascii or latex)","ascii");
//   Optionjl<std::string> svm_type_opt("svmt", "svmtype", "Type of SVM (C_SVC, nu_SVC,one_class, epsilon_SVR, nu_SVR)","C_SVC");
//   Optionjl<std::string> kernel_type_opt("kt", "kerneltype", "Type of kernel function (linear,polynomial,radial,sigmoid) ","radial");
//   Optionjl<unsigned short> kernel_degree_opt("kd", "kd", "Degree in kernel function",3);
//   Optionjl<float> gamma_opt("g", "gamma", "Gamma in kernel function",1.0);
//   Optionjl<float> coef0_opt("c0", "coef0", "Coef0 in kernel function",0);
//   Optionjl<float> ccost_opt("cc", "ccost", "The parameter C of C_SVC, epsilon_SVR, and nu_SVR",1000);
//   Optionjl<float> nu_opt("nu", "nu", "The parameter nu of nu_SVC, one_class SVM, and nu_SVR",0.5);
//   Optionjl<float> epsilon_loss_opt("eloss", "eloss", "The epsilon in loss function of epsilon_SVR",0.1);
//   Optionjl<int> cache_opt("cache", "cache", "Cache memory size in MB",100);
//   Optionjl<float> epsilon_tol_opt("etol", "etol", "The tolerance of termination criterion",0.001);
//   Optionjl<bool> shrinking_opt("shrink", "shrink", "Whether to use the shrinking heuristics",false);
//   Optionjl<bool> prob_est_opt("pe", "probest", "Whether to train a SVC or SVR model for probability estimates",true,2);
//   // Optionjl<bool> weight_opt("wi", "wi", "Set the parameter C of class i to weight*C, for C_SVC",true);
//   Optionjl<unsigned short> comb_opt("comb", "comb", "How to combine bootstrap aggregation classifiers (0: sum rule, 1: product rule, 2: max rule). Also used to aggregate classes with rc option.",0);
//   Optionjl<unsigned short> bag_opt("bag", "bag", "Number of bootstrap aggregations", 1);
//   Optionjl<int> bagSize_opt("bagsize", "bagsize", "Percentage of features used from available training features for each bootstrap aggregation (one size for all classes, or a different size for each class respectively", 100);
//   Optionjl<string> classBag_opt("cb", "classbag", "Output for each individual bootstrap aggregation");
//   Optionjl<string> extent_opt("e", "extent", "Only classify within extent from polygons in vector file");
//   Optionjl<string> eoption_opt("eo","eo", "special extent options controlling rasterization: ATTRIBUTE|CHUNKYSIZE|ALL_TOUCHED|BURN_VALUE_FROM|MERGE_ALG, e.g., -eo ATTRIBUTE=fieldname");
//   Optionjl<string> mask_opt("m", "mask", "Only classify within specified mask. For raster mask, set nodata values with the option msknodata.");
//   Optionjl<short> msknodata_opt("msknodata", "msknodata", "Mask value(s) not to consider for classification. Values will be taken over in classification image.", 0);
//   Optionjl<unsigned short> nodata_opt("nodata", "nodata", "Nodata value to put where image is masked as nodata", 0);
//   Optionjl<string> colorTable_opt("ct", "ct", "Color table in ASCII format having 5 columns: id R G B ALFA (0: transparent, 255: solid)");
//   Optionjl<string> prob_opt("prob", "prob", "Probability image.");
//   Optionjl<string> entropy_opt("entropy", "entropy", "Entropy image (measure for uncertainty of classifier output","",2);
//   Optionjl<string> active_opt("active", "active", "Ogr output for active training sample.","",2);
//   Optionjl<string> ogrformat_opt("f", "f", "Output ogr format for active training sample","SQLite");
//   Optionjl<unsigned int> nactive_opt("na", "nactive", "Number of active training points",1);
//   Optionjl<string> classname_opt("c", "class", "List of class names.");
//   Optionjl<short> classvalue_opt("r", "reclass", "List of class values (use same order as in class opt).");
//   Optionjl<short> verbose_opt("v", "verbose", "Verbose level",0,2);

  // extent_opt.setHide(1);
  // eoption_opt.setHide(1);
  // band_opt.setHide(1);
  // bstart_opt.setHide(1);
  // bend_opt.setHide(1);
  // balance_opt.setHide(1);
  // minSize_opt.setHide(1);
  // bag_opt.setHide(1);
  // bagSize_opt.setHide(1);
  // comb_opt.setHide(1);
  // classBag_opt.setHide(1);
  // prob_opt.setHide(1);
  // priorimg_opt.setHide(1);
  // offset_opt.setHide(1);
  // scale_opt.setHide(1);
  // svm_type_opt.setHide(1);
  // kernel_type_opt.setHide(1);
  // kernel_degree_opt.setHide(1);
  // coef0_opt.setHide(1);
  // nu_opt.setHide(1);
  // epsilon_loss_opt.setHide(1);
  // cache_opt.setHide(1);
  // epsilon_tol_opt.setHide(1);
  // shrinking_opt.setHide(1);
  // prob_est_opt.setHide(1);
  // entropy_opt.setHide(1);
  // active_opt.setHide(1);
  // nactive_opt.setHide(1);
  // random_opt.setHide(1);
  // verbose_opt.setHide(2);

  // bool doProcess;//stop process when program was invoked with help option (-h --help)
  // try{
  //   doProcess=training_opt.retrieveOption(app);
  //   cv_opt.retrieveOption(app);
  //   cmformat_opt.retrieveOption(app);
  //   tlayer_opt.retrieveOption(app);
  //   classname_opt.retrieveOption(app);
  //   classvalue_opt.retrieveOption(app);
  //   ogrformat_opt.retrieveOption(app);
  //   colorTable_opt.retrieveOption(app);
  //   label_opt.retrieveOption(app);
  //   priors_opt.retrieveOption(app);
  //   gamma_opt.retrieveOption(app);
  //   ccost_opt.retrieveOption(app);
  //   extent_opt.retrieveOption(app);
  //   eoption_opt.retrieveOption(app);
  //   mask_opt.retrieveOption(app);
  //   msknodata_opt.retrieveOption(app);
  //   nodata_opt.retrieveOption(app);
  //   // Advanced options
  //   band_opt.retrieveOption(app);
  //   bstart_opt.retrieveOption(app);
  //   bend_opt.retrieveOption(app);
  //   balance_opt.retrieveOption(app);
  //   minSize_opt.retrieveOption(app);
  //   bag_opt.retrieveOption(app);
  //   bagSize_opt.retrieveOption(app);
  //   comb_opt.retrieveOption(app);
  //   classBag_opt.retrieveOption(app);
  //   prob_opt.retrieveOption(app);
  //   priorimg_opt.retrieveOption(app);
  //   offset_opt.retrieveOption(app);
  //   scale_opt.retrieveOption(app);
  //   svm_type_opt.retrieveOption(app);
  //   kernel_type_opt.retrieveOption(app);
  //   kernel_degree_opt.retrieveOption(app);
  //   coef0_opt.retrieveOption(app);
  //   nu_opt.retrieveOption(app);
  //   epsilon_loss_opt.retrieveOption(app);
  //   cache_opt.retrieveOption(app);
  //   epsilon_tol_opt.retrieveOption(app);
  //   shrinking_opt.retrieveOption(app);
  //   prob_est_opt.retrieveOption(app);
  //   entropy_opt.retrieveOption(app);
  //   active_opt.retrieveOption(app);
  //   nactive_opt.retrieveOption(app);
  //   verbose_opt.retrieveOption(app);
  //   random_opt.retrieveOption(app);
  //   // memory_opt.retrieveOption(app);

  //   if(!doProcess){
  //     cout << endl;
  //     std::ostringstream helpStream;
  //     helpStream << "short option -h shows basic options only, use long option --help to show all options" << std::endl;
  //     throw(helpStream.str());//help was invoked, stop processing
  //   }

  //   if(entropy_opt[0]=="")
  //     entropy_opt.clear();
  //   if(active_opt[0]=="")
  //     active_opt.clear();
  //   if(priorimg_opt[0]=="")
  //     priorimg_opt.clear();


    // std::map<std::string, svm::SVM_TYPE> svmMap;

    // svmMap["C_SVC"]=svm::C_SVC;
    // svmMap["nu_SVC"]=svm::nu_SVC;
    // svmMap["one_class"]=svm::one_class;
    // svmMap["epsilon_SVR"]=svm::epsilon_SVR;
    // svmMap["nu_SVR"]=svm::nu_SVR;

    // std::map<std::string, svm::KERNEL_TYPE> kernelMap;

    // kernelMap["linear"]=svm::linear;
    // kernelMap["polynomial"]=svm::polynomial;
    // kernelMap["radial"]=svm::radial;
    // kernelMap["sigmoid;"]=svm::sigmoid;

    // if(training_opt.empty()){
    //   string errorString="Error: no training vector dataset provided";
    //   throw(errorString);
    // }

    // if(verbose_opt[0]>=1){
    //   if(mask_opt.size())
    //     std::cout << "mask filename: " << mask_opt[0] << std::endl;
    //   std::cout << "training vector file: " << std::endl;
    //   for(unsigned int ifile=0;ifile<training_opt.size();++ifile)
    //     std::cout << training_opt[ifile] << std::endl;
    //   std::cout << "verbose: " << verbose_opt[0] << std::endl;
    // }
    // unsigned short nbag=(training_opt.size()>1)?training_opt.size():bag_opt[0];
    // if(verbose_opt[0]>=1)
    //   std::cout << "number of bootstrap aggregations: " << nbag << std::endl;

    // ImgReaderOgr extentReader;
    // Jim maskReader;
    // // OGRLayer  *readLayer;

    // double ulx=0;
    // double uly=0;
    // double lrx=0;
    // double lry=0;

    // // bool maskIsVector=false;
    // if(extent_opt.size()){
    //   if(mask_opt.size()){
    //     string errorString="Error: can only either mask or extent, not both";
    //     throw(errorString);
    //   }
    //   extentReader.open(extent_opt[0]);
    //   // readLayer = extentReader.getDataSource()->GetLayer(0);
    //   if(!(extentReader.getExtent(ulx,uly,lrx,lry))){
    //     cerr << "Error: could not get extent from " << mask_opt[0] << endl;
    //     exit(1);
    //   }
    //   maskReader.open(this->nrOfCol(),this->nrOfRow(),1,GDT_Float64);
    //   double gt[6];
    //   this->getGeoTransform(gt);
    //   maskReader.setGeoTransform(gt);
    //   maskReader.setProjection(this->getProjection());
    //   // vector<double> burnValues(1,1);//burn value is 1 (single band)
    //   // maskReader.rasterizeBuf(extentReader);
    //   maskReader.rasterizeBuf(extentReader,nodata_opt[0],eoption_opt);
    //   maskReader.GDALSetNoDataValue(nodata_opt[0]);
    //   extentReader.close();
    // }

    // ImgWriterOgr activeWriter;
    // if(active_opt.size()){
    //   prob_est_opt[0]=true;
    //   ImgReaderOgr trainingReader(training_opt[0]);
    //   activeWriter.open(active_opt[0],ogrformat_opt[0]);
    //   activeWriter.createLayer(active_opt[0],trainingReader.getProjection(),wkbPoint,NULL);
    //   activeWriter.copyFields(trainingReader);
    // }
    // vector<PosValue> activePoints(nactive_opt[0]);
    // for(unsigned int iactive=0;iactive<activePoints.size();++iactive){
    //   activePoints[iactive].value=1.0;
    //   activePoints[iactive].posx=0.0;
    //   activePoints[iactive].posy=0.0;
    // }

    // unsigned int totalSamples=0;
    // unsigned int nactive=0;
    // vector<struct svm_model*> svm(nbag);
    // vector<struct svm_parameter> param(nbag);

    // unsigned int nclass=0;
    // unsigned int nband=0;
    // unsigned int startBand=2;//first two bands represent X and Y pos

    // //normalize priors from command line
    // if(priors_opt.size()>1){//priors from argument list
    //   priors.resize(priors_opt.size());
    //   double normPrior=0;
    //   for(unsigned short iclass=0;iclass<priors_opt.size();++iclass){
    //     priors[iclass]=priors_opt[iclass];
    //     normPrior+=priors[iclass];
    //   }
    //   //normalize
    //   for(unsigned short iclass=0;iclass<priors_opt.size();++iclass)
    //     priors[iclass]/=normPrior;
    // }

    // //convert start and end band options to vector of band indexes
    // if(bstart_opt.size()){
    //   if(bend_opt.size()!=bstart_opt.size()){
    //     string errorstring="Error: options for start and end band indexes must be provided as pairs, missing end band";
    //     throw(errorstring);
    //   }
    //   band_opt.clear();
    //   for(int ipair=0;ipair<bstart_opt.size();++ipair){
    //     if(bend_opt[ipair]<=bstart_opt[ipair]){
    //       string errorstring="Error: index for end band must be smaller then start band";
    //       throw(errorstring);
    //     }
    //     for(unsigned int iband=bstart_opt[ipair];iband<=bend_opt[ipair];++iband)
    //       band_opt.push_back(iband);
    //   }
    // }

    // //sort bands
    // if(band_opt.size())
    //   std::sort(band_opt.begin(),band_opt.end());

    // map<string,short> classValueMap;
    // vector<std::string> nameVector;
    // if(classname_opt.size()){
    //   assert(classname_opt.size()==classvalue_opt.size());
    //   for(unsigned int iclass=0;iclass<classname_opt.size();++iclass)
    //     classValueMap[classname_opt[iclass]]=classvalue_opt[iclass];
    // }

    // //----------------------------------- Training -------------------------------
    // confusionmatrix::ConfusionMatrix cm;
    // vector< vector<double> > offset(nbag);
    // vector< vector<double> > scale(nbag);
    // map<string,Vector2d<float> > trainingMap;
    // vector< Vector2d<float> > trainingPixels;//[class][sample][band]
    // vector<string> fields;

    // vector<struct svm_problem> prob(nbag);
    // vector<struct svm_node *> x_space(nbag);

    // for(int ibag=0;ibag<nbag;++ibag){
    //   //organize training data
    //   if(ibag<training_opt.size()){//if bag contains new training pixels
    //     trainingMap.clear();
    //     trainingPixels.clear();
    //     if(verbose_opt[0]>=1)
    //       std::cout << "reading imageVector file " << training_opt[0] << std::endl;
    //     ImgReaderOgr trainingReaderBag(training_opt[ibag]);
    //     if(band_opt.size())
    //       //todo: when tlayer_opt is provided, readDataImageOgr does not read any layer
    //       totalSamples=trainingReaderBag.readDataImageOgr(trainingMap,fields,band_opt,label_opt[0],tlayer_opt,verbose_opt[0]);
    //     else
    //       totalSamples=trainingReaderBag.readDataImageOgr(trainingMap,fields,0,0,label_opt[0],tlayer_opt,verbose_opt[0]);
    //     if(trainingMap.size()<2){
    //       string errorstring="Error: could not read at least two classes from training file, did you provide class labels in training sample (see option label)?";
    //       throw(errorstring);
    //     }
    //     trainingReaderBag.close();
    //     //convert map to vector
    //     // short iclass=0;
    //     if(verbose_opt[0]>1)
    //       std::cout << "training pixels: " << std::endl;
    //     map<string,Vector2d<float> >::iterator mapit=trainingMap.begin();
    //     while(mapit!=trainingMap.end()){
    //       //delete small classes
    //       if((mapit->second).size()<minSize_opt[0]){
    //         trainingMap.erase(mapit);
    //         continue;
    //       }
    //       trainingPixels.push_back(mapit->second);
    //       if(verbose_opt[0]>1)
    //         std::cout << mapit->first << ": " << (mapit->second).size() << " samples" << std::endl;
    //       ++mapit;
    //     }
    //     if(!ibag){
    //       nclass=trainingPixels.size();
    //       if(classname_opt.size())
    //         assert(nclass==classname_opt.size());
    //       nband=trainingPixels[0][0].size()-2;//X and Y//trainingPixels[0][0].size();
    //     }
    //     else{
    //       assert(nclass==trainingPixels.size());
    //       assert(nband==trainingPixels[0][0].size()-2);
    //     }

    //     //do not remove outliers here: could easily be obtained through ogr2ogr -where 'B2<110' output.shp input.shp
    //     //balance training data
    //     if(balance_opt[0]>0){
    //       while(balance_opt.size()<nclass)
    //         balance_opt.push_back(balance_opt.back());
    //       if(random_opt[0])
    //         srand(time(NULL));
    //       totalSamples=0;
    //       for(short iclass=0;iclass<nclass;++iclass){
    //         if(trainingPixels[iclass].size()>balance_opt[iclass]){
    //           while(trainingPixels[iclass].size()>balance_opt[iclass]){
    //             int index=rand()%trainingPixels[iclass].size();
    //             trainingPixels[iclass].erase(trainingPixels[iclass].begin()+index);
    //           }
    //         }
    //         else{
    //           int oldsize=trainingPixels[iclass].size();
    //           for(int isample=trainingPixels[iclass].size();isample<balance_opt[iclass];++isample){
    //             int index = rand()%oldsize;
    //             trainingPixels[iclass].push_back(trainingPixels[iclass][index]);
    //           }
    //         }
    //         totalSamples+=trainingPixels[iclass].size();
    //       }
    //     }

    //     //set scale and offset
    //     offset[ibag].resize(nband);
    //     scale[ibag].resize(nband);
    //     if(offset_opt.size()>1)
    //       assert(offset_opt.size()==nband);
    //     if(scale_opt.size()>1)
    //       assert(scale_opt.size()==nband);
    //     for(int iband=0;iband<nband;++iband){
    //       if(verbose_opt[0]>=1)
    //         std::cout << "scaling for band" << iband << std::endl;
    //       offset[ibag][iband]=(offset_opt.size()==1)?offset_opt[0]:offset_opt[iband];
    //       scale[ibag][iband]=(scale_opt.size()==1)?scale_opt[0]:scale_opt[iband];
    //       //search for min and maximum
    //       if(scale[ibag][iband]<=0){
    //         float theMin=trainingPixels[0][0][iband+startBand];
    //         float theMax=trainingPixels[0][0][iband+startBand];
    //         for(short iclass=0;iclass<nclass;++iclass){
    //           for(int isample=0;isample<trainingPixels[iclass].size();++isample){
    //             if(theMin>trainingPixels[iclass][isample][iband+startBand])
    //               theMin=trainingPixels[iclass][isample][iband+startBand];
    //             if(theMax<trainingPixels[iclass][isample][iband+startBand])
    //               theMax=trainingPixels[iclass][isample][iband+startBand];
    //           }
    //         }
    //         offset[ibag][iband]=theMin+(theMax-theMin)/2.0;
    //         scale[ibag][iband]=(theMax-theMin)/2.0;
    //         if(verbose_opt[0]>=1){
    //           std::cout << "Extreme image values for band " << iband << ": [" << theMin << "," << theMax << "]" << std::endl;
    //           std::cout << "Using offset, scale: " << offset[ibag][iband] << ", " << scale[ibag][iband] << std::endl;
    //           std::cout << "scaled values for band " << iband << ": [" << (theMin-offset[ibag][iband])/scale[ibag][iband] << "," << (theMax-offset[ibag][iband])/scale[ibag][iband] << "]" << std::endl;
    //         }
    //       }
    //     }
    //   }
    //   else{//use same offset and scale
    //     offset[ibag].resize(nband);
    //     scale[ibag].resize(nband);
    //     for(int iband=0;iband<nband;++iband){
    //       offset[ibag][iband]=offset[0][iband];
    //       scale[ibag][iband]=scale[0][iband];
    //     }
    //   }

      // if(!ibag){
      //   if(priors_opt.size()==1){//default: equal priors for each class
      //     priors.resize(nclass);
      //     for(short iclass=0;iclass<nclass;++iclass)
      //       priors[iclass]=1.0/nclass;
      //   }
      //   assert(priors_opt.size()==1||priors_opt.size()==nclass);

      //   //set bagsize for each class if not done already via command line
      //   while(bagSize_opt.size()<nclass)
      //     bagSize_opt.push_back(bagSize_opt.back());

      //   if(verbose_opt[0]>=1){
      //     std::cout << "number of bands: " << nband << std::endl;
      //     std::cout << "number of classes: " << nclass << std::endl;
      //     if(priorimg_opt.empty()){
      //       std::cout << "priors:";
      //       for(short iclass=0;iclass<nclass;++iclass)
      //         std::cout << " " << priors[iclass];
      //       std::cout << std::endl;
      //     }
      //   }
      //   map<string,Vector2d<float> >::iterator mapit=trainingMap.begin();
      //   bool doSort=true;
      //   while(mapit!=trainingMap.end()){
      //     nameVector.push_back(mapit->first);
      //     if(classValueMap.size()){
      //       //check if name in training is covered by classname_opt (values can not be 0)
      //       if(classValueMap[mapit->first]>0){
      //         if(cm.getClassIndex(type2string<short>(classValueMap[mapit->first]))<0){
      //           cm.pushBackClassName(type2string<short>(classValueMap[mapit->first]),doSort);
      //         }
      //       }
      //       else{
      //         std::cerr << "Error: names in classname option are not complete, please check names in training vector and make sure classvalue is > 0" << std::endl;
      //         exit(1);
      //       }
      //     }
      //     else
      //       cm.pushBackClassName(mapit->first,doSort);
      //     ++mapit;
      //   }
      //   if(classname_opt.empty()){
      //     //std::cerr << "Warning: no class name and value pair provided for all " << nclass << " classes, using string2type<int> instead!" << std::endl;
      //     for(int iclass=0;iclass<nclass;++iclass){
      //       if(verbose_opt[0])
      //         std::cout << iclass << " " << cm.getClass(iclass) << " -> " << string2type<short>(cm.getClass(iclass)) << std::endl;
      //       classValueMap[cm.getClass(iclass)]=string2type<short>(cm.getClass(iclass));
      //     }
      //   }

      //   // if(priors_opt.size()==nameVector.size()){
      //   //  std::cerr << "Warning: please check if priors are provided in correct order!!!" << std::endl;
      //   //  for(int iclass=0;iclass<nameVector.size();++iclass)
      //   //    std::cerr << nameVector[iclass] << " " << priors_opt[iclass] << std::endl;
      //   // }
      // }//if(!ibag)

      // //Calculate features of training set
      // vector< Vector2d<float> > trainingFeatures(nclass);
      // for(short iclass=0;iclass<nclass;++iclass){
      //   int nctraining=0;
      //   if(verbose_opt[0]>=1)
      //     std::cout << "calculating features for class " << iclass << std::endl;
      //   if(random_opt[0])
      //     srand(time(NULL));
      //   nctraining=(bagSize_opt[iclass]<100)? trainingPixels[iclass].size()/100.0*bagSize_opt[iclass] : trainingPixels[iclass].size();//bagSize_opt[iclass] given in % of training size
      //   if(nctraining<=0)
      //     nctraining=1;
      //   assert(nctraining<=trainingPixels[iclass].size());
      //   if(bagSize_opt[iclass]<100)
      //     random_shuffle(trainingPixels[iclass].begin(),trainingPixels[iclass].end());
      //   if(verbose_opt[0]>1)
      //     std::cout << "nctraining (class " << iclass << "): " << nctraining << std::endl;
      //   trainingFeatures[iclass].resize(nctraining);
      //   for(int isample=0;isample<nctraining;++isample){
      //     //scale pixel values according to scale and offset!!!
      //     for(int iband=0;iband<nband;++iband){
      //       float value=trainingPixels[iclass][isample][iband+startBand];
      //       trainingFeatures[iclass][isample].push_back((value-offset[ibag][iband])/scale[ibag][iband]);
      //     }
      //   }
      //   assert(trainingFeatures[iclass].size()==nctraining);
      // }

    //   unsigned int nFeatures=trainingFeatures[0][0].size();
    //   if(verbose_opt[0]>=1)
    //     std::cout << "number of features: " << nFeatures << std::endl;
    //   unsigned int ntraining=0;
    //   for(short iclass=0;iclass<nclass;++iclass)
    //     ntraining+=trainingFeatures[iclass].size();
    //   if(verbose_opt[0]>=1)
    //     std::cout << "training size over all classes: " << ntraining << std::endl;

    //   prob[ibag].l=ntraining;
    //   prob[ibag].y = Malloc(double,prob[ibag].l);
    //   prob[ibag].x = Malloc(struct svm_node *,prob[ibag].l);
    //   x_space[ibag] = Malloc(struct svm_node,(nFeatures+1)*ntraining);
    //   unsigned long int spaceIndex=0;
    //   int lIndex=0;
    //   for(short iclass=0;iclass<nclass;++iclass){
    //     for(int isample=0;isample<trainingFeatures[iclass].size();++isample){
    //       prob[ibag].x[lIndex]=&(x_space[ibag][spaceIndex]);
    //       for(int ifeature=0;ifeature<nFeatures;++ifeature){
    //         x_space[ibag][spaceIndex].index=ifeature+1;
    //         x_space[ibag][spaceIndex].value=trainingFeatures[iclass][isample][ifeature];
    //         ++spaceIndex;
    //       }
    //       x_space[ibag][spaceIndex++].index=-1;
    //       prob[ibag].y[lIndex]=iclass;
    //       ++lIndex;
    //     }
    //   }
    //   assert(lIndex==prob[ibag].l);

    //   //set SVM parameters through command line options
    //   param[ibag].svm_type = svmMap[svm_type_opt[0]];
    //   param[ibag].kernel_type = kernelMap[kernel_type_opt[0]];
    //   param[ibag].degree = kernel_degree_opt[0];
    //   param[ibag].gamma = (gamma_opt[0]>0)? gamma_opt[0] : 1.0/nFeatures;
    //   param[ibag].coef0 = coef0_opt[0];
    //   param[ibag].nu = nu_opt[0];
    //   param[ibag].cache_size = cache_opt[0];
    //   param[ibag].C = ccost_opt[0];
    //   param[ibag].eps = epsilon_tol_opt[0];
    //   param[ibag].p = epsilon_loss_opt[0];
    //   param[ibag].shrinking = (shrinking_opt[0])? 1 : 0;
    //   param[ibag].probability = (prob_est_opt[0])? 1 : 0;
    //   param[ibag].nr_weight = 0;//not used: I use priors and balancing
    //   param[ibag].weight_label = NULL;
    //   param[ibag].weight = NULL;
    //   param[ibag].verbose=(verbose_opt[0]>1)? true:false;

    //   if(verbose_opt[0]>1)
    //     std::cout << "checking parameters" << std::endl;
    //   svm_check_parameter(&prob[ibag],&param[ibag]);
    //   if(verbose_opt[0])
    //     std::cout << "parameters ok, training" << std::endl;
    //   svm[ibag]=svm_train(&prob[ibag],&param[ibag]);
    //   if(verbose_opt[0]>1)
    //     std::cout << "SVM is now trained" << std::endl;
    //   if(cv_opt[0]>1){
    //     if(verbose_opt[0]>1)
    //       std::cout << "Cross validating" << std::endl;
    //     double *target = Malloc(double,prob[ibag].l);
    //     svm_cross_validation(&prob[ibag],&param[ibag],cv_opt[0],target);
    //     assert(param[ibag].svm_type != EPSILON_SVR&&param[ibag].svm_type != NU_SVR);//only for regression

    //     for(int i=0;i<prob[ibag].l;i++){
    //       string refClassName=nameVector[prob[ibag].y[i]];
    //       string className=nameVector[target[i]];
    //       if(classValueMap.size())
    //         cm.incrementResult(type2string<short>(classValueMap[refClassName]),type2string<short>(classValueMap[className]),1.0/nbag);
    //       else
    //         cm.incrementResult(cm.getClass(prob[ibag].y[i]),cm.getClass(target[i]),1.0/nbag);
    //     }
    //     free(target);
    //   }
    //   // *NOTE* Because svm_model contains pointers to svm_problem, you can
    //   // not free the memory used by svm_problem if you are still using the
    //   // svm_model produced by svm_train().
    // }//for ibag
    // if(cv_opt[0]>1){
    //   assert(cm.nReference());
    //   cm.setFormat(cmformat_opt[0]);
    //   cm.reportSE95(false);
    //   std::cout << cm << std::endl;
    // }

    // //--------------------------------- end of training -----------------------------------

    // const char* pszMessage;
    // void* pProgressArg=NULL;
    // GDALProgressFunc pfnProgress=GDALTermProgress;
    // float progress=0;
    // if(!verbose_opt[0])
    //   MyProgressFunc(progress,pszMessage,pProgressArg);
    // //-------------------------------- open image file ------------------------------------
    // // bool inputIsRaster=false;
    // bool inputIsRaster=true;
    // // ImgReaderOgr imgReaderOgr;
    // // //todo: will not work in GDAL v2.0
    // // try{
    // //   imgReaderOgr.open(input_opt[0]);
    // //   imgReaderOgr.close();
    // // }
    // // catch(string errorString){
    // //   inputIsRaster=true;
    // // }
    // // if(inputIsRaster){
    // int nrow=nrOfRow();
    // int ncol=nrOfCol();
    // if(this->isInit()){
    //   if(verbose_opt[0]>=1)
    //     std::cout << "opening image" << std::endl;
    //   // imgWriter.open(*this,false);
    //   imgWriter.open(ncol,nrow,1,GDT_Byte);
    //   imgWriter.GDALSetNoDataValue(nodata_opt[0]);
    //   imgWriter.copyGeoTransform(*this);
    //   imgWriter.setProjection(this->getProjection());
    //   if(colorTable_opt.size())
    //     imgWriter.setColorTable(colorTable_opt[0],0);
    //   Jim priorReader;
    //   if(priorimg_opt.size()){
    //     if(verbose_opt[0]>=1)
    //       std::cout << "opening prior image " << priorimg_opt[0] << std::endl;
    //     priorReader.open(priorimg_opt[0]);
    //     assert(priorReader.nrOfCol()==ncol);
    //     assert(priorReader.nrOfRow()==nrow);
    //   }

    //   vector<char> classOut(ncol);//classified line for writing to image file

    //   //   assert(nband==imgWriter.nrOfBand());
    //   Jim classImageBag;
    //   // Jim imgWriter.
    //   Jim probImage;
    //   Jim entropyImage;

    //   string imageType=imgWriter.getImageType();
    //   if(classBag_opt.size()){
    //     classImageBag.open(classBag_opt[0],ncol,nrow,nbag,GDT_Byte,imageType);
    //     classImageBag.GDALSetNoDataValue(nodata_opt[0]);
    //     classImageBag.copyGeoTransform(imgWriter);
    //     classImageBag.setProjection(this->getProjection());
    //   }
    //   if(prob_opt.size()){
    //     probImage.open(prob_opt[0],ncol,nrow,nclass,GDT_Byte,imageType);
    //     probImage.GDALSetNoDataValue(nodata_opt[0]);
    //     probImage.copyGeoTransform(imgWriter);
    //     probImage.setProjection(this->getProjection());
    //   }
    //   if(entropy_opt.size()){
    //     entropyImage.open(entropy_opt[0],ncol,nrow,1,GDT_Byte,imageType);
    //     entropyImage.GDALSetNoDataValue(nodata_opt[0]);
    //     entropyImage.copyGeoTransform(imgWriter);
    //     entropyImage.setProjection(this->getProjection());
    //   }

      // if(mask_opt.size()){
      //   if(verbose_opt[0]>=1)
      //     std::cout << "opening mask image file " << mask_opt[0] << std::endl;
      //   maskReader.open(mask_opt[0]);
      // }

      // for(unsigned int iline=0;iline<nrow;++iline){
      //   vector<float> buffer(ncol);
      //   vector<short> lineMask;
      //   Vector2d<float> linePrior;
      //   if(priorimg_opt.size())
      //     linePrior.resize(nclass,ncol);//prior prob for each class
      //   Vector2d<float> hpixel(ncol);
      //   Vector2d<float> probOut(nclass,ncol);//posterior prob for each (internal) class
      //   vector<float> entropy(ncol);
      //   Vector2d<char> classBag;//classified line for writing to image file
      //   if(classBag_opt.size())
      //     classBag.resize(nbag,ncol);
      //   if(band_opt.size()){
      //     for(unsigned int iband=0;iband<band_opt.size();++iband){
      //       if(verbose_opt[0]==2)
      //         std::cout << "reading band " << band_opt[iband] << std::endl;
      //       assert(band_opt[iband]>=0);
      //       assert(band_opt[iband]<imgWriter.nrOfBand());
      //       readData(buffer,iline,band_opt[iband]);
      //       for(unsigned int icol=0;icol<ncol;++icol)
      //         hpixel[icol].push_back(buffer[icol]);
      //     }
      //   }
      //   else{
      //     for(unsigned int iband=0;iband<nband;++iband){
      //       if(verbose_opt[0]==2)
      //         std::cout << "reading band " << iband << std::endl;
      //       assert(iband>=0);
      //       assert(iband<imgWriter.nrOfBand());
      //       readData(buffer,iline,iband);
      //       for(unsigned int icol=0;icol<ncol;++icol)
      //         hpixel[icol].push_back(buffer[icol]);
      //     }
      //   }
      //   assert(nband==hpixel[0].size());
      //   if(verbose_opt[0]>1)
      //     std::cout << "used bands: " << nband << std::endl;
      //   //read prior
      //   if(priorimg_opt.size()){
      //     for(unsigned int iclass=0;iclass<nclass;++iclass){
      //       if(verbose_opt.size()>1)
      //         std::cout << "Reading " << priorimg_opt[0] << " band " << iclass << " line " << iline << std::endl;
      //       priorReader.readData(linePrior[iclass],iline,iclass);
      //     }
      //   }
      //   double oldRowMask=-1;//keep track of row mask to optimize number of line readings
      //   //process per pixel
      //   for(int icol=0;icol<ncol;++icol){
      //     assert(hpixel[icol].size()==nband);
      //     bool doClassify=true;
      //     bool masked=false;
      //     double geox=0;
      //     double geoy=0;
      //     if(maskReader.isInit()){
      //       //read mask
      //       double colMask=0;
      //       double rowMask=0;

      //       imgWriter.image2geo(icol,iline,geox,geoy);
      //       maskReader.geo2image(geox,geoy,colMask,rowMask);
      //       colMask=static_cast<int>(colMask);
      //       rowMask=static_cast<int>(rowMask);
      //       if(rowMask>=0&&rowMask<maskReader.nrOfRow()&&colMask>=0&&colMask<maskReader.nrOfCol()){
      //         if(static_cast<int>(rowMask)!=static_cast<int>(oldRowMask)){
      //           assert(rowMask>=0&&rowMask<maskReader.nrOfRow());
      //           // maskReader.readData(lineMask[imask],GDT_Int32,static_cast<int>(rowMask));
      //           maskReader.readData(lineMask,static_cast<unsigned int>(rowMask));
      //           oldRowMask=rowMask;
      //         }
      //         short theMask=0;
      //         for(short ivalue=0;ivalue<msknodata_opt.size();++ivalue){
      //           // if(msknodata_opt[ivalue]>=0){//values set in msknodata_opt are invalid
      //           if(lineMask[colMask]==msknodata_opt[ivalue]){
      //             theMask=lineMask[colMask];
      //             masked=true;
      //             break;
      //           }
      //           // }
      //           // else{//only values set in msknodata_opt are valid
      //           //   if(lineMask[colMask]!=-msknodata_opt[ivalue]){
      //           //     theMask=lineMask[colMask];
      //           //     masked=true;
      //           //   }
      //           //   else{
      //           //     masked=false;
      //           //     break;
      //           //   }
      //           // }
      //         }
      //         if(masked){
      //           if(classBag_opt.size())
      //             for(int ibag=0;ibag<nbag;++ibag)
      //               classBag[ibag][icol]=theMask;
      //           classOut[icol]=theMask;
      //           continue;
      //         }
      //       }
      //       bool valid=false;
      //       for(int iband=0;iband<hpixel[icol].size();++iband){
      //         if(hpixel[icol][iband]){
      //           valid=true;
      //           break;
      //         }
      //       }
      //       if(!valid)
      //         doClassify=false;
      //     }
      //     for(short iclass=0;iclass<nclass;++iclass)
      //       probOut[iclass][icol]=0;
      //     if(!doClassify){
      //       if(classBag_opt.size())
      //         for(int ibag=0;ibag<nbag;++ibag)
      //           classBag[ibag][icol]=nodata_opt[0];
      //       classOut[icol]=nodata_opt[0];
      //       continue;//next column
      //     }
          // //----------------------------------- classification -------------------
          // for(int ibag=0;ibag<nbag;++ibag){
          //   vector<double> result(nclass);
          //   struct svm_node *x;
          //   x = (struct svm_node *) malloc((nband+1)*sizeof(struct svm_node));
          //   for(int iband=0;iband<nband;++iband){
          //     x[iband].index=iband+1;
          //     x[iband].value=(hpixel[icol][iband]-offset[ibag][iband])/scale[ibag][iband];
          //   }
          //   x[nband].index=-1;//to end svm feature vector
          //   double predict_label=0;
          //   vector<float> prValues(nclass);
          //   float maxP=0;
          //   if(!prob_est_opt[0]){
          //     predict_label = svm_predict(svm[ibag],x);
          //     for(short iclass=0;iclass<nclass;++iclass){
          //       if(iclass==static_cast<short>(predict_label))
          //         result[iclass]=1;
          //       else
          //         result[iclass]=0;
          //     }
          //   }
          //   else{
          //     if(!svm_check_probability_model(svm[ibag])){
          //       string errorstring="Error: svm probability model check failed";
          //       throw(errorstring);
          //     }
          //     predict_label = svm_predict_probability(svm[ibag],x,&(result[0]));
          //   }
          //   //calculate posterior prob of bag
          //   if(classBag_opt.size()){
          //     //search for max prob within bag
          //     maxP=0;
          //     classBag[ibag][icol]=0;
          //   }
          //   double normPrior=0;
          //   if(priorimg_opt.size()){
          //     for(short iclass=0;iclass<nclass;++iclass)
          //       normPrior+=linePrior[iclass][icol];
          //   }
          //   for(short iclass=0;iclass<nclass;++iclass){
          //     if(priorimg_opt.size())
          //       priors[iclass]=linePrior[iclass][icol]/normPrior;//todo: check if correct for all cases... (automatic classValueMap and manual input for names and values)
          //     switch(comb_opt[0]){
          //     default:
          //     case(0)://sum rule
          //       probOut[iclass][icol]+=result[iclass]*priors[iclass];//add probabilities for each bag
          //     break;
          //     case(1)://product rule
          //       probOut[iclass][icol]*=pow(static_cast<float>(priors[iclass]),static_cast<float>(1.0-nbag)/nbag)*result[iclass];//multiply probabilities for each bag
          //       break;
          //     case(2)://max rule
          //       if(priors[iclass]*result[iclass]>probOut[iclass][icol])
          //         probOut[iclass][icol]=priors[iclass]*result[iclass];
          //       break;
          //     }
          //     if(classBag_opt.size()){
          //       //search for max prob within bag
          //       // if(prValues[iclass]>maxP){
          //       //   maxP=prValues[iclass];
          //       //   classBag[ibag][icol]=iclass;
          //       // }
          //       if(result[iclass]>maxP){
          //         maxP=result[iclass];
          //         classBag[ibag][icol]=iclass;
          //       }
          //     }
          //   }
          //   free(x);
          // }//ibag

          // //search for max class prob
          // float maxBag1=0;//max probability
          // float maxBag2=0;//second max probability
          // float normBag=0;
          // for(short iclass=0;iclass<nclass;++iclass){
          //   if(probOut[iclass][icol]>maxBag1){
          //     maxBag1=probOut[iclass][icol];
          //     classOut[icol]=classValueMap[nameVector[iclass]];
          //   }
          //   else if(probOut[iclass][icol]>maxBag2)
          //     maxBag2=probOut[iclass][icol];
          //   normBag+=probOut[iclass][icol];
          // }
          // //normalize probOut and convert to percentage
          // entropy[icol]=0;
          // for(short iclass=0;iclass<nclass;++iclass){
          //   float prv=probOut[iclass][icol];
          //   prv/=normBag;
          //   entropy[icol]-=prv*log(prv)/log(2.0);
          //   prv*=100.0;

          //   probOut[iclass][icol]=static_cast<short>(prv+0.5);
          //   // assert(classValueMap[nameVector[iclass]]<probOut.size());
          //   // assert(classValueMap[nameVector[iclass]]>=0);
          //   // probOut[classValueMap[nameVector[iclass]]][icol]=static_cast<short>(prv+0.5);
          // }
          // entropy[icol]/=log(static_cast<double>(nclass))/log(2.0);
//           entropy[icol]=static_cast<short>(100*entropy[icol]+0.5);
//           if(active_opt.size()){
//             if(entropy[icol]>activePoints.back().value){
//               activePoints.back().value=entropy[icol];//replace largest value (last)
//               activePoints.back().posx=icol;
//               activePoints.back().posy=iline;
//               std::sort(activePoints.begin(),activePoints.end(),Decrease_PosValue());//sort in descending order (largest first, smallest last)
//               if(verbose_opt[0])
//                 std::cout << activePoints.back().posx << " " << activePoints.back().posy << " " << activePoints.back().value << std::endl;
//             }
//           }
//         }//icol
//         //----------------------------------- write output ------------------------------------------
//         if(classBag_opt.size())
//           for(int ibag=0;ibag<nbag;++ibag)
//             classImageBag.writeData(classBag[ibag],iline,ibag);
//         if(prob_opt.size()){
//           for(short iclass=0;iclass<nclass;++iclass)
//             probImage.writeData(probOut[iclass],iline,iclass);
//         }
//         if(entropy_opt.size()){
//           entropyImage.writeData(entropy,iline);
//         }
//         imgWriter.writeData(classOut,iline);
//         if(!verbose_opt[0]){
//           progress=static_cast<float>(iline+1.0)/imgWriter.nrOfRow();
//           MyProgressFunc(progress,pszMessage,pProgressArg);
//         }
//       }
//       //write active learning points
//       if(active_opt.size()){
//         for(int iactive=0;iactive<activePoints.size();++iactive){
//           std::map<string,double> pointMap;
//           for(int iband=0;iband<imgWriter.nrOfBand();++iband){
//             double value;
//             readData(value,static_cast<int>(activePoints[iactive].posx),static_cast<int>(activePoints[iactive].posy),iband);
//             ostringstream fs;
//             fs << "B" << iband;
//             pointMap[fs.str()]=value;
//           }
//           pointMap[label_opt[0]]=0;
//           double x, y;
//           imgWriter.image2geo(activePoints[iactive].posx,activePoints[iactive].posy,x,y);
//           std::string fieldname="id";//number of the point
//           activeWriter.addPoint(x,y,pointMap,fieldname,++nactive);
//         }
//       }

//       // imgWriter.close();
//       if(mask_opt.size())
//         maskReader.close();
//       if(priorimg_opt.size())
//         priorReader.close();
//       if(prob_opt.size())
//         probImage.close();
//       if(entropy_opt.size())
//         entropyImage.close();
//       if(classBag_opt.size())
//         classImageBag.close();
//       // imgWriter.close();
//     }
//       if(cm.nReference()){
//         std::cout << cm << std::endl;
//         cout << "class #samples userAcc prodAcc" << endl;
//         double se95_ua=0;
//         double se95_pa=0;
//         double se95_oa=0;
//         double dua=0;
//         double dpa=0;
//         double doa=0;
//         for(short iclass=0;iclass<cm.nClasses();++iclass){
//           dua=cm.ua_pct(cm.getClass(iclass),&se95_ua);
//           dpa=cm.pa_pct(cm.getClass(iclass),&se95_pa);
//           cout << cm.getClass(iclass) << " " << cm.nReference(cm.getClass(iclass)) << " " << dua << " (" << se95_ua << ")" << " " << dpa << " (" << se95_pa << ")" << endl;
//         }
//         std::cout << "Kappa: " << cm.kappa() << std::endl;
//         doa=cm.oa(&se95_oa);
//         std::cout << "Overall Accuracy: " << 100*doa << " (" << 100*se95_oa << ")"  << std::endl;
//       }
//       if(active_opt.size())
//         activeWriter.close();

//       for(int ibag=0;ibag<nbag;++ibag){
//         // svm_destroy_param[ibag](&param[ibag]);
//         svm_destroy_param(&param[ibag]);
//         free(prob[ibag].y);
//         free(prob[ibag].x);
//         free(x_space[ibag]);
//         svm_free_and_destroy_model(&(svm[ibag]));
//       }
//       return(CE_None);
//   }
//   catch(BadConversion conversionString){
//     std::cerr << "Error: did you provide class pairs names (-c) and integer values (-r) for each class in training vector?" << std::endl;
//     return(CE_Failure);
//   }
//   catch(string predefinedString){
//     std::cout << predefinedString << std::endl;
//     return(CE_Failure);
//   }
// }
