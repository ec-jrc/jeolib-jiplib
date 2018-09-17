/**********************************************************************
jlann_lib.cc: classify raster image using Artificial Neural Network
Copyright (C) 2008-2018 Pieter Kempeneers

This file is part of pktools

pktools is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

pktools is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with pktools.  If not, see <http://www.gnu.org/licenses/>.
***********************************************************************/
#include <stdlib.h>
#include <vector>
#include <map>
#include <algorithm>
#include "imageclasses/Jim.h"
#include "imageclasses/VectorOgr.h"
#include "base/Optionjl.h"
#include "base/PosValue.h"
#include "algorithms/ConfusionMatrix.h"
#include "floatfann.h"
#include "algorithms/myfann_cpp.h"

using namespace std;
using namespace app;

/**
 * @param app application specific option arguments
 * @return output Vector
 **/
shared_ptr<VectorOgr> VectorOgr::classifyANN(app::AppFactory& app){
  std::shared_ptr<VectorOgr> ogrWriter=VectorOgr::createVector();
  if(classifyANN(*ogrWriter, app)!=OGRERR_NONE){
    std::cerr << "Failed to extract" << std::endl;
  }
  return(ogrWriter);
}

/**
 * @param imgWriter output classified raster dataset
 * @param app application specific option arguments
 * @return CE_None if successful, CE_Failure if failed
 **/
OGRErr VectorOgr::classifyANN(VectorOgr& ogrWriter, app::AppFactory& app){
  //--------------------------- command line options ------------------------------------
  Optionjl<std::string> model_opt("model", "model", "Model filename to save trained classifier.");
  Optionjl<unsigned int> band_opt("b", "band", "band index (starting from 0, either use band option or use start to end)");
  Optionjl<std::string> bandNames_opt("bn", "bandname", "Band name(s) to use. Leave empty to use all bands");
  Optionjl<unsigned int> bstart_opt("sband", "startband", "Start band sequence number");
  Optionjl<unsigned int> bend_opt("eband", "endband", "End band sequence number");
  Optionjl<double> offset_opt("offset", "offset", "offset value for each spectral band input features: refl[band]=(DN[band]-offset[band])/scale[band]", 0.0);
  Optionjl<double> scale_opt("scale", "scale", "scale value for each spectral band input features: refl=(DN[band]-offset[band])/scale[band] (use 0 if scale min and max in each band to -1.0 and 1.0)", 0.0);
  Optionjl<double> priors_opt("prior", "prior", "prior probabilities for each class (e.g., -p 0.3 -p 0.3 -p 0.2 )");
  Optionjl<string> output_opt("o", "output", "Filename of classified vector dataset");
  Optionjl<string> ogrformat_opt("f", "f", "Output ogr format for active training sample","SQLite");
  Optionjl<std::string> option_opt("co", "co", "Creation option for output file. Multiple options can be specified.");
  Optionjl<string> copyFields_opt("copy", "copy", "copy these fields from input to output vector dataset");
  Optionjl<short> verbose_opt("v", "verbose", "set to: 0 (results only), 1 (confusion matrix), 2 (debug)",0,2);
  // Optionjl<string> cmformat_opt("cmf","cmf","Format for confusion matrix (ascii or latex)","ascii");
  // Optionjl<std::string> label_opt("label", "label", "Attribute name for reference class label used for validation.","label");
  // Optionjl<string> classname_opt("c", "class", "list of class names.");
  // Optionjl<short> classvalue_opt("r", "reclass", "list of class values (use same order as in class opt).");

  band_opt.setHide(1);
  bandNames_opt.setHide(1);
  bstart_opt.setHide(1);
  bend_opt.setHide(1);
  option_opt.setHide(1);
  verbose_opt.setHide(2);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=model_opt.retrieveOption(app);
    copyFields_opt.retrieveOption(app);
    ogrformat_opt.retrieveOption(app);
    option_opt.retrieveOption(app);
    priors_opt.retrieveOption(app);
    // Advanced options
    band_opt.retrieveOption(app);
    bandNames_opt.retrieveOption(app);
    bstart_opt.retrieveOption(app);
    bend_opt.retrieveOption(app);
    offset_opt.retrieveOption(app);
    scale_opt.retrieveOption(app);
    output_opt.retrieveOption(app);
    verbose_opt.retrieveOption(app);
    // cmformat_opt.retrieveOption(app);
    // label_opt.retrieveOption(app);
    // classname_opt.retrieveOption(app);
    // classvalue_opt.retrieveOption(app);

    if(!doProcess){
      std::cout << std::endl;
      std::ostringstream helpStream;
      helpStream << "short option -h shows basic options only, use long option --help to show all options" << std::endl;
      throw(helpStream.str());//help was invoked, stop processing
    }

    if(model_opt.empty()){
      std::ostringstream errorStream;
      errorStream << "Error: model is not defined, please use option --model" << std::endl;
      throw(errorStream.str());
    }

    if(verbose_opt[0]>=1)
      std::cout << "model file: " << model_opt[0] << std::endl;

    char **papszOptions=NULL;
    for(std::vector<std::string>::const_iterator optionIt=option_opt.begin();optionIt!=option_opt.end();++optionIt)
      papszOptions=CSLAddString(papszOptions,optionIt->c_str());

    bool initWriter=true;
    if(ogrWriter.open(output_opt[0],ogrformat_opt[0])!=OGRERR_NONE)
      initWriter=false;

    std::vector<double> offset;
    std::vector<double> scale;
    map<string,short> classValueMap;
    vector<std::string> nameVector;
    vector<std::string> bandNames;

    FANN::neural_net net;//the neural network

    if(!net.create_from_file(model_opt[0],offset,scale,classValueMap,nameVector,bandNames)){
      std::ostringstream errorStream;
      errorStream << "Error: could not create neural network from file " << model_opt[0];
      throw(errorStream.str());
    }

    //todo: check if get_num_output() is really the number of classes
    unsigned int nclass=net.get_num_output();
    unsigned int nband=bandNames.size()?bandNames.size():offset.size();//todo: find a more elegant way to define number of bands (in model?)

    if(verbose_opt[0]){
      std::cout << "nclass: " << nclass << std::endl;
      std::cout << "nband: " << nband << std::endl;
      std::map<std::string,short>::const_iterator mapit=classValueMap.begin();
      std::cout << "classValueMap:" << std::endl;
      while(mapit!=classValueMap.end()){
        std::cout << mapit->first << ": " << mapit->second << std::endl;
        ++mapit;
      }
      std::cout << "nameVector:" << std::endl;
      for(int index=0;index<nameVector.size();++index)
        std::cout << nameVector[index] << std::endl;
      std::cout << "bandNames:" << std::endl;
      for(int index=0;index<bandNames.size();++index)
        std::cout << bandNames[index] << std::endl;
    }

    //normalize priors from command line
    vector<double> priors;
    if(priors_opt.size()>1){//priors from argument list
      priors.resize(priors_opt.size());
      double normPrior=0;
      for(unsigned int iclass=0;iclass<priors_opt.size();++iclass){
        priors[iclass]=priors_opt[iclass];
        normPrior+=priors[iclass];
      }
      //normalize
      for(unsigned int iclass=0;iclass<priors_opt.size();++iclass)
        priors[iclass]/=normPrior;
    }

    //convert start and end band options to vector of band indexes
    if(bstart_opt.size()){
      if(bend_opt.size()!=bstart_opt.size()){
        string errorstring="Error: options for start and end band indexes must be provided as pairs, missing end band";
        throw(errorstring);
      }
      band_opt.clear();
      for(unsigned int ipair=0;ipair<bstart_opt.size();++ipair){
        if(bend_opt[ipair]<=bstart_opt[ipair]){
          string errorstring="Error: index for end band must be smaller then start band";
          throw(errorstring);
        }
        for(unsigned int iband=bstart_opt[ipair];iband<=bend_opt[ipair];++iband)
          band_opt.push_back(iband);
      }
    }

    if(band_opt.size()){
      //sort bands
      // std::sort(band_opt.begin(),band_opt.end());
      if(nband!=band_opt.size()){
        string errorstring="Error: index for end band must be smaller then start band";
        throw(errorstring);
      }
    }
    else{
      unsigned short iband=0;
      while(band_opt.size()<nband)
        band_opt.push_back(iband++);
    }

    // map<string,short> classValueMap;
    // vector<std::string> nameVector;
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
      pfnProgress(progress,pszMessage,pProgressArg);

    // cm.clearResults();
    //notice that fields have already been set by readDataImageOgr (taking into account appropriate bands)
    int nvalidation=1;
    for(int ivalidation=0;ivalidation<nvalidation;++ivalidation){
      if(verbose_opt[0])
        cout << "number of layers in input ogr file: " << getLayerCount() << std::endl;
      for(unsigned int ilayer=0;ilayer<getLayerCount();++ilayer){
        if(verbose_opt[0])
          std::cout << "processing input layer " << ilayer << std::endl;
        if(initWriter){
          if(ogrWriter.pushLayer(getLayerName(ilayer),getProjection(ilayer),getGeometryType(),papszOptions)!=OGRERR_NONE){
            ostringstream fs;
            fs << "push layer to ogrWriter with polygons failed ";
            fs << "layer name: "<< getLayerName(ilayer) << std::endl;
            throw(fs.str());
          }
          // ogrWriter.pushLayer(getLayer()->GetName(),getProjection(),getGeometryType(),NULL);
          if(copyFields_opt.size()){
            if(verbose_opt[0])
              std::cout << "copy fields" << std::endl;
            ogrWriter.copyFields(*this,copyFields_opt,ilayer);
          }
          if(verbose_opt[0])
            std::cout << "creating field class" << std::endl;
          if(classValueMap.size())
            ogrWriter.createField("class",OFTInteger,ilayer);
          else
            ogrWriter.createField("class",OFTString,ilayer);
        }

        //make sure to use resize and setFeature instead of pushFeature when in processing in parallel!!!
        ogrWriter.resize(getFeatureCount(ilayer),ilayer);

        unsigned int nFeatures=getFeatureCount(ilayer);
        progress=0;
        pfnProgress(progress,pszMessage,pProgressArg);
#if PKTOOLS_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
        for(unsigned int ifeature=0;ifeature<getFeatureCount(ilayer);++ifeature){
          OGRFeature *poFeature=cloneFeature(ifeature,ilayer);
          if(verbose_opt[0]>1)
            std::cout << "feature " << ifeature << std::endl;
          if( poFeature == NULL ){
            cout << "Warning: could not read feature " << ifeature << " in layer " << getLayerName(ilayer) << endl;
            continue;
          }
          OGRFeature *poDstFeature = NULL;
          poDstFeature=ogrWriter.createFeature(ilayer);
          if( poDstFeature->SetFrom( poFeature, TRUE ) != OGRERR_NONE ){
              OGRFeature::DestroyFeature( poFeature );
              OGRFeature::DestroyFeature( poDstFeature );
              std::ostringstream errorStream;
              errorStream << "Error: Unable to translate feature " << poFeature->GetFID() << "from layer " << ogrWriter.getLayerName(ilayer).c_str() << std::endl;
              throw(errorStream.str());
          }
          // poDstFeature->SetGeometry(poFeature->GetGeometryRef());
          std::vector<float> validationFeature;
          std::vector<float> probOut(nclass);//posterior prob for each (internal) class
          for(int iField=0;iField<m_features[ilayer][ifeature]->GetFieldCount();++iField){
            std::string fieldname=m_features[ilayer][ifeature]->GetFieldDefnRef(iField)->GetNameRef();
            if(bandNames.size()){
              if(find(bandNames.begin(),bandNames.end(),fieldname)!=bandNames.end()){
                double theValue=m_features[ilayer][ifeature]->GetFieldAsDouble(iField);
                validationFeature.push_back((theValue-offset[iField])/scale[iField]);
              }
            }
            else if(fieldname!="fid"){
              double theValue=m_features[ilayer][ifeature]->GetFieldAsDouble(iField);
              validationFeature.push_back((theValue-offset[iField])/scale[iField]);
            }
          }
          vector<float> result(nclass);
          result=net.run(validationFeature);

          //calculate posterior prob and calculate max class prob
          float max=0;//max probability
          std::string classOut="Unclassified";
          for(short iclass=0;iclass<nclass;++iclass){
            result[iclass]=(result[iclass]+1.0)/2.0;//bring back to scale [0,1]
            if(priors_opt.size())
              probOut[iclass]=priors[iclass]*result[iclass];
            else
              probOut[iclass]=result[iclass];
            if(verbose_opt[0]>1)
              std::cout << "feature " << ifeature << " probOut " << iclass << ": " << probOut[iclass] << std::endl;
            if(probOut[iclass]>max){
              max=probOut[iclass];
              if(classValueMap.size())
                classOut=nameVector[iclass];//classOut=classValueMap[nameVector[iclass]];
              else
                classOut=type2string<short>(iclass);
            }
          }
          if(verbose_opt[0]>1)
            std::cout << "feature " << ifeature << " classOut " << classOut << std::endl;

          if(classValueMap.size())
            poDstFeature->SetField("class",classValueMap[classOut]);
          else
            poDstFeature->SetField("class",classOut.c_str());
          //todo: might not be needed due to SetFrom
          poDstFeature->SetFID( poFeature->GetFID() );

          // int labelIndex=poFeature->GetFieldIndex(label_opt[0].c_str());
          // if(labelIndex>=0){
          //   string classRef=poFeature->GetFieldAsString(labelIndex);
          //   if(classRef!="0"){
          //     if(classValueMap.size())
          //       cm.incrementResult(type2string<short>(classValueMap[classRef]),type2string<short>(classValueMap[classOut]),1);
          //     else
          //       cm.incrementResult(classRef,classOut,1);
          //   }
          // }

          //make sure to use setFeature instead of pushFeature when in processing in parallel!!!
          ogrWriter.setFeature(ifeature,poDstFeature,ilayer);
          if(!verbose_opt[0]){
            progress=static_cast<float>(ifeature+1.0)/nFeatures;
            pfnProgress(progress,pszMessage,pProgressArg);
          }
        }//get next feature
      }//next layer
    }
    // if(cm.nReference()){
    //   std::cout << cm << std::endl;
    //   std::cout << "class #samples userAcc prodAcc" << std::endl;
    //   double se95_ua=0;
    //   double se95_pa=0;
    //   double se95_oa=0;
    //   double dua=0;
    //   double dpa=0;
    //   double doa=0;
    //   for(short iclass=0;iclass<cm.nClasses();++iclass){
    //     dua=cm.ua_pct(cm.getClass(iclass),&se95_ua);
    //     dpa=cm.pa_pct(cm.getClass(iclass),&se95_pa);
    //     cout << cm.getClass(iclass) << " " << cm.nReference(cm.getClass(iclass)) << " " << dua << " (" << se95_ua << ")" << " " << dpa << " (" << se95_pa << ")" << std::endl;
    //   }
    //   std::cout << "Kappa: " << cm.kappa() << std::endl;
    //   doa=cm.oa_pct(&se95_oa);
    //   std::cout << "Overall Accuracy: " << doa << " (" << se95_oa << ")"  << std::endl;
    // }
    return(CE_None);
  }
  catch(BadConversion conversionString){
    std::cerr << "Error: did you provide class pairs names (-c) and integer values (-r) for each class in training vector?" << std::endl;
    return(CE_Failure);
  }
  catch(string predefinedString){
    std::cout << predefinedString << std::endl;
    return(CE_Failure);
  }
}
