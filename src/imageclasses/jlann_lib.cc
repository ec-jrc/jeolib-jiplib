/**********************************************************************
jlann_lib.cc: classify raster image using Artificial Neural Network
Copyright (C) 2008-2016 Pieter Kempeneers

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
#include "Jim.h"
#include "VectorOgr.h"
// #include "ImgReaderOgr.h"
// #include "ImgWriterOgr.h"
#include "base/Optionjl.h"
#include "base/PosValue.h"
#include "algorithms/ConfusionMatrix.h"
#include "floatfann.h"
#include "algorithms/myfann_cpp.h"

using namespace std;
using namespace app;

std::string VectorOgr::trainANN(app::AppFactory& app){
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
  Optionjl<unsigned int> nneuron_opt("nn", "nneuron", "number of neurons in hidden layers in neural network (multiple hidden layers are set by defining multiple number of neurons: -n 15 -n 1, default is one hidden layer with 5 neurons)", 5);
  Optionjl<float> connection_opt("\0", "connection", "connection rate (default: 1.0 for a fully connected network)", 1.0);
  Optionjl<float> learning_opt("l", "learning", "learning rate (default: 0.7)", 0.7);
  Optionjl<float> weights_opt("w", "weights", "weights for neural network. Apply to fully connected network only, starting from first input neuron to last output neuron, including the bias neurons (last neuron in each but last layer)", 0.0);
  Optionjl<unsigned int> maxit_opt("\0", "maxit", "number of maximum iterations (epoch) (default: 500)", 500);
  Optionjl<std::string> classname_opt("c", "class", "List of class names.");
  Optionjl<short> classvalue_opt("r", "reclass", "List of class values (use same order as in class opt).");
  Optionjl<short> verbose_opt("v", "verbose", "Verbose level",0,2);

  random_opt.setHide(1);
  minSize_opt.setHide(1);
  // band_opt.setHide(1);
  // bstart_opt.setHide(1);
  // bend_opt.setHide(1);
  connection_opt.setHide(1);
  learning_opt.setHide(1);
  weights_opt.setHide(1);
  maxit_opt.setHide(1);
  verbose_opt.setHide(2);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    // doProcess=model_opt.retrieveOption(app);
    doProcess=label_opt.retrieveOption(app);
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
    nneuron_opt.retrieveOption(app);
    connection_opt.retrieveOption(app);
    learning_opt.retrieveOption(app);
    weights_opt.retrieveOption(app);
    maxit_opt.retrieveOption(app);
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
    FANN::neural_net net;//the neural network

    // if(model_opt.empty()){
    //   std::string errorString="Error: filename to save model is emtpy";
    //   throw(errorString);
    // }

    unsigned int nclass=0;
    unsigned int nband=0;
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

    const unsigned int num_layers = nneuron_opt.size()+2;
    const float desired_error = 0.0003;
    const unsigned int iterations_between_reports = (verbose_opt[0])? maxit_opt[0]+1:0;
    if(verbose_opt[0]>=1){
      std::cout << "number of features: " << nFeatures << std::endl;
      std::cout << "creating artificial neural network with " << nneuron_opt.size() << " hidden layer, having " << std::endl;
      for(unsigned int ilayer=0;ilayer<nneuron_opt.size();++ilayer)
        std::cout << nneuron_opt[ilayer] << " ";
      std::cout << "neurons" << std::endl;
      std::cout << "connection_opt[0]: " << connection_opt[0] << std::endl;
      std::cout << "num_layers: " << num_layers << std::endl;
      std::cout << "nFeatures: " << nFeatures << std::endl;
      std::cout << "nneuron_opt[0]: " << nneuron_opt[0] << std::endl;
      std::cout << "number of classes (nclass): " << nclass << std::endl;
    }
    switch(num_layers){
    case(3):{
      // net.create_sparse(connection_opt[0],num_layers, nFeatures, nneuron_opt[0], nclass);//replace all create_sparse with create_sparse_array due to bug in FANN!
      unsigned int layers[3];
      layers[0]=nFeatures;
      layers[1]=nneuron_opt[0];
      layers[2]=nclass;
      net.create_sparse_array(connection_opt[0],num_layers,layers);
      break;
    }
    case(4):{
      unsigned int layers[4];
      layers[0]=nFeatures;
      layers[1]=nneuron_opt[0];
      layers[2]=nneuron_opt[1];
      layers[3]=nclass;
      // layers.push_back(nFeatures);
      // for(int ihidden=0;ihidden<nneuron_opt.size();++ihidden)
      //  layers.push_back(nneuron_opt[ihidden]);
      // layers.push_back(nclass);
      net.create_sparse_array(connection_opt[0],num_layers,layers);
      break;
    }
    default:
      cerr << "Only 1 or 2 hidden layers are supported!" << std::endl;
      exit(1);
      break;
    }
    if(verbose_opt[0]>=1)
      std::cout << "network created" << std::endl;

    net.set_learning_rate(learning_opt[0]);

    //   net.set_activation_steepness_hidden(1.0);
    //   net.set_activation_steepness_output(1.0);

    net.set_activation_function_hidden(FANN::SIGMOID_SYMMETRIC_STEPWISE);
    net.set_activation_function_output(FANN::SIGMOID_SYMMETRIC_STEPWISE);

    // Set additional properties such as the training algorithm
    //   net.set_training_algorithm(FANN::TRAIN_QUICKPROP);

    // Output network type and parameters
    if(verbose_opt[0]>=1){
      std::cout << std::endl << "Network Type                         :  ";
      switch (net.get_network_type())
        {
        case FANN::LAYER:
          std::cout << "LAYER" << std::endl;
          break;
        case FANN::SHORTCUT:
          std::cout << "SHORTCUT" << std::endl;
          break;
        default:
          std::cout << "UNKNOWN" << std::endl;
          break;
        }
      net.print_parameters();
    }

    if(cv_opt[0]>1){
      if(verbose_opt[0])
        std::cout << "cross validation" << std::endl;
      vector<unsigned short> referenceVector;
      vector<unsigned short> outputVector;
      float rmse=net.cross_validation(trainingFeatures,
                                      ntraining,
                                      cv_opt[0],
                                      maxit_opt[0],
                                      desired_error,
                                      referenceVector,
                                      outputVector,
                                      verbose_opt[0]);
      map<string,Vector2d<float> >::iterator mapit=trainingMap.begin();
      for(unsigned int isample=0;isample<referenceVector.size();++isample){
        string refClassName=nameVector[referenceVector[isample]];
        string className=nameVector[outputVector[isample]];
        if(classValueMap.size())
          cm.incrementResult(type2string<short>(classValueMap[refClassName]),type2string<short>(classValueMap[className]),1.0);
        else
          cm.incrementResult(cm.getClass(referenceVector[isample]),cm.getClass(outputVector[isample]),1.0);
      }
    }

    if(verbose_opt[0]>=1)
      std::cout << std::endl << "Set training data" << std::endl;

    if(verbose_opt[0]>=1)
      std::cout << std::endl << "Training network" << std::endl;

    if(verbose_opt[0]>=1){
      std::cout << "Max Epochs " << setw(8) << maxit_opt[0] << ". "
                << "Desired Error: " << left << desired_error << right << std::endl;
    }
    if(weights_opt.size()==net.get_total_connections()){//no new training needed (same training sample)
      vector<fann_connection> convector;
      net.get_connection_array(convector);
      for(unsigned int i_connection=0;i_connection<net.get_total_connections();++i_connection)
        convector[i_connection].weight=weights_opt[i_connection];
      net.set_weight_array(convector);
    }
    else{
      bool initWeights=true;
      net.train_on_data(trainingFeatures,ntraining,initWeights, maxit_opt[0],
                        iterations_between_reports, desired_error);
    }


    if(verbose_opt[0]>=2){
      net.print_connections();
      vector<fann_connection> convector;
      net.get_connection_array(convector);
      for(unsigned int i_connection=0;i_connection<net.get_total_connections();++i_connection)
        std::cout << "connection " << i_connection << ": " << convector[i_connection].weight << std::endl;

    }
    if(cv_opt[0]>1){
      assert(cm.nReference());
      cm.setFormat(cmformat_opt[0]);
      cm.reportSE95(false);
      std::cout << cm << std::endl;
    }
    //--------------------------------- end of training -----------------------------------
    // net.save(model_opt[0], offset, scale, classValueMap, nameVector, bandNames_opt);
    std::ostringstream outputStream;
    net.ann_serialize_model(outputStream, offset, scale, classValueMap, nameVector, bandNames_opt);
    return outputStream.str();
  }
  catch(std::string errorString){
    std::cerr << errorString << std::endl;
    throw;
  }
}

/**
 * @param app application specific option arguments
 * @return output classified raster dataset
 **/
shared_ptr<Jim> Jim::classifyANN(app::AppFactory& app){
  try{
    shared_ptr<Jim> imgWriter=createImg();
    classifyANN(*imgWriter, app);
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
// shared_ptr<Jim> Jim::ann(app::AppFactory& app){
//   try{
//     shared_ptr<Jim> imgWriter=createImg();
//     ann(*imgWriter, app);
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
CPLErr Jim::classifyANN(Jim& imgWriter, app::AppFactory& app){
  vector<double> priors;

  //--------------------------- command line options ------------------------------------
  Optionjl<std::string> model_opt("model", "model", "Model filename to save trained classifier.");
  Optionjl<unsigned int> band_opt("b", "band", "band index (starting from 0, either use band option or use start to end)");
  // Optionjl<std::string> bandNames_opt("bn", "bandname", "Band name(s) to use. Leave empty to use all bands");
  Optionjl<unsigned int> bstart_opt("sband", "startband", "Start band sequence number");
  Optionjl<unsigned int> bend_opt("eband", "endband", "End band sequence number");
  Optionjl<double> offset_opt("offset", "offset", "offset value for each spectral band input features: refl[band]=(DN[band]-offset[band])/scale[band]", 0.0);
  Optionjl<double> scale_opt("scale", "scale", "scale value for each spectral band input features: refl=(DN[band]-offset[band])/scale[band] (use 0 if scale min and max in each band to -1.0 and 1.0)", 0.0);
  Optionjl<double> priors_opt("prior", "prior", "prior probabilities for each class (e.g., -p 0.3 -p 0.3 -p 0.2 )", 0.0);
  Optionjl<string> priorimg_opt("pim", "priorimg", "prior probability image (multi-band img with band for each class","",2);
  Optionjl<string> extent_opt("e", "extent", "Only classify within extent from polygons in vector file");
  Optionjl<string> eoption_opt("eo","eo", "special extent options controlling rasterization: ATTRIBUTE|CHUNKYSIZE|ALL_TOUCHED|BURN_VALUE_FROM|MERGE_ALG, e.g., -eo ATTRIBUTE=fieldname");
  Optionjl<string> mask_opt("m", "mask", "Only classify within specified mask (vector or raster). For raster mask, set nodata values with the option msknodata.");
  Optionjl<short> msknodata_opt("msknodata", "msknodata", "mask value(s) not to consider for classification. Values will be taken over in classification image. Default is 0", 0);
  Optionjl<double> srcnodata_opt("srcnodata", "srcnodata", "Nodata value in source",0);
  Optionjl<unsigned short> dstnodata_opt("dstnodata", "dstnodata", "Nodata value to put where image is masked as nodata", 0);
  // Optionjl<unsigned short> nodata_opt("nodata", "nodata", "nodata value to put where image is masked as nodata", 0);
  Optionjl<string> colorTable_opt("ct", "ct", "colour table in ASCII format having 5 columns: id R G B ALFA (0: transparent, 255: solid)");
  Optionjl<string> prob_opt("\0", "prob", "probability image. Default is no probability image");
  Optionjl<string> entropy_opt("entropy", "entropy", "entropy image (measure for uncertainty of classifier output","",2);
  Optionjl<short> verbose_opt("v", "verbose", "set to: 0 (results only), 1 (confusion matrix), 2 (debug)",0,2);
  extent_opt.setHide(1);
  eoption_opt.setHide(1);
  band_opt.setHide(1);
  bstart_opt.setHide(1);
  bend_opt.setHide(1);
  prob_opt.setHide(1);
  priorimg_opt.setHide(1);
  offset_opt.setHide(1);
  scale_opt.setHide(1);
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
    // bandNames_opt.setHide(1);
    bstart_opt.retrieveOption(app);
    bend_opt.retrieveOption(app);
    prob_opt.retrieveOption(app);
    priorimg_opt.retrieveOption(app);
    offset_opt.retrieveOption(app);
    scale_opt.retrieveOption(app);
    priors_opt.retrieveOption(app);
    entropy_opt.retrieveOption(app);
    verbose_opt.retrieveOption(app);

    if(!doProcess){
      std::cout << std::endl;
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

    // ImgReaderOgr extentReader;
    VectorOgr extentReader;
    Jim maskReader;
    // OGRLayer  *readLayer;

    double ulx=0;
    double uly=0;
    double lrx=0;
    double lry=0;

    // bool maskIsVector=false;
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
      maskReader.rasterizeBuf(extentReader,dstnodata_opt[0],eoption_opt);
      maskReader.GDALSetNoDataValue(dstnodata_opt[0]);
      extentReader.close();
    }

    unsigned int totalSamples=0;
    FANN::neural_net net;//the neural network


    vector<double> offset;
    vector<double> scale;
    map<string,short> classValueMap;
    vector<std::string> nameVector;
    vector<std::string> bandNames;

    if(!net.create_from_file(model_opt[0],offset,scale,classValueMap,nameVector,bandNames)){
      std::ostringstream errorStream;
      errorStream << "Error: could not create neural network from file " << model_opt[0];
      throw(errorStream.str());
    }

    //todo: check if get_num_output() is really the number of classes
    unsigned int nclass=net.get_num_output();
    unsigned int nband=bandNames.size()?bandNames.size():offset.size();//todo: find a more elegant way to define number of bands (in model?)

    //normalize priors from command line
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
      for(unsigned int ipair=0;ipair<bstart_opt.size();++ipair){
        if(bend_opt[ipair]<=bstart_opt[ipair]){
          string errorstring="Error: index for end band must be smaller then start band";
          throw(errorstring);
        }
        for(unsigned int iband=bstart_opt[ipair];iband<=bend_opt[ipair];++iband)
          band_opt.push_back(iband);
      }
    }

    //sort bands
    if(band_opt.size())
      std::sort(band_opt.begin(),band_opt.end());
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
      imgWriter.GDALSetNoDataValue(dstnodata_opt[0]);
      imgWriter.copyGeoTransform(*this);
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

      string imageType=imgWriter.getImageType();
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
        Vector2d<float> fpixel(ncol);
        Vector2d<float> probOut(nclass,ncol);//posterior prob for each (internal) class
        vector<float> entropy(ncol);
        if(band_opt.size()){
          for(unsigned int iband=0;iband<band_opt.size();++iband){
            if(verbose_opt[0]==2)
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
            if(verbose_opt[0]==2)
              std::cout << "reading band " << iband << std::endl;
            assert(iband>=0);
            assert(iband<nrOfBand());
            readData(buffer,iline,iband);
            for(unsigned int icol=0;icol<ncol;++icol)
              hpixel[icol].push_back(buffer[icol]);
          }
        }
        assert(nband==hpixel[0].size());
        if(verbose_opt[0]==2)
          std::cout << "used bands: " << nband << std::endl;
        //read prior
        if(priorimg_opt.size()){
          for(short iclass=0;iclass<nclass;++iclass){
            if(verbose_opt.size()>1)
              std::cout << "Reading " << priorimg_opt[0] << " band " << iclass << " line " << iline << std::endl;
            priorReader.readData(linePrior[iclass],iline,iclass);
          }
        }
        double oldRowMask=-1;//keep track of row mask to optimize number of line readings
        //process per pixel
        for(unsigned int icol=0;icol<ncol;++icol){
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
                // maskReader.readData(lineMask[imask],GDT_Int32,static_cast<unsigned int>(rowMask));
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
          //calculate image features
          fpixel[icol].clear();
          for(int iband=0;iband<nband;++iband)
            fpixel[icol].push_back((hpixel[icol][iband]-offset[iband])/scale[iband]);
          vector<float> result(nclass);
          result=net.run(fpixel[icol]);
          vector<float> prValues(nclass);
          float maxP=0;

          double normPrior=0;
          if(priorimg_opt.size()){
            for(short iclass=0;iclass<nclass;++iclass)
              normPrior+=linePrior[iclass][icol];
          }
          for(short iclass=0;iclass<nclass;++iclass){
            result[iclass]=(result[iclass]+1.0)/2.0;//bring back to scale [0,1]
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
        }//icol
        //----------------------------------- write output ------------------------------------------
        if(prob_opt.size()){
          for(unsigned int iclass=0;iclass<nclass;++iclass)
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
    return(CE_None);
  }
  catch(BadConversion conversionString){
    std::cerr << "Error: did you provide class pairs names (-c) and integer values (-r) for each class in training vector?" << std::endl;
    throw;
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
// CPLErr Jim::ann(Jim& imgWriter, app::AppFactory& app){
//   vector<double> priors;

//   //--------------------------- command line options ------------------------------------
//   Optionjl<string> training_opt("t", "training", "training vector file. A single vector file contains all training features (must be set as: B0, B1, B2,...) for all classes (class numbers identified by label option). Use multiple training files for bootstrap aggregation (alternative to the bag and bsize options, where a random subset is taken from a single training file)");
//   Optionjl<string> tlayer_opt("tln", "tln", "training layer name(s)");
//   Optionjl<string> label_opt("label", "label", "identifier for class label in training vector file.","label");
//   Optionjl<unsigned int> balance_opt("bal", "balance", "balance the input data to this number of samples for each class", 0);
//   Optionjl<bool> random_opt("random", "random", "in case of balance, randomize input data", true,2);
//   Optionjl<int> minSize_opt("min", "min", "if number of training pixels is less then min, do not take this class into account (0: consider all classes)", 0);
//   Optionjl<unsigned int> band_opt("b", "band", "band index (starting from 0, either use band option or use start to end)");
//   Optionjl<unsigned int> bstart_opt("sband", "startband", "Start band sequence number");
//   Optionjl<unsigned int> bend_opt("eband", "endband", "End band sequence number");
//   Optionjl<double> offset_opt("offset", "offset", "offset value for each spectral band input features: refl[band]=(DN[band]-offset[band])/scale[band]", 0.0);
//   Optionjl<double> scale_opt("scale", "scale", "scale value for each spectral band input features: refl=(DN[band]-offset[band])/scale[band] (use 0 if scale min and max in each band to -1.0 and 1.0)", 0.0);
//   Optionjl<double> priors_opt("prior", "prior", "prior probabilities for each class (e.g., -p 0.3 -p 0.3 -p 0.2 )", 0.0);
//   Optionjl<string> priorimg_opt("pim", "priorimg", "prior probability image (multi-band img with band for each class","",2);
//   Optionjl<unsigned short> cv_opt("cv", "cv", "n-fold cross validation mode",0);
//   Optionjl<string> cmformat_opt("cmf","cmf","Format for confusion matrix (ascii or latex)","ascii");
//   Optionjl<unsigned short> comb_opt("comb", "comb", "how to combine bootstrap aggregation classifiers (0: sum rule, 1: product rule, 2: max rule). Also used to aggregate classes with rc option. Default is sum rule (0)",0);
//   Optionjl<unsigned short> bag_opt("bag", "bag", "Number of bootstrap aggregations (default is no bagging: 1)", 1);
//   Optionjl<int> bagSize_opt("bs", "bsize", "Percentage of features used from available training features for each bootstrap aggregation (one size for all classes, or a different size for each class respectively", 100);
//   Optionjl<string> classBag_opt("cb", "classbag", "output for each individual bootstrap aggregation (default is blank)");
//   Optionjl<string> extent_opt("e", "extent", "Only classify within extent from polygons in vector file");
//   Optionjl<string> eoption_opt("eo","eo", "special extent options controlling rasterization: ATTRIBUTE|CHUNKYSIZE|ALL_TOUCHED|BURN_VALUE_FROM|MERGE_ALG, e.g., -eo ATTRIBUTE=fieldname");
//   Optionjl<string> mask_opt("m", "mask", "Only classify within specified mask (vector or raster). For raster mask, set nodata values with the option msknodata.");
//   Optionjl<short> msknodata_opt("msknodata", "msknodata", "mask value(s) not to consider for classification. Values will be taken over in classification image. Default is 0", 0);
//   Optionjl<unsigned short> nodata_opt("nodata", "nodata", "nodata value to put where image is masked as nodata", 0);
//   Optionjl<string> output_opt("o", "output", "output classification image");
//   Optionjl<string>  otype_opt("ot", "otype", "Data type for output image ({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}). Empty string: inherit type from input image");
//   Optionjl<string> colorTable_opt("ct", "ct", "colour table in ASCII format having 5 columns: id R G B ALFA (0: transparent, 255: solid)");
//   Optionjl<string> prob_opt("\0", "prob", "probability image. Default is no probability image");
//   Optionjl<string> entropy_opt("entropy", "entropy", "entropy image (measure for uncertainty of classifier output","",2);
//   Optionjl<string> active_opt("active", "active", "ogr output for active training sample.","",2);
//   Optionjl<string> ogrformat_opt("f", "f", "Output ogr format for active training sample","SQLite");
//   Optionjl<unsigned int> nactive_opt("na", "nactive", "number of active training points",1);
//   Optionjl<string> classname_opt("c", "class", "list of class names.");
//   Optionjl<short> classvalue_opt("r", "reclass", "list of class values (use same order as in class opt).");
//   Optionjl<short> verbose_opt("v", "verbose", "set to: 0 (results only), 1 (confusion matrix), 2 (debug)",0,2);
//   Optionjl<unsigned int> nneuron_opt("nn", "nneuron", "number of neurons in hidden layers in neural network (multiple hidden layers are set by defining multiple number of neurons: -n 15 -n 1, default is one hidden layer with 5 neurons)", 5);
//   Optionjl<float> connection_opt("\0", "connection", "connection reate (default: 1.0 for a fully connected network)", 1.0);
//   Optionjl<float> learning_opt("l", "learning", "learning rate (default: 0.7)", 0.7);
//   Optionjl<float> weights_opt("w", "weights", "weights for neural network. Apply to fully connected network only, starting from first input neuron to last output neuron, including the bias neurons (last neuron in each but last layer)", 0.0);
//   Optionjl<unsigned int> maxit_opt("\0", "maxit", "number of maximum iterations (epoch) (default: 500)", 500);
//   extent_opt.setHide(1);
//   eoption_opt.setHide(1);
//   band_opt.setHide(1);
//   bstart_opt.setHide(1);
//   bend_opt.setHide(1);
//   balance_opt.setHide(1);
//   minSize_opt.setHide(1);
//   bag_opt.setHide(1);
//   bagSize_opt.setHide(1);
//   comb_opt.setHide(1);
//   classBag_opt.setHide(1);
//   minSize_opt.setHide(1);
//   prob_opt.setHide(1);
//   priorimg_opt.setHide(1);
//   minSize_opt.setHide(1);
//   offset_opt.setHide(1);
//   scale_opt.setHide(1);
//   connection_opt.setHide(1);
//   weights_opt.setHide(1);
//   maxit_opt.setHide(1);
//   learning_opt.setHide(1);
//   verbose_opt.setHide(2);

//   bool doProcess;//stop process when program was invoked with help option (-h --help)
//   try{
//     doProcess=training_opt.retrieveOption(app);
//     cv_opt.retrieveOption(app);
//     cmformat_opt.retrieveOption(app);
//     tlayer_opt.retrieveOption(app);
//     classname_opt.retrieveOption(app);
//     classvalue_opt.retrieveOption(app);
//     ogrformat_opt.retrieveOption(app);
//     colorTable_opt.retrieveOption(app);
//     label_opt.retrieveOption(app);
//     priors_opt.retrieveOption(app);
//     extent_opt.retrieveOption(app);
//     balance_opt.retrieveOption(app);
//     random_opt.retrieveOption(app);
//     minSize_opt.retrieveOption(app);
//     band_opt.retrieveOption(app);
//     bstart_opt.retrieveOption(app);
//     bend_opt.retrieveOption(app);
//     offset_opt.retrieveOption(app);
//     scale_opt.retrieveOption(app);
//     priorimg_opt.retrieveOption(app);
//     comb_opt.retrieveOption(app);
//     bag_opt.retrieveOption(app);
//     bagSize_opt.retrieveOption(app);
//     classBag_opt.retrieveOption(app);
//     mask_opt.retrieveOption(app);
//     msknodata_opt.retrieveOption(app);
//     nodata_opt.retrieveOption(app);
//     output_opt.retrieveOption(app);
//     otype_opt.retrieveOption(app);
//     prob_opt.retrieveOption(app);
//     entropy_opt.retrieveOption(app);
//     active_opt.retrieveOption(app);
//     nactive_opt.retrieveOption(app);
//     verbose_opt.retrieveOption(app);
//     nneuron_opt.retrieveOption(app);
//     connection_opt.retrieveOption(app);
//     weights_opt.retrieveOption(app);
//     learning_opt.retrieveOption(app);
//     maxit_opt.retrieveOption(app);

//     if(!doProcess){
//       std::cout << std::endl;
//       std::ostringstream helpStream;
//       helpStream << "short option -h shows basic options only, use long option --help to show all options" << std::endl;
//       throw(helpStream.str());//help was invoked, stop processing
//     }

//     if(entropy_opt[0]=="")
//       entropy_opt.clear();
//     if(active_opt[0]=="")
//       active_opt.clear();
//     if(priorimg_opt[0]=="")
//       priorimg_opt.clear();

//     if(training_opt.empty()){
//       string errorString="Error: no training vector dataset provided";
//       throw(errorString);
//     }

//     if(verbose_opt[0]>=1){
//       if(mask_opt.size())
//         std::cout << "mask filename: " << mask_opt[0] << std::endl;
//       std::cout << "training vector file: " << std::endl;
//       for(unsigned int ifile=0;ifile<training_opt.size();++ifile)
//         std::cout << training_opt[ifile] << std::endl;
//     }
//     unsigned short nbag=(training_opt.size()>1)?training_opt.size():bag_opt[0];
//     if(verbose_opt[0]>=1)
//       std::cout << "number of bootstrap aggregations: " << nbag << std::endl;

//     ImgReaderOgr extentReader;
//     Jim maskReader;
//     // OGRLayer  *readLayer;

//     double ulx=0;
//     double uly=0;
//     double lrx=0;
//     double lry=0;

//     // bool maskIsVector=false;
//     if(extent_opt.size()){
//       if(mask_opt.size()){
//         string errorString="Error: can only either mask or extent, not both";
//         throw(errorString);
//       }
//       extentReader.open(extent_opt[0]);
//       // readLayer = extentReader.getDataSource()->GetLayer(0);
//       if(!(extentReader.getExtent(ulx,uly,lrx,lry))){
//         cerr << "Error: could not get extent from " << mask_opt[0] << endl;
//         exit(1);
//       }
//       maskReader.open(this->nrOfCol(),this->nrOfRow(),1,GDT_Float64);
//       double gt[6];
//       this->getGeoTransform(gt);
//       maskReader.setGeoTransform(gt);
//       maskReader.setProjection(this->getProjection());
//       // vector<double> burnValues(1,1);//burn value is 1 (single band)
//       maskReader.rasterizeBuf(extentReader,nodata_opt[0],eoption_opt);
//       maskReader.GDALSetNoDataValue(nodata_opt[0]);
//       extentReader.close();
//     }

//     ImgWriterOgr activeWriter;
//     if(active_opt.size()){
//       ImgReaderOgr trainingReader(training_opt[0]);
//       activeWriter.open(active_opt[0],ogrformat_opt[0]);
//       activeWriter.createLayer(active_opt[0],trainingReader.getProjection(),wkbPoint,NULL);
//       activeWriter.copyFields(trainingReader);
//     }
//     vector<PosValue> activePoints(nactive_opt[0]);
//     for(unsigned int iactive=0;iactive<activePoints.size();++iactive){
//       activePoints[iactive].value=1.0;
//       activePoints[iactive].posx=0.0;
//       activePoints[iactive].posy=0.0;
//     }

//     unsigned int totalSamples=0;
//     unsigned int nactive=0;
//     vector<FANN::neural_net> net(nbag);//the neural network

//     unsigned int nclass=0;
//     unsigned int nband=0;
//     unsigned int startBand=2;//first two bands represent X and Y pos

//     //normalize priors from command line
//     if(priors_opt.size()>1){//priors from argument list
//       priors.resize(priors_opt.size());
//       double normPrior=0;
//       for(unsigned int iclass=0;iclass<priors_opt.size();++iclass){
//         priors[iclass]=priors_opt[iclass];
//         normPrior+=priors[iclass];
//       }
//       //normalize
//       for(unsigned int iclass=0;iclass<priors_opt.size();++iclass)
//         priors[iclass]/=normPrior;
//     }

//     //convert start and end band options to vector of band indexes
//     if(bstart_opt.size()){
//       if(bend_opt.size()!=bstart_opt.size()){
//         string errorstring="Error: options for start and end band indexes must be provided as pairs, missing end band";
//         throw(errorstring);
//       }
//       band_opt.clear();
//       for(unsigned int ipair=0;ipair<bstart_opt.size();++ipair){
//         if(bend_opt[ipair]<=bstart_opt[ipair]){
//           string errorstring="Error: index for end band must be smaller then start band";
//           throw(errorstring);
//         }
//         for(unsigned int iband=bstart_opt[ipair];iband<=bend_opt[ipair];++iband)
//           band_opt.push_back(iband);
//       }
//     }

//     //sort bands
//     if(band_opt.size())
//       std::sort(band_opt.begin(),band_opt.end());

//     map<string,short> classValueMap;
//     vector<std::string> nameVector;
//     if(classname_opt.size()){
//       assert(classname_opt.size()==classvalue_opt.size());
//       for(unsigned int iclass=0;iclass<classname_opt.size();++iclass)
//         classValueMap[classname_opt[iclass]]=classvalue_opt[iclass];
//     }

//     //----------------------------------- Training -------------------------------
//     confusionmatrix::ConfusionMatrix cm;
//     vector< vector<double> > offset(nbag);
//     vector< vector<double> > scale(nbag);
//     map<string,Vector2d<float> > trainingMap;
//     vector< Vector2d<float> > trainingPixels;//[class][sample][band]
//     vector<string> fields;
//     for(unsigned int ibag=0;ibag<nbag;++ibag){
//       //organize training data
//       if(ibag<training_opt.size()){//if bag contains new training pixels
//         trainingMap.clear();
//         trainingPixels.clear();
//         if(verbose_opt[0]>=1)
//           std::cout << "reading imageVector file " << training_opt[0] << std::endl;
//         ImgReaderOgr trainingReaderBag(training_opt[ibag]);
//         if(band_opt.size())
//           totalSamples=trainingReaderBag.readDataImageOgr(trainingMap,fields,band_opt,label_opt[0],tlayer_opt);
//         else
//           totalSamples=trainingReaderBag.readDataImageOgr(trainingMap,fields,0,0,label_opt[0],tlayer_opt);
//         if(trainingMap.size()<2){
//           string errorstring="Error: could not read at least two classes from training file, did you provide class labels in training sample (see option label)?";
//           throw(errorstring);
//         }
//         trainingReaderBag.close();
//         //convert map to vector
//         if(verbose_opt[0]>1)
//           std::cout << "training pixels: " << std::endl;
//         map<string,Vector2d<float> >::iterator mapit=trainingMap.begin();
//         while(mapit!=trainingMap.end()){
//           //delete small classes
//           if((mapit->second).size()<minSize_opt[0]){
//             trainingMap.erase(mapit);
//             continue;
//           }
//           trainingPixels.push_back(mapit->second);
//           if(verbose_opt[0]>1)
//             std::cout << mapit->first << ": " << (mapit->second).size() << " samples" << std::endl;
//           ++mapit;
//         }
//         if(!ibag){
//           nclass=trainingPixels.size();
//           if(classname_opt.size())
//             assert(nclass==classname_opt.size());
//           nband=(training_opt.size())?trainingPixels[0][0].size()-2:trainingPixels[0][0].size();//X and Y
//         }
//         else{
//           assert(nclass==trainingPixels.size());
//           assert(nband==(training_opt.size())?trainingPixels[0][0].size()-2:trainingPixels[0][0].size());
//         }

//         //do not remove outliers here: could easily be obtained through ogr2ogr -where 'B2<110' output.shp input.shp
//         //balance training data
//         if(balance_opt[0]>0){
//           while(balance_opt.size()<nclass)
//             balance_opt.push_back(balance_opt.back());
//           if(random_opt[0])
//             srand(time(NULL));
//           totalSamples=0;
//           for(short iclass=0;iclass<nclass;++iclass){
//             if(trainingPixels[iclass].size()>balance_opt[iclass]){
//               while(trainingPixels[iclass].size()>balance_opt[iclass]){
//                 int index=rand()%trainingPixels[iclass].size();
//                 trainingPixels[iclass].erase(trainingPixels[iclass].begin()+index);
//               }
//             }
//             else{
//               int oldsize=trainingPixels[iclass].size();
//               for(int isample=trainingPixels[iclass].size();isample<balance_opt[iclass];++isample){
//                 int index = rand()%oldsize;
//                 trainingPixels[iclass].push_back(trainingPixels[iclass][index]);
//               }
//             }
//             totalSamples+=trainingPixels[iclass].size();
//           }
//         }

//         //set scale and offset
//         offset[ibag].resize(nband);
//         scale[ibag].resize(nband);
//         if(offset_opt.size()>1)
//           assert(offset_opt.size()==nband);
//         if(scale_opt.size()>1)
//           assert(scale_opt.size()==nband);
//         for(unsigned int iband=0;iband<nband;++iband){
//           if(verbose_opt[0]>=1)
//             std::cout << "scaling for band" << iband << std::endl;
//           offset[ibag][iband]=(offset_opt.size()==1)?offset_opt[0]:offset_opt[iband];
//           scale[ibag][iband]=(scale_opt.size()==1)?scale_opt[0]:scale_opt[iband];
//           //search for min and maximum
//           if(scale[ibag][iband]<=0){
//             float theMin=trainingPixels[0][0][iband+startBand];
//             float theMax=trainingPixels[0][0][iband+startBand];
//             for(short iclass=0;iclass<nclass;++iclass){
//               for(int isample=0;isample<trainingPixels[iclass].size();++isample){
//                 if(theMin>trainingPixels[iclass][isample][iband+startBand])
//                   theMin=trainingPixels[iclass][isample][iband+startBand];
//                 if(theMax<trainingPixels[iclass][isample][iband+startBand])
//                   theMax=trainingPixels[iclass][isample][iband+startBand];
//               }
//             }
//             offset[ibag][iband]=theMin+(theMax-theMin)/2.0;
//             scale[ibag][iband]=(theMax-theMin)/2.0;
//             if(verbose_opt[0]>=1){
//               std::cout << "Extreme image values for band " << iband << ": [" << theMin << "," << theMax << "]" << std::endl;
//               std::cout << "Using offset, scale: " << offset[ibag][iband] << ", " << scale[ibag][iband] << std::endl;
//               std::cout << "scaled values for band " << iband << ": [" << (theMin-offset[ibag][iband])/scale[ibag][iband] << "," << (theMax-offset[ibag][iband])/scale[ibag][iband] << "]" << std::endl;
//             }
//           }
//         }
//       }
//       else{//use same offset and scale
//         offset[ibag].resize(nband);
//         scale[ibag].resize(nband);
//         for(int iband=0;iband<nband;++iband){
//           offset[ibag][iband]=offset[0][iband];
//           scale[ibag][iband]=scale[0][iband];
//         }
//       }

//       if(!ibag){
//         if(priors_opt.size()==1){//default: equal priors for each class
//           priors.resize(nclass);
//           for(short iclass=0;iclass<nclass;++iclass)
//             priors[iclass]=1.0/nclass;
//         }
//         assert(priors_opt.size()==1||priors_opt.size()==nclass);

//         //set bagsize for each class if not done already via command line
//         while(bagSize_opt.size()<nclass)
//           bagSize_opt.push_back(bagSize_opt.back());

//         if(verbose_opt[0]>=1){
//           std::cout << "number of bands: " << nband << std::endl;
//           std::cout << "number of classes: " << nclass << std::endl;
//           std::cout << "priors:";
//           if(priorimg_opt.empty()){
//             for(short iclass=0;iclass<nclass;++iclass)
//               std::cout << " " << priors[iclass];
//             std::cout << std::endl;
//           }
//         }
//         map<string,Vector2d<float> >::iterator mapit=trainingMap.begin();
//         bool doSort=true;
//         while(mapit!=trainingMap.end()){
//           nameVector.push_back(mapit->first);
//           if(classValueMap.size()){
//             //check if name in training is covered by classname_opt (values can not be 0)
//             if(classValueMap[mapit->first]>0){
//               if(cm.getClassIndex(type2string<short>(classValueMap[mapit->first]))<0)
//                 cm.pushBackClassName(type2string<short>(classValueMap[mapit->first]),doSort);
//             }
//             else{
//               std::cerr << "Error: names in classname option are not complete, please check names in training vector and make sure classvalue is > 0" << std::endl;
//               exit(1);
//             }
//           }
//           else
//             cm.pushBackClassName(mapit->first,doSort);
//           ++mapit;
//         }
//         if(classname_opt.empty()){
//           //std::cerr << "Warning: no class name and value pair provided for all " << nclass << " classes, using string2type<int> instead!" << std::endl;
//           for(short iclass=0;iclass<nclass;++iclass){
//             if(verbose_opt[0])
//               std::cout << iclass << " " << cm.getClass(iclass) << " -> " << string2type<short>(cm.getClass(iclass)) << std::endl;
//             classValueMap[cm.getClass(iclass)]=string2type<short>(cm.getClass(iclass));
//           }
//         }
//         // if(priors_opt.size()==nameVector.size()){
//         //   std::cerr << "Warning: please check if priors are provided in correct order!!!" << std::endl;
//         //   for(unsigned int iclass=0;iclass<nameVector.size();++iclass)
//         //     std::cerr << nameVector[iclass] << " " << priors_opt[iclass] << std::endl;
//         // }
//       }//if(!ibag)

//       //Calculate features of training set
//       vector< Vector2d<float> > trainingFeatures(nclass);
//       for(short iclass=0;iclass<nclass;++iclass){
//         int nctraining=0;
//         if(verbose_opt[0]>=1)
//           std::cout << "calculating features for class " << iclass << std::endl;
//         if(random_opt[0])
//           srand(time(NULL));
//         nctraining=(bagSize_opt[iclass]<100)? trainingPixels[iclass].size()/100.0*bagSize_opt[iclass] : trainingPixels[iclass].size();//bagSize_opt[iclass] given in % of training size
//         if(nctraining<=0)
//           nctraining=1;
//         assert(nctraining<=trainingPixels[iclass].size());
//         if(bagSize_opt[iclass]<100)
//           random_shuffle(trainingPixels[iclass].begin(),trainingPixels[iclass].end());
//         if(verbose_opt[0]>1)
//           std::cout << "nctraining (class " << iclass << "): " << nctraining << std::endl;
//         trainingFeatures[iclass].resize(nctraining);
//         for(int isample=0;isample<nctraining;++isample){
//           //scale pixel values according to scale and offset!!!
//           for(int iband=0;iband<nband;++iband){
//             float value=trainingPixels[iclass][isample][iband+startBand];
//             trainingFeatures[iclass][isample].push_back((value-offset[ibag][iband])/scale[ibag][iband]);
//           }
//         }
//         assert(trainingFeatures[iclass].size()==nctraining);
//       }

//       unsigned int nFeatures=trainingFeatures[0][0].size();
//       if(verbose_opt[0]>=1)
//         std::cout << "number of features: " << nFeatures << std::endl;
//       unsigned int ntraining=0;
//       for(short iclass=0;iclass<nclass;++iclass)
//         ntraining+=trainingFeatures[iclass].size();

//       const unsigned int num_layers = nneuron_opt.size()+2;
//       const float desired_error = 0.0003;
//       const unsigned int iterations_between_reports = (verbose_opt[0])? maxit_opt[0]+1:0;
//       if(verbose_opt[0]>=1){
//         std::cout << "number of features: " << nFeatures << std::endl;
//         std::cout << "creating artificial neural network with " << nneuron_opt.size() << " hidden layer, having " << std::endl;
//         for(unsigned int ilayer=0;ilayer<nneuron_opt.size();++ilayer)
//           std::cout << nneuron_opt[ilayer] << " ";
//         std::cout << "neurons" << std::endl;
//         std::cout << "connection_opt[0]: " << connection_opt[0] << std::endl;
//         std::cout << "num_layers: " << num_layers << std::endl;
//         std::cout << "nFeatures: " << nFeatures << std::endl;
//         std::cout << "nneuron_opt[0]: " << nneuron_opt[0] << std::endl;
//         std::cout << "number of classes (nclass): " << nclass << std::endl;
//       }
//       switch(num_layers){
//       case(3):{
//         // net[ibag].create_sparse(connection_opt[0],num_layers, nFeatures, nneuron_opt[0], nclass);//replace all create_sparse with create_sparse_array due to bug in FANN!
//         unsigned int layers[3];
//         layers[0]=nFeatures;
//         layers[1]=nneuron_opt[0];
//         layers[2]=nclass;
//         net[ibag].create_sparse_array(connection_opt[0],num_layers,layers);
//         break;
//       }
//       case(4):{
//         unsigned int layers[4];
//         layers[0]=nFeatures;
//         layers[1]=nneuron_opt[0];
//         layers[2]=nneuron_opt[1];
//         layers[3]=nclass;
//         // layers.push_back(nFeatures);
//         // for(int ihidden=0;ihidden<nneuron_opt.size();++ihidden)
//         //  layers.push_back(nneuron_opt[ihidden]);
//         // layers.push_back(nclass);
//         net[ibag].create_sparse_array(connection_opt[0],num_layers,layers);
//         break;
//       }
//       default:
//         cerr << "Only 1 or 2 hidden layers are supported!" << std::endl;
//         exit(1);
//         break;
//       }
//       if(verbose_opt[0]>=1)
//         std::cout << "network created" << std::endl;

//       net[ibag].set_learning_rate(learning_opt[0]);

//       //   net.set_activation_steepness_hidden(1.0);
//       //   net.set_activation_steepness_output(1.0);

//       net[ibag].set_activation_function_hidden(FANN::SIGMOID_SYMMETRIC_STEPWISE);
//       net[ibag].set_activation_function_output(FANN::SIGMOID_SYMMETRIC_STEPWISE);

//       // Set additional properties such as the training algorithm
//       //   net.set_training_algorithm(FANN::TRAIN_QUICKPROP);

//       // Output network type and parameters
//       if(verbose_opt[0]>=1){
//         std::cout << std::endl << "Network Type                         :  ";
//         switch (net[ibag].get_network_type())
//           {
//           case FANN::LAYER:
//             std::cout << "LAYER" << std::endl;
//             break;
//           case FANN::SHORTCUT:
//             std::cout << "SHORTCUT" << std::endl;
//             break;
//           default:
//             std::cout << "UNKNOWN" << std::endl;
//             break;
//           }
//         net[ibag].print_parameters();
//       }

//       if(cv_opt[0]>1){
//         if(verbose_opt[0])
//           std::cout << "cross validation" << std::endl;
//         vector<unsigned short> referenceVector;
//         vector<unsigned short> outputVector;
//         float rmse=net[ibag].cross_validation(trainingFeatures,
//                                               ntraining,
//                                               cv_opt[0],
//                                               maxit_opt[0],
//                                               desired_error,
//                                               referenceVector,
//                                               outputVector,
//                                               verbose_opt[0]);
//         map<string,Vector2d<float> >::iterator mapit=trainingMap.begin();
//         for(unsigned int isample=0;isample<referenceVector.size();++isample){
//           string refClassName=nameVector[referenceVector[isample]];
//           string className=nameVector[outputVector[isample]];
//           if(classValueMap.size())
//             cm.incrementResult(type2string<short>(classValueMap[refClassName]),type2string<short>(classValueMap[className]),1.0/nbag);
//           else
//             cm.incrementResult(cm.getClass(referenceVector[isample]),cm.getClass(outputVector[isample]),1.0/nbag);
//         }
//       }

//       if(verbose_opt[0]>=1)
//         std::cout << std::endl << "Set training data" << std::endl;

//       if(verbose_opt[0]>=1)
//         std::cout << std::endl << "Training network" << std::endl;

//       if(verbose_opt[0]>=1){
//         std::cout << "Max Epochs " << setw(8) << maxit_opt[0] << ". "
//                   << "Desired Error: " << left << desired_error << right << std::endl;
//       }
//       if(weights_opt.size()==net[ibag].get_total_connections()){//no new training needed (same training sample)
//         vector<fann_connection> convector;
//         net[ibag].get_connection_array(convector);
//         for(unsigned int i_connection=0;i_connection<net[ibag].get_total_connections();++i_connection)
//           convector[i_connection].weight=weights_opt[i_connection];
//         net[ibag].set_weight_array(convector);
//       }
//       else{
//         bool initWeights=true;
//         net[ibag].train_on_data(trainingFeatures,ntraining,initWeights, maxit_opt[0],
//                                 iterations_between_reports, desired_error);
//       }


//       if(verbose_opt[0]>=2){
//         net[ibag].print_connections();
//         vector<fann_connection> convector;
//         net[ibag].get_connection_array(convector);
//         for(unsigned int i_connection=0;i_connection<net[ibag].get_total_connections();++i_connection)
//           std::cout << "connection " << i_connection << ": " << convector[i_connection].weight << std::endl;

//       }
//     }//for ibag
//     if(cv_opt[0]>1){
//       assert(cm.nReference());
//       cm.setFormat(cmformat_opt[0]);
//       cm.reportSE95(false);
//       std::cout << cm << std::endl;
//       // std::cout << "class #samples userAcc prodAcc" << std::endl;
//       // double se95_ua=0;
//       // double se95_pa=0;
//       // double se95_oa=0;
//       // double dua=0;
//       // double dpa=0;
//       // double doa=0;
//       // for(unsigned int iclass=0;iclass<cm.nClasses();++iclass){
//       //   dua=cm.ua_pct(cm.getClass(iclass),&se95_ua);
//       //   dpa=cm.pa_pct(cm.getClass(iclass),&se95_pa);
//       //   std::cout << cm.getClass(iclass) << " " << cm.nReference(cm.getClass(iclass)) << " " << dua << " (" << se95_ua << ")" << " " << dpa << " (" << se95_pa << ")" << std::endl;
//       // }
//       // std::cout << "Kappa: " << cm.kappa() << std::endl;
//       // doa=cm.oa_pct(&se95_oa);
//       // std::cout << "Overall Accuracy: " << doa << " (" << se95_oa << ")"  << std::endl;
//     }
//     //--------------------------------- end of training -----------------------------------

//     const char* pszMessage;
//     void* pProgressArg=NULL;
//     GDALProgressFunc pfnProgress=GDALTermProgress;
//     float progress=0;
//     if(!verbose_opt[0])
//       MyProgressFunc(progress,pszMessage,pProgressArg);
//     //-------------------------------- open image file ------------------------------------
//     // bool inputIsRaster=false;
//     bool inputIsRaster=true;
//     // ImgReaderOgr imgReaderOgr;
//     // //todo: will not work in GDAL v2.0
//     // try{
//     //   imgReaderOgr.open(input_opt[0]);
//     //   imgReaderOgr.close();
//     // }
//     // catch(string errorString){
//     //   inputIsRaster=true;
//     // }
//     // if(inputIsRaster){
//     int nrow=nrOfRow();
//     int ncol=nrOfCol();
//     if(this->isInit()){
//       if(verbose_opt[0]>=1)
//         std::cout << "opening image" << std::endl;
//       // imgWriter.open(*this,false);
//       imgWriter.open(ncol,nrow,1,GDT_Byte);
//       imgWriter.GDALSetNoDataValue(nodata_opt[0]);
//       imgWriter.copyGeoTransform(*this);
//       imgWriter.setProjection(this->getProjection());
//       if(colorTable_opt.size())
//         imgWriter.setColorTable(colorTable_opt[0],0);
//       Jim priorReader;
//       if(priorimg_opt.size()){
//         if(verbose_opt[0]>=1)
//           std::cout << "opening prior image " << priorimg_opt[0] << std::endl;
//         priorReader.open(priorimg_opt[0]);
//         assert(priorReader.nrOfCol()==ncol);
//         assert(priorReader.nrOfRow()==nrow);
//       }

//       vector<char> classOut(ncol);//classified line for writing to image file

//       //   assert(nband==imgWriter.nrOfBand());
//       Jim classImageBag;
//       // Jim classImageOut;
//       Jim probImage;
//       Jim entropyImage;

//       string imageType=imgWriter.getImageType();
//       if(classBag_opt.size()){
//         classImageBag.open(classBag_opt[0],ncol,nrow,nbag,GDT_Byte,imageType);
//         classImageBag.GDALSetNoDataValue(nodata_opt[0]);
//         classImageBag.copyGeoTransform(imgWriter);
//         classImageBag.setProjection(this->getProjection());
//       }
//       if(prob_opt.size()){
//         probImage.open(prob_opt[0],ncol,nrow,nclass,GDT_Byte,imageType);
//         probImage.GDALSetNoDataValue(nodata_opt[0]);
//         probImage.copyGeoTransform(imgWriter);
//         probImage.setProjection(this->getProjection());
//       }
//       if(entropy_opt.size()){
//         entropyImage.open(entropy_opt[0],ncol,nrow,1,GDT_Byte,imageType);
//         entropyImage.GDALSetNoDataValue(nodata_opt[0]);
//         entropyImage.copyGeoTransform(imgWriter);
//         entropyImage.setProjection(this->getProjection());
//       }

//       // if(maskIsVector){
//       //   //todo: produce unique name or perform in memory solving issue on flush memory buffer (check gdal development list on how to retrieve gdal mem buffer)
//       //   maskWriter.open("/vsimem/mask.tif",ncol,nrow,1,GDT_Float32,imageType,memory_opt[0],option_opt);
//       //   maskWriter.GDALSetNoDataValue(nodata_opt[0]);
//       //   maskWriter.copyGeoTransform(testImage);
//       //   maskWriter.setProjection(imgWriter.getProjection());
//       //   vector<double> burnValues(1,1);//burn value is 1 (single band)
//       //   maskWriter.rasterizeOgr(extentReader,burnValues);
//       //   extentReader.close();
//       //   maskWriter.close();
//       //   mask_opt.clear();
//       //   mask_opt.push_back("/vsimem/mask.tif");
//       // }
//       // // Jim maskReader;
//       if(mask_opt.size()){
//         if(verbose_opt[0]>=1)
//           std::cout << "opening mask image file " << mask_opt[0] << std::endl;
//         maskReader.open(mask_opt[0]);
//       }

//       for(unsigned int iline=0;iline<nrow;++iline){
//         vector<float> buffer(ncol);
//         vector<short> lineMask;
//         Vector2d<float> linePrior;
//         if(priorimg_opt.size())
//           linePrior.resize(nclass,ncol);//prior prob for each class
//         Vector2d<float> hpixel(ncol);
//         Vector2d<float> fpixel(ncol);
//         Vector2d<float> probOut(nclass,ncol);//posterior prob for each (internal) class
//         vector<float> entropy(ncol);
//         Vector2d<char> classBag;//classified line for writing to image file
//         if(classBag_opt.size())
//           classBag.resize(nbag,ncol);
//         //read all bands of all pixels in this line in hline
//         if(band_opt.size()){
//           for(unsigned int iband=0;iband<band_opt.size();++iband){
//             if(verbose_opt[0]==2)
//               std::cout << "reading band " << band_opt[iband] << std::endl;
//             assert(band_opt[iband]>=0);
//             assert(band_opt[iband]<imgWriter.nrOfBand());
//             readData(buffer,iline,band_opt[iband]);
//             for(unsigned int icol=0;icol<ncol;++icol)
//               hpixel[icol].push_back(buffer[icol]);
//           }
//         }
//         else{
//           for(unsigned int iband=0;iband<nband;++iband){
//             if(verbose_opt[0]==2)
//               std::cout << "reading band " << iband << std::endl;
//             assert(iband>=0);
//             assert(iband<imgWriter.nrOfBand());
//             readData(buffer,iline,iband);
//             for(unsigned int icol=0;icol<ncol;++icol)
//               hpixel[icol].push_back(buffer[icol]);
//           }
//         }
//         assert(nband==hpixel[0].size());
//         if(verbose_opt[0]==2)
//           std::cout << "used bands: " << nband << std::endl;
//         //read prior
//         if(priorimg_opt.size()){
//           for(short iclass=0;iclass<nclass;++iclass){
//             if(verbose_opt.size()>1)
//               std::cout << "Reading " << priorimg_opt[0] << " band " << iclass << " line " << iline << std::endl;
//             priorReader.readData(linePrior[iclass],iline,iclass);
//           }
//         }
//         double oldRowMask=-1;//keep track of row mask to optimize number of line readings
//         //process per pixel
//         for(unsigned int icol=0;icol<ncol;++icol){
//           assert(hpixel[icol].size()==nband);
//           bool doClassify=true;
//           bool masked=false;
//           double geox=0;
//           double geoy=0;
//           if(maskReader.isInit()){
//             //read mask
//             double colMask=0;
//             double rowMask=0;

//             imgWriter.image2geo(icol,iline,geox,geoy);
//             maskReader.geo2image(geox,geoy,colMask,rowMask);
//             colMask=static_cast<int>(colMask);
//             rowMask=static_cast<int>(rowMask);
//             if(rowMask>=0&&rowMask<maskReader.nrOfRow()&&colMask>=0&&colMask<maskReader.nrOfCol()){
//               if(static_cast<int>(rowMask)!=static_cast<int>(oldRowMask)){
//                 assert(rowMask>=0&&rowMask<maskReader.nrOfRow());
//                 // maskReader.readData(lineMask[imask],GDT_Int32,static_cast<unsigned int>(rowMask));
//                 maskReader.readData(lineMask,static_cast<unsigned int>(rowMask));
//                 oldRowMask=rowMask;
//               }
//               short theMask=0;
//               for(short ivalue=0;ivalue<msknodata_opt.size();++ivalue){
//                 // if(msknodata_opt[ivalue]>=0){//values set in msknodata_opt are invalid
//                 if(lineMask[colMask]==msknodata_opt[ivalue]){
//                   theMask=lineMask[colMask];
//                   masked=true;
//                   break;
//                 }
//                 // }
//                 // else{//only values set in msknodata_opt are valid
//                 //  if(lineMask[colMask]!=-msknodata_opt[ivalue]){
//                 //    theMask=lineMask[colMask];
//                 //    masked=true;
//                 //  }
//                 //  else{
//                 //    masked=false;
//                 //    break;
//                 //  }
//                 // }
//               }
//               if(masked){
//                 if(classBag_opt.size())
//                   for(unsigned int ibag=0;ibag<nbag;++ibag)
//                     classBag[ibag][icol]=theMask;
//                 classOut[icol]=theMask;
//                 continue;
//               }
//             }
//             bool valid=false;
//             for(unsigned int iband=0;iband<hpixel[icol].size();++iband){
//               if(hpixel[icol][iband]){
//                 valid=true;
//                 break;
//               }
//             }
//             if(!valid)
//               doClassify=false;
//           }
//           for(short iclass=0;iclass<nclass;++iclass)
//             probOut[iclass][icol]=0;
//           if(!doClassify){
//             if(classBag_opt.size())
//               for(int ibag=0;ibag<nbag;++ibag)
//                 classBag[ibag][icol]=nodata_opt[0];
//             classOut[icol]=nodata_opt[0];
//             continue;//next column
//           }
//           //----------------------------------- classification -------------------
//           for(int ibag=0;ibag<nbag;++ibag){
//             //calculate image features
//             fpixel[icol].clear();
//             for(int iband=0;iband<nband;++iband)
//               fpixel[icol].push_back((hpixel[icol][iband]-offset[ibag][iband])/scale[ibag][iband]);
//             vector<float> result(nclass);
//             result=net[ibag].run(fpixel[icol]);
//             vector<float> prValues(nclass);
//             float maxP=0;

//             //calculate posterior prob of bag
//             if(classBag_opt.size()){
//               //search for max prob within bag
//               maxP=0;
//               classBag[ibag][icol]=0;
//             }
//             double normPrior=0;
//             if(priorimg_opt.size()){
//               for(short iclass=0;iclass<nclass;++iclass)
//                 normPrior+=linePrior[iclass][icol];
//             }
//             for(short iclass=0;iclass<nclass;++iclass){
//               result[iclass]=(result[iclass]+1.0)/2.0;//bring back to scale [0,1]
//               if(priorimg_opt.size())
//                 priors[iclass]=linePrior[iclass][icol]/normPrior;//todo: check if correct for all cases... (automatic classValueMap and manual input for names and values)
//               switch(comb_opt[0]){
//               default:
//               case(0)://sum rule
//                 probOut[iclass][icol]+=result[iclass]*priors[iclass];//add probabilities for each bag
//               break;
//               case(1)://product rule
//                 probOut[iclass][icol]*=pow(static_cast<float>(priors[iclass]),static_cast<float>(1.0-nbag)/nbag)*result[iclass];//multiply probabilities for each bag
//                 break;
//               case(2)://max rule
//                 if(priors[iclass]*result[iclass]>probOut[iclass][icol])
//                   probOut[iclass][icol]=priors[iclass]*result[iclass];
//                 break;
//               }
//               if(classBag_opt.size()){
//                 //search for max prob within bag
//                 // if(prValues[iclass]>maxP){
//                 //   maxP=prValues[iclass];
//                 //   classBag[ibag][icol]=vcode[iclass];
//                 // }
//                 if(result[iclass]>maxP){
//                   maxP=result[iclass];
//                   classBag[ibag][icol]=iclass;
//                 }
//               }
//             }
//           }//ibag

//           //search for max class prob
//           float maxBag1=0;//max probability
//           float maxBag2=0;//second max probability
//           float normBag=0;
//           for(short iclass=0;iclass<nclass;++iclass){
//             if(probOut[iclass][icol]>maxBag1){
//               maxBag1=probOut[iclass][icol];
//               classOut[icol]=classValueMap[nameVector[iclass]];
//             }
//             else if(probOut[iclass][icol]>maxBag2)
//               maxBag2=probOut[iclass][icol];
//             normBag+=probOut[iclass][icol];
//           }
//           //normalize probOut and convert to percentage
//           entropy[icol]=0;
//           for(short iclass=0;iclass<nclass;++iclass){
//             float prv=probOut[iclass][icol];
//             prv/=normBag;
//             entropy[icol]-=prv*log(prv)/log(2.0);
        //     prv*=100.0;

        //     probOut[iclass][icol]=static_cast<short>(prv+0.5);
        //     // assert(classValueMap[nameVector[iclass]]<probOut.size());
        //     // assert(classValueMap[nameVector[iclass]]>=0);
        //     // probOut[classValueMap[nameVector[iclass]]][icol]=static_cast<short>(prv+0.5);
        //   }
        //   entropy[icol]/=log(static_cast<double>(nclass))/log(2.0);
        //   entropy[icol]=static_cast<short>(100*entropy[icol]+0.5);
        //   if(active_opt.size()){
        //     if(entropy[icol]>activePoints.back().value){
        //       activePoints.back().value=entropy[icol];//replace largest value (last)
        //       activePoints.back().posx=icol;
        //       activePoints.back().posy=iline;
        //       std::sort(activePoints.begin(),activePoints.end(),Decrease_PosValue());//sort in descending order (largest first, smallest last)
        //       if(verbose_opt[0])
        //         std::cout << activePoints.back().posx << " " << activePoints.back().posy << " " << activePoints.back().value << std::endl;
        //     }
        //   }
        // }//icol
        //----------------------------------- write output ------------------------------------------
    //     if(classBag_opt.size())
    //       for(unsigned int ibag=0;ibag<nbag;++ibag)
    //         classImageBag.writeData(classBag[ibag],iline,ibag);
    //     if(prob_opt.size()){
    //       for(unsigned int iclass=0;iclass<nclass;++iclass)
    //         probImage.writeData(probOut[iclass],iline,iclass);
    //     }
    //     if(entropy_opt.size()){
    //       entropyImage.writeData(entropy,iline);
    //     }
    //     imgWriter.writeData(classOut,iline);
    //     if(!verbose_opt[0]){
    //       progress=static_cast<float>(iline+1.0)/imgWriter.nrOfRow();
    //       MyProgressFunc(progress,pszMessage,pProgressArg);
    //     }
    //   }
    //   //write active learning points
    //   if(active_opt.size()){
    //     for(int iactive=0;iactive<activePoints.size();++iactive){
    //       std::map<string,double> pointMap;
    //       for(int iband=0;iband<imgWriter.nrOfBand();++iband){
    //         double value;
    //         readData(value,static_cast<unsigned int>(activePoints[iactive].posx),static_cast<unsigned int>(activePoints[iactive].posy),iband);
    //         ostringstream fs;
    //         fs << "B" << iband;
    //         pointMap[fs.str()]=value;
    //       }
    //       pointMap[label_opt[0]]=0;
    //       double x, y;
    //       imgWriter.image2geo(activePoints[iactive].posx,activePoints[iactive].posy,x,y);
    //       std::string fieldname="id";//number of the point
    //       activeWriter.addPoint(x,y,pointMap,fieldname,++nactive);
    //     }
    //   }

    //   // imgWriter.close();
    //   if(mask_opt.size())
    //     maskReader.close();
    //   if(priorimg_opt.size())
    //     priorReader.close();
    //   if(prob_opt.size())
    //     probImage.close();
    //   if(entropy_opt.size())
    //     entropyImage.close();
    //   if(classBag_opt.size())
    //     classImageBag.close();
    //   // classImageOut.close();
    // }
//     if(cm.nReference()){
//       std::cout << cm << std::endl;
//       std::cout << "class #samples userAcc prodAcc" << std::endl;
//       double se95_ua=0;
//       double se95_pa=0;
//       double se95_oa=0;
//       double dua=0;
//       double dpa=0;
//       double doa=0;
//       for(short iclass=0;iclass<cm.nClasses();++iclass){
//         dua=cm.ua_pct(cm.getClass(iclass),&se95_ua);
//         dpa=cm.pa_pct(cm.getClass(iclass),&se95_pa);
//         cout << cm.getClass(iclass) << " " << cm.nReference(cm.getClass(iclass)) << " " << dua << " (" << se95_ua << ")" << " " << dpa << " (" << se95_pa << ")" << std::endl;
//       }
//       std::cout << "Kappa: " << cm.kappa() << std::endl;
//       doa=cm.oa_pct(&se95_oa);
//       std::cout << "Overall Accuracy: " << doa << " (" << se95_oa << ")"  << std::endl;
//     }
//     if(active_opt.size())
//       activeWriter.close();
//     return(CE_None);
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
