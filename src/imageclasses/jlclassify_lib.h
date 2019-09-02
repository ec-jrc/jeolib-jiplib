/**********************************************************************
jlclassify_lib.h: classify raster image
Author(s): Pieter.Kempeneers@ec.europa.eu
Copyright (c) 2016-2019 European Union (Joint Research Centre)
License EUPLv1.2

This file is part of jiplib
***********************************************************************/
#pragma once

enum CLASSIFIER { NONE = 0, SVM = 1, ANN = 2, SML = 3};

static CLASSIFIER getClassifier(const std::string &method){
  std::map<std::string,CLASSIFIER> methodMap;
  methodMap["svm"]=SVM;
  methodMap["ann"]=ANN;
  methodMap["sml"]=SML;
  if(methodMap.count(method))
    return(methodMap[method]);
  else
    return(NONE);
};

