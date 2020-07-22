/**********************************************************************
jlclassify_lib.h: classify raster image
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
#pragma once

enum CLASSIFIER { NONE = 0, SVM = 1, ANN = 2, SML = 3};

static CLASSIFIER getClassifier(const std::string &method){
  std::map<std::string,CLASSIFIER> methodMap;
  methodMap["SVM"]=SVM;
  methodMap["svm"]=SVM;
  methodMap["ANN"]=ANN;
  methodMap["ann"]=ANN;
  methodMap["SML"]=SML;
  methodMap["sml"]=SML;
  if(methodMap.count(method))
    return(methodMap[method]);
  else
    return(NONE);
};

