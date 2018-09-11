/**********************************************************************
pkclassify_lib.h: classify raster image
Copyright (C) 2018 Pieter Kempeneers

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

