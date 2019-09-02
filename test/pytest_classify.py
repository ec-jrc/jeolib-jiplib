###############################################################################
# pytest_classify.py: classify
# Author(s): Pieter.Kempeneers@ec.europa.eu
# Copyright (c) 2016-2019 European Union (Joint Research Centre)
# License EUPLv1.2
# 
# This file is part of jiplib
###############################################################################

# History
# 2017/10/24 - Created by Pieter Kempeneers (pieter.kempeneers@ec.europa.eu)
# Change log


import argparse
import os
import math
import jiplib as jl

parser=argparse.ArgumentParser()
parser.add_argument("-input","--input",help="Path of the input raster dataset",dest="input",required=True,type=str)
parser.add_argument("-vector","--vector",help="Path of the sample vector dataset with labels",dest="vector",required=True,type=str)
parser.add_argument("-model","--model",help="Path of the model output filename used for training",dest="model",required=True,type=str)
parser.add_argument("-output","--output",help="Path of the classification output raster dataset",dest="output",required=True,type=str)
parser.add_argument("-classifier","--classifier",help="classifier (svm, ann)",dest="classifier",required=False,type=str,default="svm")
args = parser.parse_args()

try:
    print("createJim")
    jim=jl.createJim(args.input)
    print("createVector")
    sample=jl.createVector();
    print("open vector",args.vector)
    sample.open(args.vector)
    print("extractOgr")
    training=jim.extractOgr(sample,{'output':'training','oformat':'Memory','copy':'label'})
    if args.classifier == 'svm':
        #SVM classification
        print("training")
        training.train({'method':'svm','label':'label','model':args.model})
        print("classification")
        jim_classify=jim.classify({'method':'svm','model':args.model})
        jim_classify.write({'filename':args.output})
        jim_classify.close()
    else:
        #ANN classification
        training.train({'method':'ann','label':'label','model':args.model})
        jim_classify=jim.classify({'method':'ann','model':args.model})
        jim_classify.write({'filename':args.output})
        jim_classify.close()
    sample.close()
    training.close()
    jim.close()
    print("Success: classify")
except:
    print("Failed: classify")
