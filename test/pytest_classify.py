# pytest_classify.py: classify
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
args = parser.parse_args()

try:
    jim=jl.createJim({'filename':args.input})
    sample=jim.extractOgr({'sample':args.vector,'output':'training','oformat':'Memory'})
    #SVM classification
    sample.train({'method':'svm','label':'label','model':args.model})
    jim_classify=jim.classify({'method':'svm','model':args.model})
    jim_classify.write({'filename':args.output})
    jim_classify.close()
    sample.train({'method':'ann','label':'label','model':args.model})
    jim_classify=jim.classify({'method':'ann','model':args.model})
    jim_classify.write({'filename':args.output})
    jim_classify.close()
    sample.close()
    jim.close()
    print("Success: classify")
except:
    print("Failed: classify")
