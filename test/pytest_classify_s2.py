# pytest_classify_s2.py: classify
# History
# 2018/01/22 - Created by Pieter Kempeneers (pieter.kempeneers@ec.europa.eu)
# Change log


import argparse
import os
import math
import jiplib as jl

parser=argparse.ArgumentParser()
parser.add_argument("-input","--input",help="Path of the input raster dataset",dest="input",required=True,type=str)
parser.add_argument("-reference","--reference",help="Path of the reference raster dataset",dest="reference",required=True,type=str)
parser.add_argument("-model","--model",help="Path of the model output filename used for training",dest="model",required=False,type=str)
parser.add_argument("-output","--output",help="Path of the classification output raster dataset",dest="output",required=True,type=str)
parser.add_argument("-sampleSize","--sampleSize",help="Sample size used for training svm or ann",dest="sampleSize",required=False,type=int,default=100)
parser.add_argument("-classifier","--classifier",help="classifier (sml, svm, ann)",dest="classifier",required=True,type=str,default="sml")
args = parser.parse_args()

# try:
if True:
    jim=jl.createJim({'filename':args.input})
    #preparation of reference
    classDict={}
    classDict['urban']=2
    classDict['agriculture']=12
    classDict['forest']=25
    classDict['water']=41
    classDict['rest']=50
    classFrom=range(0,50)
    classTo=[50]*50
    for i in range(0,50):
        if i>=1 and i<10:
            classTo[i]=classDict['urban']
        elif i>=11 and i<22:
            classTo[i]=classDict['agriculture']
        elif i>=23 and i<25:
            classTo[i]=classDict['forest']
        elif i>=40 and i<45:
            classTo[i]=classDict['water']
        else:
            classTo[i]=classDict['rest']

    jim_ref=jl.createJim({'filename':args.reference,'dx':jim.getDeltaX(),'dy':jim.getDeltaY(),'ulx':jim.getUlx(),'uly':jim.getUly(),'lrx':jim.getLrx(),'lry':jim.getLry()})
    jim_ref=jim_ref.reclass({'class':classFrom,'reclass':classTo})

    if args.classifier == "sml":
        reflist=jl.JimList([jim_ref])
        sml=jim.classifySML(reflist,{'class':sorted(classDict.values())})
        #preparation of output
        sml_class=sml.statProfile({'function':'maxindex'}).reclass({'class':range(0,sml.nrOfBand()),'reclass':sorted(classDict.values())})
        if args.output:
            sml_class.write({'filename':args.output})
        sml_class.close()
    else:
        trainingfn='training.shp'
        dict={'output':trainingfn}
        dict.update({'oformat':'Memory'})
        dict.update({'class':sorted(classDict.values())})
        sampleSize=-args.sampleSize #use negative values for absolute and positive values for percentage values
        dict.update({'threshold':sampleSize})
        srcnodata=[0]
        dict.update({'srcnodata':srcnodata})
        dict.update({'bandname':['B02','B03','B04','B08']})
        dict.update({'band':[0,1,2,3]})
        sample=jim.extractImg(jim_ref,dict)
        if args.classifier == "svm":
            #training
            sample.train({'method':args.classifier,'label':'label','model':args.model})
            #explicitly define all band names to use for training
            #sample.train({'method':args.classifier,'label':'label','bandname':['B02','B03','B04','B08'],'model':args.model})
            #define selection of band names to use for training
            #sample.train({'method':'svm','label':'label','bandname':['B03','B08'],'model':args.model})
            sample.close()
            #classification
            svm_class=jim.classify({'method':'svm','model':args.model})
            #explicitly define all band indexes to use for classification
            #svm_class=jim.classify({'method':'svm','model':args.model,'band':[0,1,2,3]})
            #define selection of band indexes to use for classification
            #svm_class=jim.classify({'method':'svm','model':args.model,'band':[1,3]})
            if args.output:
                svm_class.write({'filename':args.output})
            svm_class.close()
        elif args.classifier == "ann":
            #training
            sample.train({'method':args.classifier,'label':'label','model':args.model})
            #explicitly define all band names to use for training
            #sample.train({'method':'ann','label':'label','bandname':['B02','B03','B04','B08'],'model':args.model})
            #define selection of band names to use for training
            #sample.train({'method':'ann','label':'label','bandname':['B02','B04'],'model':args.model})
            sample.close()
            #classification
            ann_class=jim.classify({'method':'ann','model':args.model})
            #explicitly define all band indexes to use for classification
            #svm_class=jim.classify({'method':'ann','model':args.model,'band':[0,1,2,3]})
            #define selection of band indexes to use for classification
            #ann_class=jim.classify({'method':'ann','model':args.model,'band':[0,2]})
            if args.output:
                ann_class.write({'filename':args.output})
            ann_class.close()
        else:
            print("Error: classifier",args.classifier,"not implemented")
            throw()

    jim.close()
    print("Success: classify")
try:
    print("Success: classify")
except:
    print("Failed: classify")
    jim.close()
