# pytest_extractogr.py: extractogr
# History
# 2017/10/24 - Created by Pieter Kempeneers (pieter.kempeneers@ec.europa.eu)
# Change log


import argparse
import os
import math
import jiplib as jl

parser=argparse.ArgumentParser()
parser.add_argument("-input","--input",help="Path of the raster dataset",dest="input",required=False,type=str)
parser.add_argument("-ulx","--ulx",help="left margin of bounding box",dest="ulx",required=False,type=float,default=16.1)
parser.add_argument("-lrx","--lrx",help="lrx margin of bounding box",dest="lrx",required=False,type=float,default=16.6)
parser.add_argument("-uly","--uly",help="uly margin of bounding box",dest="uly",required=False,type=float,default=48.6)
parser.add_argument("-lry","--lry",help="lry margin of bounding box",dest="lry",required=False,type=float,default=47.2)
parser.add_argument("-dx","--dx",help="Resolution in x",dest="dx",required=False,type=float)
parser.add_argument("-dy","--dy",help="Resolution in y",dest="dy",required=False,type=float)
parser.add_argument("-t_srs","--t_srs",help="Target spatial reference system of bounding box",dest="t_srs",required=False,type=str,default='4326')
parser.add_argument("-output","--output",help="Path of the output vector dataset",dest="output",required=True,type=str)
parser.add_argument("-random","--random",help="Number of random pixels to select",dest="random",required=False,type=int,default=10)
args = parser.parse_args()

try:
    rules=['median']
    if args.output:
        output=args.output
        oformat='SQLite'
    else:
        output='mem01'
        oformat='Memory'

    openDict={'t_srs':'epsg:'+args.t_srs}
    openDict.update({'ulx':args.ulx,'uly':args.uly,'lrx':args.lrx,'lry':args.lry})
    if(args.dx):
        openDict.update({'dx':args.dx})
    if(args.dy):
        openDict.update({'dy':args.dy})
    if args.input:
        refpath=args.input
    else:
        refpath='/eos/jeodpp/data/base/Landcover/EUROPE/CorineLandCover/CLC2012/VER18-5/Data/GeoTIFF/250m/g250_clc12_V18_5.tif'
        openDict.update({'s_srs':'epsg:3035'})

    openDict.update({'filename':refpath})
    jim_ref=jl.createJim(openDict)
    # jim_ref=jim_ref.warp({'t_srs':'epsg:'+args.t_srs})

    classDict={}
    classDict['urban']=2
    classDict['agriculture']=12
    classDict['forest']=25
    classDict['water']=41
    classDict['rest']=50
    sorted(classDict.values())

    print(classDict)

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

    jim_ref=jim_ref.reclass({'class':classFrom,'reclass':classTo})
    jim_ref.write({'filename':'/vsimem/reference.tif'})

    labels=classDict.values()
    print(labels)
    print("open vector file")
    v=jl.createVector({'filename':output,'oformat':oformat})
    for classname in classDict:
        print("class: ",classname)
        label=classDict[classname]
        srcnodata=classDict.values()
        srcnodata.remove(label)
        srcnodata.append(255)
        print(srcnodata)
        print("extract")
        jim_ref.extractSample({'ln':classname,'random':args.random,'rule':rules,'output':output,'oformat':oformat,'bandname':['label'],'mask':'/vsimem/reference.tif','msknodata':srcnodata,'buffer':1}).write()
    v.close()
    jim_ref.close()
    print("Success: extractogr")
except:
    print("Failed: extractogr")
