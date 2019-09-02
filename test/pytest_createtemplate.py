###############################################################################
# pytest_createtemplate.py: Create image using template
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
import jiplib as jl

parser=argparse.ArgumentParser()
parser.add_argument("-nrow","--nrow",help="Number of rows",dest="nrow",required=False,type=int,default=1024)
parser.add_argument("-ncol","--ncol",help="Number of cols",dest="ncol",required=False,type=int,default=1024)
parser.add_argument("-min","--min",help="min value for pixel values",dest="min",required=False,type=float,default=1000)
parser.add_argument("-max","--max",help="max value for pixel values",dest="max",required=False,type=float,default=2000)
args = parser.parse_args()

ULX=600000.0
ULY=4000020.0
LRX=709800.0
LRY=3890220.0
projection='epsg:32612'
dict={'ulx':ULX,'uly':ULY,'lrx':LRX,'lry':LRY,'a_srs':projection}
dict.update({'nrow':args.nrow,'ncol':args.ncol})
dict.update({'uniform':[args.min,args.max+1]})
dict.update({'otype':'GDT_UInt16'})
dict.update({'seed':10915})
# try:
if True:
    jim0=jl.createJim(**dict)
    jim0.d_pointOpBlank(500)
    #create a copy without copying pixel values
    jim1=jl.createJim(jim0,copyData=False)
    #images should have same geoTransform
    print("jim0.getGeoTransform:{}".format(jim0.getGeoTransform()))
    print("jim1.getGeoTransform:{}".format(jim1.getGeoTransform()))
    if jim0.getGeoTransform() != jim1.getGeoTransform():
        print("Failed: geoTransform")
        throw()
    print("stats jim0:{}".format(jim0.getStats({'function':['min','max','mean']})))
    print("stats jim1:{}".format(jim1.getStats({'function':['min','max','mean']})))
    #images should not be identical in pixel values
    if jim0.isEqual(jim1):
        print("Failed: isEqual")
        throw()
    print("Success: create image using template")
try:
    print("ok")
except:
    print("Failed: create image using template")
jim1.close()
