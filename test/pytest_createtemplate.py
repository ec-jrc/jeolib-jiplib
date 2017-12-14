# pytest_createtemplate.py: Create image using template
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
try:
    jim0=jl.createJim(dict)
    jim0.d_pointOpBlank(500)
    #create a copy without copying pixel values
    jim1=jl.createJim(jim0,False)
    #images should have same geoTransform
    if jim0.getGeoTransform() != jim0.getGeoTransform():
        print("Failed: geoTransform")
        throw()
    #images should not be identical in pixel values
    if jim0.isEqual(jim1):
        print("Failed: isEqual")
        throw()
    print("Success: create image using template")
except:
    print("Failed: create image using template")
jim1.close()