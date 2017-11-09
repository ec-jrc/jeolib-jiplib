# pytest_createreference.py: Create a reference image
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
    #create a reference
    jim1=jim0
    jim0.d_pointOpBlank(500)
    if not jim1.isEqual(jim0):
        print("Failed: isEqual")
        throw()
    jim0=None
    theStats=jim1.getStats({'function':['max']})
    if theStats['max']!=500:
        print("Failed: max")
        throw()
    print("Success: Create a reference image")
except:
    print("Failed: Create a reference image")
jim1.close()
