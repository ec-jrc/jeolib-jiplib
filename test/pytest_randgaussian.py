# pytest_randgaussian.py: Create a georeferenced image with random pixel values
# History
# 2017/10/23 - Created by Pieter Kempeneers (pieter.kempeneers@ec.europa.eu)
# Change log

import argparse
import jiplib as jl

parser=argparse.ArgumentParser()
parser.add_argument("-nrow","--nrow",help="Number of rows",dest="nrow",required=False,type=int,default=1024)
parser.add_argument("-ncol","--ncol",help="Number of cols",dest="ncol",required=False,type=int,default=1024)
parser.add_argument("-mean","--mean",help="mean value for pixel values",dest="mean",required=False,type=float,default=1000)
parser.add_argument("-stdev","--stdev",help="stdev value for pixel values",dest="stdev",required=False,type=float,default=100)
args = parser.parse_args()

ULX=600000.0
ULY=4000020.0
LRX=709800.0
LRY=3890220.0
projection='epsg:32612'
dict={'ulx':ULX,'uly':ULY,'lrx':LRX,'lry':LRY,'a_srs':projection}
dict.update({'otype':'GDT_Float32'})
dict.update({'nrow':args.nrow,'ncol':args.ncol})
dict.update({'mean':args.mean,'stdev':args.stdev})
jim0=jl.createJim(**dict)
theStats=jim0.getStats({'function':['mean','stdev']})
print(theStats)
if theStats['mean']<args.mean-1 or theStats['mean']>args.mean+1:
    print("Failed: mean")
if theStats['stdev']<args.stdev-1 or theStats['stdev']>args.stdev+1:
    print("Failed: stdev")
else:
    print("Success: create georeferenced image with random pixel values")
jim0.close()
