# pytest_bb.py: Create a georeferenced image with bounding box
# History
# 2017/10/23 - Created by Pieter Kempeneers (pieter.kempeneers@ec.europa.eu)
# Change log

import argparse
import jiplib as jl

parser=argparse.ArgumentParser()
parser.add_argument("-nrow","--nrow",help="Number of rows",dest="nrow",required=False,type=int,default=1098)
parser.add_argument("-ncol","--ncol",help="Number of cols",dest="ncol",required=False,type=int,default=1098)
args = parser.parse_args()

ULX=600000.0
ULY=4000020.0
LRX=709800.0
LRY=3890220.0
projection='epsg:32612'
dict={'ulx':ULX,'uly':ULY,'lrx':LRX,'lry':LRY,'a_srs':projection}
dict.update({'otype':'GDT_UInt16'})
dict.update({'nrow':args.nrow,'ncol':args.ncol})
jim0=jl.createJim(dict)
if jim0.getDeltaX()!=100:
    print("Failed: deltaX",jim0.getDeltaX())
if jim0.getDeltaY()!=100:
    print("Failed: deltaY",jim0.getDeltaY())
else:
    print("Success: create georeferenced image with bounding box")
jim0.close()

