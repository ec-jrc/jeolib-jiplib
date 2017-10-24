# pytest7.cc: Create a georeferenced image
# History
# 2017/10/23 - Created by Pieter Kempeneers (pieter.kempeneers@ec.europa.eu)
# Change log


import argparse
import jiplib as jl

parser=argparse.ArgumentParser()
parser.add_argument("-dx","--dx",help="spatial resolution in x (in m)",dest="dx",required=False,type=int,default=100)
parser.add_argument("-dy","--dy",help="spatial resolution in y (in m)",dest="dy",required=False,type=int,default=100)
args = parser.parse_args()

ULX=600000.0
ULY=4000020.0
LRX=709800.0
LRY=3890220.0
projection='epsg:32612'
dict={'ulx':ULX,'uly':ULY,'lrx':LRX,'lry':LRY,'a_srs':projection}
dict.update({'otype':'GDT_UInt16'})
dict.update({'dy':args.dy,'dx':args.dx})
jim0=jl.createJim(dict)
if jim0.nrOfCol()!=1098:
    print("Failed: number of cols")
if jim0.nrOfRow()!=1098:
    print("Failed: number of rows")
else:
    print("Success: create georeferenced image")
jim0.close()

