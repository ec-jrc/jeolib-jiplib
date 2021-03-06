###############################################################################
# pytest_init.py: Create a georeferenced image with initialized pixel values
# Author(s): Pieter.Kempeneers@ec.europa.eu
# Copyright (c) 2016-2019 European Union (Joint Research Centre)
# License EUPLv1.2
# 
# This file is part of jiplib
###############################################################################

# History
# 2017/10/23 - Created by Pieter Kempeneers (pieter.kempeneers@ec.europa.eu)
# Change log

import argparse
import jiplib as jl

parser=argparse.ArgumentParser()
parser.add_argument("-nrow","--nrow",help="Number of rows",dest="nrow",required=False,type=int,default=1024)
parser.add_argument("-ncol","--ncol",help="Number of cols",dest="ncol",required=False,type=int,default=1024)
parser.add_argument("-value","--value",help="Initial value for image pixels",dest="value",required=False,type=float,default=1000)
args = parser.parse_args()

ULX=600000.0
ULY=4000020.0
LRX=709800.0
LRY=3890220.0
projection='epsg:32612'
dict={'ulx':ULX,'uly':ULY,'lrx':LRX,'lry':LRY,'a_srs':projection}
dict.update({'otype':'GDT_Float32'})
nrow=1098
ncol=1098
# dict.update({'nrow':args.nrow,'ncol':args.ncol,'mean':args.value})
dict.update({'nrow':args.nrow,'ncol':args.ncol})
jim0=jl.createJim(**dict)
print("gds before pointOpBlank:{}".format(jim0.getDataset()))
jim0=jim0.pointOpBlank(args.value)
print("gds after pointOpBlank:{}".format(jim0.getDataset()))
theStats=jim0.getStats({'function':['min','max']})
print("gds:{}".format(jim0.getDataset()))
print("statistics:{}".format(theStats))
if theStats['max']!=args.value:
    print("Failed: max")
if theStats['min']!=args.value:
    print("Failed: min")
else:
    print("Success: create georeferenced image with initialized pixel values")
print("gds:{}".format(jim0.getDataset()))
jim0.close()
