###############################################################################
# pytest_setthreshold.py: setThreshold to create mask
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
import os
import jiplib as jl

parser=argparse.ArgumentParser()
parser.add_argument("-input","--input",help="Path of the input file",dest="input",required=True,type=str)
parser.add_argument("-min","--min",help="min value for threshold",dest="min",required=True,type=int)
parser.add_argument("-max","--max",help="max value for threshold",dest="max",required=True,type=int)
args = parser.parse_args()

jim0=jl.createJim(args.input)
if(args.min==0 and args.max==1):
    try:
        jim0=jim0.pushNoDataValue(1)
        # Create binary mask where all values not within [0,1[ are set to 1 (i.e., all pixel values > 0 are set to 1)
        jim0=jim0.setThreshold({'min':args.min,'max':args.max})
        if jim0.getNvalid()!=87957:
            print("Failed: nvalid")
            throw()
        if jim0.getNinvalid()!=174187:
            print("Failed: ninvalid")
            throw()
        print("Success: masking")
    except:
        print("Failed: masking")
elif(args.min==10 and args.max==50):
    try:
        jim0=jim0.pushNoDataValue(0)
        jim1=jim0.setThreshold({'min':10,'max':50})
        theStats=jim1.getStats({'function':['min','max'],'nodata':0})
        print(theStats)
        if theStats['min'][0]!=10:
            print("Failed: min",theStats['min'])
            throw()
        if theStats['max'][0]!=50:
            print("Failed: max",theStats['max'])
            throw()
        jim1=jim0.setThreshold({'min':10,'max':50,'value':100}).clearNoData()
        theStats=jim1.getStats({'function':['min','max']})
        if theStats['min'][0]!=0:
            print("Failed: min",theStats['min'])
            throw()
        if theStats['max'][0]!=100:
            print("Failed: max",theStats['max'])
            throw()
        print("Success: masking")
    except:
        print("Failed: masking")
else:
    try:
        jim0=jim0.pushNoDataValue(0)
        jim1=jim0.setThreshold({'min':args.min,'max':args.max})
        theStats=jim1.getStats({'function':['min','max'],'nodata':0})
        print(theStats)
        print("Success: masking")
    except:
        print("Failed: masking")
    jim1.close()
jim0.close()
