###############################################################################
# pytest_statprofile.py: statprofile
# Author(s): Pieter.Kempeneers@ec.europa.eu
# Copyright (c) 2016-2019 European Union (Joint Research Centre)
# License EUPLv1.2
# 
# This file is part of jiplib
###############################################################################

# History
# 2017/12/13 - Created by Pieter Kempeneers (pieter.kempeneers@ec.europa.eu)
# Change log


import argparse
import os
import math
import jiplib as jl

parser=argparse.ArgumentParser()
parser.add_argument("-input","--input",help="Path of the raster dataset",dest="input",required=True,type=str)
parser.add_argument("-function","--function",help="Statistical functions to perform",dest="function",required=True,type=str,nargs='+')
parser.add_argument("-perc","--perc",help="Percentiles to calculate",dest="percentile",required=False,type=str,nargs='*')
parser.add_argument("-output","--output",help="Path of the output vector dataset",dest="output",required=False,type=str)
args = parser.parse_args()

try:
    print('calculating functions:',args.function)
    jim0=jl.createJim(args.input)
    if args.percentile:
        jim1=jim0.statProfile({'function':args.function,'perc':args.percentile})
    else:
        jim1=jim0.statProfile({'function':args.function})
    if jim1.nrOfBand() < len(args.function):
        throw()
    if args.output:
        jim1.write({'filename':args.output})
    jim1.close()
    jim0.close()
    print("Success: statprofile")
except:
    print("Failed: statprofile")
