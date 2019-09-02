###############################################################################
# pytest_open.py: open raster dataset and calculate basic statistics
# Author(s): Pieter.Kempeneers@ec.europa.eu
# Copyright (c) 2016-2019 European Union (Joint Research Centre)
# License EUPLv1.2
#
# This file is part of jiplib
###############################################################################

# History
# 2017/10/23 - Created by Pieter Kempeneers (pieter.kempeneers@ec.europa.eu)
# Change log

# Test1: simple open and close
# Open an existing JP2 raster dataset
# get maximum pixel value in raster dataset
# close raster dataset

import argparse
import jiplib as jl

parser=argparse.ArgumentParser()
parser.add_argument("-input","--input",help="Path of the input file",dest="input",required=True,type=str)
args = parser.parse_args()

jim0=jl.createJim(args.input)
# Get basic statistics of the raster datasets
theStats=jim0.getStats({'function':['min','max','mean']})
if max(theStats['min'])!=0:
    print("Failed: getMin()")
elif min(theStats['max'])!=73:
    print("Failed: getMax()")
elif theStats['mean'][0]<16.9 or theStats['mean'][0]>17:
    print("Failed: getMean()")
else:
    print("Success: open raster dataset and calculate basic statistics")
jim0.close()
