# pytest16.cc: setThreshold to create mask
# History
# 2017/10/24 - Created by Pieter Kempeneers (pieter.kempeneers@ec.europa.eu)
# Change log


import argparse
import os
import jiplib as jl

parser=argparse.ArgumentParser()
parser.add_argument("-input","--input",help="Path of the input file",dest="input",required=True,type=str)
args = parser.parse_args()

try:
    jim0=jl.createJim({'filename':args.input}).pushNoDataValue(1)
    # Create binary mask where all values not within [0,1[ are set to 1 (i.e., all pixel values > 0 are set to 1)
    jim0=jim0.setThreshold(0,1)
    if jim0.getNvalid()!=87957:
        print("Failed: nvalid")
        throw()
    if jim0.getNinvalid()!=174187:
        print("Failed: ninvalid")
        throw()
    jim0.close()
    print("Success: masking")
except:
    print("Failed: masking")
    jim0.close()
