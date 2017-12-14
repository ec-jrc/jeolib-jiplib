# pytest_statprofile.py: statprofile
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
parser.add_argument("-output","--output",help="Path of the output vector dataset",dest="output",required=False,type=str)
args = parser.parse_args()

try:
    print('calculating functions:',args.function)
    jim0=jl.createJim({'filename':args.input})
    jim1=jim0.statProfile({'function':args.function})
    if jim1.nrOfBand() != len(args.function):
        throw()
    if args.output:
        jim1.write({'filename':args.output})
    jim1.close()
    jim0.close()
    print("Success: statprofile")
except:
    print("Failed: statprofile")
