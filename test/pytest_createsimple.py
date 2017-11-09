# pytest_createsimple.py: Create a simple image
# History
# 2017/10/23 - Created by Pieter Kempeneers (pieter.kempeneers@ec.europa.eu)
# Change log


import argparse
import os
import jiplib as jl

parser=argparse.ArgumentParser()
parser.add_argument("-nrow","--nrow",help="Number of rows",dest="nrow",required=False,type=int,default=1024)
parser.add_argument("-ncol","--ncol",help="Number of cols",dest="ncol",required=False,type=int,default=1024)
args = parser.parse_args()

dict={'nrow':args.nrow,'ncol':args.ncol}
jim0=jl.createJim(dict)
if jim0.nrOfCol()!=args.ncol:
    print("Failed: number of cols")
if jim0.nrOfRow()!=args.nrow:
    print("Failed: number of rows")
else:
    print("Success: create simple image with nrow and ncol")
jim0.close()

