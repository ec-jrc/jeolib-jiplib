# pytest15.cc: Masking
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
    jim0=jl.createJim({'filename':args.input}).pushNoDataValue(0)
    jim1=jim0.setThreshold(10,50)
    theStats=jim1.getStats({'function':['min','max']})
    if theStats['min']!=10:
        print("Failed: min",theStats['min'])
        throw()
    if theStats['max']!=50:
        print("Failed: max",theStats['max'])
        throw()
    jim1=jim0.setThreshold(10,50,100).clearNoData()
    theStats=jim1.getStats({'function':['min','max']})
    if theStats['min']!=0:
        print("Failed: min",theStats['min'])
        throw()
    if theStats['max']!=100:
        print("Failed: max",theStats['max'])
        throw()
    print("Success: masking")
except:
    print("Failed: masking")
jim0.close()
jim1.close()
