# pytest19.cc: setMask with vector
# History
# 2017/10/24 - Created by Pieter Kempeneers (pieter.kempeneers@ec.europa.eu)
# Change log


import argparse
import os
import jiplib as jl

parser=argparse.ArgumentParser()
parser.add_argument("-input","--input",help="Path of the raster dataset",dest="input",required=True,type=str)
parser.add_argument("-vectormask","--vectormask",help="Path of the vector dataset",dest="vm",required=True,type=str)
args = parser.parse_args()

try:
    jim0=jl.createJim({'filename':args.input})
    v0=jl.createVector()
    v0.open({'filename':args.vm})
    jim1=jim0.setMask({'vectormask':args.vm,'nodata':1})
    if jim1.getNvalid()!=248716:
        print("Failed: nvalid",jim1.getNvalid())
        throw()
    if jim1.getNinvalid()!=13428:
        print("Failed: ninvalid",jim1.getNinvalid())
        throw()
    jim0.close()
    jim1.close()
    print("Success: setMask with vector")
except:
    print("Failed: setMask with vector")
    jim0.close()
