# pytest_getmask.py: getMask
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
    jim0=jl.createJim(args.input)
    theStats=jim0.getStats({'function':'max'})
    # The method getMask sets values within [min,max] to value defined by 'data', else to 'nodata'
    # The min and max values can be defined as a list
    jim1=jim0.getMask({'min':1,'max':theStats['max'],'nodata':0,'data':1}).pushNoDataValue(1)
    if jim1.getNvalid()!=87957:
        print("Failed: nvalid",jim1.getNvalid())
        throw()
    if jim1.getNinvalid()!=174187:
        print("Failed: ninvalid",jim1.getNinvalid())
        throw()
    # The min and max values can be defined as a list (min and max values should correspond element wise)
    # We create a mask with pixel value=1 for pixels within [1,1000] and [2000,maxValue]
    # Pixels not within those ranges get a no data value 0
    jim1=jim0.getMask({'min':[1,30],'max':[10,theStats['max']],'nodata':0,'data':1}).setNoDataValue(0)
    if jim1.getNvalid()!=112924:
        print("Failed: nvalid",jim1.getNvalid())
        throw()
    if jim1.getNinvalid()!=149220:
        print("Failed: ninvalid",jim1.getNinvalid())
        throw()
    jim0.close()
    jim1.close()
    print("Success: getMask")
except:
    print("Failed: getMask")
    jim0.close()
