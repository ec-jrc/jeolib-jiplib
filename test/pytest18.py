# pytest18.cc: setMask
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
    jim0=jl.createJim({'filename':args.input})
    theStats=jim0.getStats({'function':'max'})
    # The method setMask sets values within [min,max] to value defined by 'data', else to 'nodata'
    # The min and max values can be defined as a list
    jim1=jim0.getMask({'min':1,'max':20,'nodata':0,'data':1}).pushNoDataValue(1)
    jl1=jl.JimList([jim1])
    jim2=jim0.setMask(jl1,{'msknodata':0})
    if jim2.getNvalid()!=68335:
        print("Failed: nvalid",jim1.getNvalid())
        throw()
    if jim2.getNinvalid()!=193809:
        print("Failed: ninvalid",jim1.getNinvalid())
        throw()
    jim0.close()
    jim1.close()
    print("Success: setMask")
except:
    print("Failed: setMask")
    jim0.close()
