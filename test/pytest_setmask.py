# pytest_setmask.py: setMask with vector
# History
# 2017/10/24 - Created by Pieter Kempeneers (pieter.kempeneers@ec.europa.eu)
# Change log


import argparse
import os
import jiplib as jl

parser=argparse.ArgumentParser()
parser.add_argument("-input","--input",help="Path of the raster dataset",dest="input",required=True,type=str)
parser.add_argument("-vectormask","--vectormask",help="Path of the vector mask dataset",dest="vm",required=False,type=str)
args = parser.parse_args()

jim0=jl.createJim({'filename':args.input})
if args.vm:
    try:
        print("create vector")
        v0=jl.createVector()
        print("open vector", args.vm)
        v0.open({'filename':args.vm,'noread':True})
        print("setMask")
        jim1=jim0.setMask(v0,{'nodata':255,'eo':'ALL_TOUCHED'})
        if jim1.getNvalid()!=250568:
            print(jim1.getNvalid())
            print("Failed: nvalid",jim1.getNvalid())
            throw()
        if jim1.getNinvalid()!=11576:
            print(jim1.getNinvalid())
            print("Failed: ninvalid",jim1.getNinvalid())
            throw()
        print("Success: setMask with vector")
    except:
        print("Failed: setMask with vector")
else:
    try:
        jim1=jim0.getMask({'min':1,'max':20,'nodata':0,'data':1}).pushNoDataValue(1)
        jl1=jl.JimList([jim1])
        jim2=jim0.setMask(jl1,{'msknodata':0})
        if jim2.getNvalid()!=68335:
            print("Failed: nvalid",jim1.getNvalid())
            throw()
        if jim2.getNinvalid()!=193809:
            print("Failed: ninvalid",jim1.getNinvalid())
            throw()
        print("Success: setMask")
    except:
        print("Failed: setMask")
jim0.close()
jim1.close()
