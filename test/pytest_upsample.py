# pytest_upsample.py: 5 m)
# History
# 2017/10/23 - Created by Pieter Kempeneers (pieter.kempeneers@ec.europa.eu)
# Change log


import argparse
import os
import jiplib as jl

parser=argparse.ArgumentParser()
parser.add_argument("-input","--input",help="Path of the input file",dest="input",required=True,type=str)
args = parser.parse_args()

jim0=jl.createJim({'filename':args.input,'noread':True})
ULX=jim0.getUlx()
ULY=jim0.getUly()
LRX=ULX+1000
LRY=ULY-1000
jim0.close()
jim0=jl.createJim({'filename':args.input,'ulx':ULX,'uly':ULY,'lrx':LRX,'lry':LRY,'dx':5,'dy':5})
if jim0.nrOfCol() != 200:
    print("Failed: createJim with upsample")
elif jim0.nrOfRow() != 200:
    print("Failed: createJim with upsample")
else:
    print("Success: createJim with upsample")
jim0.close()

