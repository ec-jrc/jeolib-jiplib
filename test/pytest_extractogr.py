# pytest_extractogr.py: extractogr
# History
# 2017/10/24 - Created by Pieter Kempeneers (pieter.kempeneers@ec.europa.eu)
# Change log


import argparse
import os
import math
import jiplib as jl

parser=argparse.ArgumentParser()
parser.add_argument("-input","--input",help="Path of the raster dataset",dest="input",required=True,type=str)
parser.add_argument("-vector","--vector",help="Path of the vector dataset",dest="vector",required=True,type=str)
parser.add_argument("-output","--output",help="Path of the output vector dataset",dest="output",required=True,type=str)
args = parser.parse_args()

try:
    jim0=jl.createJim({'filename':args.input})
    rules=['min','max','mean','stdev']
    v01=jim0.extractOgr({'random':20,'buffer':3,'rule':rules,'output':'mem01','oformat':'Memory'})
    v01.close()
    npoint=100
    gridsize=int(jim0.nrOfCol()*jim0.getDeltaX()/math.sqrt(npoint))
    print(gridsize)
    v02=jim0.extractOgr({'grid':gridsize,'buffer':3,'rule':rules,'output':'mem02','oformat':'Memory'})
    v02.close()
    # check: some segmentation fault due to NULL feature in m_features?
    # v1=jim0.extractOgr({'grid':gridsize,'rule':'point','output':'/tmp/grid.sqlite','oformat':'SQLite'})
    # v1.write()
    # v1.close()
    v2=jim0.extractOgr({'sample':args.vector,'rule':rules[2],'output':args.output,'oformat':'SQLite'})
    v2.write()
    v2.close()
    jim0.close()
    print("Success: extractogr")
except:
    print("Failed: extractogr")