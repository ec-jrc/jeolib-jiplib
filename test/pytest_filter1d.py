# pytest_filter1d.py: filter1d
# History
# 2017/11/09 - Created by Pieter Kempeneers (pieter.kempeneers@ec.europa.eu)
# Change log


import argparse
import os
import math
import jiplib as jl

parser=argparse.ArgumentParser()
parser.add_argument("-input","--input",help="Path of the input raster dataset",dest="input",required=True,type=str)
parser.add_argument("-filter","--filter",help="filter function",dest="filter",required=False,type=str,default='median')
parser.add_argument("-nodata","--nodata",help="no data value",dest="nodata",required=False,type=int,nargs='+')
parser.add_argument("-dz","--dz",help="kernel size",dest="dz",required=False,type=int,default=3)
parser.add_argument("-threshold","--threshold",help="threshold for dwt cut",dest="threshold",required=False,type=float,default=10)
parser.add_argument("-output","--output",help="Path of the classification output raster dataset",dest="output",required=False,type=str)
parser.add_argument("-interp","--interp",help="Interpolation type",dest="interp",required=False,type=str,default="linear")
args = parser.parse_args()

try:
    jim=jl.createJim({'filename':args.input})
    if args.nodata:
        jim_filtered=jim.filter1d({'filter':args.filter,'dz':args.dz,'threshold':args.threshold,'nodata':args.nodata,'interp':args.interp})
    else:
        jim_filtered=jim.filter1d({'filter':args.filter,'dz':args.dz,'threshold':args.threshold})
    if args.output:
        jim_filtered.write({'filename':args.output})
    jim_filtered.close()
    jim.close()
    print("Success: filter1d")
except:
    print("Failed: filter1d")
