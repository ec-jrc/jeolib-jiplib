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
parser.add_argument("-output","--output",help="Path of the classification output raster dataset",dest="output",required=False,type=str)
args = parser.parse_args()

try:
    jim=jl.createJim({'filename':args.input})
    if args.nodata:
        jim_filtered=jim.filter2d({'filter':args.filter,'dx':3,'dy':3,'nodata':args.nodata})
    else:
        jim_filtered=jim.filter2d({'filter':args.filter,'dx':3,'dy':3})
    if args.output:
        jim_filtered.write({'filename':args.output})
    jim_filtered.close()
    jim.close()
    print("Success: filter1d")
except:
    print("Failed: filter1d")
