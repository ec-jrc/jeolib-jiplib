# pytest_filter2d.py: filter2d
# History
# 2017/11/07 - Created by Pieter Kempeneers (pieter.kempeneers@ec.europa.eu)
# Change log


import argparse
import os
import math
import jiplib as jl

parser=argparse.ArgumentParser()
parser.add_argument("-input","--input",help="Path of the input raster dataset",dest="input",required=True,type=str)
parser.add_argument("-filter","--filter",help="filter function",dest="filter",required=False,type=str,default='median')
parser.add_argument("-band","--band",help="band to filter (leave empty to filter all bands)",dest="band",required=False,type=int,nargs='+')
parser.add_argument("-size","--size",help="no data value",dest="size",required=False,type=int,default=3)
parser.add_argument("-cut","--cut",help="cut percentage of wavelet coefficients",dest="cut",required=False,type=int,default=0)
parser.add_argument("-nodata","--nodata",help="no data value",dest="nodata",required=False,type=int,nargs='+')
parser.add_argument("-output","--output",help="Path of the classification output raster dataset",dest="output",required=False,type=str)
args = parser.parse_args()

try:
    jim=jl.createJim(args.input)
    if args.band:
        jim=jim.crop({'band':args.band})
    if 'dwt' in args.filter:
        if args.cut > 0:
            jim_filtered=jim.filter2d({'filter':'dwt_cut','otype':'Int16','threshold':args.cut}).pushNoDataValue(0).setThreshold({'min':0,'max':255}).convert({'otype':'Byte'})
        else:
            jim_filtered=jim.filter2d({'filter':'dwt','otype':'Int16'})
            jim_filtered=jim_filtered.filter2d({'filter':'dwti'}).pushNoDataValue(0).setThreshold(0,255).crop({'otype':'Byte'})
    else:
        if args.nodata:
            jim_filtered=jim.filter2d({'filter':args.filter,'dx':args.size,'dy':args.size,'nodata':args.nodata})
        else:
            jim_filtered=jim.filter2d({'filter':args.filter,'dx':args.size,'dy':args.size})
    if args.output:
        jim_filtered.write({'filename':args.output,'co':['COMPRESS=LZW','TILED=YES']})
    jim_filtered.close()
    jim.close()
    print("Success: filter2d")
except:
    print("Failed: filter2d")
