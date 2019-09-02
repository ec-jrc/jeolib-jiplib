###############################################################################
# pytest_pytest_np.py: jim2np and np2jim
# Author(s): Pieter.Kempeneers@ec.europa.eu
# Copyright (c) 2016-2019 European Union (Joint Research Centre)
# License EUPLv1.2
# 
# This file is part of jiplib
###############################################################################

# History
# 2018/10/04 - Created by Pieter Kempeneers (pieter.kempeneers@ec.europa.eu)
# Change log


import argparse
import os
import math
import numpy as np
import jiplib as jl

parser=argparse.ArgumentParser()
parser.add_argument("-input","--input",help="Path of the input raster dataset",dest="input",required=True,type=str)
parser.add_argument("-filter","--filter",help="filter function",dest="filter",required=False,type=str)
parser.add_argument("-band","--band",help="band to filter (leave empty to filter all bands)",dest="band",required=False,type=int,nargs='+')
parser.add_argument("-size","--size",help="size",dest="size",required=False,type=int,default=3)
parser.add_argument("-nodata","--nodata",help="no data value",dest="nodata",required=False,type=int,nargs='+')
parser.add_argument("-output","--output",help="Path of the classification output raster dataset",dest="output",required=False,type=str)
args = parser.parse_args()

# try:
if True:
    jimlist=jl.createJimList()
    if args.band:
        for band in args.band:
            jim0=jl.createJim(filename=args.input,band=band)
            print("mean of jim0:{}".format(jim0.getStats({'function':'mean'})))
            projection=jim0.getProjection()
            gt=jim0.getGeoTransform()
            np0=jl.jim2np(jim0)
            jim=jl.np2jim(np0)
            print("mean of jim:{}".format(jim.getStats({'function':'mean'})))
            print("number of cols in jim:{}".format(jim.nrOfCol()))
            print("number of rows in jim:{}".format(jim.nrOfRow()))
            # jim.setGeoTransform(gt)
            # jim.setProjection(projection)
            if args.filter:
                if args.nodata:
                    print("filter with nodata value {}".format(args.nodata))
                    jimlist.pushImage(jim.filter2d({'filter':args.filter,'dx':args.size,'dy':args.size,'nodata':args.nodata}))
                else:
                    print("filter")
                    jimlist.pushImage(jim.filter2d({'filter':args.filter,'dx':args.size,'dy':args.size}))
            else:
                print("jim.crop.nrOfCol():{}".format(jim.crop({}).nrOfCol()))
                jimlist.pushImage(jim.crop({}))
        print("stacking {} bands".format(jimlist.getSize()))
        jim=jimlist.stack({'verbose':2})
        print("number of cols in jim:{}".format(jim.nrOfCol()))
        print("number of rows in jim:{}".format(jim.nrOfRow()))
        print("mean of stacked jim:{}".format(jim.getStats({'function':'mean'})))
    else:
        jim0=jl.createJim(filename=args.input)
        projection=jim0.getProjection()
        gt=jim0.getGeoTransform()
        np0=jl.jim2np(jim0)
        jim=jl.np2jim(np0)
        # jim.setGeoTransform(gt)
        # jim.setProjection(projection)
        if args.filter:
            if args.nodata:
                print("filter with nodata value {}".format(args.nodata))
                jim=jim.filter2d({'filter':args.filter,'dx':args.size,'dy':args.size,'nodata':args.nodata})
            else:
                print("filter")
                jim=jim.filter2d({'filter':args.filter,'dx':args.size,'dy':args.size})
    print("number of cols in jim:{}".format(jim.nrOfCol()))
    print("number of rows in jim:{}".format(jim.nrOfRow()))
    print("mean of stacked jim:{}".format(jim.getStats({'function':'mean'})))
    print("number of output bands:{}".format(jim.nrOfBand()))
    print(jim.getStats({'function':'mean'}))
    if args.output:
        print("write output to {}".format(args.output))
        jim.write({'filename':args.output,'co':['COMPRESS=LZW','TILED=YES']})
    jim.close()
    print("Success: filter2d")
try:
    print("ok")
except:
    print("Failed: filter2d")
