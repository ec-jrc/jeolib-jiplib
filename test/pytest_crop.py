# pytest_crop.py: crop
# History
# 2018/03/01 - Created by Pieter Kempeneers (pieter.kempeneers@ec.europa.eu)
# Change log


import argparse
import os
import math
import jiplib as jl

parser=argparse.ArgumentParser()
parser.add_argument("-input","--input",help="Path of the raster dataset",dest="input",required=True,type=str)
parser.add_argument("-extent","--extent",help="Path of the vector dataset",dest="extent",required=False,type=str)
parser.add_argument("-output","--output",help="Path of the output vector dataset",dest="output",required=False,type=str)
parser.add_argument("-nodata","--nodata",help="nodata value to put in cut region",dest="nodata",required=False,default=0,type=int)
parser.add_argument("-cut_to_cutline","--cut_to_cutline",help="Crop the extent of the target dataset to the extent of the cutline, setting the outside area to nodata",dest="cut_to_cutline",required=False,type=bool,default=False)
parser.add_argument("-cut_in_cutline","--cut_in_cutline",help="Crop the extent of the target dataset to the extent of the cutline, setting the inner area to nodata",dest="cut_in_cutline",required=False,type=bool,default=False)
args = parser.parse_args()

# try:
if True:
    print("createJim")
    jim0=jl.createJim(args.input)
    rules=['centroid','min','max','mean','stdev']
    print("createVector")
    jlv=jl.createVector(args.extent);
    if os.path.basename(args.extent)=='nuts_italy.sqlite':
        if args.cut_to_cutline:
            print("cut out")
            print("milano")
            jim_milano=jim0.cropOgr(jlv,{'ln':'milano','crop_to_cutline':True,'nodata':args.nodata,'align':True})
            print("lodi")
            jim_lodi=jim0.cropOgr(jlv,{'ln':'lodi','crop_to_cutline':True,'nodata':args.nodata,'align':True})
        elif args.cut_in_cutline:
            print("cut within")
            # to keep region
            jim_milano=jim0.cropOgr(jlv,{'ln':'milano','crop_in_cutline':True,'nodata':args.nodata,'align':True})
            jim_lodi=jim0.cropOgr(jlv,{'ln':'lodi','crop_in_cutline':True,'nodata':args.nodata,'align':True})
        milanofn=os.path.join(os.path.dirname(args.output),'milano.tif')
        print("write")
        lodifn=os.path.join(os.path.dirname(args.output),'lodi.tif')
        jim_milano.write({'filename':milanofn})
        jim_milano.close()
        jim_lodi.write({'filename':lodifn})
        jim_lodi.close()
    else:
        jim0.cropOgr(jlv,{'align':True}).write({'filename':args.output}).close()
    jim0.close()
    jlv.close()
try:
    print("ok")
    print("Success: crop")
except:
    print("Failed: crop")
