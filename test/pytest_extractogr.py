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
parser.add_argument("-vector","--vector",help="Path of the vector dataset",dest="vector",required=False,type=str)
parser.add_argument("-output","--output",help="Path of the output vector dataset",dest="output",required=False,type=str)
parser.add_argument("-noread","--noread",help="Postpone reading raster dataset",dest="noread",required=False,type=bool,default=False)
args = parser.parse_args()

try:
    jim0=jl.createJim({'filename':args.input})
    rules=['centroid','min','max','mean','stdev']
    if not args.vector:
        v01=jim0.extractSample({'random':20,'buffer':3,'rule':rules,'output':'mem01','oformat':'Memory','verbose':1})
        v01.close()
        npoint=100
        gridsize=int(jim0.nrOfCol()*jim0.getDeltaX()/math.sqrt(npoint))
        print("gridsize: ",gridsize)
        v02=jim0.extractSample({'grid':gridsize,'buffer':3,'rule':rules,'output':'mem02','oformat':'Memory'})
        v02.close()
        if args.output:
            v1=jim0.extractSample({'grid':gridsize,'rule':'point','output':args.output,'oformat':'SQLite'})
            v1.write()
            v1.close()
    else:
        sample=jl.createVector(args.vector);
        if os.path.basename(args.vector)=='nuts_italy.sqlite':
            if args.noread:
                v0=jl.createVector()
                jim_milano=jim0.crop({'extent':args.vector,'ln':'milano','align':True})
                milanofn=os.path.join(os.path.dirname(args.output),'milano.tif')
                jim_milano.write({'filename':milanofn})
                jim_milano.close()
                jim_lodi=jim0.crop({'extent':args.vector,'ln':'lodi','align':True})
                lodifn=os.path.join(os.path.dirname(args.output),'lodi.tif')
                jim_lodi.write({'filename':lodifn})
                jim_lodi.close()
                jimlist=jl.JimList([jl.createJim({'filename':milanofn,'noread':True}),jl.createJim({'filename':lodifn,'noread':True})])
                v2=jimlist.extractOgr(sample,{'rule':rules[3],'output':args.output,'oformat':'SQLite','co':'OVERWRITE=YES','all_covered':True})
                v2.write()
                v2.close()
                jim_milano.close()
                jim_lodi.close()
            else:
                v2=jim0.extractOgr(sample,{'rule':rules[3],'output':args.output,'oformat':'SQLite','co':'OVERWRITE=YES'})
                v2.write()
                v2.close()
        else:
            v2=jim0.extractOgr(sample,{'rule':rules[2],'output':args.output,'oformat':'SQLite','co':'OVERWRITE=YES'})
            v2.write()
            v2.close()
        sample.close()
    jim0.close()
    print("Success: extractogr")
except:
    print("Failed: extractogr")
