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
parser.add_argument("-random","--random",help="Number of random pixels to select",dest="random",required=False,type=int,default=100)
args = parser.parse_args()

# try:
if True:
    jim0=jl.createJim(args.input)
    rules=['centroid','min','max','mean','stdev']
    if not args.vector:
        v01=jim0.extractSample({'random':20,'buffer':3,'rule':rules,'output':'mem01','oformat':'Memory'})
        v01.close()
        npoint=100
        gridsize=int(jim0.nrOfCol()*jim0.getDeltaX()/math.sqrt(npoint))
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
                print("createVector")
                v0=jl.createVector(args.vector)
                print("crop Milano")
                jim_milano=jim0.cropOgr(v0,{'ln':'milano','align':True})
                milanofn=os.path.join(os.path.dirname(args.output),'milano.tif')
                print("write Milano")
                jim_milano.write({'filename':milanofn})
                print("close Milano")
                jim_milano.close()
                print("crop Lodi")
                jim_lodi=jim0.cropOgr(v0,{'ln':'lodi','align':True})
                lodifn=os.path.join(os.path.dirname(args.output),'lodi.tif')
                print("write Lodi")
                jim_lodi.write({'filename':lodifn})
                print("close Lodi")
                jim_lodi.close()
                jimlist=jl.JimList([jl.createJim(milanofn,noread=True),jl.createJim(lodifn,noread=True)])
                print("extractOgr")
                v2=jimlist.extractOgr(sample,{'rule':rules[3],'output':args.output,'oformat':'SQLite','co':'OVERWRITE=YES','all_covered':True})
                print("write v2")
                v2.write()
                print("close v2")
                v2.close()
                print("close jim_milano")
                jim_milano.close()
                print("close jim_lodi")
                jim_lodi.close()
                v0.close()
            else:
                v2=jim0.extractOgr(sample,{'rule':rules[3],'output':args.output,'oformat':'SQLite','co':'OVERWRITE=YES'})
                v2.write()
                v2.close()
        else:
            v=jl.createVector()
            for band in range(0,11):
                print(band)
                print("create jimlist")
                jl0=jl.JimList([jim0.crop({'band':band})])
                bandname='B'+str(band)
                print("bandname: ",bandname)
                if not band:
                    print("first time")
                    print("extractOgr")
                    v=jl0.extractOgr(sample,{'rule':'mean','output':args.output,'oformat':'SQLite','co':['OVERWRITE=YES'],'bandname':bandname,'fid':'fid'})
                    v.write()
                    v.close()
                else:
                    v1=jl.createVector(args.output)
                    v2=jl0.extractOgr(sample,{'rule':'mean','output':'/vsimem/v2.sqlite','oformat':'SQLite','co':['OVERWRITE=YES'],'bandname':bandname,'fid':'fid'})
                    v=v1.join(v2,{'output':args.output,'oformat':'SQLite','co':['OVERWRITE=YES'],'key':['fid']});
                    v1.close()
                    v2.close()
                    v.write()
                    v.close()
            jl0.close()
        sample.close()
    jim0.close()
    print("Success: extractogr")
try:
    print("ok")
except:
    print("Failed: extractogr")
