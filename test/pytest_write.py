# pytest_write.py: open raster dataset and write to file
# History
# 2017/10/23 - Created by Pieter Kempeneers (pieter.kempeneers@ec.europa.eu)
# Change log


import argparse
import os
import jiplib as jl

parser=argparse.ArgumentParser()
parser.add_argument("-input","--input",help="Path of the input file",dest="input",required=True,type=str)
parser.add_argument("-output","--output",help="Path of the output file",dest="output",required=True,type=str)
args = parser.parse_args()

try:
    jim0=jl.createJim(args.input)
    jim0.write({'filename':args.output, 'oformat': 'GTiff', 'co':['COMPRESS=LZW','TILED=YES']}).close()
    if os.path.isfile(args.output):
        print("Success: write to file")
    else:
        throw()
except:
    print("Failed: write()")

