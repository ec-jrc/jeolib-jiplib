# pytest_open.py: open raster dataset and calculate basic statistics
# History
# 2017/10/23 - Created by Pieter Kempeneers (pieter.kempeneers@ec.europa.eu)
# Change log

# Test1: simple open and close
# Open an existing JP2 raster dataset
# get maximum pixel value in raster dataset
# close raster dataset

import argparse
import jiplib as jl

parser=argparse.ArgumentParser()
parser.add_argument("-input","--input",help="Path of the input file",dest="input",required=True,type=str)
args = parser.parse_args()

jim0=jl.createJim({'filename':args.input})
# Get basic statistics of the raster datasets
theStats=jim0.getStats({'function':['min','max','mean']})
if theStats['min']!=0:
    print("Failed: getMin()")
elif theStats['max']!=73:
    print("Failed: getMax()")
elif theStats['mean']<16.9 or theStats['mean']>17:
    print("Failed: getMean()")
else:
    print("Success: open raster dataset and calculate basic statistics")
jim0.close()
