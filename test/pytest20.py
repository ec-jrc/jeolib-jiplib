# pytest20.cc: composite min/max
# History
# 2017/10/24 - Created by Pieter Kempeneers (pieter.kempeneers@ec.europa.eu)
# Change log


import argparse
import os
import jiplib as jl

parser=argparse.ArgumentParser()
parser.add_argument("-input","--input",help="Path of the raster dataset",dest="input",required=True,type=str)
args = parser.parse_args()

try:
    jim0=jl.createJim({'filename':args.input})
    stats0=jim0.getStats({'function':['min','max'], 'band':0})
    stats5=jim0.getStats({'function':['min','max'], 'band':5})
    stats10=jim0.getStats({'function':['min','max'], 'band':10})
    jl0=jl.JimList([jim0.crop({'band':0}),jim0.crop({'band':5}),jim0.crop({'band':10})])
    jim1=jl0.composite({'crule':'minband'}).clearNoData()
    statmin=jim1.getStats({'function':['min']})
    if statmin['min']>min(stats0['min'],stats5['min'],stats10['min']):
        print("Failed: min",statmin['min'],stats0['min'],stats5['min'],stats10['min'])
        throw()
    jimsp=jl0.statProfile({'function':'min'})
    if not jim1.isEqual(jimsp):
        print("Failed: statProfile min")
        throw()
    jim1=jl0.composite({'crule':'maxband'}).clearNoData()
    statmax=jim1.getStats({'function':['max']})
    if statmax['max']<max(stats0['min'],stats5['min'],stats10['min']):
        print("Failed: max",statsc['max'],stats0['max'],stats5['max'],stats10['max'])
        throw()
    jimsp=jl0.statProfile({'function':'max'})
    if not jim1.isEqual(jimsp):
        print("Failed: statProfile max")
        throw()
    jl0.close()
    jim1.close()
    jimsp.close()
    print("Success: composite min/max")
except:
    print("Failed: composite min/max")
