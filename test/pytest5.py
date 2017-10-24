# pytest5.cc: Open spatial subset based on extent of vector dataset
# History
# 2017/10/23 - Created by Pieter Kempeneers (pieter.kempeneers@ec.europa.eu)
# Change log


import argparse
import os
import jiplib as jl

parser=argparse.ArgumentParser()
parser.add_argument("-input","--input",help="Path of the input file",dest="input",required=True,type=str)
parser.add_argument("-extent","--extent",help="Path of the extent file",dest="extent",required=True,type=str)
args = parser.parse_args()


jim0=jl.createJim({'filename':args.input,'extent':args.extent})
v0=jl.createVector()
v0.open({'filename':args.extent})
if jim0.getUlx()<v0.getUlx()-jim0.getDeltaX() or jim0.getUlx()>v0.getUlx()+jim0.getDeltaX():
    print("1Failed: get spatial extent ulx")
elif jim0.getUly()>v0.getUly()+jim0.getDeltaY() or jim0.getUly()<v0.getUly()-jim0.getDeltaY():
    print("2Failed: get spatial extent uly")
if jim0.getLrx()<v0.getLrx()-jim0.getDeltaX() or jim0.getLrx()>v0.getLrx()+jim0.getDeltaX():
    print("3Failed: get spatial extent lrx")
elif jim0.getLry()>v0.getLry()+jim0.getDeltaY() or jim0.getLry()<v0.getLry()-jim0.getDeltaY():
    print("4Failed: get spatial extent lry")
else:
    print("Success: createJim with extent")
jim0.close()
v0.close()

