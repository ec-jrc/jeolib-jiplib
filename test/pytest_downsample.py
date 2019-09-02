###############################################################################
# pytest_downsample.py: 100 m)
# Author(s): Pieter.Kempeneers@ec.europa.eu
# Copyright (c) 2016-2019 European Union (Joint Research Centre)
# License EUPLv1.2
# 
# This file is part of jiplib
###############################################################################

# History
# 2017/10/23 - Created by Pieter Kempeneers (pieter.kempeneers@ec.europa.eu)
# Change log


import argparse
import os
import jiplib as jl

parser=argparse.ArgumentParser()
parser.add_argument("-input","--input",help="Path of the input file",dest="input",required=True,type=str)
args = parser.parse_args()

# jim0=jl.createJim(filename=args.input,dx=1000,dy=1000)
jim0=jl.Jim({'filename':args.input,'dx':1000,'dy':1000})
if jim0.nrOfCol() == 256:
        print("Success: createJim with downsample")
else:
    print("Failed: createJim with downsample, ncol:",jim0.nrOfCol())

