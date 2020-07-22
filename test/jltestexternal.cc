/**********************************************************************
pktestexternal: test program for Jim constructor with external data
Author(s): Pieter.Kempeneers@ec.europa.eu
Copyright (C) 2016-2020 European Union (Joint Research Centre)

This file is part of jiplib.

jiplib is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

jiplib is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with jiplib.  If not, see <https://www.gnu.org/licenses/>.
***********************************************************************/
#include <math.h>
#include <string>
#include <fstream>
#include <assert.h>
#include "base/Optionjl.h"
#include "imageclasses/Jim.h"

using namespace std;

int main(int argc, char *argv[])
{
  Optionjl<string> input_opt("i", "input", "Input shape file");
  Optionjl<string> output_opt("o", "output", "Output vector dataset");
  Optionjl<string> attribute_opt("af", "afilter", "attribute filter");
  Optionjl<short> verbose_opt("v", "verbose", "verbose (Default: 0)", 0,2);

  verbose_opt.setHide(2);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=input_opt.retrieveOption(argc,argv);
    output_opt.retrieveOption(argc,argv);
    verbose_opt.retrieveOption(argc,argv);
  }
  catch(string predefinedString){
    std::cout << predefinedString << std::endl;
    exit(0);
  }
  if(!doProcess){
    cout << endl;
    cout << "Usage: pktestogr -i input [-o output]" << endl;
    cout << endl;
    std::cout << "short option -h shows basic options only, use long option --help to show all options" << std::endl;
    exit(0);//help was invoked, stop processing
  }

  if(input_opt.empty()){
    std::cerr << "No input file provided (use option -i). Use --help for help information" << std::endl;
    exit(0);
  }
  if(output_opt.empty()){
    std::cerr << "No output file provided (use option -o). Use --help for help information" << std::endl;
    exit(0);
  }

  try{
    Jim inputReader(input_opt[0]);
    std::vector<void*> vdata(inputReader.nrOfBand());
    for(size_t iband=0;iband<inputReader.nrOfBand();++iband)
      vdata[iband]=inputReader.getDataPointer(iband);
    if(verbose_opt[0])
      std::cout << "construct externalRaster" << std::endl;
    Jim externalRaster;
    externalRaster.open(vdata,inputReader.nrOfCol(),inputReader.nrOfRow(),inputReader.nrOfPlane(),inputReader.getGDALDataType());
    externalRaster.setExternalData(true);
    externalRaster.setFile(output_opt[0],"GTiff");
    externalRaster.setProjectionProj4(inputReader.getProjection());
    double gt[6];
    inputReader.getGeoTransform(gt);
    externalRaster.setGeoTransform(gt);
    if(verbose_opt[0])
      std::cout << "write externalRaster" << std::endl;
    externalRaster.write();
    if(verbose_opt[0])
      std::cout << "close externalRaster" << std::endl;
    externalRaster.close();
    if(verbose_opt[0])
      std::cout << "close inputReader" << std::endl;
    inputReader.close();
  }
  catch(string errorstring){
    cerr << errorstring << endl;
  }
}

