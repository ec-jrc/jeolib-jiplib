/**********************************************************************
pktestexternal: test program for Jim constructor with external data
History
2018/04/19 - Created by Pieter Kempeneers
Change log
***********************************************************************/
#include <math.h>
#include <string>
#include <fstream>
#include <assert.h>
#include "base/Optionpk.h"
#include "jim.h"

using namespace std;
using namespace jiplib;

int main(int argc, char *argv[])
{
  Optionpk<string> input_opt("i", "input", "Input shape file");
  Optionpk<string> output_opt("o", "output", "Output vector dataset");
  Optionpk<string> attribute_opt("af", "afilter", "attribute filter");
  Optionpk<short> verbose_opt("v", "verbose", "verbose (Default: 0)", 0,2);

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

