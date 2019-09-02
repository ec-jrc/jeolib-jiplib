/**********************************************************************
testCreateImg.cc: test app creating random image for jiblib
Author(s): Pieter.Kempeneers@ec.europa.eu
Copyright (c) 2016-2019 European Union (Joint Research Centre)
License EUPLv1.2

This file is part of jiplib
***********************************************************************/
#include <memory>
#include <string>
#include "base/Optionjl.h"
#include "algorithms/StatFactory.h"
#include "imageclasses/Jim.h"

using namespace std;
using namespace statfactory;

int main(int argc, char *argv[])
{
  Optionjl<string> output_opt("o", "output", "Output image file");
  Optionjl<string> oformat_opt("of", "oformat", "Output image format (see also gdal_translate).","GTiff");
  Optionjl<string> option_opt("co", "co", "Creation option for output file. Multiple options can be specified.");
  Optionjl<unsigned long int>  memory_opt("mem", "mem", "Buffer size (in MB) to read image data blocks in memory",0,1);

  memory_opt.setHide(1);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  app::AppFactory app(argc,argv);
  try{
    doProcess=output_opt.retrieveOption(app);
    oformat_opt.retrieveOption(app);
    option_opt.retrieveOption(app);
    memory_opt.retrieveOption(app);

    if(doProcess&&output_opt.empty()){
      if(output_opt.empty()){
        std::cerr << "Error: no output file provided (use option -o). Use --help for help information" << std::endl;
        exit(1);
      }
    }
    Jim imgRaster(app);
    imgRaster.setFile(output_opt[0],oformat_opt[0],memory_opt[0],option_opt);
    // Jim imgRaster(output_opt[0],oformat_opt[0],memory_opt[0],option_opt);
    // Jim::createImg(imgRaster,app);
    imgRaster.close();
  }
  catch(string helpString){
    cerr << helpString << endl;
    cout << "Usage: testCreateImg --ncol columns --ncol rows [--nband bands] -o output" << endl;
    return(1);
  }
  std::cout << "test1: done" << std::endl;
  return(0);
}
