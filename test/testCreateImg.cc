/**********************************************************************
testCreateImg.cc: test app creating random image for jiblib
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
