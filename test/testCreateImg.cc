/**********************************************************************
testCreateImg.cc: test app creating random image for jiblib
History
2016/09/12 - Created by Pieter Kempeneers
Change log
***********************************************************************/
#include <memory>
#include <string>
#include "base/Optionpk.h"
#include "algorithms/StatFactory.h"
#include "imageclasses/ImgRaster.h"

using namespace std;
using namespace statfactory;

int main(int argc, char *argv[])
{
  Optionpk<string> output_opt("o", "output", "Output image file");
  Optionpk<string> oformat_opt("of", "oformat", "Output image format (see also gdal_translate).","GTiff");
  Optionpk<string> option_opt("co", "co", "Creation option for output file. Multiple options can be specified.");
  Optionpk<unsigned long int>  memory_opt("mem", "mem", "Buffer size (in MB) to read image data blocks in memory",0,1);

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
    ImgRaster imgRaster(app);
    imgRaster.setFile(output_opt[0],oformat_opt[0],memory_opt[0],option_opt);
    // ImgRaster imgRaster(output_opt[0],oformat_opt[0],memory_opt[0],option_opt);
    // ImgRaster::createImg(imgRaster,app);
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
