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
#include "jim.h"

using namespace std;
using namespace jiplib;
using namespace statfactory;

int main(int argc, char *argv[])
{
  Optionpk<string> output_opt("o", "output", "Output image file");
  Optionpk<string> oformat_opt("of", "oformat", "Output image format (see also gdal_translate).","GTiff");
  Optionpk<string> option_opt("co", "co", "Creation option for output file. Multiple options can be specified.");
  Optionpk<unsigned long int>  memory_opt("mem", "mem", "Buffer size (in MB) to read image data blocks in memory",0,1);

  memory_opt.setHide(1);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=output_opt.retrieveOption(argc,argv);
    oformat_opt.retrieveOption(argc,argv);
    option_opt.retrieveOption(argc,argv);
    memory_opt.retrieveOption(argc,argv);

    app::AppFactory app(argc,argv);

    if(doProcess&&output_opt.empty()){
      if(output_opt.empty()){
        std::cerr << "Error: no output file provided (use option -o). Use --help for help information" << std::endl;
        exit(1);
      }
    }
    shared_ptr<ImgRaster> pRaster=Jim::createImg();
    pRaster->setFile(output_opt[0],oformat_opt[0],memory_opt[0],option_opt);
    ImgRaster::createImg(pRaster,app);
    pRaster->close();
  }
  catch(string helpString){
    cerr << helpString << endl;
    cout << "Usage: testCreateImg -ns samples -nl lines [-b bands] -o output" << endl;
    return(1);
  }
  std::cout << "success" << std::endl;
  return(0);
}
