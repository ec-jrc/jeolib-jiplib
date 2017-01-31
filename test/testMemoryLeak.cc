/**********************************************************************
testMemoryLeak.cc: test app for jiblib detecting memory leaks
History
2017/01/23 - Created by Pieter Kempeneers
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
  Optionpk<int> number_opt("n", "number", "Number of images to create for memory test",10);
  Optionpk<unsigned long int>  memory_opt("mem", "mem", "Buffer size (in MB) to read image data blocks in memory",0,1);

  memory_opt.setHide(1);
  try{
    bool doProcess;//stop process when program was invoked with help option (-h --help)
    app::AppFactory app(argc,argv);

    doProcess=number_opt.retrieveOption(app);
    memory_opt.retrieveOption(app);

    std::shared_ptr<Jim> imgRaster1;
    std::shared_ptr<Jim> imgRaster2;
    for(int i=0;i<number_opt[0];++i){
      std::cout << "Creating shared pointer to image " << i << std::endl;
      // std::shared_ptr<Jim> imgRaster=jiplib::Jim::createImg(app);
      imgRaster1=jiplib::Jim::createImg(app);
      imgRaster2=jiplib::Jim::createImg(app);
      std::cout << "Number of rows, cols, bands: "  << imgRaster1->nrOfRow() << ", " << imgRaster1->nrOfCol() << ", " << imgRaster1->nrOfBand() << std::endl;
      // std::cout << "getMax(): "  << imgRaster->getMax() << std::endl;
      imgRaster1->close();
      std::cout << "Number of rows, cols, bands: "  << imgRaster2->nrOfRow() << ", " << imgRaster2->nrOfCol() << ", " << imgRaster2->nrOfBand() << std::endl;
      // imgRaster2->close();
    }
  }
  catch(string helpString){
    cerr << helpString << endl;
    cout << "Usage: testMemoryLeak -nrow rows -ncol rows [-nband bands] -n number" << endl;
    return(1);
  }
  std::cout << "testMemoryLeak: done" << std::endl;
  return(0);
}
