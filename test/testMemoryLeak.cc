/**********************************************************************
testMemoryLeak.cc: test app for jiblib detecting memory leaks
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
  Optionjl<int> number_opt("n", "number", "Number of images to create for memory test",10);
  Optionjl<unsigned long int>  memory_opt("mem", "mem", "Buffer size (in MB) to read image data blocks in memory",0,1);

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
      imgRaster1=Jim::createImg(app);
      imgRaster2=Jim::createImg(app);
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
