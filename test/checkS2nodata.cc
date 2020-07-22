/**********************************************************************
test1.cc: test app for jiblib
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
#include "base/Optionjl.h"
#include "algorithms/Filter2d.h"
#include "apps/AppFactory.h"
#include "imageclasses/ImgCollection.h"
#include "jim.h"

using namespace std;

int main(int argc, char *argv[])
{
Optionjl<string>  input_opt("i", "input", "Input image file(s). If input contains multiple images, a multi-band output is created");
  Optionjl<unsigned short>  nodata_opt("nodata", "nodata", "Nodata value to check in image.",0);
  Optionjl<unsigned long int>  memory_opt("mem", "mem", "Buffer size (in MB) to read image data blocks in memory",0,1);

  memory_opt.setHide(1);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=input_opt.retrieveOption(argc,argv);
    // output_opt.retrieveOption(argc,argv);
    nodata_opt.retrieveOption(argc,argv);
    memory_opt.retrieveOption(argc,argv);
  }
  catch(string predefinedString){
    std::cout << predefinedString << std::endl;
    exit(0);
  }
  if(doProcess){
    if(input_opt.empty()){
      std::cerr << "Error: no input file provided (use option -i). Use --help for help information" << std::endl;
      exit(1);
    }
  }

  app::AppFactory app(argc,argv);
  // app.setOption("dx","100");
  // app.setOption("dy","100");

  filter2d::Filter2d filter;
  try{
    Jim inputImg(input_opt[0]);
    shared_ptr<Jim> mask=make_shared<Jim>(inputImg,true);
    shared_ptr<Jim> marker=make_shared<Jim>(inputImg,false);
    mask->pushNoDataValue(1);
    mask->setThreshold(nodata_opt[0],nodata_opt[0],0);
    int theValue=1;
    marker->writeData(theValue,static_cast<unsigned int>(1),static_cast<unsigned int>(1),static_cast<unsigned int>(0));
    marker->rdil(mask,8,1);
    if(marker!=mask)
      std::cout << "Error: check not passed for image " << input_opt[0] << std::endl;
    else
      std::cout << "Check passed for image " << input_opt[0] << std::endl;
    inputImg.close();
    mask->close();
    marker->close();
  }
  catch(string helpString){//help was invoked
    std::cout << helpString << std::endl;
    return(1);
  }
  return(0);
}
