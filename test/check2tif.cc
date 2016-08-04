/**********************************************************************
test1.cc: test app for jiblib
History
2016/06/24 - Created by Pieter Kempeneers
Change log
***********************************************************************/
#include <memory>
#include "base/Optionpk.h"
#include "apps/AppFactory.h"
#include "imageclasses/ImgCollection.h"
#include "jim.h"

using namespace std;
using namespace jiplib;

int main(int argc, char *argv[])
{
  Optionpk<string>  input_opt("i", "input", "Input image file(s). If input contains multiple images, a multi-band output is created");
  Optionpk<unsigned short>  nodata_opt("nodata", "nodata", "Nodata value to check in image.",0);
  Optionpk<string>  output_opt("o", "output", "Output image file");
  Optionpk<string>  oformat_opt("of", "oformat", "Output image format (see also gdal_translate).","GTiff");
  Optionpk<string> option_opt("co", "co", "Creation option for output file. Multiple options can be specified.");
  Optionpk<unsigned long int>  memory_opt("mem", "mem", "Buffer size (in MB) to read image data blocks in memory",0,1);

  memory_opt.setHide(1);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=input_opt.retrieveOption(argc,argv);
    nodata_opt.retrieveOption(argc,argv);
    output_opt.retrieveOption(argc,argv);
    oformat_opt.retrieveOption(argc,argv);
    option_opt.retrieveOption(argc,argv);
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

  try{
    Jim inputImg(input_opt[0]);
    Jim mask(inputImg,true);
    mask.setThreshold(nodata_opt[0],nodata_opt[0],0,1);
    Jim marker(inputImg,false);
    int theValue=1;
    marker.writeData(theValue,static_cast<unsigned int>(1),static_cast<unsigned int>(1),static_cast<unsigned int>(0));
    marker.rdil(mask,8,1);
    CPLErr imagesDiffer=marker.imequalp(mask);
    mask.close();
    marker.close();
    if(imagesDiffer!=CE_None)
      std::cout << "Error: check not passed for image " << input_opt[0] << std::endl;
    else{
      std::cout << "Converting " << input_opt[0] << std::endl;
      // Jim outputImg(inputImg,true);
      //test
      Jim outputImg(inputImg,false);
      IMAGE inputIM=inputImg.getMIA(0);
      IMAGE outputIM=outputImg.getMIA(0);
      //bug: imequalp does not seem to work...
      std::cout << "image equal?: " << imequalp(&inputIM,&outputIM) << std::endl;
      std::cout << "image equal?: " << imequalp(&inputIM,&inputIM) << std::endl;
      //test
      iminfo(&inputIM);
      dumpxyz(&inputIM,1500,1500,0,5,5);
      
      if(inputImg.imequalp(outputImg)!=CE_None)
        std::cout << "input != output" << std::endl;
      else
        std::cout << "input = output" << std::endl;
      std::cout << "outputImg.nrOfCol(): " << outputImg.nrOfCol() << std::endl;
      std::cout << "outputImg.nrOfRow(): " << outputImg.nrOfRow() << std::endl;
      outputImg.readData(theValue,1500,1500,0);
      std::cout << "theValue: " << theValue << std::endl;
      outputImg.setFile(output_opt[0],oformat_opt[0],memory_opt[0],option_opt);
      outputImg.close();
    }
    inputImg.close();
  }
  catch(string helpString){//help was invoked
    std::cout << helpString << std::endl;
    return(1);
  }
  return(0);
}  
