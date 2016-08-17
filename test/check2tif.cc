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
    inputImg.readData();
    Jim mask(inputImg,true);
    Jim marker(inputImg,false);
    std::vector<unsigned short> lineBuffer(marker.nrOfCol());
    std::vector<unsigned short> zeroBuffer(marker.nrOfCol());
    marker.writeData(zeroBuffer,0,0);
    for(int icol=0;icol<marker.nrOfCol();++icol){
      if(icol<1||icol>marker.nrOfCol()-2)
        lineBuffer[icol]=0;
      else
        lineBuffer[icol]=1;
    }
    for(int irow=1;irow<marker.nrOfRow()-1;++irow)
      marker.writeData(lineBuffer,irow,0);
    marker.writeData(zeroBuffer,marker.nrOfRow()-1,0);

    mask.writeData(nodata_opt[0],1500,1500,0);
    mask.setThreshold(nodata_opt[0],nodata_opt[0],0,1);

    marker.rero(mask,8,1);
    marker.setFile("/scratch/test/marker_cc.tif",oformat_opt[0],memory_opt[0],option_opt);
    mask.setFile("/scratch/test/mask_cc.tif",oformat_opt[0],memory_opt[0],option_opt);
    if(marker.isEqual(mask))
      std::cout << "Check passed for image " << input_opt[0] << std::endl;
    else
      std::cout << "Error: check not passed for image " << input_opt[0] << std::endl;
    Jim outputImg(inputImg,true);
    outputImg.setFile(output_opt[0],oformat_opt[0],memory_opt[0],option_opt);
    if(inputImg==outputImg)
      cout << "created image identical to input image" << endl;
    else
      cout << "Error: created image different then input image" << endl;
    outputImg.close();
    mask.close();
    marker.close();
    inputImg.close();
  }
  catch(string helpString){//help was invoked
    std::cout << helpString << std::endl;
    return(1);
  }
  return(0);
}  
