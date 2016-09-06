/**********************************************************************
test1.cc: test app for jiblib
History
2016/06/24 - Created by Pieter Kempeneers
Change log
***********************************************************************/
#include <memory>
#include "base/Optionpk.h"
#include "algorithms/Filter2d.h"
#include "apps/AppFactory.h"
#include "imageclasses/ImgCollection.h"
#include "jim.h"

using namespace std;
using namespace jiplib;

int main(int argc, char *argv[])
{
Optionpk<string>  input_opt("i", "input", "Input image file(s). If input contains multiple images, a multi-band output is created");
  Optionpk<string>  output_opt("o", "output", "Output image file");
  Optionpk<string>  oformat_opt("of", "oformat", "Output image format (see also gdal_translate).","GTiff");
  Optionpk<string> option_opt("co", "co", "Creation option for output file. Multiple options can be specified.");
  Optionpk<string>  projection_opt("a_srs", "a_srs", "Override the spatial reference for the output file (leave blank to copy from input file, use epsg:3035 to use European projection and force to European grid");
  Optionpk<double> scale_opt("scale", "scale", "output=scale*input+offset");
  Optionpk<double> offset_opt("offset", "offset", "output=scale*input+offset");
  Optionpk<unsigned long int>  memory_opt("mem", "mem", "Buffer size (in MB) to read image data blocks in memory",0,1);

  option_opt.setHide(1);
  scale_opt.setHide(1);
  offset_opt.setHide(1);
  memory_opt.setHide(1);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=input_opt.retrieveOption(argc,argv);
    output_opt.retrieveOption(argc,argv);
    oformat_opt.retrieveOption(argc,argv);
    option_opt.retrieveOption(argc,argv);
    projection_opt.retrieveOption(argc,argv);
    scale_opt.retrieveOption(argc,argv);
    offset_opt.retrieveOption(argc,argv);
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
    else if(output_opt.empty()){
      std::cerr << "Error: no output file provided (use option -o). Use --help for help information" << std::endl;
      exit(1);
    }
  }

  app::AppFactory app(argc,argv);
  // // app.setOption("dx","500");
  // // app.setOption("dy","500");
  
  filter2d::Filter2d filter;
  ImgCollection inputCollection;
  try{
    shared_ptr<Jim> inputImg(new Jim);
    inputImg->open(input_opt[0]);
    inputCollection.pushImage(inputImg);
    double theMin=0;
    double theMax=0;
    inputCollection[0]->getMinMax(theMin,theMax,0);
    cout << "min, max: " << theMin << ", " << theMax << endl;
    string imageType;
    if(oformat_opt.size())//default
      imageType=oformat_opt[0];
    else
      imageType=inputCollection[0]->getImageType();

    app.showOptions();

    Jim outputImg;
    shared_ptr<Jim> imgPointer=make_shared<Jim>(outputImg);
    imgPointer=static_pointer_cast<Jim>(inputCollection.crop(app));
    // inputCollection.crop(imgPointer,app);
    cout << "smoothing" << endl;
    filter.smooth(imgPointer,imgPointer,5);


    // for(int iband=0;iband<imgPointer->nrOfBand();++iband){
    //   IMAGE mia1=imgPointer->getMIA(iband);
    //   IMAGE mia2=imgPointer->getMIA(iband);
    //   arith(&mia1, &mia2, ADD_op);
    // }

    cout << "performing arithmetic operation" << endl;
    imgPointer->arith(imgPointer,ADD_op);
    imgPointer->rdil(imgPointer,1,1);

    cout << "performing erosion" << endl;
    filter.morphology(imgPointer,imgPointer,"erode",3,3);
    // cout << "performing dilation" << endl;
    // filter.morphology(*imgPointer,*imgPointer,"dilate",3,3);

    inputImg->close();
    imgPointer->setFile(output_opt[0],imageType);
    imgPointer->close();
  }
  catch(string helpString){//help was invoked
    std::cout << helpString << std::endl;
    return(1);
  }
  return(0);
}  
