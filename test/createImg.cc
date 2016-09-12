/**********************************************************************
createImg.cc: test app creating random image for jiblib
History
2016/09/12 - Created by Pieter Kempeneers
Change log
***********************************************************************/
#include <memory>
#include "base/Optionpk.h"
#include "algorithms/StatFactory.h"
#include "jim.h"

using namespace std;
using namespace jiplib;
using namespace statfactory;

int main(int argc, char *argv[])
{
  Optionpk<string> output_opt("o", "output", "Output image file");
  Optionpk<unsigned int> nsample_opt("ns", "nsample", "Number of samples");
  Optionpk<unsigned int> nline_opt("nl", "nline", "Number of lines");
  Optionpk<unsigned int> nband_opt("b", "nband", "Number of bands",1);
  Optionpk<double> ulx_opt("ulx", "ulx", "Upper left x value bounding box", 0.0);
  Optionpk<double> uly_opt("uly", "uly", "Upper left y value bounding box", 0.0);
  Optionpk<double> lrx_opt("lrx", "lrx", "Lower right x value bounding box", 0.0);
  Optionpk<double> lry_opt("lry", "lry", "Lower right y value bounding box", 0.0);
  Optionpk<double> dx_opt("dx", "dx", "Resolution in x");
  Optionpk<double> dy_opt("dy", "dy", "Resolution in y");
  Optionpk<string> oformat_opt("of", "oformat", "Output image format (see also gdal_translate).","GTiff");
  Optionpk<string> otype_opt("ot", "otype", "Data type for output image ({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}). Empty string: inherit type from input image","");
  Optionpk<string> option_opt("co", "co", "Creation option for output file. Multiple options can be specified.");
  Optionpk<double> nodata_opt("nodata", "nodata", "Nodata value to put in image if out of bounds.");
  Optionpk<unsigned long int> seed_opt("seed", "seed", "seed value for random generator",0);
  Optionpk<double> mean_opt("mean", "mean", "Mean value for random generator",0);
  Optionpk<double> sigma_opt("sigma", "sigma", "Sigma value for random generator",1);
  Optionpk<string> description_opt("d", "description", "Set image description");
  Optionpk<string> projection_opt("a_srs", "a_srs", "Override the spatial reference for the output file (leave blank to copy from input file, use epsg:3035 to use European projection and force to European grid");
  Optionpk<unsigned long int>  memory_opt("mem", "mem", "Buffer size (in MB) to read image data blocks in memory",0,1);

  memory_opt.setHide(1);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=output_opt.retrieveOption(argc,argv);
    nsample_opt.retrieveOption(argc,argv);
    nline_opt.retrieveOption(argc,argv);
    nband_opt.retrieveOption(argc,argv);
    ulx_opt.retrieveOption(argc,argv);
    uly_opt.retrieveOption(argc,argv);
    lrx_opt.retrieveOption(argc,argv);
    lry_opt.retrieveOption(argc,argv);
    dx_opt.retrieveOption(argc,argv);
    dy_opt.retrieveOption(argc,argv);
    oformat_opt.retrieveOption(argc,argv);
    otype_opt.retrieveOption(argc,argv);
    option_opt.retrieveOption(argc,argv);
    nodata_opt.retrieveOption(argc,argv);
    seed_opt.retrieveOption(argc,argv);
    mean_opt.retrieveOption(argc,argv);
    sigma_opt.retrieveOption(argc,argv);
    description_opt.retrieveOption(argc,argv);
    projection_opt.retrieveOption(argc,argv);
    memory_opt.retrieveOption(argc,argv);
    if(doProcess&&output_opt.empty()){
      if(output_opt.empty()){
        std::cerr << "Error: no output file provided (use option -o). Use --help for help information" << std::endl;
        exit(1);
      }
      else if(nsample_opt.empty()){
        std::cerr << "Error: no number of samples (use option -ns). Use --help for help information" << std::endl;
        exit(1);
      }
      else if(nline_opt.empty()){
        std::cerr << "Error: no number of lines (use option -nl). Use --help for help information" << std::endl;
        exit(1);
      }
    }
    std::shared_ptr<Jim> imgWriter = std::make_shared<Jim>();
    imgWriter->setFile(output_opt[0],oformat_opt[0],memory_opt[0],option_opt);
    GDALDataType theType=GDT_Unknown;
    for(int iType = 0; iType < GDT_TypeCount; ++iType){
      if( GDALGetDataTypeName((GDALDataType)iType) != NULL
          && EQUAL(GDALGetDataTypeName((GDALDataType)iType),
                   otype_opt[0].c_str()))
        theType=(GDALDataType) iType;
    }
    imgWriter->open(nsample_opt[0],nline_opt[0],nband_opt[0],theType);
    imgWriter->setNoData(nodata_opt);
    if(description_opt.size())
      imgWriter->setImageDescription(description_opt[0]);
    double gt[6];
    if(ulx_opt[0]<lrx_opt[0])
      gt[0]=ulx_opt[0];
    else
      gt[0]=0;
    if(dx_opt.size())
      gt[1]=dx_opt[0];
    else if(lrx_opt[0]>0){
      gt[1]=lrx_opt[0]-ulx_opt[0];
      gt[1]/=imgWriter->nrOfCol();
    }
    else
      gt[1]=1;
    gt[2]=0;
    if(uly_opt[0]>lry_opt[0])
      gt[3]=uly_opt[0];
    else
      gt[3]=0;
    gt[4]=0;
    if(dy_opt.size())
      gt[5]=-dy_opt[0];
    else if(lry_opt[0]>0){
      gt[5]=lry_opt[0]-uly_opt[0];
      gt[5]/=imgWriter->nrOfRow();
    }
    else
      gt[5]=1;
    imgWriter->setGeoTransform(gt);
    if(projection_opt.size())
      imgWriter->setProjectionProj4(projection_opt[0]);
    StatFactory stat;
    gsl_rng* rndgen=stat.getRandomGenerator(seed_opt[0]);
    vector<double> lineBuffer(imgWriter->nrOfCol());
    for(unsigned int iband=0;iband<imgWriter->nrOfBand();++iband){
      for(unsigned int irow=0;irow<imgWriter->nrOfRow();++irow){
        for(unsigned int icol=0;icol<imgWriter->nrOfCol();++icol){
          double value=stat.getRandomValue(rndgen,"gaussian",mean_opt[0],sigma_opt[0]);
          lineBuffer[icol]=value;
        }
        imgWriter->writeData(lineBuffer,irow,iband);
      }
    }
    imgWriter->close();
  }
  catch(string helpString){//help was invoked
    std::cout << helpString << std::endl;
    return(1);
  }
  std::cout << "success" << std::endl;
  return(0);
}
