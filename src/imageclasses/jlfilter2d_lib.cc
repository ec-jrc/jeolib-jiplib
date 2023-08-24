/**********************************************************************
jlfilter2d_lib.cc: program to filter raster images in spatial domain
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
#include <assert.h>
#include <iostream>
#include <string>
#include <fstream>
#include <memory>
// #include <math.h>
#include <cmath>
#include <sys/types.h>
#include <stdio.h>
#include "base/Optionjl.h"
#include "fileclasses/FileReaderAscii.h"
#include "imageclasses/Jim.h"
#include "algorithms/StatFactory.h"
#include "algorithms/Filter.h"
#include "algorithms/Filter2d.h"
#include "apps/AppFactory.h"
#include "jlfilter2d_lib.h"

using namespace std;
using namespace app;
using namespace filter;

// shared_ptr<Jim> Jim::filter2dFast(const app::AppFactory& app){
//   try{
//     shared_ptr<Jim> imgWriter=createImg();
//     filter2dFast(*imgWriter, app);
//     return(imgWriter);
//   }
//   catch(string helpString){
//     cerr << helpString << endl;
//     return(0);
//   }
// }

/**
 * @param filter (type: std::string) filter function (nvalid, median, var, min, max, sum, mean, dilate, erode, close, open, homog (central pixel must be identical to all other pixels within window), heterog (central pixel must be different than all other pixels within window), sobelx (horizontal edge detection), sobely (vertical edge detection), sobelxy (diagonal edge detection NE-SW),sobelyx (diagonal edge detection NW-SE), density, countid, mode (majority voting), only for classes), smooth, smoothnodata (smooth nodata values only) values, ismin, ismax, order (rank pixels in order), stdev, mrf, dwt, dwti, scramble, shift, percentile, proportion)
 * @param dx (type: double) (default: 3) filter kernel size in x, use odd values only
 * @param dy (type: double) (default: 3) filter kernel size in y, use odd values only
 * @param nodata (type: double) nodata value(s) (e.g., used for smoothnodata filter)
 * @param resampling-method (type: std::string) (default: near) Resampling method for shifting operation (near: nearest neighbour, bilinear: bi-linear interpolation).
 * @param wavelet (type: std::string) (default: daubechies) wavelet type: daubechies,daubechies_centered, haar, haar_centered, bspline, bspline_centered
 * @param family (type: int) (default: 4) wavelet family (vanishing moment, see also http://www.gnu.org/software/gsl/manual/html_node/DWT-Initialization.html)
 * @param class (type: short) class value(s) to use for density, erosion, dilation, openening and closing, thresholding
 * @param threshold (type: double) (default: 0) threshold value(s) to use for threshold filter (one for each class), or threshold to cut for dwt_cut (use 0 to keep all) or dwt_cut_from, or sigma for shift
 * @param tap (type: std::string) text file containing taps used for spatial filtering (from ul to lr). Use dimX and dimY to specify tap dimensions in x and y. Leave empty for not using taps
 * @param pad (type: std::string) (default: symmetric) Padding method for filtering (how to handle edge effects). Choose between: symmetric, replicate, circular, zero (pad with 0).
 * @param down (type: short) (default: 1) down sampling factor. Use value 1 for no downsampling. Use value n>1 for downsampling (aggregation)
 * @param beta (type: std::string) ASCII file with beta for each class transition in Markov Random Field
 * @param otype (type: std::string) Data type for output image ({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}). Empty string: inherit type from input image
 * @param ct (type: std::string) color table (file with 5 columns: id R G B ALFA (0: transparent, 255: solid). Use none to ommit color table
 * @param circular (type: bool) (default: 0) circular disc kernel for dilation and erosion
 * @return shared pointer to image object
 **/
shared_ptr<Jim> Jim::filter2d(const app::AppFactory& app){
  try{
    shared_ptr<Jim> imgWriter=createImg();
    filter2d(*imgWriter, app);
    return(imgWriter);
  }
  catch(string helpString){
    cerr << helpString << endl;
    throw;
  }
}


/**
 * @param filter (type: std::string) filter function (nvalid, median, var, min, max, sum, mean, dilate, erode, close, open, homog (central pixel must be identical to all other pixels within window), heterog (central pixel must be different than all other pixels within window), sobelx (horizontal edge detection), sobely (vertical edge detection), sobelxy (diagonal edge detection NE-SW),sobelyx (diagonal edge detection NW-SE), density, countid, mode (majority voting), only for classes), smooth, smoothnodata (smooth nodata values only) values, ismin, ismax, order (rank pixels in order), stdev, mrf, dwt, dwti, scramble, shift, percentile, proportion)
 * @param dx (type: double) (default: 3) filter kernel size in x, use odd values only
 * @param dy (type: double) (default: 3) filter kernel size in y, use odd values only
 * @param nodata (type: double) nodata value(s) (e.g., used for smoothnodata filter)
 * @param resampling-method (type: std::string) (default: near) Resampling method for shifting operation (near: nearest neighbour, bilinear: bi-linear interpolation).
 * @param wavelet (type: std::string) (default: daubechies) wavelet type: daubechies,daubechies_centered, haar, haar_centered, bspline, bspline_centered
 * @param family (type: int) (default: 4) wavelet family (vanishing moment, see also http://www.gnu.org/software/gsl/manual/html_node/DWT-Initialization.html)
 * @param class (type: short) class value(s) to use for density, erosion, dilation, openening and closing, thresholding
 * @param threshold (type: double) (default: 0) threshold value(s) to use for threshold filter (one for each class), or threshold to cut for dwt_cut (use 0 to keep all) or dwt_cut_from, or sigma for shift
 * @param tap (type: std::string) text file containing taps used for spatial filtering (from ul to lr). Use dimX and dimY to specify tap dimensions in x and y. Leave empty for not using taps
 * @param pad (type: std::string) (default: symmetric) Padding method for filtering (how to handle edge effects). Choose between: symmetric, replicate, circular, zero (pad with 0).
 * @param down (type: short) (default: 1) down sampling factor. Use value 1 for no downsampling. Use value n>1 for downsampling (aggregation)
 * @param beta (type: std::string) ASCII file with beta for each class transition in Markov Random Field
 * @param otype (type: std::string) Data type for output image ({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}). Empty string: inherit type from input image
 * @param ct (type: std::string) color table (file with 5 columns: id R G B ALFA (0: transparent, 255: solid). Use none to ommit color table
 * @param circular (type: bool) (default: 0) circular disc kernel for dilation and erosion
 **/
void Jim::filter2d(Jim& imgWriter, const app::AppFactory& app){
  Optionjl<bool> disc_opt("circ", "circular", "circular disc kernel for dilation and erosion", false);
  // Optionjl<double> angle_opt("a", "angle", "angle used for directional filtering in dilation (North=0, East=90, South=180, West=270).");
  Optionjl<std::string> method_opt("f", "filter", "filter function (nvalid, median, var, min, max, sum, mean, dilate, erode, close, open, homog (central pixel must be identical to all other pixels within window), heterog (central pixel must be different than all other pixels within window), sobelx (horizontal edge detection), sobely (vertical edge detection), sobelxy (diagonal edge detection NE-SW),sobelyx (diagonal edge detection NW-SE), density, countid, mode (majority voting), only for classes), smooth, smoothnodata (smooth nodata values only) values, ismin, ismax, order (rank pixels in order), stdev, mrf, dwt, dwti, scramble, shift, percentile, proportion)");
  Optionjl<std::string> resample_opt("r", "resampling-method", "Resampling method for shifting operation (near: nearest neighbour, bilinear: bi-linear interpolation).", "near");
  Optionjl<double> dimX_opt("dx", "dx", "filter kernel size in x, use odd values only", 3);
  Optionjl<double> dimY_opt("dy", "dy", "filter kernel size in y, use odd values only", 3);
  Optionjl<std::string> wavelet_type_opt("wt", "wavelet", "wavelet type: daubechies,daubechies_centered, haar, haar_centered, bspline, bspline_centered", "daubechies");
  Optionjl<int> family_opt("wf", "family", "wavelet family (vanishing moment, see also http://www.gnu.org/software/gsl/manual/html_node/DWT-Initialization.html)", 4);
  Optionjl<short> class_opt("class", "class", "class value(s) to use for density, erosion, dilation, openening and closing, thresholding");
  Optionjl<double> threshold_opt("t", "threshold", "threshold value(s) to use for threshold filter (one for each class), or threshold to cut for dwt_cut (use 0 to keep all) or dwt_cut_from, or sigma for shift", 0);
  Optionjl<double> nodata_opt("nodata", "nodata", "nodata value(s) (e.g., used for smoothnodata filter)");
  Optionjl<double> tap_opt("tap", "tap", "taps used for spatial filtering (from ul to lr). Use dx and dy to specify tap dimensions in x and y. Leave empty for not using taps");
  Optionjl<bool> abs_opt("abs", "abs", "use absolute values when filtering",false);
  Optionjl<bool> norm_opt("norm", "norm", "normalize tap values values when filtering",false);
  Optionjl<string> padding_opt("pad","pad", "Padding method for filtering (how to handle edge effects). Choose between: symmetric, replicate, circular, zero (pad with 0).", "symmetric");
  Optionjl<std::string>  otype_opt("ot", "otype", "Data type for output image ({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}). Empty string: inherit type from input image");
  Optionjl<string>  colorTable_opt("ct", "ct", "color table (file with 5 columns: id R G B ALFA (0: transparent, 255: solid). Use none to ommit color table");
  Optionjl<short> down_opt("d", "down", "down sampling factor. Use value 1 for no downsampling. Use value n>1 for downsampling (aggregation)", 1);
  Optionjl<string> beta_opt("beta", "beta", "ASCII file with beta for each class transition in Markov Random Field");
  Optionjl<short> verbose_opt("v", "verbose", "verbose mode if > 0", 0,2);

  resample_opt.setHide(1);
  wavelet_type_opt.setHide(1);
  family_opt.setHide(1);
  class_opt.setHide(1);
  threshold_opt.setHide(1);
  tap_opt.setHide(1);
  abs_opt.setHide(1);
  norm_opt.setHide(1);
  padding_opt.setHide(1);
  down_opt.setHide(1);
  beta_opt.setHide(1);
  otype_opt.setHide(1);
  colorTable_opt.setHide(1);
  disc_opt.setHide(1);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=method_opt.retrieveOption(app);
    // angle_opt.retrieveOption(app);
    dimX_opt.retrieveOption(app);
    dimY_opt.retrieveOption(app);
    nodata_opt.retrieveOption(app);
    resample_opt.retrieveOption(app);
    wavelet_type_opt.retrieveOption(app);
    family_opt.retrieveOption(app);
    class_opt.retrieveOption(app);
    threshold_opt.retrieveOption(app);
    tap_opt.retrieveOption(app);
    abs_opt.retrieveOption(app);
    norm_opt.retrieveOption(app);
    padding_opt.retrieveOption(app);
    down_opt.retrieveOption(app);
    beta_opt.retrieveOption(app);
    otype_opt.retrieveOption(app);
    colorTable_opt.retrieveOption(app);
    disc_opt.retrieveOption(app);
    verbose_opt.retrieveOption(app);
    if(!doProcess){
      cout << endl;
      std::ostringstream helpStream;
      helpStream << "short option -h shows basic options only, use long option --help to show all options" << std::endl;
      throw(helpStream.str());//help was invoked, stop processing
    }

    //not implemented yet, must debug first...
    vector<double> angle_opt;

    GDALDataType theType=GDT_Unknown;
    if(otype_opt.size()){
      theType=string2GDAL(otype_opt[0]);
      if(theType==GDT_Unknown)
        std::cout << "Warning: unknown output pixel type: " << otype_opt[0] << ", using input type as default" << std::endl;
    }

    if(theType==GDT_Unknown)
      theType=this->getGDALDataType();

    if(verbose_opt[0])
      std::cout << "Output pixel type:  " << GDALGetDataTypeName(theType) << endl;

    string errorString;
    unsigned int nband=this->nrOfBand();

    if(tap_opt.size())
      nband=this->nrOfBand();
    else{
      if(method_opt.empty()){
        errorString="Error: no filter selected, use option -f";
        throw(errorString);
      }
      else if(verbose_opt[0])
        std::cout << "filter method: " << method_opt[0] << "=" << filter2d::Filter2d::getFilterType(method_opt[0]) << std::endl;
      switch(filter2d::Filter2d::getFilterType(method_opt[0])){
      case(filter2d::dilate):
      case(filter2d::erode):
      case(filter2d::close):
      case(filter2d::open):
      case(filter2d::smooth):
        //implemented in spectral/temporal domain (dimZ>1) and spatial domain
        nband=this->nrOfBand();
      break;
      case(filter2d::dwt):
      case(filter2d::dwti):
      case(filter2d::dwt_cut):
      case(filter2d::smoothnodata):
        //implemented in spectral/temporal/spatial domain and nband always this->nrOfBand()
        nband=this->nrOfBand();
      break;
      case(filter2d::mrf)://deliberate fall through
        assert(class_opt.size()>1);
        if(verbose_opt[0])
          std::cout << "opening output image" << std::endl;
        nband=class_opt.size();
      case(filter2d::ismin):
      case(filter2d::ismax):
      case(filter2d::shift):
      case(filter2d::scramble):
      case(filter2d::mode):
      case(filter2d::sobelx):
      case(filter2d::sobely):
      case(filter2d::sobelxy):
      case(filter2d::countid):
      case(filter2d::order):
      case(filter2d::density):
      case(filter2d::homog):
      case(filter2d::heterog):
      case(filter2d::sauvola):
        //only implemented in spatial domain
      break;
      case(filter2d::sum):
      case(filter2d::mean):
      case(filter2d::min):
      case(filter2d::max):
      case(filter2d::var):
      case(filter2d::stdev):
      case(filter2d::nvalid):
      case(filter2d::median):
      case(filter2d::percentile):
      case(filter2d::proportion):
        //implemented in spectral/temporal/spatial domain
        nband=this->nrOfBand();
      break;
      default:{
        cout << endl;
        std::ostringstream errorStream;
        errorStream << "filter method: " << method_opt[0] << "=" << filter2d::Filter2d::getFilterType(method_opt[0]) << " not implemented"<< std::endl;
        // errorStream << "filter " << method_opt[0] << " (" << )"<< " not implemented";
        throw(errorStream.str());
        break;
      }
      }
    }
    imgWriter.open((this->nrOfCol()+down_opt[0]-1)/down_opt[0],(this->nrOfRow()+down_opt[0]-1)/down_opt[0],nband,theType);
    imgWriter.setProjection(this->getProjection());
    double gt[6];
    this->getGeoTransform(gt);
    gt[1]*=down_opt[0];//dx
    gt[5]*=down_opt[0];//dy
    imgWriter.setGeoTransform(gt);

    if(colorTable_opt.size()){
      if(colorTable_opt[0]!="none"){
        if(verbose_opt[0])
          cout << "set colortable " << colorTable_opt[0] << endl;
        assert(imgWriter.getDataType()==GDT_Byte);
        imgWriter.setColorTable(colorTable_opt[0]);
      }
    }
    else if(this->getColorTable()!=NULL)
      imgWriter.setColorTable(this->getColorTable());

    if(nodata_opt.size()){
      for(unsigned int iband=0;iband<imgWriter.nrOfBand();++iband)
        imgWriter.GDALSetNoDataValue(nodata_opt[0],iband);
    }

    filter2d::Filter2d filter2d;
    if(verbose_opt[0])
      cout << "Set padding to " << padding_opt[0] << endl;
    if(class_opt.size()){
      if(verbose_opt[0])
        std::cout<< "class values: ";
      for(int iclass=0;iclass<class_opt.size();++iclass){
        filter2d.pushClass(class_opt[iclass]);
        if(verbose_opt[0])
          std::cout<< class_opt[iclass] << " ";
      }
      if(verbose_opt[0])
        std::cout<< std::endl;
    }

    if(nodata_opt.size()){
      if(verbose_opt[0])
        std::cout<< "mask values: ";
      for(unsigned int imask=0;imask<nodata_opt.size();++imask){
        if(verbose_opt[0])
          std::cout<< nodata_opt[imask] << " ";
        filter2d.pushNoDataValue(nodata_opt[imask]);
      }
      if(verbose_opt[0])
        std::cout<< std::endl;
    }
    filter2d.setThresholds(threshold_opt);

    if(tap_opt.size()){
      // ifstream tapfile(tap_opt[0].c_str());
      // assert(tapfile);
      Vector2d<double> taps(dimY_opt[0],dimX_opt[0]);
      if(tap_opt.size()!=dimX_opt[0]*dimY_opt[0]){
        std::string errorString="Error: wrong dimensions of taps";
        throw(errorString);
      }

      for(unsigned int j=0;j<dimY_opt[0];++j){
        for(unsigned int i=0;i<dimX_opt[0];++i){
          taps[j][i]=tap_opt[j*dimX_opt[0]+i];
          // tapfile >> taps[j][i];
        }
      }
      if(verbose_opt[0]){
        std::cout << "taps: ";
        for(unsigned int j=0;j<dimY_opt[0];++j){
          for(unsigned int i=0;i<dimX_opt[0];++i){
            std::cout<< taps[j][i] << " ";
          }
          std::cout<< std::endl;
        }
      }
      filter2d.setTaps(taps);
      filter2d.filter(*this,imgWriter,abs_opt[0],norm_opt[0]);
      // filter2d.filterLB(*this,imgWriter,abs_opt[0],norm_opt[0]);
      // tapfile.close();
    }
    else{
      switch(filter2d::Filter2d::getFilterType(method_opt[0])){
      case(filter2d::dilate):
        if(down_opt[0]!=1){
          std::cerr << "Error: down option not supported for morphological operator" << std::endl;
          exit(1);
        }
        filter2d.morphology(*this,imgWriter,"dilate",dimX_opt[0],dimY_opt[0],angle_opt,disc_opt[0]);
        break;
      case(filter2d::erode):
        if(down_opt[0]!=1){
          std::cerr << "Error: down option not supported for morphological operator" << std::endl;
          exit(1);
        }
        filter2d.morphology(*this,imgWriter,"erode",dimX_opt[0],dimY_opt[0],angle_opt,disc_opt[0]);
        break;
      case(filter2d::close):{//closing
        if(down_opt[0]!=1){
          std::cerr << "Error: down option not supported for morphological operator" << std::endl;
          exit(1);
        }
        filter2d.morphology(*this,imgWriter,"dilate",dimX_opt[0],dimY_opt[0],angle_opt,disc_opt[0]);
        filter2d.morphology(imgWriter,imgWriter,"erode",dimX_opt[0],dimY_opt[0],angle_opt,disc_opt[0]);
        break;
      }
      case(filter2d::open):{//opening
        if(down_opt[0]!=1){
          std::cerr << "Error: down option not supported for morphological operator" << std::endl;
          exit(1);
        }
        filter2d.morphology(*this,imgWriter,"erode",dimX_opt[0],dimY_opt[0],angle_opt,disc_opt[0]);
        filter2d.morphology(imgWriter,imgWriter,"dilate",dimX_opt[0],dimY_opt[0],angle_opt,disc_opt[0]);
        break;
      }
      case(filter2d::homog):{//spatially homogeneous
        filter2d.doit(*this,imgWriter,"homog",dimX_opt[0],dimY_opt[0],down_opt[0],disc_opt[0]);
        break;
      }
      case(filter2d::heterog):{//spatially heterogeneous
        filter2d.doit(*this,imgWriter,"heterog",dimX_opt[0],dimY_opt[0],down_opt[0],disc_opt[0]);
        break;
      }
      case(filter2d::sauvola):{//Implements Sauvola's thresholding method (http://fiji.sc/Auto_Local_Threshold)
        //test
        Vector2d<unsigned short> inBuffer;
        for(unsigned int iband=0;iband<this->nrOfBand();++iband){
          this->readDataBlock(inBuffer,0,this->nrOfCol()-1,0,this->nrOfRow()-1,iband);
        }
        filter2d.doit(*this,imgWriter,"sauvola",dimX_opt[0],dimY_opt[0],down_opt[0],disc_opt[0]);
        break;
      }
      case(filter2d::shift):{//shift
        if(down_opt[0]!=1){
          std::cerr << "Error: down option not supported for shift operator" << std::endl;
          exit(1);
        }
        assert(this->nrOfBand());
        assert(this->nrOfCol());
        assert(this->nrOfRow());
        filter2d.shift(*this,imgWriter,dimX_opt[0],dimY_opt[0],threshold_opt[0],filter2d::Filter2d::getResampleType(resample_opt[0]));
        break;
      }
      case(filter2d::mrf):{//Markov Random Field
        if(verbose_opt[0])
          std::cout << "Markov Random Field filtering" << std::endl;
        if(beta_opt.size()){
          //in file: classFrom classTo
          //in variable: beta[classTo][classFrom]
          FileReaderAscii betaReader(beta_opt[0]);
          Vector2d<double> beta(class_opt.size(),class_opt.size());
          vector<int> cols(class_opt.size());
          for(int iclass=0;iclass<class_opt.size();++iclass)
            cols[iclass]=iclass;
          betaReader.readData(beta,cols);
          if(verbose_opt[0]){
            std::cout << "using values for beta:" << std::endl;
            for(int iclass1=0;iclass1<class_opt.size();++iclass1)
              std::cout << "      " << iclass1 << " (" << class_opt[iclass1] << ")";
            std::cout << std::endl;
            for(int iclass1=0;iclass1<class_opt.size();++iclass1){
              std::cout << iclass1 << " (" << class_opt[iclass1] << ")";
              for(int iclass2=0;iclass2<class_opt.size();++iclass2)
                std::cout << " " << beta[iclass2][iclass1] << " (" << class_opt[iclass2] << ")";
              std::cout << std::endl;
            }
          }
          filter2d.mrf(*this, imgWriter, dimX_opt[0], dimY_opt[0], beta, true, down_opt[0], verbose_opt[0]);
        }
        else
          filter2d.mrf(*this, imgWriter, dimX_opt[0], dimY_opt[0], 1, true, down_opt[0], verbose_opt[0]);
        break;
      }
      case(filter2d::sobelx):{//Sobel edge detection in X
        if(down_opt[0]!=1){
          std::cerr << "Error: down option not supported for sobel edge detection" << std::endl;
          exit(1);
        }
        Vector2d<double> theTaps(3,3);
        theTaps[0][0]=-1.0;
        theTaps[0][1]=0.0;
        theTaps[0][2]=1.0;
        theTaps[1][0]=-2.0;
        theTaps[1][1]=0.0;
        theTaps[1][2]=2.0;
        theTaps[2][0]=-1.0;
        theTaps[2][1]=0.0;
        theTaps[2][2]=1.0;
        filter2d.setTaps(theTaps);
        filter2d.filter(*this,imgWriter,true,true);//absolute and normalize
        break;
      }
      case(filter2d::sobely):{//Sobel edge detection in Y
        if(down_opt[0]!=1){
          std::cerr << "Error: down option not supported for sobel edge detection" << std::endl;
          exit(1);
        }
        Vector2d<double> theTaps(3,3);
        theTaps[0][0]=1.0;
        theTaps[0][1]=2.0;
        theTaps[0][2]=1.0;
        theTaps[1][0]=0.0;
        theTaps[1][1]=0.0;
        theTaps[1][2]=0.0;
        theTaps[2][0]=-1.0;
        theTaps[2][1]=-2.0;
        theTaps[2][2]=-1.0;
        filter2d.setTaps(theTaps);
        filter2d.filter(*this,imgWriter,true,true);//absolute and normalize
        break;
      }
      case(filter2d::sobelxy):{//Sobel edge detection in XY
        if(down_opt[0]!=1){
          std::cerr << "Error: down option not supported for sobel edge detection" << std::endl;
          exit(1);
        }
        Vector2d<double> theTaps(3,3);
        theTaps[0][0]=0.0;
        theTaps[0][1]=1.0;
        theTaps[0][2]=2.0;
        theTaps[1][0]=-1.0;
        theTaps[1][1]=0.0;
        theTaps[1][2]=1.0;
        theTaps[2][0]=-2.0;
        theTaps[2][1]=-1.0;
        theTaps[2][2]=0.0;
        filter2d.setTaps(theTaps);
        filter2d.filter(*this,imgWriter,true,true);//absolute and normalize
        break;
      }
      case(filter2d::sobelyx):{//Sobel edge detection in XY
        if(down_opt[0]!=1){
          std::cerr << "Error: down option not supported for sobel edge detection" << std::endl;
          exit(1);
        }
        Vector2d<double> theTaps(3,3);
        theTaps[0][0]=2.0;
        theTaps[0][1]=1.0;
        theTaps[0][2]=0.0;
        theTaps[1][0]=1.0;
        theTaps[1][1]=0.0;
        theTaps[1][2]=-1.0;
        theTaps[2][0]=0.0;
        theTaps[2][1]=-1.0;
        theTaps[2][2]=-2.0;
        filter2d.setTaps(theTaps);
        filter2d.filter(*this,imgWriter,true,true);//absolute and normalize
        break;
      }
      case(filter2d::smooth):{//Smoothing filter
        if(down_opt[0]!=1){
          std::cerr << "Error: down option not supported for this filter" << std::endl;
          exit(1);
        }
        filter2d.smooth(*this,imgWriter,dimX_opt[0],dimY_opt[0]);
        break;
      }
      case(filter2d::smoothnodata):{//Smoothing filter
        if(down_opt[0]!=1){
          std::cerr << "Error: down option not supported for this filter" << std::endl;
          exit(1);
        }
        if(verbose_opt[0])
          std::cout<< "2-D filtering: smooth" << std::endl;
        filter2d.smoothNoData(*this,imgWriter,dimX_opt[0],dimY_opt[0]);
        break;
      }
      case(filter2d::dwt):
        if(down_opt[0]!=1){
          std::cerr << "Error: down option not supported for this filter" << std::endl;
          exit(1);
        }
        filter2d.dwtForward(*this, imgWriter, wavelet_type_opt[0], family_opt[0]);
        break;
      case(filter2d::dwti):
        if(down_opt[0]!=1){
          std::cerr << "Error: down option not supported for this filter" << std::endl;
          exit(1);
        }
        filter2d.dwtInverse(*this, imgWriter, wavelet_type_opt[0], family_opt[0]);
        break;
      case(filter2d::dwt_cut):
        if(down_opt[0]!=1){
          std::cerr << "Error: down option not supported for this filter" << std::endl;
          exit(1);
        }
        filter2d.dwtCut(*this, imgWriter, wavelet_type_opt[0], family_opt[0], threshold_opt[0]);
        break;
      case(filter2d::percentile)://deliberate fall through
      case(filter2d::threshold)://deliberate fall through
        assert(threshold_opt.size());
      filter2d.setThresholds(threshold_opt);
      case(filter2d::density)://deliberate fall through
        filter2d.setClasses(class_opt);
        if(verbose_opt[0])
          std::cout << "classes set" << std::endl;
      default:
        filter2d.doit(*this,imgWriter,method_opt[0],dimX_opt[0],dimY_opt[0],down_opt[0],disc_opt[0]);
        break;
      }
    }
  }
  catch(string predefinedString){
    std::cout << predefinedString << std::endl;
    throw;
  }
}

std::shared_ptr<Jim> Jim::firfilter2d(app::AppFactory& app){
  try{
    shared_ptr<Jim> imgWriter=createImg();
    firfilter2d(*imgWriter, app);
    return(imgWriter);
  }
  catch(std::string helpString){
    cerr << helpString << endl;
    throw;
  }
}

void Jim::firfilter2d(Jim& imgWriter, app::AppFactory& app){
  imgWriter.open(nrOfCol(),nrOfRow(),nrOfBand(),nrOfPlane(),getGDALDataType());
  imgWriter.setProjection(getProjection());
  double gt[6];
  this->getGeoTransform(gt);
  imgWriter.setGeoTransform(gt);
  switch(getDataType()){
  case(GDT_Byte):
    firfilter2d_t<unsigned char>(imgWriter,app);
    break;
  case(GDT_Int16):
    firfilter2d_t<short>(imgWriter,app);
    break;
  case(GDT_UInt16):
    firfilter2d_t<unsigned short>(imgWriter,app);
    break;
  case(GDT_Int32):
    firfilter2d_t<int>(imgWriter,app);
    break;
  case(GDT_UInt32):
    firfilter2d_t<unsigned int>(imgWriter,app);
    break;
  case(GDT_Float32):
    firfilter2d_t<float>(imgWriter,app);
    break;
  case(GDT_Float64):
    firfilter2d_t<double>(imgWriter,app);
    break;
  default:
    std::string errorString="Error: data type not supported";
    throw(errorString);
    break;
  }
}

shared_ptr<Jim> Jim::dwt2d(app::AppFactory& app){
  try{
    shared_ptr<Jim> imgWriter=clone();
    imgWriter->d_dwt2d(app);
    return(imgWriter);
  }
  catch(std::string helpString){
    cerr << helpString << endl;
    throw;
  }
}

void Jim::d_dwt2d(app::AppFactory& app){
  Optionjl<std::string> wavelet_type_opt("wt", "wavelet", "wavelet type: daubechies,daubechies_centered, haar, haar_centered, bspline, bspline_centered", "daubechies");
  Optionjl<int> family_opt("wf", "family", "wavelet family (vanishing moment, see also https://www.gnu.org/software/gsl/doc/html/dwt.html)", 4);
  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=wavelet_type_opt.retrieveOption(app);
    family_opt.retrieveOption(app);
    if(!doProcess){
      cout << endl;
      std::ostringstream helpStream;
      helpStream << "short option -h shows basic options only, use long option --help to show all options" << std::endl;
      throw(helpStream.str());//help was invoked, stop processing
    }

    if(nrOfPlane()!=1){
      std::ostringstream errorStream;
      errorStream << "Error: multi-plane images are not supported" << std::endl;
      throw(errorStream.str());
    }
    if(getDataType()!=GDT_Float64){
      std::ostringstream errorStream;
      errorStream << "Error: data type must be GDT_Float64" << std::endl;
      throw(errorStream.str());
    }
    size_t oldrow=nrOfRow();
    size_t oldcol=nrOfCol();
    size_t newrow=nrOfRow();
    size_t newcol=nrOfCol();
    while(newrow&(newrow-1))
      ++newrow;
    while(newcol&(newcol-1))
      ++newcol;
    std::vector<int> box{0,static_cast<int>(newcol-oldcol),0,static_cast<int>(newrow-oldrow),0,0};
#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
    for(size_t iband=0;iband<nrOfBand();++iband){
      //make sure data size if power of 2
      //dangerous, we are not adapting the ncol and nrow, do not use nrOfCol() nor nrOfRow() within this block (before d_imageFrameSubtract)
      d_imageFrameAdd(box.data(),0.0);
      int nsize=newrow*newcol;
      gsl_wavelet *w;
      gsl_wavelet_workspace *work;
      w=gsl_wavelet_alloc(filter::Filter::getWaveletType(wavelet_type_opt[0]),family_opt[0]);
      work=gsl_wavelet_workspace_alloc(nsize);
      double* pin=static_cast<double*>(getDataPointer(iband));
      gsl_wavelet2d_nstransform_forward (w, pin, newrow, newrow, newcol, work);
      gsl_wavelet_free (w);
      gsl_wavelet_workspace_free (work);
      d_imageFrameSubtract(box.data());
    }
  }
  catch(std::string predefinedString ){
    std::cout << predefinedString << std::endl;
    throw;
  }
}

shared_ptr<Jim> Jim::dwti2d(app::AppFactory& app){
  try{
    shared_ptr<Jim> imgWriter=clone();
    imgWriter->d_dwti2d(app);
    return(imgWriter);
  }
  catch(std::string helpString){
    cerr << helpString << endl;
    throw;
  }
}

void Jim::d_dwti2d(app::AppFactory& app){
  Optionjl<std::string> wavelet_type_opt("wt", "wavelet", "wavelet type: daubechies,daubechies_centered, haar, haar_centered, bspline, bspline_centered", "daubechies");
  Optionjl<int> family_opt("wf", "family", "wavelet family (vanishing moment, see also https://www.gnu.org/software/gsl/doc/html/dwt.html)", 4);
  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=wavelet_type_opt.retrieveOption(app);
    family_opt.retrieveOption(app);
    if(!doProcess){
      cout << endl;
      std::ostringstream helpStream;
      helpStream << "short option -h shows basic options only, use long option --help to show all options" << std::endl;
      throw(helpStream.str());//help was invoked, stop processing
    }

    if(nrOfPlane()!=1){
      std::ostringstream errorStream;
      errorStream << "Error: multi-plane images are not supported" << std::endl;
      throw(errorStream.str());
    }
    if(getDataType()!=GDT_Float64){
      std::ostringstream errorStream;
      errorStream << "Error: data type must be GDT_Float64" << std::endl;
      throw(errorStream.str());
    }
    size_t oldrow=nrOfRow();
    size_t oldcol=nrOfCol();
    size_t newrow=nrOfRow();
    size_t newcol=nrOfCol();
    while(newrow&(newrow-1))
      ++newrow;
    while(newcol&(newcol-1))
      ++newcol;
    std::vector<int> box{0,static_cast<int>(newcol-oldcol),0,static_cast<int>(newrow-oldrow),0,0};
#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
    for(size_t iband=0;iband<nrOfBand();++iband){
      //make sure data size if power of 2
      //dangerous, we are not adapting the ncol and nrow, do not use nrOfCol() nor nrOfRow() within this block (before d_imageFrameSubtract)
      d_imageFrameAdd(box.data(),0.0);
      int nsize=newrow*newcol;
      gsl_wavelet *w;
      gsl_wavelet_workspace *work;
      w=gsl_wavelet_alloc(filter::Filter::getWaveletType(wavelet_type_opt[0]),family_opt[0]);
      work=gsl_wavelet_workspace_alloc(nsize);
      double* pin=static_cast<double*>(getDataPointer(iband));
      gsl_wavelet2d_nstransform_inverse (w, pin, newrow, newrow, newcol, work);
      gsl_wavelet_free (w);
      gsl_wavelet_workspace_free (work);
      d_imageFrameSubtract(box.data());
    }
  }
  catch(std::string predefinedString ){
    std::cout << predefinedString << std::endl;
    throw;
  }
}
