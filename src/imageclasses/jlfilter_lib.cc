/**********************************************************************
jlfilter_lib.cc: program to filter raster images: median, min/max, morphological, filtering
Author(s): Pieter.Kempeneers@ec.europa.eu
Copyright (c) 2016-2018 European Union (Joint Research Centre)
License EUPLv1.2

This file is part of jiplib
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

using namespace std;
using namespace app;
using namespace filter;

/**
 * @param app application specific option arguments
 * @return output filtered raster dataset
 **/
shared_ptr<Jim> Jim::filter2dFast(const app::AppFactory& app){
  try{
    shared_ptr<Jim> imgWriter=createImg();
    filter2dFast(*imgWriter, app);
    return(imgWriter);
  }
  catch(string helpString){
    cerr << helpString << endl;
    throw;
  }
}

/**
 * @param app application specific option arguments
 * @return output filtered raster dataset
 **/
shared_ptr<Jim> Jim::filter(app::AppFactory& app){
  try{
    shared_ptr<Jim> imgWriter=createImg();
    filter(*imgWriter, app);
    return(imgWriter);
  }
  catch(string helpString){
    cerr << helpString << endl;
    throw;
  }
}

/**
 * @param imgWriter output filtered raster dataset
 * @param app application specific option arguments
 **/
void Jim::filter(Jim& imgWriter, app::AppFactory& app){
  Optionjl<bool> disc_opt("circ", "circular", "circular disc kernel for dilation and erosion", false);
  // Optionjl<double> angle_opt("a", "angle", "angle used for directional filtering in dilation (North=0, East=90, South=180, West=270).");
  Optionjl<std::string> method_opt("f", "filter", "filter function (nvalid, median, var, min, max, sum, mean, dilate, erode, close, open, homog (central pixel must be identical to all other pixels within window), heterog (central pixel must be different than all other pixels within window), sauvola, sobelx (horizontal edge detection), sobely (vertical edge detection), sobelxy (diagonal edge detection NE-SW),sobelyx (diagonal edge detection NW-SE), density, countid, mode (majority voting), only for classes), smooth, smoothnodata (smooth nodata values only) values, ismin, ismax, order (rank pixels in order), stdev, mrf, dwt, dwti, dwt_cut, dwt_cut_from, scramble, shift, savgolay, percentile, proportion)");
  Optionjl<std::string> resample_opt("r", "resampling-method", "Resampling method for shifting operation (near: nearest neighbour, bilinear: bi-linear interpolation).", "near");
  Optionjl<double> dimX_opt("dx", "dx", "filter kernel size in x, use odd values only", 3);
  Optionjl<double> dimY_opt("dy", "dy", "filter kernel size in y, use odd values only", 3);
  Optionjl<int> dimZ_opt("dz", "dz", "filter kernel size in z (spectral/temporal dimension), must be odd (example: 3).. Set dz>0 if 1-D filter must be used in band domain");
  Optionjl<std::string> wavelet_type_opt("wt", "wavelet", "wavelet type: daubechies,daubechies_centered, haar, haar_centered, bspline, bspline_centered", "daubechies");
  Optionjl<int> family_opt("wf", "family", "wavelet family (vanishing moment, see also http://www.gnu.org/software/gsl/manual/html_node/DWT-Initialization.html)", 4);
  Optionjl<int> savgolay_nl_opt("nl", "nl", "Number of leftward (past) data points used in Savitzky-Golay filter)", 2);
  Optionjl<int> savgolay_nr_opt("nr", "nr", "Number of rightward (future) data points used in Savitzky-Golay filter)", 2);
  Optionjl<int> savgolay_ld_opt("ld", "ld", "order of the derivative desired in Savitzky-Golay filter (e.g., ld=0 for smoothed function)", 0);
  Optionjl<int> savgolay_m_opt("m", "m", "order of the smoothing polynomial in Savitzky-Golay filter, also equal to the highest conserved moment; usual values are m = 2 or m = 4)", 2);
  Optionjl<short> class_opt("class", "class", "class value(s) to use for density, erosion, dilation, openening and closing, thresholding");
  Optionjl<double> threshold_opt("t", "threshold", "threshold value(s) to use for threshold filter (one for each class), or threshold to cut for dwt_cut (use 0 to keep all) or dwt_cut_from, or sigma for shift", 0);
  Optionjl<double> nodata_opt("nodata", "nodata", "nodata value(s) (e.g., used for smoothnodata filter)");
  Optionjl<std::string> tap_opt("tap", "tap", "text file containing taps used for spatial filtering (from ul to lr). Use dimX and dimY to specify tap dimensions in x and y. Leave empty for not using taps");
  Optionjl<double> tapz_opt("tapz", "tapz", "taps used for spectral filtering");
  Optionjl<string> padding_opt("pad","pad", "Padding method for filtering (how to handle edge effects). Choose between: symmetric, replicate, circular, zero (pad with 0).", "symmetric");
  Optionjl<double> fwhm_opt("fwhm", "fwhm", "list of full width half to apply spectral filtering (-fwhm band1 -fwhm band2 ...)");
  Optionjl<std::string> srf_opt("srf", "srf", "list of ASCII files containing spectral response functions (two columns: wavelength response)");
  Optionjl<double> wavelengthIn_opt("win", "wavelengthIn", "list of wavelengths in input spectrum (-win band1 -win band2 ...)");
  Optionjl<double> wavelengthOut_opt("wout", "wavelengthOut", "list of wavelengths in output spectrum (-wout band1 -wout band2 ...)");
  Optionjl<std::string> interpolationType_opt("interp", "interp", "type of interpolation for spectral filtering (see http://www.gnu.org/software/gsl/manual/html_node/Interpolation-Types.html)","akima");
  Optionjl<std::string>  otype_opt("ot", "otype", "Data type for output image ({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}). Empty string: inherit type from input image");
  Optionjl<string>  colorTable_opt("ct", "ct", "color table (file with 5 columns: id R G B ALFA (0: transparent, 255: solid). Use none to ommit color table");
  Optionjl<short> down_opt("d", "down", "down sampling factor. Use value 1 for no downsampling. Use value n>1 for downsampling (aggregation)", 1);
  Optionjl<string> beta_opt("beta", "beta", "ASCII file with beta for each class transition in Markov Random Field");
  // Optionjl<double> eps_opt("eps","eps", "error marging for linear feature",0);
  // Optionjl<bool> l1_opt("l1","l1", "obtain longest object length for linear feature",false);
  // Optionjl<bool> l2_opt("l2","l2", "obtain shortest object length for linear feature",false,2);
  // Optionjl<bool> a1_opt("a1","a1", "obtain angle found for longest object length for linear feature",false);
  // Optionjl<bool> a2_opt("a2","a2", "obtain angle found for shortest object length for linear feature",false);
  Optionjl<short> verbose_opt("v", "verbose", "verbose mode if > 0", 0,2);

  resample_opt.setHide(1);
  wavelet_type_opt.setHide(1);
  family_opt.setHide(1);
  savgolay_nl_opt.setHide(1);
  savgolay_nr_opt.setHide(1);
  savgolay_ld_opt.setHide(1);
  savgolay_m_opt.setHide(1);
  class_opt.setHide(1);
  threshold_opt.setHide(1);
  tap_opt.setHide(1);
  tapz_opt.setHide(1);
  padding_opt.setHide(1);
  wavelengthIn_opt.setHide(1);
  wavelengthOut_opt.setHide(1);
  down_opt.setHide(1);
  beta_opt.setHide(1);
  // eps_opt.setHide(1);
  // l1_opt.setHide(1);
  // l2_opt.setHide(1);
  // a1_opt.setHide(1);
  // a2_opt.setHide(1);
  interpolationType_opt.setHide(1);
  otype_opt.setHide(1);
  colorTable_opt.setHide(1);
  disc_opt.setHide(1);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=method_opt.retrieveOption(app);
    // angle_opt.retrieveOption(app);
    srf_opt.retrieveOption(app);
    fwhm_opt.retrieveOption(app);
    dimX_opt.retrieveOption(app);
    dimY_opt.retrieveOption(app);
    dimZ_opt.retrieveOption(app);
    nodata_opt.retrieveOption(app);
    resample_opt.retrieveOption(app);
    wavelet_type_opt.retrieveOption(app);
    family_opt.retrieveOption(app);
    savgolay_nl_opt.retrieveOption(app);
    savgolay_nr_opt.retrieveOption(app);
    savgolay_ld_opt.retrieveOption(app);
    savgolay_m_opt.retrieveOption(app);
    class_opt.retrieveOption(app);
    threshold_opt.retrieveOption(app);
    tap_opt.retrieveOption(app);
    tapz_opt.retrieveOption(app);
    padding_opt.retrieveOption(app);
    wavelengthIn_opt.retrieveOption(app);
    wavelengthOut_opt.retrieveOption(app);
    down_opt.retrieveOption(app);
    beta_opt.retrieveOption(app);
    // eps_opt.retrieveOption(app);
    // l1_opt.retrieveOption(app);
    // l2_opt.retrieveOption(app);
    // a1_opt.retrieveOption(app);
    // a2_opt.retrieveOption(app);
    interpolationType_opt.retrieveOption(app);
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

    if(fwhm_opt.size())
      nband=fwhm_opt.size();
    else if(srf_opt.size())
      nband=srf_opt.size();
    else if(tap_opt.size()||tapz_opt.size())
      nband=this->nrOfBand();
    else{
      if(method_opt.empty()){
        errorString="Error: no filter selected, use option -f";
        throw(errorString);
      }
      else if(verbose_opt[0])
        std::cout << "filter method: " << method_opt[0] << "=" << filter2d::Filter2d::getFilterType(method_opt[0]) << std::endl;
      // std::cout << "filter method: "<< filter2d::Filter2d::getFilterType(method_opt[0]) << std::endl;
      switch(filter2d::Filter2d::getFilterType(method_opt[0])){
      case(filter2d::dilate):
      case(filter2d::erode):
      case(filter2d::close):
      case(filter2d::open):
      case(filter2d::smooth):
        //implemented in spectral/temporal domain (dimZ>1) and spatial domain
        if(dimZ_opt.size())
          assert(dimZ_opt[0]>1);
      nband=this->nrOfBand();
      break;
      case(filter2d::dwt):
      case(filter2d::dwti):
      case(filter2d::dwt_cut):
      case(filter2d::smoothnodata):
        //implemented in spectral/temporal/spatial domain and nband always this->nrOfBand()
        nband=this->nrOfBand();
      break;
      case(filter2d::savgolay):
        nband=this->nrOfBand();
        if(dimZ_opt.empty())
          dimZ_opt.push_back(1);
      case(filter2d::dwt_cut_from):
        //only implemented in spectral/temporal domain
        if(dimZ_opt.size()){
          nband=this->nrOfBand();
          assert(threshold_opt.size());
        }
        else{
          errorString="filter not implemented in spatial domain";
          throw(errorString);
        }
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
        if(dimZ_opt.size()){
          errorString="filter not implemented in spectral/temporal domain";
          throw(errorString);
        }
      break;
      // case(filter2d::percentile):
      //  //implemented in spectral/temporal/spatial domain and nband 1 if dimZ>0
      //  if(dimZ_opt.size()){
      //    dimZ_opt[0]=1;
      //    nband=1;
      //  }
      //  else
      //    nband=this->nrOfBand();
      //  break;
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
        //implemented in spectral/temporal/spatial domain and nband 1 if dimZ==1
        if(dimZ_opt.size()==1)
          if(dimZ_opt[0]==1)
            nband=1;
          else
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
    // imgWriter.open(output_opt[0],(this->nrOfCol()+down_opt[0]-1)/down_opt[0],(this->nrOfRow()+down_opt[0]-1)/down_opt[0],nband,theType,imageType,memory_opt[0],option_opt);
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
    filter::Filter filter1d;
    if(verbose_opt[0])
      cout << "Set padding to " << padding_opt[0] << endl;
    filter1d.setPadding(padding_opt[0]);
    if(class_opt.size()){
      if(verbose_opt[0])
        std::cout<< "class values: ";
      for(int iclass=0;iclass<class_opt.size();++iclass){
        if(!dimZ_opt.size())
          filter2d.pushClass(class_opt[iclass]);
        else
          filter1d.pushClass(class_opt[iclass]);
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
        filter1d.pushNoDataValue(nodata_opt[imask]);
        filter2d.pushNoDataValue(nodata_opt[imask]);
      }
      if(verbose_opt[0])
        std::cout<< std::endl;
    }
    filter1d.setThresholds(threshold_opt);
    filter2d.setThresholds(threshold_opt);

    if(tap_opt.size()){
      ifstream tapfile(tap_opt[0].c_str());
      assert(tapfile);
      Vector2d<double> taps(dimY_opt[0],dimX_opt[0]);

      for(unsigned int j=0;j<dimY_opt[0];++j){
        for(unsigned int i=0;i<dimX_opt[0];++i){
          tapfile >> taps[j][i];
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
      //todo: this->filter2D.filter(imgWriter);
      filter2d.filter(*this,imgWriter);
      // filter2d.filter(input,output);
      tapfile.close();
    }
    else if(tapz_opt.size()){
      if(verbose_opt[0]){
        std::cout << "taps: ";
        for(int itap=0;itap<tapz_opt.size();++itap)
          std::cout<< tapz_opt[itap] << " ";
        std::cout<< std::endl;
      }
      filter1d.setTaps(tapz_opt);
      //todo:this->filter3D.filter(imgWriter);
      filter1d.filter(*this,imgWriter);
      // filter1d.filter(input,output);
    }
    else if(fwhm_opt.size()){
      if(verbose_opt[0])
        std::cout << "spectral filtering to " << fwhm_opt.size() << " bands with provided fwhm " << std::endl;
      assert(wavelengthOut_opt.size()==fwhm_opt.size());
      assert(wavelengthIn_opt.size());

      Vector2d<double> lineInput(this->nrOfBand(),this->nrOfCol());
      Vector2d<double> lineOutput(wavelengthOut_opt.size(),this->nrOfCol());
      const char* pszMessage;
      void* pProgressArg=NULL;
      GDALProgressFunc pfnProgress=GDALTermProgress;
      double progress=0;
      pfnProgress(progress,pszMessage,pProgressArg);
      for(unsigned int y=0;y<this->nrOfRow();++y){
        if((y+1+down_opt[0]/2)%down_opt[0])
          continue;
        for(unsigned int iband=0;iband<this->nrOfBand();++iband)
          this->readData(lineInput[iband],y,iband);
        filter1d.applyFwhm<double>(wavelengthIn_opt,lineInput,wavelengthOut_opt,fwhm_opt, interpolationType_opt[0], lineOutput, down_opt[0], verbose_opt[0]);
        for(unsigned int iband=0;iband<imgWriter.nrOfBand();++iband){
          imgWriter.writeData(lineOutput[iband],y/down_opt[0],iband);
        }
        progress=(1.0+y)/imgWriter.nrOfRow();
        pfnProgress(progress,pszMessage,pProgressArg);
      }
    }
    else if(srf_opt.size()){
      if(verbose_opt[0])
        std::cout << "spectral filtering to " << srf_opt.size() << " bands with provided SRF " << std::endl;
      assert(wavelengthIn_opt.size());
      vector< Vector2d<double> > srf(srf_opt.size());//[0] srf_nr, [1]: wavelength, [2]: response
      ifstream srfFile;
      for(int isrf=0;isrf<srf_opt.size();++isrf){
        srf[isrf].resize(2);
        srfFile.open(srf_opt[isrf].c_str());
        double v;
        //add 0 to make sure srf is 0 at boundaries after interpolation step
        srf[isrf][0].push_back(0);
        srf[isrf][1].push_back(0);
        srf[isrf][0].push_back(1);
        srf[isrf][1].push_back(0);
        while(srfFile >> v){
          srf[isrf][0].push_back(v);
          srfFile >> v;
          srf[isrf][1].push_back(v);
        }
        srfFile.close();
        //add 0 to make sure srf[isrf] is 0 at boundaries after interpolation step
        srf[isrf][0].push_back(srf[isrf][0].back()+1);
        srf[isrf][1].push_back(0);
        srf[isrf][0].push_back(srf[isrf][0].back()+1);
        srf[isrf][1].push_back(0);
        if(verbose_opt[0])
          cout << "srf file details: " << srf[isrf][0].size() << " wavelengths defined" << endl;
      }
      assert(imgWriter.nrOfBand()==srf.size());
      double centreWavelength=0;
      Vector2d<double> lineInput(this->nrOfBand(),this->nrOfCol());
      const char* pszMessage;
      void* pProgressArg=NULL;
      GDALProgressFunc pfnProgress=GDALTermProgress;
      double progress=0;
      pfnProgress(progress,pszMessage,pProgressArg);
      for(unsigned int y=0;y<this->nrOfRow();++y){
        if((y+1+down_opt[0]/2)%down_opt[0])
          continue;
        for(unsigned int iband=0;iband<this->nrOfBand();++iband)
          this->readData(lineInput[iband],y,iband);
        for(unsigned int isrf=0;isrf<srf.size();++isrf){
          vector<double> lineOutput(imgWriter.nrOfCol());
          double delta=1.0;
          bool normalize=true;
          centreWavelength=filter1d.applySrf<double>(wavelengthIn_opt,lineInput,srf[isrf], interpolationType_opt[0], lineOutput, delta, normalize);
          if(verbose_opt[0])
            std::cout << "centre wavelength srf " << isrf << ": " << centreWavelength << std::endl;
          imgWriter.writeData(lineOutput,y/down_opt[0],isrf);
        }
        progress=(1.0+y)/imgWriter.nrOfRow();
        pfnProgress(progress,pszMessage,pProgressArg);
      }

    }
    else{
      switch(filter2d::Filter2d::getFilterType(method_opt[0])){
      case(filter2d::dilate):
        if(down_opt[0]!=1){
          std::cerr << "Error: down option not supported for morphological operator" << std::endl;
          exit(1);
        }
        if(dimZ_opt.size()){
          if(verbose_opt[0])
            std::cout<< "1-D filtering: dilate" << std::endl;
          //todo:this->filter3D.morphology(imgWriter,"dilate",dimZ_opt[0],verbose_opt[0]);
          filter1d.morphology(*this,imgWriter,"dilate",dimZ_opt[0],verbose_opt[0]);
          // filter1d.morphology(input,output,"dilate",dimZ_opt[0],verbose_opt[0]);
        }
        else{
          //todo:this=>filter2D.morphology(imgWriter,"dilate",dimX_opt[0],dimY_opt[0],angle_opt,disc_opt[0]);
          filter2d.morphology(*this,imgWriter,"dilate",dimX_opt[0],dimY_opt[0],angle_opt,disc_opt[0]);
          // filter2d.morphology(input,output,"dilate",dimX_opt[0],dimY_opt[0],angle_opt,disc_opt[0]);
        }
        break;
      case(filter2d::erode):
        if(down_opt[0]!=1){
          std::cerr << "Error: down option not supported for morphological operator" << std::endl;
          exit(1);
        }
        if(dimZ_opt.size()>0){
          if(verbose_opt[0])
            std::cout<< "1-D filtering: erode" << std::endl;
          //todo:this->filter3D.morphology(imgRaster,"erode",dimZ_opt[0],verbose_opt[0]);
          filter1d.morphology(*this,imgWriter,"erode",dimZ_opt[0],verbose_opt[0]);
          // filter1d.morphology(input,output,"erode",dimZ_opt[0],verbose_opt[0]);
        }
        else{
          //todo:this->filter2D.morphology(imgRaster,"erode",dimX_opt[0],dimY_opt[0],angle_opt,disc_opt[0]);
          filter2d.morphology(*this,imgWriter,"erode",dimX_opt[0],dimY_opt[0],angle_opt,disc_opt[0]);
          // filter2d.morphology(input,output,"erode",dimX_opt[0],dimY_opt[0],angle_opt,disc_opt[0]);
        }
        break;
      case(filter2d::close):{//closing
        if(down_opt[0]!=1){
          std::cerr << "Error: down option not supported for morphological operator" << std::endl;
          exit(1);
        }

        if(dimZ_opt.size()){
          //todo:this->filter3D.morphology(imgRaster,"dilate",dimZ_opt[0]);
          filter1d.morphology(*this,imgWriter,"dilate",dimZ_opt[0]);
          // filter1d.morphology(input,output,"dilate",dimZ_opt[0]);
        }
        else{
          //todo:this->filter2D.morphology(imgRaster,"dilate",dimX_opt[0],dimY_opt[0],angle_opt,disc_opt[0]);
          filter2d.morphology(*this,imgWriter,"dilate",dimX_opt[0],dimY_opt[0],angle_opt,disc_opt[0]);
          // filter2d.morphology(input,output,"dilate",dimX_opt[0],dimY_opt[0],angle_opt,disc_opt[0]);
        }
        if(dimZ_opt.size()){
          filter1d.morphology(imgWriter,imgWriter,"erode",dimZ_opt[0]);
          // filter1d.morphology(output,output,"erode",dimZ_opt[0]);
        }
        else{
          filter2d.morphology(imgWriter,imgWriter,"erode",dimX_opt[0],dimY_opt[0],angle_opt,disc_opt[0]);
          // filter2d.morphology(output,output,"erode",dimX_opt[0],dimY_opt[0],angle_opt,disc_opt[0]);
        }
        break;
      }
      case(filter2d::open):{//opening
        if(down_opt[0]!=1){
          std::cerr << "Error: down option not supported for morphological operator" << std::endl;
          exit(1);
        }
        if(dimZ_opt.size()){
          filter1d.morphology(*this,imgWriter,"erode",dimZ_opt[0],verbose_opt[0]);
          // filter1d.morphology(input,output,"erode",dimZ_opt[0],verbose_opt[0]);
        }
        else{
          filter2d.morphology(*this,imgWriter,"erode",dimX_opt[0],dimY_opt[0],angle_opt,disc_opt[0]);
          // filter2d.morphology(input,output,"erode",dimX_opt[0],dimY_opt[0],angle_opt,disc_opt[0]);
        }
        if(dimZ_opt.size()){
          filter1d.morphology(imgWriter,imgWriter,"dilate",dimZ_opt[0]);
          // filter1d.morphology(output,output,"dilate",dimZ_opt[0]);
        }
        else{
          filter2d.morphology(imgWriter,imgWriter,"dilate",dimX_opt[0],dimY_opt[0],angle_opt,disc_opt[0]);
          // filter2d.morphology(output,output,"dilate",dimX_opt[0],dimY_opt[0],angle_opt,disc_opt[0]);
        }
        break;
      }
      case(filter2d::homog):{//spatially homogeneous
        filter2d.doit(*this,imgWriter,"homog",dimX_opt[0],dimY_opt[0],down_opt[0],disc_opt[0]);
        // filter2d.doit(input,output,"homog",dimX_opt[0],dimY_opt[0],down_opt[0],disc_opt[0]);
        break;
      }
      case(filter2d::heterog):{//spatially heterogeneous
        filter2d.doit(*this,imgWriter,"heterog",dimX_opt[0],dimY_opt[0],down_opt[0],disc_opt[0]);
        // filter2d.doit(input,output,"heterog",dimX_opt[0],dimY_opt[0],down_opt[0],disc_opt[0]);
        break;
      }
      case(filter2d::sauvola):{//Implements Sauvola's thresholding method (http://fiji.sc/Auto_Local_Threshold)

        //test
        Vector2d<unsigned short> inBuffer;
        for(unsigned int iband=0;iband<this->nrOfBand();++iband){
          this->readDataBlock(inBuffer,0,this->nrOfCol()-1,0,this->nrOfRow()-1,iband);
        }
        filter2d.doit(*this,imgWriter,"sauvola",dimX_opt[0],dimY_opt[0],down_opt[0],disc_opt[0]);
        // filter2d.doit(input,output,"sauvola",dimX_opt[0],dimY_opt[0],down_opt[0],disc_opt[0]);
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
        // filter2d.shift(*this,imgWriter,dimX_opt[0],dimY_opt[0],threshold_opt[0],filter2d::Filter2d::getResResampleType(resample_opt[0]));
        break;
      }
        // case(filter2d::linearfeature):{
        //   if(down_opt[0]!=1){
        //  std::cerr << "Error: down option not supported for linear feature" << std::endl;
        //  exit(1);
        //   }
        //   assert(this->nrOfBand());
        //   assert(this->nrOfCol());
        //   assert(this->nrOfRow());
        //   float theAngle=361;
        //   if(angle_opt.size())
        //  theAngle=angle_opt[0];
        //   if(verbose_opt[0])
        //  std::cout << "using angle " << theAngle << std::endl;
        //   try{
        //  //using an angle step of 5 degrees and no maximum distance
        //     filter2d.linearFeature(input,output,theAngle,5,0,eps_opt[0],l1_opt[0],a1_opt[0],l2_opt[0],a2_opt[0],0,verbose_opt[0]);
        //   }
        //   catch(string errorstring){
        //     cerr << errorstring << endl;
        //   }
        //   break;
        // }
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
          //filter2d.mrf(input, output, dimX_opt[0], dimY_opt[0], beta, true, down_opt[0], verbose_opt[0]);
        }
        else
          filter2d.mrf(*this, imgWriter, dimX_opt[0], dimY_opt[0], 1, true, down_opt[0], verbose_opt[0]);
        // filter2d.mrf(input, output, dimX_opt[0], dimY_opt[0], 1, true, down_opt[0], verbose_opt[0]);
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
        // filter2d.filter(input,output,true,true);//absolute and normalize
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
        // filter2d.filter(input,output,true,true);//absolute and normalize
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
        // filter2d.filter(input,output,true,true);//absolute and normalize
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
        // filter2d.filter(input,output,true,true);//absolute and normalize
        break;
      }
      case(filter2d::smooth):{//Smoothing filter
        if(down_opt[0]!=1){
          std::cerr << "Error: down option not supported for this filter" << std::endl;
          exit(1);
        }
        if(dimZ_opt.size()){
          if(verbose_opt[0])
            std::cout<< "1-D filtering: smooth" << std::endl;
          filter1d.smooth(*this,imgWriter,dimZ_opt[0]);
        }
        else{
          filter2d.smooth(*this,imgWriter,dimX_opt[0],dimY_opt[0]);
        }
        break;
      }
      case(filter2d::smoothnodata):{//Smoothing filter
        if(down_opt[0]!=1){
          std::cerr << "Error: down option not supported for this filter" << std::endl;
          exit(1);
        }
        if(dimZ_opt.size()){
          if(verbose_opt[0])
            std::cout<< "1-D filtering: smooth" << std::endl;
          filter1d.smoothNoData(*this,interpolationType_opt[0],imgWriter);
          // filter1d.smoothNoData(input,interpolationType_opt[0],output);
        }
        else{
          if(verbose_opt[0])
            std::cout<< "2-D filtering: smooth" << std::endl;
          filter2d.smoothNoData(*this,imgWriter,dimX_opt[0],dimY_opt[0]);
          // filter2d.smoothNoData(input,output,dimX_opt[0],dimY_opt[0]);
        }
        break;
      }
      case(filter2d::dwt):
        if(down_opt[0]!=1){
          std::cerr << "Error: down option not supported for this filter" << std::endl;
          exit(1);
        }
        if(dimZ_opt.size()){
          if(verbose_opt[0])
            std::cout<< "DWT in spectral domain" << std::endl;
          filter1d.dwtForward(*this, imgWriter, wavelet_type_opt[0], family_opt[0]);
          // filter1d.dwtForward(input, output, wavelet_type_opt[0], family_opt[0]);
        }
        else
          filter2d.dwtForward(*this, imgWriter, wavelet_type_opt[0], family_opt[0]);
        // filter2d.dwtForward(input, output, wavelet_type_opt[0], family_opt[0]);
        break;
      case(filter2d::dwti):
        if(down_opt[0]!=1){
          std::cerr << "Error: down option not supported for this filter" << std::endl;
          exit(1);
        }
        if(dimZ_opt.size()){
          if(verbose_opt[0])
            std::cout<< "inverse DWT in spectral domain" << std::endl;
          filter1d.dwtInverse(*this, imgWriter, wavelet_type_opt[0], family_opt[0]);
          // filter1d.dwtInverse(input, output, wavelet_type_opt[0], family_opt[0]);
        }
        else
          filter2d.dwtInverse(*this, imgWriter, wavelet_type_opt[0], family_opt[0]);
        // filter2d.dwtInverse(input, output, wavelet_type_opt[0], family_opt[0]);
        break;
      case(filter2d::dwt_cut):
        if(down_opt[0]!=1){
          std::cerr << "Error: down option not supported for this filter" << std::endl;
          exit(1);
        }
        if(dimZ_opt.size()){
          if(verbose_opt[0])
            std::cout<< "DWT approximation in spectral domain" << std::endl;
          filter1d.dwtCut(*this, imgWriter, wavelet_type_opt[0], family_opt[0], threshold_opt[0]);
          // filter1d.dwtCut(input, output, wavelet_type_opt[0], family_opt[0], threshold_opt[0]);
        }
        else
          filter2d.dwtCut(*this, imgWriter, wavelet_type_opt[0], family_opt[0], threshold_opt[0]);
        // filter2d.dwtCut(input, output, wavelet_type_opt[0], family_opt[0], threshold_opt[0]);
        break;
      case(filter2d::dwt_cut_from):
        if(down_opt[0]!=1){
          std::cerr << "Error: down option not supported for this filter" << std::endl;
          exit(1);
        }
        if(dimZ_opt.size()){
          if(verbose_opt[0])
            std::cout<< "DWT approximation in spectral domain" << std::endl;
          filter1d.dwtCutFrom(*this,imgWriter, wavelet_type_opt[0], family_opt[0], static_cast<int>(threshold_opt[0]));
          //filter1d.dwtCutFrom(input, output, wavelet_type_opt[0], family_opt[0], static_cast<int>(thres
        }
        else{
          string errorString="Error: this filter is not supported in 2D";
          throw(errorString);
        }
        break;
      case(filter2d::savgolay):{
        assert(savgolay_nl_opt.size());
        assert(savgolay_nr_opt.size());
        assert(savgolay_ld_opt.size());
        assert(savgolay_m_opt.size());
        if(verbose_opt[0])
          std::cout << "Calculating Savitzky-Golay coefficients: " << endl;
        filter1d.getSavGolayCoefficients(tapz_opt, this->nrOfBand(), savgolay_nl_opt[0], savgolay_nr_opt[0], savgolay_ld_opt[0], savgolay_m_opt[0]);
        if(verbose_opt[0]){
          std::cout << "taps (size is " << tapz_opt.size() << "): ";
          for(int itap=0;itap<tapz_opt.size();++itap)
            std::cout<< tapz_opt[itap] << " ";
          std::cout<< std::endl;
        }
        filter1d.setTaps(tapz_opt);
        filter1d.filter(*this,imgWriter);
        // filter1d.filter(input,output);
        break;
      }
      case(filter2d::percentile)://deliberate fall through
      case(filter2d::threshold)://deliberate fall through
        assert(threshold_opt.size());
      if(dimZ_opt.size())
        filter1d.setThresholds(threshold_opt);
      else
        filter2d.setThresholds(threshold_opt);
      case(filter2d::density)://deliberate fall through
        filter2d.setClasses(class_opt);
        if(verbose_opt[0])
          std::cout << "classes set" << std::endl;
      default:
        if(dimZ_opt.size()){
          if(dimZ_opt[0]==1)
            filter1d.stat(*this,imgWriter,method_opt[0]);
          // filter1d.stat(input,output,method_opt[0]);
          else{
            assert(down_opt[0]==1);//not implemented yet...
            filter1d.filter(*this,imgWriter,method_opt[0],dimZ_opt[0]);
            // filter1d.filter(input,output,method_opt[0],dimZ_opt[0]);
          }
        }
        else if(getBlockSize()==nrOfRow())
          filter2d.doit(*this,imgWriter,method_opt[0],dimX_opt[0],dimY_opt[0],down_opt[0],disc_opt[0]);
        else
          filter2d.doit(*this,imgWriter,method_opt[0],dimX_opt[0],dimY_opt[0],down_opt[0],disc_opt[0]);
        // filter2d.doit(input,output,method_opt[0],dimX_opt[0],dimY_opt[0],down_opt[0],disc_opt[0]);
        break;
      }
    }
    // this->close();
    // imgWriter.close();
  }
  catch(string predefinedString){
    std::cout << predefinedString << std::endl;
    throw;
  }
}

///Fast implementation of filter2d, not taking care of boundary effects (values at the boundary will be initialized as 0 as they will be cut of by caller)
void Jim::filter2dFast(Jim& imgWriter, const app::AppFactory& app)
{
  Optionjl<unsigned int>  band_opt("b", "band", "band index to crop (leave empty to retain all bands)");
  Optionjl<bool> disc_opt("circ", "circular", "circular disc kernel for dilation and erosion", false);
  // Optionjl<double> angle_opt("a", "angle", "angle used for directional filtering in dilation (North=0, East=90, South=180, West=270).");
  Optionjl<std::string> method_opt("f", "filter", "filter function (nvalid, median, var, min, max, sum, mean, dilate, erode, close, open, homog (central pixel must be identical to all other pixels within window), heterog (central pixel must be different than all other pixels within window), sauvola, sobelx (horizontal edge detection), sobely (vertical edge detection), sobelxy (diagonal edge detection NE-SW),sobelyx (diagonal edge detection NW-SE), density, countid, mode (majority voting), only for classes), smooth, smoothnodata (smooth nodata values only) values, ismin, ismax, order (rank pixels in order), stdev, mrf, dwt, dwti, dwt_cut, dwt_cut_from, scramble, shift, savgolay, percentile, proportion)");
  Optionjl<std::string> resample_opt("r", "resampling-method_opt[0]", "Resampling method_opt[0] for shifting operation (near: nearest neighbour, bilinear: bi-linear interpolation).", "near");
  Optionjl<double> dimX_opt("dx", "dx", "filter kernel size in x, use odd values only", 3);
  Optionjl<double> dimY_opt("dy", "dy", "filter kernel size in y, use odd values only", 3);
  Optionjl<std::string> wavelet_type_opt("wt", "wavelet", "wavelet type: daubechies,daubechies_centered, haar, haar_centered, bspline, bspline_centered", "daubechies");
  Optionjl<int> family_opt("wf", "family", "wavelet family (vanishing moment, see also http://www.gnu.org/software/gsl/manual/html_node/DWT-Initialization.html)", 4);
  Optionjl<short> class_opt("class", "class", "class value(s) to use for density, erosion, dilation, openening and closing, thresholding");
  Optionjl<double> threshold_opt("t", "threshold", "threshold value(s) to use for threshold filter (one for each class), or threshold to cut for dwt_cut (use 0 to keep all) or dwt_cut_from, or sigma for shift", 0);
  Optionjl<double> nodata_opt("nodata", "nodata", "nodata value(s) (e.g., used for smoothnodata filter)");
  Optionjl<std::string> tap_opt("tap", "tap", "text file containing taps used for spatial filtering (from ul to lr). Use dimX and dimY to specify tap dimensions in x and y. Leave empty for not using taps");
  Optionjl<std::string> interpolationType_opt("interp", "interp", "type of interpolation for spectral filtering (see http://www.gnu.org/software/gsl/manual/html_node/Interpolation-Types.html)","akima");
  Optionjl<std::string>  otype_opt("ot", "otype", "Data type for output image ({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}). Empty string: inherit type from input image");
  Optionjl<string> beta_opt("beta", "beta", "ASCII file with beta for each class transition in Markov Random Field");
  Optionjl<short> verbose_opt("v", "verbose", "verbose mode if > 0", 0,2);

  resample_opt.setHide(1);
  wavelet_type_opt.setHide(1);
  family_opt.setHide(1);
  class_opt.setHide(1);
  threshold_opt.setHide(1);
  tap_opt.setHide(1);
  beta_opt.setHide(1);
  interpolationType_opt.setHide(1);
  otype_opt.setHide(1);
  disc_opt.setHide(1);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=method_opt.retrieveOption(app);
    band_opt.retrieveOption(app);
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
    beta_opt.retrieveOption(app);
    interpolationType_opt.retrieveOption(app);
    otype_opt.retrieveOption(app);
    disc_opt.retrieveOption(app);
    verbose_opt.retrieveOption(app);
    if(!doProcess){
      cout << endl;
      std::ostringstream helpStream;
      helpStream << "short option -h shows basic options only, use long option --help to show all options" << std::endl;
      throw(helpStream.str());//help was invoked, stop processing
    }
    const char* pszMessage;
    void* pProgressArg=NULL;
    GDALProgressFunc pfnProgress=GDALTermProgress;
    double progress=0;
    // MyProgressFunc(progress,pszMessage,pProgressArg);
    pfnProgress(progress,pszMessage,pProgressArg);

    statfactory::StatFactory stat;
    double noDataValue=0;
    if(m_noDataValues.size()){
      stat.setNoDataValues(m_noDataValues);
      noDataValue=m_noDataValues[0];
    }
    int iband=0;
    if(!band_opt.size())
      while(band_opt.size()<nrOfBand())
        band_opt.push_back(iband++);
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
    try{
      imgWriter.open(nrOfCol(),nrOfRow(),band_opt.size(),theType);
      imgWriter.setProjection(this->getProjection());
      double gt[6];
      this->getGeoTransform(gt);
      imgWriter.setGeoTransform(gt);

      imgWriter.setNoData(nodata_opt);
      // if(nodata_opt.size()){
      //   imgWriter.setNoData(nodata_opt);
      // }
    }
    catch(string errorstring){
      cout << errorstring << endl;
      throw;
    }

    for(int iband=0;iband<band_opt.size();++iband){
      double writeValue=0;
      for(int y=dimY_opt[0]/2;y<nrOfRow()-dimY_opt[0]/2;++y){
        for(int x=dimX_opt[0]/2;x<nrOfCol()-dimX_opt[0]/2;++x){
          std::vector<double> windowBuffer;
          std::map<int,int> occurrence;
          int centre=dimX_opt[0]*(dimY_opt[0]-1)/2+(dimX_opt[0]-1)/2;
          for(int j=-(dimY_opt[0]-1)/2;j<=dimY_opt[0]/2;++j){
            for(int i=-(dimX_opt[0]-1)/2;i<=dimX_opt[0]/2;++i){
              int indexJ=(dimY_opt[0]-1)/2+j;
              int indexI=x+i;
              //check if out of bounds
              if(indexI<0)
                indexI=-indexI;
              else if(indexI>=nrOfCol())
                indexI=nrOfCol()-i;
              if(y+j<0)
                indexJ=-j;
              else if(y+j>=nrOfRow())
                indexJ=(dimY_opt[0]>2) ? (dimY_opt[0]-1)/2-j : 0;
              // windowBuffer.push_back(inBuffer[indexJ][indexI]);
              windowBuffer.push_back(readData(x,y,band_opt[iband]));
              switch(filter2d::Filter2d::getFilterType(method_opt[0])){
              case(filter2d::density):
              case(filter2d::countid):
              case(filter2d::mode):
              case(filter2d::threshold):
                if(!stat.isNoData(readData(x,y,band_opt[iband]))){
                  std::vector<short>::const_iterator vit=class_opt.begin();
                  //todo: test if this works (only add occurrence if within defined classes)!
                  if(!class_opt.size())
                    ++occurrence[readData(x,y,band_opt[iband])];
                  else{
                    while(vit!=class_opt.end()){
                      if(readData(x,y,band_opt[iband]))
                        ++occurrence[readData(x,y,band_opt[iband])];
                    }
                  }
                  break;
                }
              default:
                break;
              }
            }
          }
          switch(filter2d::Filter2d::getFilterType(method_opt[0])){
          case(filter2d::nvalid):
            imgWriter.writeData(stat.nvalid(windowBuffer),x,y,iband);
            break;
          case(filter2d::median):
            imgWriter.writeData(stat.median(windowBuffer),x,y,iband);
            break;
          case(filter2d::var):{
            imgWriter.writeData(stat.var(windowBuffer),x,y,iband);
            break;
          }
          case(filter2d::stdev):{
            double varValue=stat.var(windowBuffer);
            if(stat.isNoData(varValue))
              imgWriter.writeData(noDataValue,x,y,iband);
            else
              imgWriter.writeData(sqrt(varValue),x,y,iband);
            break;
          }
          case(filter2d::mean):{
            if(windowBuffer.empty())
              imgWriter.writeData(noDataValue,x,y,iband);
            else
              imgWriter.writeData(stat.mean(windowBuffer),x,y,iband);
            break;
          }
          case(filter2d::min):{
            imgWriter.writeData(stat.mymin(windowBuffer),x,y,iband);
            break;
          }
          case(filter2d::ismin):{
            double minValue=stat.mymin(windowBuffer);
            if(stat.isNoData(minValue))
              imgWriter.writeData(noDataValue,x,y,iband);
            else
              imgWriter.writeData((windowBuffer[centre]==minValue)? 1:0,x,y,iband);
            break;
          }
          case(filter2d::minmax):{
            double min=0;
            double max=0;
            stat.minmax(windowBuffer,windowBuffer.begin(),windowBuffer.end(),min,max);
            if(min!=max)
              imgWriter.writeData(0,x,y,iband);
            else
              imgWriter.writeData(windowBuffer[centre],x,y,iband);//centre pixels
            break;
          }
          case(filter2d::max):{
            imgWriter.writeData(stat.mymax(windowBuffer),x,y,iband);
            break;
          }
          case(filter2d::ismax):{
            double maxValue=stat.mymax(windowBuffer);
            if(stat.isNoData(maxValue))
              imgWriter.writeData(noDataValue,x,y,iband);
            else
              imgWriter.writeData((windowBuffer[centre]==maxValue)? 1:0,x,y,iband);
            break;
          }
          case(filter2d::order):{
            stat.eraseNoData(windowBuffer);
            if(windowBuffer.empty())
              imgWriter.writeData(noDataValue,x,y,iband);
            else{
              double lbound=0;
              double ubound=dimX_opt[0]*dimY_opt[0];
              double theMin=stat.mymin(windowBuffer);
              double theMax=stat.mymax(windowBuffer);
              double scale=(ubound-lbound)/(theMax-theMin);
              imgWriter.writeData(static_cast<short>(scale*(windowBuffer[centre]-theMin)+lbound),x,y,iband);
            }
            break;
          }
          case(filter2d::sum):{
            imgWriter.writeData(stat.sum(windowBuffer),x,y,iband);
            break;
          }
          case(filter2d::percentile):{
            assert(threshold_opt.size());
            imgWriter.writeData(stat.percentile(windowBuffer,windowBuffer.begin(),windowBuffer.end(),threshold_opt[0]),x,y,iband);
            break;
          }
          case(filter2d::proportion):{
            stat.eraseNoData(windowBuffer);
            double sum=stat.sum(windowBuffer);
            if(sum)
              imgWriter.writeData(windowBuffer[centre]/sum,x,y,iband);
            else
              imgWriter.writeData(noDataValue,x,y,iband);
            break;
          }
          case(filter2d::homog):{
            // double centreValue=inBuffer[(dimY-1)/2][x];
            double centreValue=readData(x,y,band_opt[iband]);
            bool isHomog=true;
            stat.eraseNoData(windowBuffer);
            typename std::vector<double>::const_iterator wit;
            for(wit=windowBuffer.begin();wit!=windowBuffer.end();++wit){
              if(*wit==centreValue)
                continue;
              else{
                isHomog=false;
                break;
              }
            }
            if(isHomog)
              imgWriter.writeData(1,x,y,iband);
            else
              imgWriter.writeData(noDataValue,x,y,iband);
            break;
          }
          case(filter2d::heterog):{
            // double centreValue=inBuffer[(dimY-1)/2][x];
            double centreValue=readData(x,y,band_opt[iband]);
            bool isHeterog=true;
            stat.eraseNoData(windowBuffer);
            typename std::vector<double>::const_iterator wit;
            for(wit=windowBuffer.begin();wit!=windowBuffer.end();++wit){
              if(*wit!=centreValue)
                continue;
              else{
                isHeterog=false;
                break;
              }
            }
            if(isHeterog)
              imgWriter.writeData(1,x,y,iband);
            else
              imgWriter.writeData(noDataValue,x,y,iband);
            break;
          }
          case(filter2d::sauvola):{
            try{
              double theMean=0;
              double theStdev=0;
              bool invalid=false;
              // double centreValue=inBuffer[(dimY-1)/2][x];
              double centreValue=readData(x,y,band_opt[iband]);
              if(windowBuffer.empty()||stat.isNoData(centreValue)){
                invalid=true;
                throw(invalid);
              }
              stat.meanVar(windowBuffer,theMean,theStdev);
              theStdev=sqrt(theStdev);
              double kValue=0.5;
              double rValue=128;
              if(threshold_opt.size()==2){
                kValue=threshold_opt[0];
                rValue=threshold_opt[1];
              }
              //from http://fiji.sc/Auto_Local_Threshold
              //pixel = ( pixel > mean * ( 1 + k * ( standard_deviation / r - 1 ) ) ) ? object : background
              double theThreshold=theMean*(1+kValue*(theStdev/rValue - 1));
              //isdata value hardcoded as 1 for now
              imgWriter.writeData((centreValue>theThreshold) ? 1 : noDataValue,x,y,iband);
            }
            catch(bool invalid){
              imgWriter.writeData(noDataValue,x,y,iband);
            }
            break;
          }
          case(filter2d::density):{
            int nvalid=stat.nvalid(windowBuffer);
            if(nvalid){
              std::vector<short>::const_iterator vit=class_opt.begin();
              while(vit!=class_opt.end())
                imgWriter.writeData(imgWriter.readData(x,y,iband)+100.0*occurrence[*(vit++)]/nvalid,x,y,iband);
            }
            else
              imgWriter.writeData(noDataValue,x,y,iband);
            break;
          }
          case(filter2d::countid):{
            if(occurrence.size())
              imgWriter.writeData(occurrence.size(),x,y,iband);
            else
              imgWriter.writeData(noDataValue,x,y,iband);
            break;
          }
          case(filter2d::mode):{
            if(occurrence.size()){
              std::map<int,int>::const_iterator maxit=occurrence.begin();
              for(std::map<int,int>::const_iterator mit=occurrence.begin();mit!=occurrence.end();++mit){
                if(mit->second>maxit->second)
                  maxit=mit;
              }
              // if(occurrence[inBuffer[(dimY-1)/2][x]]<maxit->second)//
              if(occurrence[readData(x,y,band_opt[iband])]<maxit->second)
                imgWriter.writeData(maxit->first,x,y,iband);
              else//favorize original value in case of ties
                imgWriter.writeData(readData(x,y,band_opt[iband]),x,y,iband);
              // outBuffer[x/down_opt[0]+y/down_opt[0]*nrOfCol()]=inBuffer[(dimY-1)/2][x];
            }
            else
              imgWriter.writeData(noDataValue,x,y,iband);
            break;
          }
          case(filter2d::threshold):{
            assert(class_opt.size()==threshold_opt.size());
            int nvalid=stat.nvalid(windowBuffer);
            if(nvalid>0){
              // outBuffer[x/down_opt[0]+y/down_opt[0]*nrOfCol()]=inBuffer[(dimY-1)/2][x];//initialize with original value (in case thresholds not met)
              imgWriter.writeData(readData(x,y,band_opt[iband]),x,y,iband);//initialize with original value (in case thresholds not met)
              for(int iclass=0;iclass<class_opt.size();++iclass){
                if(100.0*(occurrence[class_opt[iclass]])/nvalid>threshold_opt[iclass])
                  imgWriter.writeData(class_opt[iclass],x,y,iband);
              }
            }
            else
              imgWriter.writeData(noDataValue,x,y,iband);
            break;
          }
          case(filter2d::scramble):{//could be done more efficiently window by window with random shuffling entire buffer and assigning entire buffer at once to output image...
            if(windowBuffer.size()){
              int randomIndex=std::rand()%windowBuffer.size();
              if(randomIndex>=windowBuffer.size())
                imgWriter.writeData(windowBuffer.back(),x,y,iband);
              else if(randomIndex<0)
                imgWriter.writeData(windowBuffer[0],x,y,iband);
              else
                imgWriter.writeData(windowBuffer[randomIndex],x,y,iband);
            }
            else
              imgWriter.writeData(noDataValue,x,y,iband);
            break;
          }
          default:
            break;
          }
        }
        progress=(1.0+y);
        progress/=nrOfRow();
        // MyProgressFunc(progress,pszMessage,pProgressArg);
        pfnProgress(progress,pszMessage,pProgressArg);
      }
    }
  }
  catch(string predefinedString){
    std::cout << predefinedString << std::endl;
    throw;
  }
}
