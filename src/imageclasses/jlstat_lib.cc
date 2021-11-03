/**********************************************************************
jlstat_lib.cc: program to calculate basic statistics from raster dataset
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
#include <iostream>
#include <fstream>
// #include <math.h>
#include <cmath>
#include "base/Optionjl.h"
#include "algorithms/StatFactory.h"
#include "algorithms/ImgRegression.h"
#if JIPLIB_BUILD_WITH_PYTHON==1
#include <Python.h>
#endif
/******************************************************************************/
using namespace std;
using namespace app;

/**
 * @param app application specific option arguments
 * @return ostringstream with statistics
 **/
// CPLErr Jim::getStats(AppFactory& app){
std::multimap<std::string,std::string> Jim::getStats(AppFactory& app){
  JimList singleList;
  std::shared_ptr<Jim> imgReader=shared_from_this();
  singleList.pushImage(imgReader);
  return(singleList.getStats(app));
  // if(singleList.getStats(app).front())
  //   return(CE_None);
  // else
  //   return(CE_Failure);
}

/**
 * @param scale (type: double) output=scale*input+offset
 * @param offset (type: double) output=scale*input+offset
 * @param mem (type: unsigned long) (default: 0) Buffer size (in MB) to read image data blocks in memory
 * @param function (type: std::string) (default: basic) Statistics function (invalid, valid, filename, basic, gdal, mean, median, var, skewness, kurtosis,stdev, sum, minmax, min, max, histogram, histogram2d, rmse, regression, regressionError, regressionPerpendicular
 * @param band (type: unsigned short) (default: 0) band(s) on which to calculate statistics
 * @param nodata (type: double) Set nodata value(s)
 * @param nbin (type: short) number of bins to calculate histogram
 * @param relative (type: bool) (default: 0) use percentiles for histogram to calculate histogram
 * @param down (type: short) (default: 1) Down sampling factor (for raster sample datasets only). Can be used to create grid points
 * @param rnd (type: unsigned int) (default: 0) generate random numbers
 * @param scale (type: double) Scale(s) for reading input image(s)
 * @param offset (type: double) Offset(s) for reading input image(s)
 * @param src_min (type: double) start reading source from this minimum value
 * @param src_max (type: double) stop reading source from this maximum value
 * @param kde (type: bool) (default: 0) Use Kernel density estimation when producing histogram. The standard deviation is estimated based on Silverman's rule of thumb
 * @return this object
 **/
// JimList& JimList::getStats(AppFactory& app){
std::multimap<std::string,std::string> JimList::getStats(AppFactory& app){
  Optionjl<unsigned short> band_opt("b","band","band(s) on which to calculate statistics");
  Optionjl<std::string> function_opt("f", "function", "Statistics function (invalid, valid, filename, basic, gdal, mean, median, var, skewness, kurtosis,stdev, sum, minmax, min, max, histogram, histogram2d, rmse, regression, regressionError, regressionPerpendicular)","basic");
  // Optionjl<double>  ulx_opt("ulx", "ulx", "Upper left x value bounding box");
  // Optionjl<double>  uly_opt("uly", "uly", "Upper left y value bounding box");
  // Optionjl<double>  lrx_opt("lrx", "lrx", "Lower right x value bounding box");
  // Optionjl<double>  lry_opt("lry", "lry", "Lower right y value bounding box");
  Optionjl<double> nodata_opt("nodata","nodata","Set nodata value(s)");
  Optionjl<short> down_opt("down", "down", "Down sampling factor (for raster sample datasets only). Can be used to create grid points", 1);
  Optionjl<unsigned int> random_opt("rnd", "rnd", "generate random numbers", 0);
  Optionjl<double>  scale_opt("scale", "scale", "Scale(s) for reading input image(s)");
  Optionjl<double>  offset_opt("offset", "offset", "Offset(s) for reading input image(s)");
  Optionjl<double> src_min_opt("src_min","src_min","start reading source from this minimum value");
  Optionjl<double> src_max_opt("src_max","src_max","stop reading source from this maximum value");
  Optionjl<short> nbin_opt("nbin","nbin","number of bins to calculate histogram");
  Optionjl<bool> relative_opt("rel","relative","use percentiles for histogram to calculate histogram",false);
  Optionjl<bool> kde_opt("kde","kde","Use Kernel density estimation when producing histogram. The standard deviation is estimated based on Silverman's rule of thumb",false);
  Optionjl<short> verbose_opt("v", "verbose", "verbose mode when positive", 0,2);
  // ulx_opt.setHide(1);
  // uly_opt.setHide(1);
  // lrx_opt.setHide(1);
  // lry_opt.setHide(1);
  down_opt.setHide(1);
  random_opt.setHide(1);
  scale_opt.setHide(1);
  offset_opt.setHide(1);
  src_min_opt.setHide(1);
  src_max_opt.setHide(1);
  kde_opt.setHide(1);

  std::multimap<std::string,std::string> mapString;
  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=function_opt.retrieveOption(app);
    //optional options
    doProcess=band_opt.retrieveOption(app);
    nodata_opt.retrieveOption(app);
    nbin_opt.retrieveOption(app);
    relative_opt.retrieveOption(app);
    //advanced options
    // ulx_opt.retrieveOption(app);
    // uly_opt.retrieveOption(app);
    // lrx_opt.retrieveOption(app);
    // lry_opt.retrieveOption(app);
    down_opt.retrieveOption(app);
    random_opt.retrieveOption(app);
    scale_opt.retrieveOption(app);
    offset_opt.retrieveOption(app);
    src_min_opt.retrieveOption(app);
    src_max_opt.retrieveOption(app);
    kde_opt.retrieveOption(app);
    verbose_opt.retrieveOption(app);

    // std::ostringstream 
    if(!doProcess){
      cout << endl;
      std::ostringstream helpStream;
      helpStream << "short option -h shows basic options only, use long option --help to show all options" << std::endl;
      throw(helpStream.str());//help was invoked, stop processing
    }

    if(empty()){
      std::ostringstream errorStream;
      errorStream << "Input collection is empty. Use --help for more help information" << std::endl;
      throw(errorStream.str());
    }


    // std::vector<std::string> badKeys;
    // app.badKeys(badKeys);
    // if(badKeys.size()){
    //   std::ostringstream errorStream;
    //   if(badKeys.size()>1)
    //     errorStream << "Error: unknown keys: ";
    //   else
    //     errorStream << "Error: unknown key: ";
    //   for(int ikey=0;ikey<badKeys.size();++ikey){
    //     errorStream << badKeys[ikey] << " ";
    //   }
    //   errorStream << std::endl;
    //   throw(errorStream.str());
    // }


    bool invalid_opt=std::find(function_opt.begin(), function_opt.end(), "invalid") != function_opt.end();//Report number of nodata values in image
    bool valid_opt=std::find(function_opt.begin(),function_opt.end(),"valid")!=function_opt.end();//Report number of nodata values (i.e., not nodata) in image
    bool filename_opt=std::find(function_opt.begin(),function_opt.end(),"filename")!=function_opt.end();//"fn", "filename", "Shows image filename ", false);
    bool stat_opt=std::find(function_opt.begin(),function_opt.end(),"basic")!=function_opt.end();//"basic", "basic", "Shows basic statistics (calculate in memory) (min,max, mean and stdev of the raster datasets)", false);
    bool fstat_opt=std::find(function_opt.begin(),function_opt.end(),"gdal")!=function_opt.end();//"gdal", "gdal", "Shows basic statistics using GDAL computeStatistics  (min,max, mean and stdev of the raster datasets)", false);
    bool mean_opt=std::find(function_opt.begin(),function_opt.end(),"mean")!=function_opt.end();//calculate mean
    bool median_opt=std::find(function_opt.begin(),function_opt.end(),"median")!=function_opt.end();//calculate median
    bool var_opt=std::find(function_opt.begin(),function_opt.end(),"var")!=function_opt.end();//calculate variance
    bool skewness_opt=std::find(function_opt.begin(),function_opt.end(),"skewness")!=function_opt.end();//calculate skewness
    bool kurtosis_opt=std::find(function_opt.begin(),function_opt.end(),"kurtosis")!=function_opt.end();//calculate kurtosis
    bool stdev_opt=std::find(function_opt.begin(),function_opt.end(),"stdev")!=function_opt.end();//calculate standard deviation
    bool sum_opt=std::find(function_opt.begin(),function_opt.end(),"sum")!=function_opt.end();//calculate sum of column
    bool minmax_opt=std::find(function_opt.begin(),function_opt.end(),"minmax")!=function_opt.end();//calculate minimum and maximum value
    bool min_opt=std::find(function_opt.begin(),function_opt.end(),"min")!=function_opt.end();//calculate minimum value
    bool max_opt=std::find(function_opt.begin(),function_opt.end(),"max")!=function_opt.end();//calculate maximum value
    bool histogram_opt=std::find(function_opt.begin(),function_opt.end(),"histogram")!=function_opt.end();//calculate histogram
    bool histogram2d_opt=std::find(function_opt.begin(),function_opt.end(),"histogram2d")!=function_opt.end();//calculate 2-dimensional histogram based on two images
    bool rmse_opt=std::find(function_opt.begin(),function_opt.end(),"rmse")!=function_opt.end();//calculate root mean square error between two raster datasets
    bool reg_opt=std::find(function_opt.begin(),function_opt.end(),"regression")!=function_opt.end();//calculate linear regression between two raster datasets and get correlation coefficient
    bool regerr_opt=std::find(function_opt.begin(),function_opt.end(),"regressionError")!=function_opt.end();//calculate linear regression between two raster datasets and get root mean square error
    bool preg_opt=std::find(function_opt.begin(),function_opt.end(),"regressionPerpendicular")!=function_opt.end();//calculate perpendicular regression between two raster datasets and get correlation coefficient


    srand(time(NULL));

    if(!band_opt.size()){
      for(size_t iband=0;iband<(this->getImage(0))->nrOfBand();++iband)
        band_opt.push_back(iband);
    }

    if(src_min_opt.size()){
      while(src_min_opt.size()<band_opt.size())
        src_min_opt.push_back(src_min_opt[0]);
    }
    if(src_max_opt.size()){
      while(src_max_opt.size()<band_opt.size())
        src_max_opt.push_back(src_max_opt[0]);
    }

    int nbin=0;
    double minX=0;
    double minY=0;
    double maxX=0;
    double maxY=0;
    double minValue=(src_min_opt.size())? src_min_opt[0] : 0;
    double maxValue=(src_max_opt.size())? src_max_opt[0] : 0;
    double meanValue=0;
    double medianValue=0;
    double stdDev=0;

    statfactory::StatFactory stat;
    imgregression::ImgRegression imgreg;
    std::vector<double> histogramOutput;
    double nsample=0;

    // Jim imgReader;
    if(scale_opt.size()){
      while(scale_opt.size()<size())
        scale_opt.push_back(scale_opt[0]);
    }
    if(offset_opt.size()){
      while(offset_opt.size()<size())
        offset_opt.push_back(offset_opt[0]);
    }
    std::vector< std::vector<double> > orignodata(size());
    std::list<std::shared_ptr<Jim> >::const_iterator imit=begin();
    size_t iimage=0;
    for(imit=begin();imit!=end();++imit){
      if(filename_opt)
        mapString.insert(std::make_pair("filename",(*imit)->getFileName()));

      if(nodata_opt.size()){
        (*imit)->getNoDataValues(orignodata[iimage]);
        for(size_t inodata=0;inodata<nodata_opt.size();++inodata)
          (*imit)->pushNoDataValue(nodata_opt[inodata]);
      }

      int nband=band_opt.size();
      std::vector<std::string> nvalids;
      std::vector<std::string> ninvalids;
      std::vector<std::string> means;
      std::vector<std::string> medians;
      std::vector<std::string> vars;
      std::vector<std::string> stdevs;
      std::vector<std::string> mins;
      std::vector<std::string> maxs;
      for(int iband=0;iband<nband;++iband){
        if(src_min_opt.size())
          minValue=(src_min_opt.size()>iband)? src_min_opt[iband] : src_min_opt[0];
        else
          minValue= 0;
        if(src_max_opt.size())
          maxValue=(src_max_opt.size()>iband)? src_max_opt[iband] : src_max_opt[0];
        else
          maxValue= 0;
        for(size_t inodata=0;inodata<nodata_opt.size();++inodata){
          if(!inodata)
            (*imit)->GDALSetNoDataValue(nodata_opt[0],band_opt[iband]);//only single no data can be set in GDALRasterBand (used for ComputeStatistics)
        }

        if(valid_opt)
          nvalids.push_back(type2string<unsigned long int>((*imit)->getNvalid(band_opt[iband])));
        // mapString.insert(std::make_pair("nvalid",type2string<unsigned long int>((*imit)->getNvalid(band_opt[iband]))));
          // mapString["nvalid"]=type2string<unsigned long int>((*imit)->getNvalid(band_opt[0]));
        // outputStream << "--nvalid " << (*imit)->getNvalid(band_opt[0]) << endl;
        if(invalid_opt)
          ninvalids.push_back(type2string<unsigned long int>((*imit)->getNinvalid(band_opt[iband])));
          // mapString.insert(std::make_pair("ninvalid",type2string<unsigned long int>((*imit)->getNinvalid(band_opt[iband]))));
          // mapString["ninvalid"]=type2string<unsigned long int>((*imit)->getNinvalid(band_opt[0]));
        // outputStream << "--ninvalid " << (*imit)->getNinvalid(band_opt[0]) << endl;
        if(stat_opt||mean_opt||median_opt||var_opt||stdev_opt||min_opt||max_opt||minmax_opt){//the hard way (in memory)
          statfactory::StatFactory stat;
          vector<double> readBuffer;
          vector<double> tmpBuffer;
          double varValue;
          //test
          // (*imit)->readData(readBuffer, 1, 0);
          // std::cout << "readBuffer.size(): " << readBuffer.size() << std::endl;
          // //test
          // for(int i=0;i<10;++i){
          //   std::cout << readBuffer[i] << " ";
          // }
          // std::cout << std::endl;
          (*imit)->readDataBlock(readBuffer, 0, (*imit)->nrOfCol()-1, 0, (*imit)->nrOfRow()-1, band_opt[iband]);
          // //test
          // for(int i=0;i<10;++i)
          //   std::cout << *(readBuffer.begin()+i) << " ";
          // std::cout << std::endl;
          // for(int j=0;j<10;++j){
          //   for(int i=0;i<10;++i)
          //     std::cout << readBuffer[j*(*imit)->nrOfCol()+i] << " ";
          //   std::cout << std::endl;
          // }
          stat.setNoDataValues(nodata_opt);
          if(src_min_opt.size())
            stat.eraseBelow(readBuffer,src_min_opt[iband]);
            // stat.eraseBelow(readBuffer,src_min_opt[band_opt[iband]]);
          if(src_max_opt.size())
            stat.eraseAbove(readBuffer,src_max_opt[iband]);
            // stat.eraseAbove(readBuffer,src_max_opt[band_opt[iband]]);
          stat.eraseNoData(readBuffer);
          // stat.meanVar(readBuffer,meanValue,varValue);
          // medianValue=stat.median(readBuffer);
          // stat.minmax(readBuffer,readBuffer.begin(),readBuffer.end(),minValue,maxValue);
          size_t stride=1;
          if(mean_opt||stat_opt){
            means.push_back(type2string<double>(gsl_stats_mean(&(readBuffer[0]),stride,readBuffer.size())));
            // mapString.insert(std::make_pair("mean",type2string<double>(gsl_stats_mean(&(readBuffer[0]),stride,readBuffer.size()))));
            // mapString["mean"]=type2string<double>(gsl_stats_mean(&(readBuffer[0]),stride,readBuffer.size()));
            // double data[10]={17,22,18,14,8,3,5,10,10,8};
            // std::vector<double> testv(10);
            // for(int i=0;i<10;++i)
            //   std::cout << *(readBuffer.begin()+i) << " ";
            // std::cout << std::endl;
            // testv.assign(readBuffer.begin(),readBuffer.begin()+10);
            // std::cout << "mean is: " << gsl_stats_mean(&(testv[0]),1,10) << std::endl;
            // std::cout << "mean is: " << gsl_stats_mean(data, 1, 10) << std::endl;
            // std::cout << "mean is: " << gsl_stats_mean(&(readBuffer[0]),1,10) << std::endl;
            // mapString["mean"]=type2string<double>(meanValue);
            // outputStream << "--mean " << meanValue << " " << std::endl;
          }
          if(median_opt){
            tmpBuffer=readBuffer;
            gsl_sort(&(tmpBuffer[0]),stride,readBuffer.size());
            medians.push_back(type2string<double>(gsl_stats_median_from_sorted_data(&(tmpBuffer[0]),stride,readBuffer.size())));
            // mapString.insert(std::make_pair("median",type2string<double>(gsl_stats_median_from_sorted_data(&(tmpBuffer[0]),stride,readBuffer.size()))));
            tmpBuffer.clear();
          }
          if(stdev_opt||stat_opt){
            stdevs.push_back(type2string<double>(gsl_stats_sd(&(readBuffer[0]),stride,readBuffer.size())));
            // mapString.insert(std::make_pair("stdev",type2string<double>(gsl_stats_sd(&(readBuffer[0]),stride,readBuffer.size()))));
            // mapString["stdev"]=type2string<double>(gsl_stats_sd(&(readBuffer[0]),stride,readBuffer.size()));
            // mapString["stdev"]=type2string<double>(sqrt(varValue));
            // outputStream << "--stdev " << sqrt(varValue) << " " << std::endl;
          }
          if(var_opt){
            vars.push_back(type2string<double>(gsl_stats_variance(&(readBuffer[0]),stride,readBuffer.size())));
            // mapString.insert(std::make_pair("var",type2string<double>(gsl_stats_variance(&(readBuffer[0]),stride,readBuffer.size()))));
            // mapString["var"]=type2string<double>(gsl_stats_variance(&(readBuffer[0]),stride,readBuffer.size()));
            // mapString["var"]=type2string<double>(varValue);
            // outputStream << "--var " << varValue << " " << std::endl;
          }
          if(min_opt||minmax_opt||stat_opt)
            mins.push_back(type2string<double>(gsl_stats_min(&(readBuffer[0]),stride,readBuffer.size())));
            // mapString.insert(std::make_pair("min",type2string<double>(gsl_stats_min(&(readBuffer[0]),stride,readBuffer.size()))));
            // mapString["min"]=type2string<double>(gsl_stats_min(&(readBuffer[0]),stride,readBuffer.size()));
          if(max_opt||minmax_opt||stat_opt)
            maxs.push_back(type2string<double>(gsl_stats_max(&(readBuffer[0]),stride,readBuffer.size())));
            // mapString.insert(std::make_pair("max",type2string<double>(gsl_stats_max(&(readBuffer[0]),stride,readBuffer.size()))));
            // mapString["max"]=type2string<double>(gsl_stats_max(&(readBuffer[0]),stride,readBuffer.size()));
          // if(stat_opt){
            // mapString["min"]=type2string<double>(minValue);
            // mapString["max"]=type2string<double>(maxValue);
            // mapString["mean"]=type2string<double>(meanValue);
            // mapString["stdev"]=type2string<double>(sqrt(varValue));
            // outputStream << "--min " << minValue << " --max " << maxValue << " --mean " << meanValue << " --stdev " << sqrt(varValue) << " " << std::endl;
          // }
        }

        if(fstat_opt){//the fast way
          assert(band_opt[iband]<(*imit)->nrOfBand());
          GDALProgressFunc pfnProgress;
          void* pProgressData;
          GDALRasterBand* rasterBand;
          rasterBand=(*imit)->getRasterBand(band_opt[iband]);
          rasterBand->ComputeStatistics(0,&minValue,&maxValue,&meanValue,&stdDev,pfnProgress,pProgressData);
          mins.push_back(type2string<double>(minValue));
          maxs.push_back(type2string<double>(maxValue));
          means.push_back(type2string<double>(meanValue));
          stdevs.push_back(type2string<double>(stdDev));
          // mapString.insert(std::make_pair("min",type2string<double>(minValue)));
          // mapString.insert(std::make_pair("max",type2string<double>(maxValue)));
          // mapString.insert(std::make_pair("mean",type2string<double>(meanValue)));
          // mapString.insert(std::make_pair("stdev",type2string<double>(stdDev)));
          // mapString["min"]=type2string<double>(minValue);
          // mapString["max"]=type2string<double>(maxValue);
          // mapString["mean"]=type2string<double>(meanValue);
          // mapString["stdev"]=type2string<double>(stdDev);
          // outputStream << "--min " << minValue << " --max " << maxValue << " --mean " << meanValue << " --stdev " << stdDev << " " << std::endl;
        }
        // if(minmax_opt||min_opt||max_opt){
        //   assert(band_opt[iband]<(*imit)->nrOfBand());

        //   if((ulx_opt.size()||uly_opt.size()||lrx_opt.size()||lry_opt.size())&&((*imit)->covers(ulx_opt[0],uly_opt[0],lrx_opt[0],lry_opt[0]))){
        //     double uli,ulj,lri,lrj;
        //     (*imit)->geo2image(ulx_opt[0],uly_opt[0],uli,ulj);
        //     (*imit)->geo2image(lrx_opt[0],lry_opt[0],lri,lrj);
        //     (*imit)->getMinMax(static_cast<int>(uli),static_cast<int>(lri),static_cast<int>(ulj),static_cast<int>(lrj),band_opt[iband],minValue,maxValue);
        //   }
        //   else{
        //     (*imit)->getMinMax(minValue,maxValue,band_opt[iband]);
        //   }
        //   if(minmax_opt){
        //     mapString["min"]=type2string<double>(minValue);
        //     mapString["max"]=type2string<double>(maxValue);
        //     // outputStream << "--min " << minValue << " --max " << maxValue << " " << std::endl;
        //   }
        //   else{
        //     if(min_opt)
        //       mapString["min"]=type2string<double>(minValue);
        //       // outputStream << "--min " << minValue << " " << std::endl;
        //     if(max_opt)
        //       mapString["max"]=type2string<double>(maxValue);
        //       // outputStream << "--max " << maxValue << " ";
        //   }
        // }
      }
      if(mins.size()){
        std::ostringstream isminvalues;
        for(size_t ivalue=0;ivalue<mins.size();++ivalue){
          if(mins.size()>1){
            if(!ivalue)
              isminvalues << "[";
            else
              isminvalues << ",";
            isminvalues << mins[ivalue];
            if(ivalue==mins.size()-1)
              isminvalues << "]";
          }
          else
            isminvalues << mins[0];
        }
        mapString.insert(std::make_pair("min",isminvalues.str()));
      }
      if(maxs.size()){
        std::ostringstream ismaxvalues;
        for(size_t ivalue=0;ivalue<maxs.size();++ivalue){
          if(maxs.size()>1){
            if(!ivalue)
              ismaxvalues << "[";
            else
              ismaxvalues << ",";
            ismaxvalues << maxs[ivalue];
            if(ivalue==maxs.size()-1)
              ismaxvalues << "]";
          }
          else
            ismaxvalues << maxs[0];
        }
        mapString.insert(std::make_pair("max",ismaxvalues.str()));
      }
      if(means.size()){
        std::ostringstream ismeanvalues;
        for(size_t ivalue=0;ivalue<means.size();++ivalue){
          if(means.size()>1){
            if(!ivalue)
              ismeanvalues << "[";
            else
              ismeanvalues << ",";
            ismeanvalues << means[ivalue];
            if(ivalue==means.size()-1)
              ismeanvalues << "]";
          }
          else
            ismeanvalues << means[0];
        }
        mapString.insert(std::make_pair("mean",ismeanvalues.str()));
      }
      if(medians.size()){
        std::ostringstream ismedianvalues;
        for(size_t ivalue=0;ivalue<medians.size();++ivalue){
          if(medians.size()>1){
            if(!ivalue)
              ismedianvalues << "[";
            else
              ismedianvalues << ",";
            ismedianvalues << medians[ivalue];
            if(ivalue==medians.size()-1)
              ismedianvalues << "]";
          }
          else
            ismedianvalues << medians[0];
        }
        mapString.insert(std::make_pair("median",ismedianvalues.str()));
      }
      if(vars.size()){
        std::ostringstream isvarvalues;
        for(size_t ivalue=0;ivalue<vars.size();++ivalue){
          if(vars.size()>1){
            if(!ivalue)
              isvarvalues << "[";
            else
              isvarvalues << ",";
            isvarvalues << vars[ivalue];
            if(ivalue==vars.size()-1)
              isvarvalues << "]";
          }
          else
            isvarvalues << vars[0];
        }
        mapString.insert(std::make_pair("var",isvarvalues.str()));
      }
      if(stdevs.size()){
        std::ostringstream isstdevvalues;
        for(size_t ivalue=0;ivalue<stdevs.size();++ivalue){
          if(stdevs.size()>1){
            if(!ivalue)
              isstdevvalues << "[";
            else
              isstdevvalues << ",";
            isstdevvalues << stdevs[ivalue];
            if(ivalue==stdevs.size()-1)
              isstdevvalues << "]";
          }
          else
            isstdevvalues << stdevs[0];
        }
        mapString.insert(std::make_pair("stdev",isstdevvalues.str()));
      }
      if(nvalids.size()){
        std::ostringstream isnvalidvalues;
        for(size_t ivalue=0;ivalue<nvalids.size();++ivalue){
          if(nvalids.size()>1){
            if(!ivalue)
              isnvalidvalues << "[";
            else
              isnvalidvalues << ",";
            isnvalidvalues << nvalids[ivalue];
            if(ivalue==nvalids.size()-1)
              isnvalidvalues << "]";
          }
          else
            isnvalidvalues << nvalids[0];
        }
        mapString.insert(std::make_pair("nvalid",isnvalidvalues.str()));
      }
      if(ninvalids.size()){
        std::ostringstream isninvalidvalues;
        for(size_t ivalue=0;ivalue<ninvalids.size();++ivalue){
          if(ninvalids.size()>1){
            if(!ivalue)
              isninvalidvalues << "[";
            else
              isninvalidvalues << ",";
            isninvalidvalues << ninvalids[ivalue];
            if(ivalue==ninvalids.size()-1)
              isninvalidvalues << "]";
          }
          else
            isninvalidvalues << ninvalids[0];
        }
        mapString.insert(std::make_pair("ninvalid",isninvalidvalues.str()));
      }

      if(histogram_opt){//aggregate results from multiple inputs, but only calculate for first selected band
        assert(band_opt[0]<(*imit)->nrOfBand());
        nbin=(nbin_opt.size())? nbin_opt[0]:0;
        (*imit)->getMinMax(minValue,maxValue,band_opt[0]);
        if(src_min_opt.size())
          minValue=src_min_opt[0];
        if(src_max_opt.size())
          maxValue=src_max_opt[0];
        if(minValue>=maxValue)
          (*imit)->getMinMax(minValue,maxValue,band_opt[0]);

        if(verbose_opt[0])
          std::cout << "number of valid pixels in image: " << (*imit)->getNvalid(band_opt[0]) << endl << std::endl;
        nsample+=(*imit)->getHistogram(histogramOutput,minValue,maxValue,nbin,band_opt[0],kde_opt[0]);

        //only output for last input file
        if(std::next(imit)==end()){
          ostringstream tmpStream;
          tmpStream.precision(10);
          for(int bin=0;bin<nbin;++bin){
            double binValue=0;
            if(nbin==maxValue-minValue+1)
              binValue=minValue+bin;
            else
              binValue=minValue+static_cast<double>(maxValue-minValue)*(bin+0.5)/nbin;
            tmpStream << binValue << " ";
            if(relative_opt[0]||kde_opt[0])
              tmpStream << 100.0*static_cast<double>(histogramOutput[bin])/static_cast<double>(nsample) << std::endl << std::endl;
            else
              tmpStream << static_cast<double>(histogramOutput[bin]) << std::endl;
          }
          mapString.insert(std::make_pair("histogram",tmpStream.str()));
          // mapString["histogram"]=tmpStream.str();
        }
      }
      if(histogram2d_opt&&size()<2){
        assert(band_opt.size()>1);
        (*imit)->getMinMax(minX,maxX,band_opt[0]);
        (*imit)->getMinMax(minY,maxY,band_opt[1]);
        if(src_min_opt.size()){
          minX=src_min_opt[0];
          minY=src_min_opt[1];
        }
        if(src_max_opt.size()){
          maxX=src_max_opt[0];
          maxY=src_max_opt[1];
        }
        nbin=(nbin_opt.size())? nbin_opt[0]:0;
        if(nbin<=1){
          std::cerr << "Warning: number of bins not defined, calculating bins from min and max value" << std::endl;
          if(minX>=maxX)
            (*imit)->getMinMax(minX,maxX,band_opt[0]);
          if(minY>=maxY)
            (*imit)->getMinMax(minY,maxY,band_opt[1]);

          minValue=(minX<minY)? minX:minY;
          maxValue=(maxX>maxY)? maxX:maxY;
          if(verbose_opt[0])
            std::cout << "min and max values: " << minValue << ", " << maxValue << std::endl << std::endl;
          nbin=maxValue-minValue+1;
        }
        assert(nbin>1);
        double sigma=0;
        //kernel density estimation as in http://en.wikipedia.org/wiki/Kernel_density_estimation
        if(kde_opt[0]){
          assert(band_opt[0]<(*imit)->nrOfBand());
          assert(band_opt[1]<(*imit)->nrOfBand());
          GDALProgressFunc pfnProgress;
          void* pProgressData;
          GDALRasterBand* rasterBand;
          double stdDev1=0;
          double stdDev2=0;
          rasterBand=(*imit)->getRasterBand(band_opt[0]);
          rasterBand->ComputeStatistics(0,&minValue,&maxValue,&meanValue,&stdDev1,pfnProgress,pProgressData);
          rasterBand=(*imit)->getRasterBand(band_opt[1]);
          rasterBand->ComputeStatistics(0,&minValue,&maxValue,&meanValue,&stdDev2,pfnProgress,pProgressData);

          double estimatedSize=1.0*(*imit)->getNvalid(band_opt[0])/down_opt[0]/down_opt[0];
          if(random_opt[0]>0)
            estimatedSize*=random_opt[0]/100.0;
          sigma=1.06*sqrt(stdDev1*stdDev2)*pow(estimatedSize,-0.2);
        }
        assert(nbin);
        if(verbose_opt[0]){
          if(sigma>0)
            std::cout << "calculating 2d kernel density estimate with sigma " << sigma << " for bands " << band_opt[0] << " and " << band_opt[1] << std::endl;
          else
            std::cout << "calculating 2d histogram for bands " << band_opt[0] << " and " << band_opt[1] << std::endl;
          std::cout << "nbin: " << nbin << std::endl;
        }


        vector< vector<double> > output;

        if(maxX<=minX)
          (*imit)->getMinMax(minX,maxX,band_opt[0]);
        if(maxY<=minY)
          (*imit)->getMinMax(minY,maxY,band_opt[1]);

        if(maxX<=minX){
          std::ostringstream s;
          s<<"Error: could not calculate distribution (minX>=maxX)";
          throw(s.str());
        }
        if(maxY<=minY){
          std::ostringstream s;
          s<<"Error: could not calculate distribution (minY>=maxY)";
          throw(s.str());
        }
        output.resize(nbin);
        for(int i=0;i<nbin;++i){
          output[i].resize(nbin);
          for(int j=0;j<nbin;++j)
            output[i][j]=0;
        }
        int binX=0;
        int binY=0;
        vector<double> inputX((*imit)->nrOfCol());
        vector<double> inputY((*imit)->nrOfCol());
        unsigned long int nvalid=0;
        for(unsigned int irow=0;irow<(*imit)->nrOfRow();++irow){
          if(irow%down_opt[0])
            continue;
          (*imit)->readData(inputX,irow,band_opt[0]);
          (*imit)->readData(inputY,irow,band_opt[1]);
          for(unsigned int icol=0;icol<(*imit)->nrOfCol();++icol){
            if(icol%down_opt[0])
              continue;
            if(random_opt[0]>0){
              double p=static_cast<double>(rand())/(RAND_MAX);
              p*=100.0;
              if(p>random_opt[0])
                continue;//do not select for now, go to next column
            }
            if((*imit)->isNoData(inputX[icol]))
              continue;
            if((*imit)->isNoData(inputY[icol]))
              continue;
            ++nvalid;
            if(inputX[icol]>=maxX)
              binX=nbin-1;
            else if(inputX[icol]<=minX)
              binX=0;
            else
              binX=static_cast<int>(static_cast<double>(inputX[icol]-minX)/(maxX-minX)*nbin);
            if(inputY[icol]>=maxY)
              binY=nbin-1;
            else if(inputY[icol]<=minX)
              binY=0;
            else
              binY=static_cast<int>(static_cast<double>(inputY[icol]-minY)/(maxY-minY)*nbin);
            assert(binX>=0);
            assert(binX<output.size());
            assert(binY>=0);
            assert(binY<output[binX].size());
            if(sigma>0){
              //create kde for Gaussian basis function
              //todo: speed up by calculating first and last bin with non-zero contriubtion...
              for(int ibinX=0;ibinX<nbin;++ibinX){
                double centerX=minX+static_cast<double>(maxX-minX)*ibinX/nbin;
                double pdfX=gsl_ran_gaussian_pdf(inputX[icol]-centerX, sigma);
                for(int ibinY=0;ibinY<nbin;++ibinY){
                  //calculate  \integral_ibinX^(ibinX+1)
                  double centerY=minY+static_cast<double>(maxY-minY)*ibinY/nbin;
                  double pdfY=gsl_ran_gaussian_pdf(inputY[icol]-centerY, sigma);
                  output[ibinX][binY]+=pdfX*pdfY;
                }
              }
            }
            else
              ++output[binX][binY];
          }
        }
        if(verbose_opt[0])
          std::cout << "number of valid pixels: " << nvalid << endl;

        ostringstream tmpStream;
        tmpStream.precision(10);
        for(int binX=0;binX<nbin;++binX){
          std::cout << endl;
          for(int binY=0;binY<nbin;++binY){
            double binValueX=0;
            if(nbin==maxX-minX+1)
              binValueX=minX+binX;
            else
              binValueX=minX+static_cast<double>(maxX-minX)*(binX+0.5)/nbin;
            double binValueY=0;
            if(nbin==maxY-minY+1)
              binValueY=minY+binY;
            else
              binValueY=minY+static_cast<double>(maxY-minY)*(binY+0.5)/nbin;

            double value=static_cast<double>(output[binX][binY]);

            if(relative_opt[0])
              value*=100.0/nvalid;

            tmpStream << binValueX << " " << binValueY << " " << value << std::endl;
            // double value=static_cast<double>(output[binX][binY])/nvalid;
            // outputStream << (maxX-minX)*bin/(nbin-1)+minX << " " << (maxY-minY)*bin/(nbin-1)+minY << " " << value << std::endl;
            mapString.insert(std::make_pair("kde",tmpStream.str()));
            // mapString["kde"]=tmpStream.str();
          }
        }
      }
      if(reg_opt&&size()<2){
        if(band_opt.size()<2)
          continue;
        imgreg.setDown(down_opt[0]);
        imgreg.setThreshold(random_opt[0]);
        double c0=0;//offset
        double c1=1;//scale
        double r2=imgreg.getR2(**imit,band_opt[0],band_opt[1],c0,c1,verbose_opt[0]);
        // outputStream << "-c0 " << c0 << " -c1 " << c1 << " -r2 " << r2 << std::endl;
        mapString.insert(std::make_pair("c0",type2string<double>(c0)));
        mapString.insert(std::make_pair("c1",type2string<double>(c1)));
        mapString.insert(std::make_pair("r2",type2string<double>(r2)));
        // mapString["c0"]=type2string<double>(c0);
        // mapString["c1"]=type2string<double>(c1);
        // mapString["r2"]=type2string<double>(r2);
      }
      if(regerr_opt&&size()<2){
        if(band_opt.size()<2)
          continue;
        imgreg.setDown(down_opt[0]);
        imgreg.setThreshold(random_opt[0]);
        double c0=0;//offset
        double c1=1;//scale
        double err=imgreg.getRMSE(**imit,band_opt[0],band_opt[1],c0,c1,verbose_opt[0]);
        // outputStream << "-c0 " << c0 << " -c1 " << c1 << " -rmse " << err << std::endl;
        mapString.insert(std::make_pair("c0",type2string<double>(c0)));
        mapString.insert(std::make_pair("c1",type2string<double>(c1)));
        mapString.insert(std::make_pair("rmse",type2string<double>(err)));
        // mapString["c0"]=type2string<double>(c0);
        // mapString["c1"]=type2string<double>(c1);
        // mapString["rmse"]=type2string<double>(err);
      }
      if(rmse_opt&&size()<2){
        if(band_opt.size()<2)
          continue;
        vector<double> xBuffer((*imit)->nrOfCol());
        vector<double> yBuffer((*imit)->nrOfCol());
        double mse=0;
        double nValid=0;
        double nPixel=(*imit)->nrOfCol()/down_opt[0]*(*imit)->nrOfRow()/down_opt[0];
        for(unsigned int irow;irow<(*imit)->nrOfRow();irow+=down_opt[0]){
          (*imit)->readData(xBuffer,irow,band_opt[0]);
          (*imit)->readData(yBuffer,irow,band_opt[1]);
          for(unsigned int icol;icol<(*imit)->nrOfCol();icol+=down_opt[0]){
            double xValue=xBuffer[icol];
            double yValue=yBuffer[icol];
            if((*imit)->isNoData(xValue)||(*imit)->isNoData(yValue)){
              continue;
            }
            if((*imit)->isNoData(xValue)||(*imit)->isNoData(yValue)){
              continue;
            }
            if(xValue<src_min_opt[0]||xValue>src_max_opt[0]||yValue<src_min_opt[0]||yValue>src_max_opt[0])
              continue;
            ++nValid;
            double e=xValue-yValue;
            if(relative_opt[0])
              e/=yValue;
            mse+=e*e/nPixel;
          }
        }
        double correctNorm=nValid;
        correctNorm/=nPixel;
        mse/=correctNorm;
        // outputStream << " -rmse " << sqrt(mse) << std::endl;
        mapString.insert(std::make_pair("rmse",type2string<double>(sqrt(mse))));
        // mapString["rmse"]=type2string<double>(sqrt(mse));
      }
      if(preg_opt&&size()<2){
        if(band_opt.size()<2)
          continue;
        imgreg.setDown(down_opt[0]);
        imgreg.setThreshold(random_opt[0]);
        double c0=0;//offset
        double c1=1;//scale
        double r2=imgreg.pgetR2(**imit,band_opt[0],band_opt[1],c0,c1,verbose_opt[0]);
        // outputStream << "-c0 " << c0 << " -c1 " << c1 << " -r2 " << r2 << std::endl;
        mapString.insert(std::make_pair("c0",type2string<double>(c0)));
        mapString.insert(std::make_pair("c1",type2string<double>(c1)));
        mapString.insert(std::make_pair("r2",type2string<double>(r2)));
        // mapString["c0"]=type2string<double>(c0);
        // mapString["c1"]=type2string<double>(c1);
        // mapString["r2"]=type2string<double>(r2);
      }
      // imgReader.close();
      ++iimage;
    }
    // if(rmse_opt&&(input_opt.size()>1)){
    //   while(band_opt.size()<input_opt.size())
    //     band_opt.push_back(band_opt[0]);
    //   if(src_min_opt.size()){
    //     while(src_min_opt.size()<input_opt.size())
    //  src_min_opt.push_back(src_min_opt[0]);
    //   }
    //   if(src_max_opt.size()){
    //     while(src_max_opt.size()<input_opt.size())
    //  src_max_opt.push_back(src_max_opt[0]);
    //   }
    //   Jim imgReader1(input_opt[0]);
    //   Jim imgReader2(input_opt[1]);

    //   if(offset_opt.size())
    //     imgReader1.setOffset(offset_opt[0],band_opt[0]);
    //   if(scale_opt.size())
    //     imgReader1.setScale(scale_opt[0],band_opt[0]);
    //   if(offset_opt.size()>1)
    //     imgReader2.setOffset(offset_opt[1],band_opt[1]);
    //   if(scale_opt.size()>1)
    //     imgReader2.setScale(scale_opt[1],band_opt[1]);

    //   for(int inodata=0;inodata<nodata_opt.size();++inodata){
    //     imgReader1.pushNoDataValue(nodata_opt[inodata]);
    //     imgReader2.pushNoDataValue(nodata_opt[inodata]);
    //   }
    //   vector<double> xBuffer(imgReader1.nrOfCol());
    //   vector<double> yBuffer(imgReader2.nrOfCol());
    //   double mse=0;
    //   double nValid=0;
    //   double nPixel=imgReader.nrOfCol()/imgReader.nrOfRow()/down_opt[0]/down_opt[0];
    //   for(unsigned int irow;irow<imgReader1.nrOfRow();irow+=down_opt[0]){
    //     double irow1=irow;
    //     double irow2=0;
    //     double icol1=0;
    //     double icol2=0;
    //     double geoX=0;
    //     double geoY=0;
    //     imgReader1.image2geo(icol1,irow1,geoX,geoY);
    //     imgReader2.geo2image(geoX,geoY,icol2,irow2);
    //     irow2=static_cast<int>(irow2);
    //     imgReader1.readData(xBuffer,irow1,band_opt[0]);
    //     imgReader2.readData(yBuffer,irow2,band_opt[1]);
    //     for(unsigned int icol;icol<imgReader.nrOfCol();icol+=down_opt[0]){
    //  icol1=icol;
    //  imgReader1.image2geo(icol1,irow1,geoX,geoY);
    //  imgReader2.geo2image(geoX,geoY,icol2,irow2);
    //  double xValue=xBuffer[icol1];
    //  double yValue=yBuffer[icol2];
    //  if(imgReader.isNoData(xValue)||imgReader.isNoData(yValue)){
    //    continue;
    //  }
    //  if(xValue<src_min_opt[0]||xValue>src_max_opt[0]||yValue<src_min_opt[1]||yValue>src_max_opt[1])
    //    continue;
    //  ++nValid;
    //  double e=xValue-yValue;
    //  if(relative_opt[0])
    //    e/=yValue;
    //  mse+=e*e/nPixel;
    //     }
    //   }
    //   double correctNorm=nValid;
    //   correctNorm/=nPixel;
    //   mse/=correctNorm;
    //   std::cout << " -rmse " << sqrt(mse) << std::endl;
    // }
    if(reg_opt&&(size()>1)){
      imgreg.setDown(down_opt[0]);
      imgreg.setThreshold(random_opt[0]);
      double c0=0;//offset
      double c1=1;//scale
      while(band_opt.size()<size())
        band_opt.push_back(band_opt[0]);
      if(src_min_opt.size()){
        while(src_min_opt.size()<size())
          src_min_opt.push_back(src_min_opt[0]);
      }
      if(src_max_opt.size()){
        while(src_max_opt.size()<size())
          src_max_opt.push_back(src_max_opt[0]);
      }
      // Jim imgReader1(input_opt[0]);
      // Jim imgReader2(input_opt[1]);

      if(offset_opt.size())
        getImage(0)->setOffset(offset_opt[0],band_opt[0]);
      if(scale_opt.size())
        getImage(0)->setScale(scale_opt[0],band_opt[0]);
      if(offset_opt.size()>1)
        getImage(1)->setOffset(offset_opt[1],band_opt[1]);
      if(scale_opt.size()>1)
        getImage(1)->setScale(scale_opt[1],band_opt[1]);

      for(int inodata=0;inodata<nodata_opt.size();++inodata){
        if(!inodata){
          getImage(0)->GDALSetNoDataValue(nodata_opt[0],band_opt[0]);//only single no data can be set in GDALRasterBand (used for ComputeStatistics)
          getImage(1)->GDALSetNoDataValue(nodata_opt[0]),band_opt[1];//only single no data can be set in GDALRasterBand (used for ComputeStatistics)
        }
        getImage(0)->pushNoDataValue(nodata_opt[inodata]);
        getImage(1)->pushNoDataValue(nodata_opt[inodata]);
      }

      double r2=imgreg.getR2(*(getImage(0)),*(getImage(1)),c0,c1,band_opt[0],band_opt[1],verbose_opt[0]);
      // outputStream << "-c0 " << c0 << " -c1 " << c1 << " -r2 " << r2 << std::endl;
      mapString.insert(std::make_pair("c0",type2string<double>(c0)));
      mapString.insert(std::make_pair("c1",type2string<double>(c1)));
      mapString.insert(std::make_pair("r2",type2string<double>(r2)));
      // mapString["c0"]=type2string<double>(c0);
      // mapString["c1"]=type2string<double>(c1);
      // mapString["r2"]=type2string<double>(r2);
      // imgReader1.close();
      // imgReader2.close();
    }
    if(preg_opt&&(size()>1)){
      imgreg.setDown(down_opt[0]);
      imgreg.setThreshold(random_opt[0]);
      double c0=0;//offset
      double c1=1;//scale
      while(band_opt.size()<size())
        band_opt.push_back(band_opt[0]);
      if(src_min_opt.size()){
        while(src_min_opt.size()<size())
          src_min_opt.push_back(src_min_opt[0]);
      }
      if(src_max_opt.size()){
        while(src_max_opt.size()<size())
          src_max_opt.push_back(src_max_opt[0]);
      }
      // Jim imgReader1(input_opt[0]);
      // Jim imgReader2(input_opt[1]);

      if(offset_opt.size())
        getImage(0)->setOffset(offset_opt[0],band_opt[0]);
      if(scale_opt.size())
        getImage(0)->setScale(scale_opt[0],band_opt[0]);
      if(offset_opt.size()>1)
        getImage(1)->setOffset(offset_opt[1],band_opt[1]);
      if(scale_opt.size()>1)
        getImage(1)->setScale(scale_opt[1],band_opt[1]);

      for(int inodata=0;inodata<nodata_opt.size();++inodata){
        if(!inodata){
          getImage(0)->GDALSetNoDataValue(nodata_opt[0],band_opt[0]);//only single no data can be set in GDALRasterBand (used for ComputeStatistics)
          getImage(1)->GDALSetNoDataValue(nodata_opt[0]),band_opt[1];//only single no data can be set in GDALRasterBand (used for ComputeStatistics)
        }
        getImage(0)->pushNoDataValue(nodata_opt[inodata]);
        getImage(1)->pushNoDataValue(nodata_opt[inodata]);
      }

      double r2=imgreg.pgetR2(*(getImage(0)),*(getImage(1)),c0,c1,band_opt[0],band_opt[1],verbose_opt[0]);
      // outputStream << "-c0 " << c0 << " -c1 " << c1 << " -r2 " << r2 << std::endl;
      mapString.insert(std::make_pair("c0",type2string<double>(c0)));
      mapString.insert(std::make_pair("c1",type2string<double>(c1)));
      mapString.insert(std::make_pair("r2",type2string<double>(r2)));
      // mapString["c0"]=type2string<double>(c0);
      // mapString["c1"]=type2string<double>(c1);
      // mapString["r2"]=type2string<double>(r2);
      // imgReader1.close();
      // imgReader2.close();
    }
    if(regerr_opt&&(size()>1)){
      imgreg.setDown(down_opt[0]);
      imgreg.setThreshold(random_opt[0]);
      double c0=0;//offset
      double c1=1;//scale
      while(band_opt.size()<size())
        band_opt.push_back(band_opt[0]);
      if(src_min_opt.size()){
        while(src_min_opt.size()<size())
          src_min_opt.push_back(src_min_opt[0]);
      }
      if(src_max_opt.size()){
        while(src_max_opt.size()<size())
          src_max_opt.push_back(src_max_opt[0]);
      }
      // Jim imgReader1(input_opt[0]);
      // Jim imgReader2(input_opt[1]);

      if(offset_opt.size())
        getImage(0)->setOffset(offset_opt[0],band_opt[0]);
      if(scale_opt.size())
        getImage(0)->setScale(scale_opt[0],band_opt[0]);
      if(offset_opt.size()>1)
        getImage(1)->setOffset(offset_opt[1],band_opt[1]);
      if(scale_opt.size()>1)
        getImage(1)->setScale(scale_opt[1],band_opt[1]);

      for(int inodata=0;inodata<nodata_opt.size();++inodata){
        if(!inodata){
          getImage(0)->GDALSetNoDataValue(nodata_opt[0],band_opt[0]);//only single no data can be set in GDALRasterBand (used for ComputeStatistics)
          getImage(1)->GDALSetNoDataValue(nodata_opt[0]),band_opt[1];//only single no data can be set in GDALRasterBand (used for ComputeStatistics)
        }
        getImage(0)->pushNoDataValue(nodata_opt[inodata]);
        getImage(1)->pushNoDataValue(nodata_opt[inodata]);
      }

      double err=imgreg.getRMSE(*(getImage(0)),*(getImage(1)),c0,c1,band_opt[0],band_opt[1],verbose_opt[0]);
      // outputStream << "-c0 " << c0 << " -c1 " << c1 << " -rmse " << err << std::endl;
      mapString.insert(std::make_pair("c0",type2string<double>(c0)));
      mapString.insert(std::make_pair("c1",type2string<double>(c1)));
      mapString.insert(std::make_pair("err",type2string<double>(err)));
      // mapString["c0"]=type2string<double>(c0);
      // mapString["c1"]=type2string<double>(c1);
      // mapString["err"]=type2string<double>(err);
      // imgReader1.close();
      // imgReader2.close();
    }
    if(rmse_opt&&(size()>1)){
      imgreg.setDown(down_opt[0]);
      imgreg.setThreshold(random_opt[0]);
      double c0=0;//offset
      double c1=1;//scale
      while(band_opt.size()<size())
        band_opt.push_back(band_opt[0]);
      if(src_min_opt.size()){
        while(src_min_opt.size()<size())
          src_min_opt.push_back(src_min_opt[0]);
      }
      if(src_max_opt.size()){
        while(src_max_opt.size()<size())
          src_max_opt.push_back(src_max_opt[0]);
      }
      // Jim imgReader1(input_opt[0]);
      // Jim imgReader2(input_opt[1]);

      if(offset_opt.size())
        getImage(0)->setOffset(offset_opt[0],band_opt[0]);
      if(scale_opt.size())
        getImage(0)->setScale(scale_opt[0],band_opt[0]);
      if(offset_opt.size()>1)
        getImage(1)->setOffset(offset_opt[1],band_opt[1]);
      if(scale_opt.size()>1)
        getImage(1)->setScale(scale_opt[1],band_opt[1]);

      for(int inodata=0;inodata<nodata_opt.size();++inodata){
        if(!inodata){
          getImage(0)->GDALSetNoDataValue(nodata_opt[0],band_opt[0]);//only single no data can be set in GDALRasterBand (used for ComputeStatistics)
          getImage(1)->GDALSetNoDataValue(nodata_opt[0]),band_opt[1];//only single no data can be set in GDALRasterBand (used for ComputeStatistics)
        }
        getImage(0)->pushNoDataValue(nodata_opt[inodata]);
        getImage(1)->pushNoDataValue(nodata_opt[inodata]);
      }

      double err=imgreg.getRMSE(*(getImage(0)),*(getImage(1)),c0,c1,band_opt[0],band_opt[1],verbose_opt[0]);
      // outputStream << "-rmse " << err << std::endl;
      mapString.insert(std::make_pair("rmse",type2string<double>(err)));
      // mapString["rmse"]=type2string<double>(err);
      // imgReader1.close();
      // imgReader2.close();
    }
    if(histogram2d_opt&&(size()>1)){
      while(band_opt.size()<size())
        band_opt.push_back(band_opt[0]);
      if(src_min_opt.size()){
        while(src_min_opt.size()<size())
          src_min_opt.push_back(src_min_opt[0]);
      }
      if(src_max_opt.size()){
        while(src_max_opt.size()<size())
          src_max_opt.push_back(src_max_opt[0]);
      }
      // Jim imgReader1(input_opt[0]);
      // Jim imgReader2(input_opt[1]);

      if(offset_opt.size())
        getImage(0)->setOffset(offset_opt[0],band_opt[0]);
      if(scale_opt.size())
        getImage(0)->setScale(scale_opt[0],band_opt[0]);
      if(offset_opt.size()>1)
        getImage(1)->setOffset(offset_opt[1],band_opt[1]);
      if(scale_opt.size()>1)
        getImage(1)->setScale(scale_opt[1],band_opt[1]);

      for(int inodata=0;inodata<nodata_opt.size();++inodata){
        if(!inodata){
          getImage(0)->GDALSetNoDataValue(nodata_opt[0],band_opt[0]);//only single no data can be set in GDALRasterBand (used for ComputeStatistics)
          getImage(1)->GDALSetNoDataValue(nodata_opt[0]),band_opt[1];//only single no data can be set in GDALRasterBand (used for ComputeStatistics)
        }
        getImage(0)->pushNoDataValue(nodata_opt[inodata]);
        getImage(1)->pushNoDataValue(nodata_opt[inodata]);
      }

      getImage(0)->getMinMax(minX,maxX,band_opt[0]);
      getImage(1)->getMinMax(minY,maxY,band_opt[1]);

      if(verbose_opt[0]){
        std::cout << "minX: " << minX << endl;
        std::cout << "maxX: " << maxX << endl;
        std::cout << "minY: " << minY << endl;
        std::cout << "maxY: " << maxY << endl;
      }

      if(src_min_opt.size()){
        minX=src_min_opt[0];
        minY=src_min_opt[1];
      }
      if(src_max_opt.size()){
        maxX=src_max_opt[0];
        maxY=src_max_opt[1];
      }

      nbin=(nbin_opt.size())? nbin_opt[0]:0;
      if(nbin<=1){
        std::cerr << "Warning: number of bins not defined, calculating bins from min and max value" << std::endl;
        // getImage(0)->getMinMax(minX,maxX,band_opt[0]);
        // imgReader2.getMinMax(minY,maxY,band_opt[0]);
        if(minX>=maxX)
          getImage(0)->getMinMax(minX,maxX,band_opt[0]);
        if(minY>=maxY)
          getImage(1)->getMinMax(minY,maxY,band_opt[1]);

        minValue=(minX<minY)? minX:minY;
        maxValue=(maxX>maxY)? maxX:maxY;
        if(verbose_opt[0])
          std::cout << "min and max values: " << minValue << ", " << maxValue << std::endl;
        nbin=maxValue-minValue+1;
      }
      assert(nbin>1);
      double sigma=0;
      //kernel density estimation as in http://en.wikipedia.org/wiki/Kernel_density_estimation
      if(kde_opt[0]){
        GDALProgressFunc pfnProgress;
        void* pProgressData;
        GDALRasterBand* rasterBand;
        double stdDev1=0;
        double stdDev2=0;
        rasterBand=getImage(0)->getRasterBand(band_opt[0]);
        rasterBand->ComputeStatistics(0,&minValue,&maxValue,&meanValue,&stdDev1,pfnProgress,pProgressData);
        rasterBand=getImage(1)->getRasterBand(band_opt[0]);
        rasterBand->ComputeStatistics(0,&minValue,&maxValue,&meanValue,&stdDev2,pfnProgress,pProgressData);

        //todo: think of smarter way how to estimate size (nodata!)
        // double estimatedSize=1.0*(*imit)->getNvalid(band_opt[0])/down_opt[0]/down_opt[0];
        double estimatedSize=1.0*getImage(0)->getNvalid(band_opt[0])/down_opt[0]/down_opt[0];
        if(random_opt[0]>0)
          estimatedSize*=random_opt[0]/100.0;
        sigma=1.06*sqrt(stdDev1*stdDev2)*pow(estimatedSize,-0.2);
      }
      assert(nbin);
      if(verbose_opt[0]){
        if(sigma>0)
          std::cout << "calculating 2d kernel density estimate with sigma " << sigma << " for datasets " << getImage(0)->getFileName() << " and " << getImage(1)->getFileName() << std::endl;
        else
          std::cout << "calculating 2d histogram for datasets " << getImage(0)->getFileName() << " and " << getImage(1)->getFileName() << std::endl;
        std::cout << "nbin: " << nbin << std::endl;
      }

      vector< vector<double> > output;

      if(maxX<=minX)
        getImage(0)->getMinMax(minX,maxX,band_opt[0]);
      if(maxY<=minY)
        getImage(1)->getMinMax(minY,maxY,band_opt[1]);

      if(maxX<=minX){
        std::ostringstream s;
        s<<"Error: could not calculate distribution (minX>=maxX)";
        throw(s.str());
      }
      if(maxY<=minY){
        std::ostringstream s;
        s<<"Error: could not calculate distribution (minY>=maxY)";
        throw(s.str());
      }
      if(verbose_opt[0]){
        std::cout << "minX: " << minX << endl;
        std::cout << "maxX: " << maxX << endl;
        std::cout << "minY: " << minY << endl;
        std::cout << "maxY: " << maxY << endl;
      }
      output.resize(nbin);
      for(int i=0;i<nbin;++i){
        output[i].resize(nbin);
        for(int j=0;j<nbin;++j)
          output[i][j]=0;
      }
      int binX=0;
      int binY=0;
      vector<double> inputX(getImage(0)->nrOfCol());
      vector<double> inputY(getImage(1)->nrOfCol());
      double nvalid=0;
      double geoX=0;
      double geoY=0;
      double icol1=0;
      double irow1=0;
      double icol2=0;
      double irow2=0;
      for(unsigned int irow=0;irow<getImage(0)->nrOfRow();++irow){
        if(irow%down_opt[0])
          continue;
        irow1=irow;
        getImage(0)->image2geo(icol1,irow1,geoX,geoY);
        getImage(1)->geo2image(geoX,geoY,icol2,irow2);
        irow2=static_cast<int>(irow2);
        getImage(0)->readData(inputX,irow1,band_opt[0]);
        getImage(1)->readData(inputY,irow2,band_opt[1]);
        for(unsigned int icol=0;icol<getImage(0)->nrOfCol();++icol){
          if(icol%down_opt[0])
            continue;
          icol1=icol;
          if(random_opt[0]>0){
            double p=static_cast<double>(rand())/(RAND_MAX);
            p*=100.0;
            if(p>random_opt[0])
              continue;//do not select for now, go to next column
          }
          if(getImage(0)->isNoData(inputX[icol]))
            continue;
          getImage(0)->image2geo(icol1,irow1,geoX,geoY);
          getImage(1)->geo2image(geoX,geoY,icol2,irow2);
          icol2=static_cast<int>(icol2);
          if(getImage(1)->isNoData(inputY[icol2]))
            continue;
          // ++nvalid;
          if(inputX[icol1]>=maxX)
            binX=nbin-1;
          else if(inputX[icol]<=minX)
            binX=0;
          else
            binX=static_cast<int>(static_cast<double>(inputX[icol1]-minX)/(maxX-minX)*nbin);
          if(inputY[icol2]>=maxY)
            binY=nbin-1;
          else if(inputY[icol2]<=minY)
            binY=0;
          else
            binY=static_cast<int>(static_cast<double>(inputY[icol2]-minY)/(maxY-minY)*nbin);
          assert(binX>=0);
          assert(binX<output.size());
          assert(binY>=0);
          assert(binY<output[binX].size());
          if(sigma>0){
            //create kde for Gaussian basis function
            //todo: speed up by calculating first and last bin with non-zero contriubtion...
            for(int ibinX=0;ibinX<nbin;++ibinX){
              double centerX=minX+static_cast<double>(maxX-minX)*ibinX/nbin;
              double pdfX=gsl_ran_gaussian_pdf(inputX[icol1]-centerX, sigma);
              for(int ibinY=0;ibinY<nbin;++ibinY){
                //calculate  \integral_ibinX^(ibinX+1)
                double centerY=minY+static_cast<double>(maxY-minY)*ibinY/nbin;
                double pdfY=gsl_ran_gaussian_pdf(inputY[icol2]-centerY, sigma);
                output[ibinX][binY]+=pdfX*pdfY;
                nvalid+=pdfX*pdfY;
              }
            }
          }
          else{
            ++output[binX][binY];
            ++nvalid;
          }
        }
      }
      if(verbose_opt[0])
        std::cout << "number of valid pixels: " << nvalid << endl;
      ostringstream tmpStream;
      tmpStream.precision(10);
      for(int binX=0;binX<nbin;++binX){
        std::cout << endl;
        for(int binY=0;binY<nbin;++binY){
          double binValueX=0;
          if(nbin==maxX-minX+1)
            binValueX=minX+binX;
          else
            binValueX=minX+static_cast<double>(maxX-minX)*(binX+0.5)/nbin;
          double binValueY=0;
          if(nbin==maxY-minY+1)
            binValueY=minY+binY;
          else
            binValueY=minY+static_cast<double>(maxY-minY)*(binY+0.5)/nbin;
          double value=static_cast<double>(output[binX][binY]);

          if(relative_opt[0]||kde_opt[0])
            value*=100.0/nvalid;

          tmpStream << binValueX << " " << binValueY << " " << value << std::endl;
          // double value=static_cast<double>(output[binX][binY])/nvalid;
          // cout << (maxX-minX)*bin/(nbin-1)+minX << " " << (maxY-minY)*bin/(nbin-1)+minY << " " << value << std::endl;
        }
      }
      mapString.insert(std::make_pair("histogram2d",tmpStream.str()));
      // mapString["histogram2d"]=tmpStream.str();
      // imgReader1.close();
      // imgReader2.close();
    }

    if(!histogram_opt||histogram2d_opt)
      std::cout << std::endl;

    // if(output_opt.size()){
    //   ofstream outputFile;
    //   outputFile.open(output_opt[0].c_str(),ios::out);
    //   outputFile << outputStream.str();
    // }
    // else{
// #if(JIPLIB_BUILD_WITH_PYTHON==1)
//       PySys_WriteStdout("%s",outputStream.str().c_str());
// #else
      // std::cout << outputStream.str();
// #endif
    // }
    //reset nodata values
    if(nodata_opt.size()){
      size_t iimage=0;
      for(imit=begin();imit!=end();++imit){
          (*imit)->setNoData(orignodata[iimage]);
          ++iimage;
      }
    }
  }
  catch(string predefinedString){
// #if(BUILD_WITH_PYTHON==1)
//     PySys_WriteStdout("%s",predefinedString.c_str());
// #else
    std::cerr << predefinedString << std::endl;
// #endif
    throw;
  }
  // return(*this);
  return(mapString);
}

// int nband=(band_opt.size()) ? band_opt.size() : imgReader.nrOfBand();

// const char* pszMessage;
// void* pProgressArg=NULL;
// GDALProgressFunc pfnProgress=GDALTermProgress;
// double progress=0;
// srand(time(NULL));


// statfactory::StatFactory stat;
// imgregression::ImgRegression imgreg;

// pfnProgress(progress,pszMessage,pProgressArg);
// for(irow=0;irow<classReader.nrOfRow();++irow){
//   if(irow%down_opt[0])
//     continue;
//   // classReader.readData(classBuffer,irow);
//   classReader.readData(classBuffer,irow);
//   double x,y;//geo coordinates
//   double iimg,jimg;//image coordinates in img image
//   for(icol=0;icol<classReader.nrOfCol();++icol){
//     if(icol%down_opt[0])
//  continue;


// if(rand_opt[0]>0){
//   gsl_rng* r=stat.getRandomGenerator(time(NULL));
//   //todo: init random number generator using time...
//   if(verbose_opt[0])
//     std::cout << "generating " << rand_opt[0] << " random numbers: " << std::endl;
//   for(unsigned int i=0;i<rand_opt[0];++i)
//     std::cout << i << " " << stat.getRandomValue(r,randdist_opt[0],randa_opt[0],randb_opt[0]) << std::endl;
// }

// imgreg.setDown(down_opt[0]);
// imgreg.setThreshold(threshold_opt[0]);
// double c0=0;//offset
// double c1=1;//scale
// double err=uncertNodata_opt[0];//start with high initial value in case we do not have first ob	err=imgreg.getRMSE(imgReaderModel1,imgReader,c0,c1,verbose_opt[0]);

//   int nband=band_opt.size();
//   if(band_opt[0]<0)
//     nband=imgReader.nrOfBand();
//   for(int iband=0;iband<nband;++iband){
//     unsigned short band_opt[iband]=(band_opt[0]<0)? iband : band_opt[iband];

//     if(minmax_opt[0]||min_opt[0]||max_opt[0]){
//  assert(band_opt[iband]<imgReader.nrOfBand());
//  if((ulx_opt.size()||uly_opt.size()||lrx_opt.size()||lry_opt.size())&&(imgReader.covers(ulx_opt[0],uly_opt[0],lrx_opt[0],lry_opt[0]))){
//    double uli,ulj,lri,lrj;
//    imgReader.geo2image(ulx_opt[0],uly_opt[0],uli,ulj);
//    imgReader.geo2image(lrx_opt[0],lry_opt[0],lri,lrj);
//    imgReader.getMinMax(static_cast<int>(uli),static_cast<int>(lri),static_cast<int>(ulj),static_cast<int>(lrj),band_opt[iband],minValue,maxValue);
//  }
//  else
//    imgReader.getMinMax(minValue,maxValue,band_opt[iband],true);
//  if(minmax_opt[0])
//    std::cout << "-min " << minValue << " -max " << maxValue << " ";
//  else{
//    if(min_opt[0])
//      std::cout << "-min " << minValue << " ";
//    if(max_opt[0])
//      std::cout << "-max " << maxValue << " ";
//  }
//     }
//   }
//   if(relative_opt[0])
//     hist_opt[0]=true;
//   if(hist_opt[0]){
//     assert(band_opt[0]<imgReader.nrOfBand());
//     unsigned int nbin=(nbin_opt.size())? nbin_opt[0]:0;
//     std::vector<unsigned long int> output;
//     minValue=0;
//     maxValue=0;
//     //todo: optimize such that getMinMax is only called once...
//     imgReader.getMinMax(minValue,maxValue,band_opt[0]);

//     if(src_min_opt.size())
//       minValue=src_min_opt[0];
//     if(src_max_opt.size())
//       maxValue=src_max_opt[0];
//     unsigned long int nsample=imgReader.getHistogram(output,minValue,maxValue,nbin,band_opt[0]);
//     std::cout.precision(10);
//     for(int bin=0;bin<nbin;++bin){
//  double binValue=0;
//  if(nbin==maxValue-minValue+1)
//    binValue=minValue+bin;
//  else
//    binValue=minValue+static_cast<double>(maxValue-minValue)*(bin+0.5)/nbin;
//  std::cout << binValue << " ";
//  if(relative_opt[0])
//    std::cout << 100.0*static_cast<double>(output[bin])/static_cast<double>(nsample) << std::endl;
//  else
//    std::cout << static_cast<double>(output[bin]) << std::endl;
//     }
//   }
