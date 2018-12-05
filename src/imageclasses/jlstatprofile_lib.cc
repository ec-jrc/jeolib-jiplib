/**********************************************************************
jlstatprofile_lib.cc: program to calculate statistics in temporal or spectral profile
Copyright (c) 2016-2018 European Union (Joint Research Centre)
License EUPLv1.2

This file is part of jiplib
***********************************************************************/
#include <assert.h>
#include <iostream>
#include <string>
#include <fstream>
#include <math.h>
#include <sys/types.h>
#include <stdio.h>
#include "base/Optionjl.h"
#include "base/Vector2d.h"
#include "algorithms/Filter2d.h"
#include "algorithms/Filter.h"
#include "imageclasses/Jim.h"
#include "imageclasses/JimList.h"
#include "algorithms/StatFactory.h"
#include "apps/AppFactory.h"
#include "config_jiplib.h"

using namespace std;
using namespace app;

/**
 * @param function (type: std::string) Statistics function (mean, median, var, stdev, min, max, sum, mode (provide classes), minindex, maxindex, proportion (provide classes), percentile, nvalid
 * @param perc (type: double) Percentile value(s) used for rule percentile
 * @param otype (type: std::string) (default: GDT_Unknown) Data type for output image ({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}). Empty string: inherit type from input image
 * @return shared pointer to image object
 **/
shared_ptr<Jim> JimList::statProfile(app::AppFactory& app){
  shared_ptr<Jim> imgWriter=Jim::createImg();
  statProfile(*imgWriter, app);
  return(imgWriter);
}

/**
 * @param function (type: std::string) Statistics function (mean, median, var, stdev, min, max, sum, mode (provide classes), minindex, maxindex, proportion (provide classes), percentile, nvalid
 * @param perc (type: double) Percentile value(s) used for rule percentile
 * @param otype (type: std::string) (default: GDT_Unknown) Data type for output image ({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}). Empty string: inherit type from input image
 * @param imgWriter output raster profile dataset
 * @param app application specific option arguments
 * @return CE_None if successful, CE_Failure if failed
 **/
void JimList::statProfile(Jim& imgWriter, app::AppFactory& app){
  Optionjl<std::string> function_opt("f", "function", "Statistics function (mean, median, var, stdev, min, max, sum, mode (provide classes), minindex, maxindex, proportion (provide classes), percentile, nvalid");
  Optionjl<double> percentile_opt("perc","perc","Percentile value(s) used for rule percentile",90);
  // Optionjl<short> class_opt("class", "class", "class value(s) to use for mode, proportion");
  Optionjl<double> nodata_opt("nodata", "nodata", "nodata value");
  Optionjl<std::string>  otype_opt("ot", "otype", "Data type for output image ({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}). Empty string: inherit type from input image","GDT_Unknown");
  // Optionjl<short> down_opt("d", "down", "down sampling factor. Use value 1 for no downsampling). Use value n>1 for downsampling (aggregation)", 1);
  Optionjl<short> verbose_opt("v", "verbose", "verbose mode if > 0", 0,2);

  // percentile_opt.setHide(1);
  // class_opt.setHide(1);
  otype_opt.setHide(1);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=function_opt.retrieveOption(app);
    percentile_opt.retrieveOption(app);
    nodata_opt.retrieveOption(app);
    otype_opt.retrieveOption(app);
    verbose_opt.retrieveOption(app);
    if(!doProcess){
      cout << endl;
      std::ostringstream helpStream;
      helpStream << "short option -h shows basic options only, use long option --help to show all options" << std::endl;
      throw(helpStream.str());//help was invoked, stop processing
    }

    std::vector<std::string> badKeys;
    app.badKeys(badKeys);
    if(badKeys.size()){
      std::ostringstream errorStream;
      if(badKeys.size()>1)
        errorStream << "Error: unknown keys: ";
      else
        errorStream << "Error: unknown key: ";
      for(int ikey=0;ikey<badKeys.size();++ikey){
        errorStream << badKeys[ikey] << " ";
      }
      errorStream << std::endl;
      throw(errorStream.str());
    }

    if(empty()){
      std::string errorString="Input collection is empty";
      throw(errorString);
    }

    if(function_opt.empty()){
      std::ostringstream errorStream;
      errorStream << "Error: no function selected, use option -f" << endl;
      throw(errorStream.str());
    }
    std::vector<double> parallel_percentile;//used for parallel processing
    if(function_opt.countSubstring("percentile")){
      while(function_opt.countSubstring("percentile")<percentile_opt.size())
        function_opt.push_back("percentile");
      parallel_percentile.resize(function_opt.size());
      std::vector<double>::const_iterator percit=percentile_opt.begin();
      for(int ifunction=0;ifunction<function_opt.size();++ifunction){
        if(function_opt[ifunction].compare("percentile"))
          parallel_percentile[ifunction]=0;
        else if(percit!=percentile_opt.end())
          parallel_percentile[ifunction]=*(percit++);
        else{
          //we should never end up here...
          std::ostringstream errorStream;
          errorStream << "Error: percentiles inconsistent" << endl;
          throw(errorStream.str());
        }
      }
    }

    GDALDataType theType=string2GDAL(otype_opt[0]);
    if(theType==GDT_Unknown)
      theType=this->front()->getGDALDataType();

    if(verbose_opt[0])
      std::cout << std::endl << "Output pixel type:  " << GDALGetDataTypeName(theType) << endl;

    if(verbose_opt[0])
      cout << "Calculating statistic metrics: " << function_opt.size() << endl;
    // imgWriter.open(output_opt[0],this->nrOfCol(),this->nrOfRow(),function_opt.size(),theType,imageType,memory_opt[0],option_opt);
    //todo: expand for collections that have different image dimensions and geotransforms?
    imgWriter.open(this->front()->nrOfCol(),this->front()->nrOfRow(),function_opt.size(),theType);
    imgWriter.setProjection(this->front()->getProjection());
    double gt[6];
    this->front()->getGeoTransform(gt);
    imgWriter.setGeoTransform(gt);

    if(nodata_opt.size()){
      for(int iband=0;iband<imgWriter.nrOfBand();++iband)
        imgWriter.setNoData(nodata_opt);
    }

    //todo:replace assert with exception
    assert(imgWriter.nrOfBand()==function_opt.size());
    Vector2d<double> lineInput(this->size(),this->front()->nrOfCol());
    assert(imgWriter.nrOfCol()==this->front()->nrOfCol());
    Vector2d<double> lineOutput(function_opt.size(),imgWriter.nrOfCol());
    statfactory::StatFactory stat;
    stat.setNoDataValues(nodata_opt);
    const char* pszMessage;
    void* pProgressArg=NULL;
    GDALProgressFunc pfnProgress=GDALTermProgress;
    double progress=0;
    MyProgressFunc(progress,pszMessage,pProgressArg);
    Vector2d<double> lineInput_transposed(this->front()->nrOfCol(),size());//band interleaved by pixel
    for(unsigned int y=0;y<this->front()->nrOfRow();++y){
      std::list<std::shared_ptr<Jim> >::const_iterator imit=begin();
      unsigned int ifile=0;
      for(imit=begin();imit!=end();++imit){
        (*imit)->readData(lineInput[ifile],y);
        ++ifile;
      }
      for(int x=0;x<imgWriter.nrOfCol();++x){
        lineInput_transposed[x]=lineInput.selectCol(x);
        if(nodata_opt.size())
          stat.eraseNoData(lineInput_transposed[x]);
      }
      //todo: check if multiple thresholds can be handled in parallel
      // int ithreshold=0;//threshold to use for percentiles
#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
      for(int imethod=0;imethod<function_opt.size();++imethod){
        int x=0;
        vector<double> lineInput_tmp;
        size_t stride=1;
        switch(filter::Filter::getFilterType(function_opt[imethod])){
        case(filter::first):
          lineOutput[imethod]=lineInput.front();
          break;
        case(filter::last):
          lineOutput[imethod]=lineInput.back();
          break;
        case(filter::nvalid):
          for(x=0;x<imgWriter.nrOfCol();++x)
            lineOutput[imethod][x]=stat.nvalid(lineInput_transposed[x]);
          break;
        case(filter::median):
          for(x=0;x<imgWriter.nrOfCol();++x){
            // lineInput_tmp.assign(lineInput_transposed[x].begin(),lineInput_transposed[x].end());
            lineInput_tmp=lineInput_transposed[x];
            gsl_sort(&(lineInput_tmp[0]),stride,lineInput_tmp.size());
            lineOutput[imethod][x]=gsl_stats_median_from_sorted_data(&(lineInput_tmp[0]),stride,lineInput_tmp.size());
          }
          break;
        case(filter::min):
          for(x=0;x<imgWriter.nrOfCol();++x)
            lineOutput[imethod][x]=gsl_stats_min(&(lineInput_transposed[x][0]),stride,lineInput_transposed[x].size());
          break;
        case(filter::minindex):
          for(x=0;x<imgWriter.nrOfCol();++x)
            lineOutput[imethod][x]=gsl_stats_min_index(&(lineInput_transposed[x][0]),stride,lineInput_transposed[x].size());
          break;
        case(filter::max):
          for(x=0;x<imgWriter.nrOfCol();++x)
            lineOutput[imethod][x]=gsl_stats_max(&(lineInput_transposed[x][0]),stride,lineInput_transposed[x].size());
          break;
        case(filter::maxindex):
          for(x=0;x<imgWriter.nrOfCol();++x)
            lineOutput[imethod][x]=gsl_stats_max_index(&(lineInput_transposed[x][0]),stride,lineInput_transposed[x].size());
          break;
        case(filter::sum):
          for(x=0;x<imgWriter.nrOfCol();++x)
            lineOutput[imethod][x]=size()*gsl_stats_mean(&(lineInput_transposed[x][0]),stride,lineInput_transposed[x].size());
          break;
        case(filter::var):
          for(x=0;x<imgWriter.nrOfCol();++x)
            lineOutput[imethod][x]=gsl_stats_variance(&(lineInput_transposed[x][0]),stride,lineInput_transposed[x].size());
          break;
        case(filter::stdev):
          for(x=0;x<imgWriter.nrOfCol();++x)
            lineOutput[imethod][x]=gsl_stats_sd(&(lineInput_transposed[x][0]),stride,lineInput_transposed[x].size());
          break;
        case(filter::mean):{
          for(x=0;x<imgWriter.nrOfCol();++x)
            lineOutput[imethod][x]=gsl_stats_mean(&(lineInput_transposed[x][0]),stride,lineInput_transposed[x].size());
          break;
        }
        case(filter::percentile):{
          // double threshold=(ithreshold<percentile_opt.size())? percentile_opt[ithreshold] : percentile_opt[0];
          double threshold=parallel_percentile[imethod];
          // double threshold=percentile_opt[0];
          for(x=0;x<imgWriter.nrOfCol();++x){
            lineInput_tmp=lineInput_transposed[x];
            gsl_sort(&(lineInput_tmp[0]),stride,lineInput_tmp.size());
            lineOutput[imethod][x]=gsl_stats_quantile_from_sorted_data(&(lineInput_tmp[0]),stride,lineInput_tmp.size(),threshold/100.0);
          }
          // ++ithreshold;
          break;
        }
        default:
          std::string errorString="method not supported";
          throw(errorString);
          break;
        }
        imgWriter.writeData(lineOutput[imethod],y,imethod);
      }
      progress=(1.0+y)/imgWriter.nrOfRow();
      MyProgressFunc(progress,pszMessage,pProgressArg);
    }
  }
  catch(string predefinedString){
    std::cout << predefinedString << std::endl;
    throw;
  }
}

/**
 * @param function (type: std::string) Statistics function (mean, median, var, stdev, min, max, sum, mode (provide classes), minindex, maxindex, proportion (provide classes), percentile, nvalid
 * @param perc (type: double) Percentile value(s) used for rule percentile
 * @param otype (type: std::string) (default: GDT_Unknown) Data type for output image ({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}). Empty string: inherit type from input image
 * @return shared pointer to image object
 **/
shared_ptr<Jim> Jim::statProfile(app::AppFactory& app){
  try{
    shared_ptr<Jim> imgWriter=Jim::createImg();
    statProfile(*imgWriter, app);
    return(imgWriter);
  }
  catch(string helpString){
    cerr << helpString << endl;
    throw;
  }
}

/**
 * @param function (type: std::string) Statistics function (mean, median, var, stdev, min, max, sum, mode (provide classes), minindex, maxindex, proportion (provide classes), percentile, nvalid
 * @param perc (type: double) Percentile value(s) used for rule percentile
 * @param otype (type: std::string) (default: GDT_Unknown) Data type for output image ({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}). Empty string: inherit type from input image
 * @param imgWriter output raster profile dataset
 * @param app application specific option arguments
 * @return CE_None if successful, CE_Failure if failed
 **/
void Jim::statProfile(Jim& imgWriter, app::AppFactory& app){
  Optionjl<std::string> function_opt("f", "function", "Statistics function (mean, median, var, stdev, min, max, sum, mode, minindex, maxindex, proportion (provide classes), percentile, nvalid");
  Optionjl<double> percentile_opt("perc","perc","Percentile value(s) used for rule percentile",90);
  // Optionjl<short> class_opt("class", "class", "class value(s) to use for mode, proportion");
  Optionjl<double> nodata_opt("nodata", "nodata", "nodata value)");
  Optionjl<std::string>  otype_opt("ot", "otype", "Data type for output image ({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}). Empty string: inherit type from input image","GDT_Unknown");
  // Optionjl<short> down_opt("d", "down", "down sampling factor. Use value 1 for no downsampling). Use value n>1 for downsampling (aggregation)", 1);
  Optionjl<short> verbose_opt("v", "verbose", "verbose mode if > 0", 0,2);

  // percentile_opt.setHide(1);
  // class_opt.setHide(1);
  otype_opt.setHide(1);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=function_opt.retrieveOption(app);
    percentile_opt.retrieveOption(app);
    nodata_opt.retrieveOption(app);
    otype_opt.retrieveOption(app);
    verbose_opt.retrieveOption(app);
    if(!doProcess){
      cout << endl;
      std::ostringstream helpStream;
      helpStream << "short option -h shows basic options only, use long option --help to show all options" << std::endl;
      throw(helpStream.str());//help was invoked, stop processing
    }

    std::vector<std::string> badKeys;
    app.badKeys(badKeys);
    if(badKeys.size()){
      std::ostringstream errorStream;
      if(badKeys.size()>1)
        errorStream << "Error: unknown keys: ";
      else
        errorStream << "Error: unknown key: ";
      for(int ikey=0;ikey<badKeys.size();++ikey){
        errorStream << badKeys[ikey] << " ";
      }
      errorStream << std::endl;
      throw(errorStream.str());
    }

    if(function_opt.empty()){
      std::ostringstream errorStream;
      errorStream << "Error: no function selected, use option -f" << endl;
      throw(errorStream.str());
    }
    std::vector<double> parallel_percentile;//used for parallel processing
    if(function_opt.countSubstring("percentile")){
      while(function_opt.countSubstring("percentile")<percentile_opt.size())
        function_opt.push_back("percentile");
      parallel_percentile.resize(function_opt.size());
      std::vector<double>::const_iterator percit=percentile_opt.begin();
      for(int ifunction=0;ifunction<function_opt.size();++ifunction){
        if(function_opt[ifunction].compare("percentile"))
          parallel_percentile[ifunction]=0;
        else if(percit!=percentile_opt.end())
          parallel_percentile[ifunction]=*(percit++);
        else{
          //we should never end up here...
          std::ostringstream errorStream;
          errorStream << "Error: percentiles inconsistent" << endl;
          throw(errorStream.str());
        }
      }
    }

    GDALDataType theType=string2GDAL(otype_opt[0]);
    if(theType==GDT_Unknown)
      theType=this->getGDALDataType();

    if(verbose_opt[0])
      std::cout << std::endl << "Output pixel type:  " << GDALGetDataTypeName(theType) << endl;

    if(verbose_opt[0])
      cout << "Calculating statistic metrics: " << function_opt.size() << endl;
    // imgWriter.open(output_opt[0],this->nrOfCol(),this->nrOfRow(),function_opt.size(),theType,imageType,memory_opt[0],option_opt);
    imgWriter.open(this->nrOfCol(),this->nrOfRow(),function_opt.size(),theType);
    imgWriter.setProjection(this->getProjection());
    double gt[6];
    this->getGeoTransform(gt);
    imgWriter.setGeoTransform(gt);

    if(nodata_opt.size()){
      for(int iband=0;iband<imgWriter.nrOfBand();++iband)
        imgWriter.GDALSetNoDataValue(nodata_opt[0],iband);
    }

    //todo:replace assert with exception
    assert(imgWriter.nrOfBand()==function_opt.size());
    Vector2d<double> lineInput(this->nrOfBand(),this->nrOfCol());
    assert(imgWriter.nrOfCol()==this->nrOfCol());
    Vector2d<double> lineOutput(function_opt.size(),imgWriter.nrOfCol());
    statfactory::StatFactory stat;
    stat.setNoDataValues(nodata_opt);
    const char* pszMessage;
    void* pProgressArg=NULL;
    GDALProgressFunc pfnProgress=GDALTermProgress;
    double progress=0;
    MyProgressFunc(progress,pszMessage,pProgressArg);
    Vector2d<double> lineInput_transposed(nrOfCol(),nrOfBand());//band interleaved by pixel
    for(unsigned int y=0;y<this->nrOfRow();++y){
      for(unsigned int iband=0;iband<this->nrOfBand();++iband)
        this->readData(lineInput[iband],y,iband);
      for(int x=0;x<nrOfCol();++x){
        lineInput_transposed[x]=lineInput.selectCol(x);
        if(nodata_opt.size())
          stat.eraseNoData(lineInput_transposed[x]);
      }
      //todo: check if multiple thresholds can be handled in parallel
      // int ithreshold=0;//threshold to use for percentiles
#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
      for(int imethod=0;imethod<function_opt.size();++imethod){
        int x=0;
        vector<double> lineInput_tmp;
        size_t stride=1;
        switch(filter::Filter::getFilterType(function_opt[imethod])){
        case(filter::first):
          lineOutput[imethod]=lineInput.front();
          break;
        case(filter::last):
          lineOutput[imethod]=lineInput.back();
          break;
        case(filter::median):
          for(x=0;x<imgWriter.nrOfCol();++x){
            lineInput_tmp=lineInput_transposed[x];
            gsl_sort(&(lineInput_tmp[0]),stride,lineInput_tmp.size());
            lineOutput[imethod][x]=gsl_stats_median_from_sorted_data(&(lineInput_tmp[0]),stride,lineInput_tmp.size());
          }
          break;
        case(filter::min):
          for(x=0;x<nrOfCol();++x)
            lineOutput[imethod][x]=gsl_stats_min(&(lineInput_transposed[x][0]),stride,lineInput_transposed[x].size());
          break;
        case(filter::minindex):
          for(x=0;x<imgWriter.nrOfCol();++x)
            lineOutput[imethod][x]=gsl_stats_min_index(&(lineInput_transposed[x][0]),stride,lineInput_transposed[x].size());
          break;
        case(filter::max):
          for(x=0;x<nrOfCol();++x)
            lineOutput[imethod][x]=gsl_stats_max(&(lineInput_transposed[x][0]),stride,lineInput_transposed[x].size());
          break;
        case(filter::maxindex):
          for(x=0;x<imgWriter.nrOfCol();++x)
            lineOutput[imethod][x]=gsl_stats_max_index(&(lineInput_transposed[x][0]),stride,lineInput_transposed[x].size());
          break;
        case(filter::sum):
          for(x=0;x<nrOfCol();++x)
            lineOutput[imethod][x]=nrOfBand()*gsl_stats_mean(&(lineInput_transposed[x][0]),stride,lineInput_transposed[x].size());
          break;
        case(filter::var):
          for(x=0;x<nrOfCol();++x)
            lineOutput[imethod][x]=gsl_stats_variance(&(lineInput_transposed[x][0]),stride,lineInput_transposed[x].size());
          break;
        case(filter::stdev):
          for(x=0;x<nrOfCol();++x)
            lineOutput[imethod][x]=gsl_stats_sd(&(lineInput_transposed[x][0]),stride,lineInput_transposed[x].size());
          break;
        case(filter::mean):{
          for(x=0;x<nrOfCol();++x)
            lineOutput[imethod][x]=gsl_stats_mean(&(lineInput_transposed[x][0]),stride,lineInput_transposed[x].size());
          break;
        }
        case(filter::percentile):{
          // double threshold=(ithreshold<percentile_opt.size())? percentile_opt[ithreshold] : percentile_opt[0];
          double threshold=parallel_percentile[imethod];
          for(x=0;x<nrOfCol();++x){
            lineInput_tmp=lineInput_transposed[x];
            gsl_sort(&(lineInput_tmp[0]),stride,lineInput_tmp.size());
            lineOutput[imethod][x]=gsl_stats_quantile_from_sorted_data(&(lineInput_tmp[0]),stride,lineInput_tmp.size(),threshold/100.0);
          }
          // ++ithreshold;
          break;
        }
        default:
          std::string errorString="method not supported";
          throw(errorString);
          break;
        }
        imgWriter.writeData(lineOutput[imethod],y,imethod);
      }
      progress=(1.0+y)/imgWriter.nrOfRow();
      MyProgressFunc(progress,pszMessage,pProgressArg);
    }
  }
  catch(string predefinedString){
    std::cout << predefinedString << std::endl;
    throw;
  }
}
