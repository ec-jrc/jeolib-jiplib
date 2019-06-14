/**********************************************************************
JimList.h: class to read raster files using GDAL API library
Author(s): Pieter.Kempeneers@ec.europa.eu
Copyright (c) 2016-2018 European Union (Joint Research Centre)
License EUPLv1.2

This file is part of jiplib
***********************************************************************/
#ifndef _JIMLIST_H_
#define _JIMLIST_H_

#include <string>
#include <vector>
#include <list>
#include <memory>
// #include "boost/date_time/posix_time/posix_time.hpp"
/* #include "ImgReaderOgr.h" */
#include "Jim.h"
#include "VectorOgr.h"
#include "apps/AppFactory.h"

#include "config_jiplib.h"

#if MIALIB == 1
extern "C" {
#include "mialib/mialib_swig.h"
#include "mialib/mialib_convolve.h"
#include "mialib/mialib_dem.h"
#include "mialib/mialib_dist.h"
#include "mialib/mialib_erodil.h"
#include "mialib/mialib_format.h"
#include "mialib/mialib_geodesy.h"
#include "mialib/mialib_geometry.h"
#include "mialib/mialib_hmt.h"
#include "mialib/mialib_imem.h"
#include "mialib/mialib_io.h"
#include "mialib/mialib_label.h"
#include "mialib/mialib_miscel.h"
#include "mialib/mialib_opclo.h"
#include "mialib/mialib_pointop.h"
#include "mialib/mialib_proj.h"
#include "mialib/mialib_segment.h"
#include "mialib/mialib_stats.h"
#include "mialib/op.h"
}
#endif

//todo: namespace jiplib
namespace app{
class AppFactory;
}

class Jim;
class VectorOgr;
/**
   This class is used to store a list of raster images
**/
/* class JimList : public std::vector<std::shared_ptr<Jim> > */
class JimList : public std::list<std::shared_ptr<Jim> >
{
public:
  enum CRULE_TYPE {overwrite=0, maxndvi=1, maxband=2, minband=3, validband=4, mean=5, mode=6, median=7,sum=8,minallbands=9,maxallbands=10,stdev=11};
  ///default constructor
  JimList(){};// : std::vector<Jim*>() {};
  ///copy constructor
  JimList(const JimList &jimlist){
    /* std::vector<std::shared_ptr<Jim> >::const_iterator pimit=jimlist.begin(); */
    std::list<std::shared_ptr<Jim> >::const_iterator pimit=jimlist.begin();
    for(pimit=jimlist.begin();pimit!=jimlist.end();++pimit)
      pushImage(*pimit);
    std::vector<double> nodata;
    jimlist.getNoDataValues(nodata);
    setNoData(nodata);
  }
  /* JimList(const std::vector<std::shared_ptr<Jim> > &coll){ */
  /*   std::vector<std::shared_ptr<Jim> >::const_iterator pimit=coll.begin(); */
  /*   for(pimit=coll.begin();pimit!=coll.end();++pimit) */
  /*     pushImage(*pimit); */
  /* } */
  JimList(unsigned int theSize);
  ///constructor using vector of images
  JimList(const std::list<std::shared_ptr<Jim> > &jimlist);
  ///constructor from an AppFactory
  JimList(app::AppFactory& theApp){open(theApp);};
  ///constructor from a JSON string
  JimList& open(const std::string& strjson);
  ///constructor from an app
  JimList& open(app::AppFactory& theApp);
  ///create a JSON string from a list
  std::string jl2json();
  ///destructor
  ~JimList(){};

  // JimList(const JimList&) = default;
  // JimList& operator=(const JimList&) = default;

  ///get bounding box of image list
  void getBoundingBox(double& ulx, double& uly, double& lrx, double& lry) const;
  ///get upper left x coordinate of image list
  double getUlx() const {double ulx, uly, lrx, lry;getBoundingBox(ulx,uly,lrx,lry);return(ulx);};
  ///get upper left y coordinate of image list
  double getUly() const {double ulx, uly, lrx, lry;getBoundingBox(ulx,uly,lrx,lry);return(uly);};
  ///get lower right x coordinate of image list
  double getLrx() const {double ulx, uly, lrx, lry;getBoundingBox(ulx,uly,lrx,lry);return(lrx);};
  ///get lower right y coordinate of image list
  double getLry() const {double ulx, uly, lrx, lry;getBoundingBox(ulx,uly,lrx,lry);return(lry);};
  // ///get begin and last time of image list
  // boost::posix_time::time_period getTimePeriod();

  // ///filter list according to period
  // void filterTime(const boost::posix_time::time_period& thePeriod){
  //   unsigned int index=0;
  //   std::vector<std::shared_ptr<Jim>>::iterator it=begin();
  //   std::vector<boost::posix_time::time_period>::iterator tit=m_time.begin();
  //   while(it!=end()&&tit!=m_time.end()){
  //     if(thePeriod.contains(m_time[m_index])){
  //       ++it;
  //       ++tit;
  //     }
  //     else{
  //       it=erase(it);
  //       tit=m_time.erase(tit);
  //     }
  //   }
  //   m_index=0;
  // };
  ///filter list according to bounding box
  JimList& selectGeo(double ulx, double uly, double lrx, double lry);
  ///filter list according to position
  JimList& selectGeo(double x, double y);
  ///push image to list
  JimList& pushImage(const std::shared_ptr<Jim> imgRaster){
    this->emplace_back(imgRaster);
    return(*this);
  };
  ///pop image from list
  JimList& popImage(){
    this->pop_back();
    return(*this);
  };
  ///get image from list (not recommended, because unlike random access vector iterator the complexity for a list iterator is linear in index)
  std::shared_ptr<Jim> getImage(int index) const{
    if(index>=this->size()){
      std::cerr << "Error: index>=list size" << std::endl;
      return(0);
    }
    std::list<std::shared_ptr<Jim>>::const_iterator it=begin();
    std::advance(it,index);
    return(*it);
  }
  size_t getSize() const{return size();};
  // ///push image to list with corresponding period
  // void pushImage(std::shared_ptr<Jim> imgRaster, boost::posix_time::time_period imgPeriod){
  //   this->emplace_back(imgRaster);
  //   // m_time.push_back(imgPeriod);
  // };
  // ///push image period
  // void pushTime(boost::posix_time::time_period imgPeriod){
  //   m_time.push_back(imgPeriod);
  // };
  // ///set image periods for list
  // void setTime(const std::vector<boost::posix_time::time_period>& timeVector){
  //   m_time=timeVector;
  // };
  // ///get image periods for list
  // void getTime(std::vector<boost::posix_time::time_period>& timeVector){
  //   timeVector=m_time;
  // };
  ///Check if a geolocation is covered by this dataset. Only the bounding box is checked, irrespective of no data values.
  bool covers(double x, double  y, OGRCoordinateTransformation *poCT=0) const;
  ///Check if a region of interest is (partially) covered by this dataset. Only the bounding box is checked, irrespective of no data values.
  bool covers(double ulx, double  uly, double lrx, double lry, bool all=false, OGRCoordinateTransformation *poCT=0) const;
  ///Check if an image raster dataset is (partially) covered by this dataset. Only the bounding box is checked, irrespective of no data values.
  bool covers(const Jim &imgRaster, bool all=false) const;
  // std::shared_ptr<Jim> getNextImage(){
  //   if(m_index<size())
  //     return(this->at(m_index++));
  //   else
  //     return(0);
  // }
  // std::shared_ptr<Jim> getNextImage(boost::posix_time::time_period& imgPeriod){
  //   if(m_index<size()){
  //     if(m_index<m_time.size())
  //       imgPeriod=m_time[m_index];
  //     return(this->at(m_index++));
  //   }
  //   else
  //     return(0);
  // }
  /* JimList& resetIterator(){m_index=0;return(*this);}; */
  JimList& clear(){std::list<std::shared_ptr<Jim> >::clear();return(*this);};
  JimList& close();
  ///Get the no data values of this dataset as a standard template library (stl) vector
  void getNoDataValues(std::vector<double>& noDataValues) const;
  ///Check if value is nodata in this dataset
  bool isNoData(double value) const{if(m_noDataValues.empty()) return false;else return find(m_noDataValues.begin(),m_noDataValues.end(),value)!=m_noDataValues.end();};
  ///Push a no data value for this dataset
  void pushNoDataValue(double noDataValue);
  ///Set the no data values of this dataset using a standard template library (stl) vector as input
  void setNoData(const std::vector<double>& nodata){m_noDataValues=nodata;};
  ///Clear the no data values
  void clearNoData(){m_noDataValues.clear();}

  ///composite image
  void composite(Jim& imgWriter, app::AppFactory& app);
  ///composite image only for in memory
  std::shared_ptr<Jim> composite(app::AppFactory& app);
  ///crop image
  /* JimList& crop(Jim& imgWriter, app::AppFactory& app); */
  ///crop image only for in memory
  /* std::shared_ptr<Jim> crop(app::AppFactory& app); */
  ///stack image
  void stackBand(Jim& imgWriter, app::AppFactory& app);
  ///stack image
  void stackBand(Jim& imgWriter){app::AppFactory app;return stackBand(imgWriter,app);};
  ///stack image only for in memory (alias for crop)
  std::shared_ptr<Jim> stackBand(app::AppFactory& app);
  ///stack image only for in memory (alias for crop)
  std::shared_ptr<Jim> stackBand(){app::AppFactory app;return stackBand(app);};
  ///stack image
  void stackPlane(Jim& imgWriter, app::AppFactory& app);
  ///stack image
  void stackPlane(Jim& imgWriter){app::AppFactory app;return stackPlane(imgWriter,app);};
  ///stack image only for in memory (alias for crop)
  std::shared_ptr<Jim> stackPlane(app::AppFactory& app);
  ///stack image only for in memory (alias for crop)
  std::shared_ptr<Jim> stackPlane(){app::AppFactory app;return stackPlane(app);};
  ///stat profile image
  void statProfile(Jim& imgWriter, app::AppFactory& app);
  ///stat profile image only for in memory
  std::shared_ptr<Jim> statProfile(app::AppFactory& app);
  ///get statistics
  std::multimap<std::string,std::string> getStats(app::AppFactory& app);
  //JimList& getStats(app::AppFactory& app);
  ///validate image based on reference vector dataset (-ref)
  JimList& validate(app::AppFactory& app);
  ///extract vector layer from list
  CPLErr extractOgr(VectorOgr& sampleReader, VectorOgr& ogrWriter, app::AppFactory& app);
  ///extract vector layer from list only for in memory
  std::shared_ptr<VectorOgr> extractOgr(VectorOgr& sampleReader, app::AppFactory& app);
  size_t extractOgrMem(VectorOgr& sampleReader, std::vector<unsigned char> &vbytes, app::AppFactory& app);
  //start insert from fun2method_imagetype_jimlist
  //end insert from fun2method_imagetype_jimlist
private:
  std::vector<double> m_noDataValues;
  // std::vector<boost::posix_time::time_period> m_time;
};
static JimList createJimList(){JimList alist; return alist;};
static JimList createJimList(const std::list<std::shared_ptr<Jim> > &jimlist){JimList alist(jimlist);return alist;};
#endif // _JIIMLIST_H_
