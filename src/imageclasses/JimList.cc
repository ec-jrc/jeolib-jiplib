/**********************************************************************
JimList.cc: class to read raster files using GDAL API library
Author(s): Pieter.Kempeneers@ec.europa.eu
Copyright (c) 2016-2019 European Union (Joint Research Centre)
License EUPLv1.2

This file is part of jiplib
***********************************************************************/
#include <vector>
#include <list>
#include <string>
#include <iostream>
#include "base/Vector2d.h"
#include "base/Optionjl.h"
#include "algorithms/StatFactory.h"
#include "algorithms/Egcs.h"
#include "apps/AppFactory.h"
#include "JimList.h"

//todo: namespace jiplib
using namespace std;
using namespace app;

JimList::JimList(const std::list<std::shared_ptr<Jim> > &jimlist){
  std::list<std::shared_ptr<Jim> >::const_iterator lit=jimlist.begin();
  for(lit=jimlist.begin();lit!=jimlist.end();++lit)
    pushImage(*lit);
}

//   ///time period covering the image list (check http://www.boost.org/doc/libs/1_55_0/doc/html/date_time/examples.html#date_time.examples.time_periods for how to use boost period)
// boost::posix_time::time_period JimList::getTimePeriod(){
//   if(m_time.size()){
//     std::vector<boost::posix_time::time_period>::iterator tit=m_time.begin();
//     boost::posix_time::time_period timePeriod=*(tit++);
//     while(tit!=m_time.end()){
//       timePeriod.span(*(tit++));
//     }
//     return(timePeriod);
//   }
// }

JimList::JimList(unsigned int theSize){
  for(unsigned int iimg=0;iimg<theSize;++iimg){
    this->emplace_back(Jim::createImg());
    /* this->emplace_back(new(Jim)); */
  }
}

///constructor using a json string coming from a custom colllection
JimList& JimList::open(const std::string& strjson){
  Json::Value custom;
  Json::Reader reader;
  bool parsedSuccess=reader.parse(strjson,custom,false);
  if(parsedSuccess){
    for(int iimg=0;iimg<custom["size"].asInt();++iimg){
      std::ostringstream os;
      os << iimg;
      Json::Value image=custom[os.str()];
      std::string filename=image["path"].asString();
      //todo: open without reading?
      app::AppFactory theApp;
      theApp.setLongOption("filename",filename);
      std::shared_ptr<Jim> theImage=Jim::createImg(theApp);
      pushImage(theImage);
    }
  }
  return(*this);
}

JimList& JimList::open(app::AppFactory& theApp){
  Optionjl<std::string> json_opt("json", "json", "The json object");
  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=json_opt.retrieveOption(theApp);
  }
  catch(std::string predefinedString){
    std::cout << predefinedString << std::endl;
  }
  if(!doProcess){
    std::cout << std::endl;
    std::ostringstream helpStream;
    helpStream << "exception thrown due to help info";
    throw(helpStream.str());//help was invoked, stop processing
  }

  std::vector<std::string> badKeys;
  theApp.badKeys(badKeys);
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
  if(json_opt.empty()){
    std::string errorString="Error: json string is empty";
    throw(errorString);
  }
  return(open(json_opt[0]));
  // JimList(std::string(""));
}

std::string JimList::jl2json(){
  Json::Value custom;
  custom["size"]=static_cast<int>(size());
  int iimg=0;
  for(std::list<std::shared_ptr<Jim> >::iterator lit=begin();lit!=end();++lit){
    Json::Value image;
    image["path"]=(*lit)->getFileName();
    std::string wktString=(*lit)->getProjectionRef();
    std::string key("EPSG");
    std::size_t foundEPSG=wktString.rfind(key);
    std::string fromEPSG=wktString.substr(foundEPSG);//EPSG","32633"]]'
    std::size_t foundFirstDigit=fromEPSG.find_first_of("0123456789");
    std::size_t foundLastDigit=fromEPSG.find_last_of("0123456789");
    std::string epsgString=fromEPSG.substr(foundFirstDigit,foundLastDigit-foundFirstDigit+1);
    image["epsg"]=atoi(epsgString.c_str());
    std::ostringstream os;
    os << iimg++;
    custom[os.str()]=image;
  }
  Json::FastWriter fastWriter;
  return(fastWriter.write(custom));
}
JimList& JimList::selectGeo(double ulx, double uly, double lrx, double lry){
  /* std::vector<std::shared_ptr<Jim>>::iterator it=begin(); */
  std::list<std::shared_ptr<Jim>>::iterator it=begin();
  // std::vector<boost::posix_time::time_period>::iterator tit=m_time.begin();
  while(it!=end()){
    if((*it)->covers(ulx, uly, lrx, lry)){
      ++it;
      // if(tit!=m_time.end())
      //   ++tit;
    }
    else{
      it=erase(it);
      // if(tit!=m_time.end())
      //   tit=m_time.erase(tit);
    }
  }
  return(*this);
};

JimList& JimList::selectGeo(double x, double y){
  /* std::vector<std::shared_ptr<Jim>>::iterator it=begin(); */
  std::list<std::shared_ptr<Jim>>::iterator it=begin();
  // std::vector<boost::posix_time::time_period>::iterator tit=m_time.begin();
  while(it!=end()){
    if((*it)->covers(x,y)){
      ++it;
      // if(tit!=m_time.end())
      //   ++tit;
    }
    else{
      it=erase(it);
      // if(tit!=m_time.end())
      //   tit=m_time.erase(tit);
    }
  }
  return(*this);
};

JimList& JimList::close(){
  /* for(std::vector<std::shared_ptr<Jim>>::iterator it=begin();it!=end();++it) */
  for(std::list<std::shared_ptr<Jim>>::iterator it=begin();it!=end();++it)
    (*it)->close();
  return(*this);
}

/**
 * @param ulx upper left coordinate in x
 * @param uly upper left coordinate in y
 * @param lrx lower left coordinate in x
 * @param lry lower left coordinate in y
 **/
void JimList::getBoundingBox(double& ulx, double& uly, double& lrx, double& lry) const{
  // std::vector<std::shared_ptr<Jim> >::const_iterator it=begin();
  std::list<std::shared_ptr<Jim> >::const_iterator it=begin();
  if(it!=end())
    (*(it++))->getBoundingBox(ulx,uly,lrx,lry);
  while(it!=end()){
    double imgulx,imguly,imglrx,imglry;
    (*(it++))->getBoundingBox(imgulx,imguly,imglrx,imglry);
    ulx=(ulx>imgulx)? imgulx : ulx;
    uly=(uly<imguly)? imguly : uly;
    lrx=(lrx<imglrx)? imglrx : lrx;
    lry=(lry>imglry)? imglry : lry;
  }
}

/**
 * @param x,y georeferenced coordinates in x and y
 * @return true if image covers the georeferenced location
 **/
bool JimList::covers(double x, double  y, OGRCoordinateTransformation *poCT) const
{
  double theULX, theULY, theLRX, theLRY;
  getBoundingBox(theULX,theULY,theLRX,theLRY);
  double ximg=x;
  double yimg=y;
  if(poCT){
    if(!poCT->Transform(1,&ximg,&yimg)){
      std::ostringstream errorStream;
      errorStream << "Error: cannot apply OGRCoordinateTransformation in JimList::covers (1)" << std::endl;
      throw(errorStream.str());
    }
  }
  return((ximg > theULX)&&
         (ximg < theLRX)&&
         (yimg < theULY)&&
         (yimg >theLRY));
}

bool JimList::covers(double ulx, double  uly, double lrx, double lry, bool all, OGRCoordinateTransformation *poCT) const
{
  double theULX, theULY, theLRX, theLRY;
  getBoundingBox(theULX,theULY,theLRX,theLRY);
  double ulximg=ulx;
  double ulyimg=uly;
  double lrximg=lrx;
  double lryimg=lry;
  if(poCT){
    std::vector<double> xvector(4);//ulx,urx,llx,lrx
    std::vector<double> yvector(4);//uly,ury,lly,lry
    xvector[0]=ulximg;
    xvector[1]=lrximg;
    xvector[2]=ulximg;
    xvector[3]=lrximg;
    yvector[0]=ulyimg;
    yvector[1]=ulyimg;
    yvector[2]=lryimg;
    yvector[3]=lryimg;
    if(!poCT->Transform(xvector.size(),&xvector[0],&yvector[0])){
      std::ostringstream errorStream;
      errorStream << "Error: cannot apply OGRCoordinateTransformation in JimList::covers (2)" << std::endl;
      throw(errorStream.str());
    }
    ulximg=std::min(xvector[0],xvector[2]);
    lrximg=std::max(xvector[1],xvector[3]);
    ulyimg=std::max(yvector[0],yvector[1]);
    lryimg=std::min(yvector[2],yvector[3]);
  }
  if(all)
    return((theULX<ulximg)&&(theULY>ulyimg)&&(theLRX>lrximg)&&(theLRY<lryimg));
  else
    return((ulximg < theLRX)&&(lrximg > theULX)&&(lryimg < theULY)&&(ulyimg > theLRY));
}

bool JimList::covers(const Jim& imgRaster, bool all) const{
  //image bounding box in SRS of the raster
  double img_ulx,img_uly,img_lrx,img_lry;
  imgRaster.getBoundingBox(img_ulx,img_uly,img_lrx,img_lry);
  OGRSpatialReference listSpatialRef=getImage(0)->getSpatialRef();
  OGRSpatialReference rasterSpatialRef=imgRaster.getSpatialRef();
  OGRCoordinateTransformation *raster2list = OGRCreateCoordinateTransformation(&rasterSpatialRef, &listSpatialRef);
  if(listSpatialRef.IsSame(&rasterSpatialRef)){
    raster2list=0;
  }
  else{
    if(!raster2list){
      std::ostringstream errorStream;
      errorStream << "Error: cannot create OGRCoordinateTransformation raster to list" << std::endl;
      throw(errorStream.str());
    }
  }
  return covers(img_ulx,img_uly,img_lrx,img_lry,all,raster2list);
}

/**
 * @param noDataValues standard template library (stl) vector containing no data values
 * @return number of no data values in this dataset
 **/
void JimList::getNoDataValues(std::vector<double>& noDataValues) const
{
  if(m_noDataValues.size()){
    noDataValues=m_noDataValues;
  }
}

/**
 * @param noDataValue no data value to be pushed for this dataset
 * @return number of no data values in this dataset
 **/
void JimList::pushNoDataValue(double noDataValue)
{
  if(find(m_noDataValues.begin(),m_noDataValues.end(),noDataValue)==m_noDataValues.end())
    m_noDataValues.push_back(noDataValue);
}

