/**********************************************************************
ImgList.cc: class to read raster files using GDAL API library
Copyright (C) 2008-2016 Pieter Kempeneers

This file is part of pktools

pktools is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

pktools is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with pktools.  If not, see <http://www.gnu.org/licenses/>.
***********************************************************************/
#include <vector>
#include <list>
#include <string>
#include <iostream>
#include "base/Vector2d.h"
#include "base/Optionpk.h"
#include "algorithms/StatFactory.h"
#include "algorithms/Egcs.h"
#include "apps/AppFactory.h"
#include "ImgList.h"

//todo: namespace jiplib
using namespace std;
using namespace app;

ImgList::ImgList(const std::list<std::shared_ptr<ImgRaster> > &jimlist){
  std::list<std::shared_ptr<ImgRaster> >::const_iterator lit=jimlist.begin();
  for(lit=jimlist.begin();lit!=jimlist.end();++lit)
    pushImage(*lit);
}

//   ///time period covering the image list (check http://www.boost.org/doc/libs/1_55_0/doc/html/date_time/examples.html#date_time.examples.time_periods for how to use boost period)
// boost::posix_time::time_period ImgList::getTimePeriod(){
//   if(m_time.size()){
//     std::vector<boost::posix_time::time_period>::iterator tit=m_time.begin();
//     boost::posix_time::time_period timePeriod=*(tit++);
//     while(tit!=m_time.end()){
//       timePeriod.span(*(tit++));
//     }
//     return(timePeriod);
//   }
// }

ImgList::ImgList(unsigned int theSize){
  for(unsigned int iimg=0;iimg<theSize;++iimg){
    this->emplace_back(ImgRaster::createImg());
    /* this->emplace_back(new(ImgRaster)); */
  }
}

///constructor using a json string coming from a custom colllection
ImgList& ImgList::open(const std::string& strjson){
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
      std::shared_ptr<ImgRaster> theImage=ImgRaster::createImg(theApp);
      pushImage(theImage);
    }
  }
  return(*this);
}

ImgList& ImgList::open(app::AppFactory& theApp){
  Optionpk<std::string> json_opt("json", "json", "The json object");
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
  // ImgList(std::string(""));
}

std::string ImgList::jl2json(){
  Json::Value custom;
  custom["size"]=static_cast<int>(size());
  int iimg=0;
  for(std::list<std::shared_ptr<ImgRaster> >::iterator lit=begin();lit!=end();++lit){
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
ImgList& ImgList::selectGeo(double ulx, double uly, double lrx, double lry){
  /* std::vector<std::shared_ptr<ImgRaster>>::iterator it=begin(); */
  std::list<std::shared_ptr<ImgRaster>>::iterator it=begin();
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

ImgList& ImgList::selectGeo(double x, double y){
  /* std::vector<std::shared_ptr<ImgRaster>>::iterator it=begin(); */
  std::list<std::shared_ptr<ImgRaster>>::iterator it=begin();
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

ImgList& ImgList::close(){
  /* for(std::vector<std::shared_ptr<ImgRaster>>::iterator it=begin();it!=end();++it) */
  for(std::list<std::shared_ptr<ImgRaster>>::iterator it=begin();it!=end();++it)
    (*it)->close();
  return(*this);
}

/**
 * @param ulx upper left coordinate in x
 * @param uly upper left coordinate in y
 * @param lrx lower left coordinate in x
 * @param lry lower left coordinate in y
 **/
const ImgList& ImgList::getBoundingBox(double& ulx, double& uly, double& lrx, double& lry) const{
  // std::vector<std::shared_ptr<ImgRaster> >::const_iterator it=begin();
  std::list<std::shared_ptr<ImgRaster> >::const_iterator it=begin();
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
  return(*this);
}

/**
 * @param x,y georeferenced coordinates in x and y
 * @return true if image covers the georeferenced location
 **/
bool ImgList::covers(double x, double  y, OGRCoordinateTransformation *poCT) const
{
  double theULX, theULY, theLRX, theLRY;
  getBoundingBox(theULX,theULY,theLRX,theLRY);
  double ximg=x;
  double yimg=y;
  if(poCT){
    if(!poCT->Transform(1,&ximg,&yimg)){
      std::ostringstream errorStream;
      errorStream << "Error: cannot apply OGRCoordinateTransformation in ImgList::covers (1)" << std::endl;
      throw(errorStream.str());
    }
  }
  return((ximg > theULX)&&
         (ximg < theLRX)&&
         (yimg < theULY)&&
         (yimg >theLRY));
}

bool ImgList::covers(double ulx, double  uly, double lrx, double lry, bool all, OGRCoordinateTransformation *poCT) const
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
      errorStream << "Error: cannot apply OGRCoordinateTransformation in ImgList::covers (2)" << std::endl;
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

bool ImgList::covers(const ImgRaster& imgRaster, bool all) const{
  //image bounding box in SRS of the raster
  double img_ulx,img_uly,img_lrx,img_lry;
  imgRaster.getBoundingBox(img_ulx,img_uly,img_lrx,img_lry);
  OGRSpatialReference listSpatialRef(getImage(0)->getProjectionRef().c_str());
  OGRSpatialReference rasterSpatialRef(imgRaster.getProjectionRef().c_str());
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
ImgList& ImgList::getNoDataValues(std::vector<double>& noDataValues)
{
  if(m_noDataValues.size()){
    noDataValues=m_noDataValues;
  }
  return(*this);
}

/**
 * @param noDataValue no data value to be pushed for this dataset
 * @return number of no data values in this dataset
 **/
ImgList& ImgList::pushNoDataValue(double noDataValue)
{
  if(find(m_noDataValues.begin(),m_noDataValues.end(),noDataValue)==m_noDataValues.end())
    m_noDataValues.push_back(noDataValue);
  return(*this);
}

