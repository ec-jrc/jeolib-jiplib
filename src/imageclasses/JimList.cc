/**********************************************************************
JimList.cc: class to read raster files using GDAL API library
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
#include "Json_compat.h"

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
JimList& JimList::open(const std::string& strjson) {
    Json::Value custom;
    std::istringstream sin(strjson);
    
    // Modern JsonCpp uses operators or CharReader, but sin >> custom 
    // is generally well-supported across versions.
    sin >> custom;

    // 1. Robustly access the "size" member
    int size = 0;
    try {
        size = json_util::get_member(custom, "size").asInt();
    } catch (...) {
        // Handle cases where "size" might be missing or not an int
        return *this; 
    }

    for (int iimg = 0; iimg < size; ++iimg) {
        // 2. Convert index to string safely
        std::string indexKey = std::to_string(iimg);
        
        // 3. Access the image object using the helper
        Json::Value image = json_util::get_member(custom, indexKey);
        
        // 4. Access the "path" member using the helper
        std::string filename = json_util::get_member(image, "path").asString();

        if (!filename.empty()) {
            app::AppFactory theApp;
            theApp.setLongOption("filename", filename);
            
            try {
                std::shared_ptr<Jim> theImage = Jim::createImg(theApp);
                pushImage(theImage);
            } catch (const std::exception& e) {
                // Log or handle individual image load failures 
                // to prevent one bad path from crashing the whole list
            }
        }
    }
    
    return *this;
}

JimList& JimList::open(app::AppFactory& theApp) {
    Optionjl<std::string> json_opt("json", "json", "The json object");
    bool doProcess = true;

    try {
        doProcess = json_opt.retrieveOption(theApp);
    }
    catch (const std::string& predefinedString) {
        std::cout << predefinedString << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Standard exception: " << e.what() << std::endl;
    }

    if (!doProcess) {
        std::cout << std::endl;
        // Using runtime_error is more robust for Python/C++ boundary
        throw std::runtime_error("exception thrown due to help info");
    }

    std::vector<std::string> badKeys;
    theApp.badKeys(badKeys);
    if (!badKeys.empty()) {
        std::ostringstream errorStream;
        errorStream << "Error: unknown key" << (badKeys.size() > 1 ? "s: " : ": ");

        for (const auto& key : badKeys) {
            errorStream << key << " ";
        }
        errorStream << std::endl;
        throw std::runtime_error(errorStream.str());
    }

    if (json_opt.empty()) {
        throw std::runtime_error("Error: json string is empty");
    }

    // This calls your newly robust open(const std::string&)
    return open(json_opt[0]);
}

std::string JimList::jl2json() {
    Json::Value custom;

    // 1. Set global size using helper
    json_util::get_member(custom, "size") = static_cast<int>(size());

    int iimg = 0;
    // Using a modern range-based loop if your list supports it,
    // otherwise sticking to the iterator for compatibility.
    for (auto lit = begin(); lit != end(); ++lit) {
        Json::Value image;

        // 2. Access "path" safely
        json_util::get_member(image, "path") = (*lit)->getFileName();

        std::string wktString = (*lit)->getProjectionRef();
        std::string key("EPSG");
        std::size_t foundEPSG = wktString.rfind(key);

        // Robustness: Check if EPSG was actually found to avoid npos crashes
        if (foundEPSG != std::string::npos) {
            std::string fromEPSG = wktString.substr(foundEPSG);
            std::size_t foundFirstDigit = fromEPSG.find_first_of("0123456789");
            std::size_t foundLastDigit = fromEPSG.find_last_of("0123456789");

            if (foundFirstDigit != std::string::npos && foundLastDigit != std::string::npos) {
                std::string epsgString = fromEPSG.substr(foundFirstDigit, foundLastDigit - foundFirstDigit + 1);
                // 3. Access "epsg" safely
                json_util::get_member(image, "epsg") = std::stoi(epsgString);
            }
        }

        // 4. Use to_string and helper for the dynamic image keys ("0", "1", etc.)
        std::string indexKey = std::to_string(iimg++);
        json_util::get_member(custom, indexKey) = image;
    }

    // 5. Handle StreamWriterBuilder specifically
    Json::StreamWriterBuilder builder;
    builder[Json::String("indentation")] = "";

    return Json::writeString(builder, custom);
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
#if GDAL_VERSION_MAJOR > 2
  listSpatialRef.SetAxisMappingStrategy(OSRAxisMappingStrategy::OAMS_TRADITIONAL_GIS_ORDER);
#endif
  OGRSpatialReference rasterSpatialRef=imgRaster.getSpatialRef();
#if GDAL_VERSION_MAJOR > 2
  rasterSpatialRef.SetAxisMappingStrategy(OSRAxisMappingStrategy::OAMS_TRADITIONAL_GIS_ORDER);
#endif
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

