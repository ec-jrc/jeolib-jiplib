/**********************************************************************
jimlist.h: class to read raster files
History
2016/12/02 - Created by Pieter Kempeneers
Change log
***********************************************************************/
#ifndef _JIMLIST_H_
#define _JIMLIST_H_

#include "jim.h"
extern "C" {
#include "config_jiplib.h"
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

namespace jiplib{
  class Jim;
  class JimList : public ImgCollection{
  public:
    JimList(){ImgCollection();};
    ///constructor using vector of images
    JimList(const std::list<std::shared_ptr<jiplib::Jim> > &jimlist);
    ///constructor from an AppFactory
    JimList(app::AppFactory& theApp){open(theApp);};
    ///constructor from a JSON string
    JimList& open(const std::string& strjson);
    ///constructor from an app
    JimList& open(app::AppFactory& theApp);
    ///create a JSON string from a list
    std::string jl2json();
    ///push image to collection
    JimList& pushImage(const std::shared_ptr<jiplib::Jim> imgRaster);
    ///pop image from collection
    JimList& popImage(){ImgCollection::popImage();return(*this);};
    ///get image from collection
    const std::shared_ptr<jiplib::Jim> getImage(int index);

    ///functions from ImgCollection in pktools

    ///select a geographical region based on bounding box
    JimList& selectGeo(double ulx, double uly, double lrx, double lry){ImgCollection::selectGeo(ulx,uly,lrx,lry);return(*this);};
    ///select a geographical region based on a position
    JimList& selectGeo(double x, double y){ImgCollection::selectGeo(x,y);return(*this);};
    ///return an empty collection
    JimList& clean(){ImgCollection::clean();return(*this);};
    ///close all images in collection
    JimList& close(){ImgCollection::close();return(*this);};
    ///Get the no data values of this dataset as a standard template library (stl) vector
    JimList& getNoDataValues(std::vector<double>& noDataValues){ImgCollection::getNoDataValues(noDataValues);return(*this);};
    ///push a no data value
    JimList& pushNoDataValue(double noDataValue){ImgCollection::pushNoDataValue(noDataValue);return(*this);};
    ///set no data values based on a vector
    JimList& setNoData(const std::vector<double>& nodata){ImgCollection::setNoData(nodata);return(*this);};
    ///Clear the no data values
    JimList& clearNoData(){ImgCollection::clearNoData();return(*this);}

    ///composite image only for in memory
    std::shared_ptr<jiplib::Jim> composite(app::AppFactory& app);
    ///crop image only for in memory
    std::shared_ptr<jiplib::Jim> crop(app::AppFactory& app);
    ///stack all images in collection to multiband image (alias for crop)
    std::shared_ptr<jiplib::Jim> stack(app::AppFactory& app);
    ///stack all images in collection to multiband image (alias for crop)
    std::shared_ptr<jiplib::Jim> stack();
    ///create statistical profile from a collection
    std::shared_ptr<jiplib::Jim> statProfile(app::AppFactory& app);
    ///get statistics on image list
    std::multimap<std::string,std::string> getStats(app::AppFactory& app);
    ///validate image based on reference vector dataset (-ref)
    JimList& validate(app::AppFactory& app);

    ///functions from mialib
    //todo: manual for now, but need to be done with Python script
    /* std::shared_ptr<jiplib::Jim> labelConstrainedCCsMultiband(Jim &imgRaster, int ox, int oy, int oz, int r1, int r2); */

    //start insert from fun2method_imagetype_jimlist
    //end insert from fun2method_imagetype_jimlist

    //automatically ported for now, but should probably better via JimList as implemented here:
    /* JimList convertRgbToHsx(int x=0);//from imrgb2hsx */
    /* JimList alphaTreeDissimGet(int alphaMax);//from alphatree */
    /* JimList histoMatchRgb();//from histrgbmatch */
    /* JimList histoMatch3dRgb();//from histrgb3dmatch */
  };
  static JimList createJimList(){JimList alist; return alist;};
  static JimList createJimList(const std::list<std::shared_ptr<jiplib::Jim> > &jimlist){JimList alist(jimlist);return alist;};
  /* JimList createJimList(app::AppFactory &theApp); */
  /* JimList createJimList(const std::list<std::shared_ptr<jiplib::Jim> > &jimlist); */
}
#endif // _JIMLIST_H_
