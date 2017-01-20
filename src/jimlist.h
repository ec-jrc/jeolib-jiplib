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
#include "config.h"
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
    ///push image to collection
    JimList& pushImage(const std::shared_ptr<jiplib::Jim> imgRaster);
    //CPLErr pushImage(const std::shared_ptr<jiplib::Jim> imgRaster);
    ///pop image from collection
    JimList& popImage();
    ///get image from collection
    const std::shared_ptr<jiplib::Jim> getImage(int index);
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
    ///functions from mialib
    JimList imrgb2hsx(int x=0);
    JimList alphaTree(int alphaMax);
    JimList histrgbmatch();
    JimList histrgb3dmatch();
  };
}
#endif // _JIMLIST_H_