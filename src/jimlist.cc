/**********************************************************************
jimlist.cc: class to read raster files
History
2016/12/05 - Created by Pieter Kempeneers
Change log
***********************************************************************/
#include "base/Optionpk.h"
#include "json/json.h"
#include "jimlist.h"

using namespace jiplib;
///constructor using vector of images
JimList::JimList(const std::list<std::shared_ptr<jiplib::Jim> > &jimlist) : ImgCollection(){
  std::list<std::shared_ptr<jiplib::Jim> >::const_iterator lit=jimlist.begin();
  for(lit=jimlist.begin();lit!=jimlist.end();++lit)
    pushImage(*lit);
  // for(int ijim=0;ijim<jimVector.size();++ijim){
    // pushImage(jimVector[ijim]);
  // }
}

///construtor using a json string coming from a custom colllection
///example:
///str = '{"size": 1, "0": {"epsg": 4326, "path":"/eos/jeodpp/data/base/Soil/GLOBAL/HWSD/VER1-2/Data/GeoTIFF/hwsd.tif"} }'
CPLErr JimList::open(const std::string& strjson){
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
      std::shared_ptr<jiplib::Jim> theImage=jiplib::Jim::createImg(theApp);
      pushImage(theImage);
    }
  }
  return(CE_None);
}

CPLErr JimList::open(app::AppFactory& theApp){
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
  // JimList(std::string(""));
}

///push image to collection
//CPLErr JimList::pushImage(const std::shared_ptr<jiplib::Jim> imgRaster){
JimList& JimList::pushImage(const std::shared_ptr<jiplib::Jim> imgRaster){
  ImgCollection::pushImage(imgRaster);
  return(*this);
  /* this->emplace_back(imgRaster); */
  /* return(CE_None); */
};
///pop image from collection
JimList& JimList::popImage(){
  ImgCollection::popImage();
  return(*this);
};

///get image from collection
const std::shared_ptr<jiplib::Jim> JimList::getImage(int index){
  return(std::dynamic_pointer_cast<jiplib::Jim>(ImgCollection::getImage(index)));
}

std::string JimList::jl2json(){
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
    image["epsg"]=epsgString;
    std::ostringstream os;
    os << iimg++;
    custom[os.str()]=image;
  }
  Json::FastWriter fastWriter;
  return(fastWriter.write(custom));
}

///composite image only for in memory
/**
 * @param input (type: std::string) Input image file(s). If input contains multiple images, a multi-band output is created
 * @param output (type: std::string) Output image file
 * @param oformat (type: std::string) (default: GTiff) Output image format (see also gdal_translate).
 * @param co (type: std::string) Creation option for output file. Multiple options can be specified.
 * @param scale (type: double) output=scale*input+offset
 * @param offset (type: double) output=scale*input+offset
 * @param mem (type: unsigned long) (default: 0) Buffer size (in MB) to read image data blocks in memory
 * @param band (type: unsigned int) band index(es) to crop (leave empty if all bands must be retained)
 * @param dx (type: double) Output resolution in x (in meter) (empty: keep original resolution)
 * @param dy (type: double) Output resolution in y (in meter) (empty: keep original resolution)
 * @param extent (type: std::string) get boundary from extent from polygons in vector file
 * @param crop_to_cutline (type: bool) (default: 0) Crop the extent of the target dataset to the extent of the cutline.
 * @param eo (type: std::string) special extent options controlling rasterization: ATTRIBUTE|CHUNKYSIZE|ALL_TOUCHED|BURN_VALUE_FROM|MERGE_ALG, e.g., -eo ATTRIBUTE=fieldname
 * @param mask (type: std::string) Use the specified file as a validity mask.
 * @param msknodata (type: float) (default: 0) Mask value not to consider for composite.
 * @param mskband (type: unsigned int) (default: 0) Mask band to read (0 indexed)
 * @param ulx (type: double) (default: 0) Upper left x value bounding box
 * @param uly (type: double) (default: 0) Upper left y value bounding box
 * @param lrx (type: double) (default: 0) Lower right x value bounding box
 * @param lry (type: double) (default: 0) Lower right y value bounding box
 * @param crule (type: std::string) (default: overwrite) Composite rule (overwrite, maxndvi, maxband, minband, mean, mode (only for byte images), median, sum, maxallbands, minallbands, stdev
 * @param cband (type: unsigned int) (default: 0) band index used for the composite rule (e.g., for ndvi, use --cband=0 --cband=1 with 0 and 1 indices for red and nir band respectively
 * @param srcnodata (type: double) invalid value(s) for input raster dataset
 * @param bndnodata (type: unsigned int) (default: 0) Band(s) in input image to check if pixel is valid (used for srcnodata, min and max options)
 * @param min (type: double) flag values smaller or equal to this value as invalid.
 * @param max (type: double) flag values larger or equal to this value as invalid.
 * @param dstnodata (type: double) (default: 0) nodata value to put in output raster dataset if not valid or out of bounds.
 * @param resampling-method (type: std::string) (default: near) Resampling method (near: nearest neighbor, bilinear: bi-linear interpolation).
 * @param otype (type: std::string) Data type for output image ({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}). Empty string: inherit type from input image
 * @param a_srs (type: std::string) Override the spatial reference for the output file (leave blank to copy from input file, use epsg:3035 to use European projection and force to European grid
 * @param file (type: short) (default: 0) write number of observations (1) or sequence nr of selected file (2) for each pixels as additional layer in composite
 * @param weight (type: short) (default: 1) Weights (type: short) for the composite, use one weight for each input file in same order as input files are provided). Use value 1 for equal weights.
 * @param class (type: short) (default: 0) classes for multi-band output image: each band represents the number of observations for one specific class. Use value 0 for no multi-band output image.
 * @param ct (type: std::string) color table file with 5 columns: id R G B ALFA (0: transparent, 255: solid)
 * @param description (type: std::string) Set image description
 * @param align (type: bool) (default: 0) Align output bounding box to input image
 * @return output image
 **/
std::shared_ptr<jiplib::Jim> JimList::composite(app::AppFactory& app){
  /* std::shared_ptr<jiplib::Jim> imgWriter=Jim::createImg(); */
  std::shared_ptr<jiplib::Jim> imgWriter=std::make_shared<jiplib::Jim>();
  ImgCollection::composite(*imgWriter, app);
  return(imgWriter);
}
/**
 * @param input (type: std::string) Input image file(s). If input contains multiple images, a multi-band output is created
 * @param output (type: std::string) Output image file
 * @param oformat (type: std::string) (default: GTiff) Output image format (see also gdal_translate).
 * @param co (type: std::string) Creation option for output file. Multiple options can be specified.
 * @param a_srs (type: std::string) Override the spatial reference for the output file (leave blank to copy from input file, use epsg:3035 to use European projection and force to European grid
 * @param mem (type: unsigned long) (default: 0) Buffer size (in MB) to read image data blocks in memory
 * @param a_srs (type: std::string) Override the projection for the output file (leave blank to copy from input file, use epsg:3035 to use European projection and force to European grid
 * @param ulx (type: double) (default: 0) Upper left x value bounding box
 * @param uly (type: double) (default: 0) Upper left y value bounding box
 * @param lrx (type: double) (default: 0) Lower right x value bounding box
 * @param lry (type: double) (default: 0) Lower right y value bounding box
 * @param band (type: unsigned int) band index to crop (leave empty to retain all bands)
 * @param startband (type: unsigned int) Start band sequence number
 * @param endband (type: unsigned int) End band sequence number
 * @param autoscale (type: double) scale output to min and max, e.g., --autoscale 0 --autoscale 255
 * @param otype (type: std::string) Data type for output image ({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}). Empty string: inherit type from input image
 * @param oformat (type: std::string) (default: GTiff) Output image format (see also gdal_translate).
 * @param ct (type: std::string) color table (file with 5 columns: id R G B ALFA (0: transparent, 255: solid)
 * @param dx (type: double) Output resolution in x (in meter) (empty: keep original resolution)
 * @param dy (type: double) Output resolution in y (in meter) (empty: keep original resolution)
 * @param resampling-method (type: std::string) (default: near) Resampling method (near: nearest neighbor, bilinear: bi-linear interpolation).
 * @param extent (type: std::string) get boundary from extent from polygons in vector file
 * @param crop_to_cutline (type: bool) (default: 0) Crop the extent of the target dataset to the extent of the cutline.
 * @param eo (type: std::string) special extent options controlling rasterization: ATTRIBUTE|CHUNKYSIZE|ALL_TOUCHED|BURN_VALUE_FROM|MERGE_ALG, e.g., -eo ATTRIBUTE=fieldname
 * @param mask (type: std::string) Use the the specified file as a validity mask (0 is nodata).
 * @param msknodata (type: double) (default: 0) Mask value not to consider for crop.
 * @param mskband (type: unsigned int) (default: 0) Mask band to read (0 indexed)
 * @param x (type: double) x-coordinate of image center to crop (in meter)
 * @param y (type: double) y-coordinate of image center to crop (in meter)
 * @param nx (type: double) image size in x to crop (in meter)
 * @param ny (type: double) image size in y to crop (in meter)
 * @param ns (type: unsigned int) number of samples  to crop (in pixels)
 * @param nl (type: unsigned int) number of lines to crop (in pixels)
 * @param scale (type: double) output=scale*input+offset
 * @param offset (type: double) output=scale*input+offset
 * @param nodata (type: double) Nodata value to put in image if out of bounds.
 * @param description (type: std::string) Set image description
 * @param align (type: bool) (default: 0) Align output bounding box to input image
 * @return output image
 **/
std::shared_ptr<jiplib::Jim> JimList::crop(app::AppFactory& app){
  /* std::shared_ptr<jiplib::Jim> imgWriter=Jim::createImg(); */
  std::shared_ptr<jiplib::Jim> imgWriter=std::make_shared<jiplib::Jim>();
  ImgCollection::crop(*imgWriter, app);
  return(imgWriter);
}
///stack all images in collection to multiband image (alias for crop)
/**
 * @param input (type: std::string) Input image file(s). If input contains multiple images, a multi-band output is created
 * @param output (type: std::string) Output image file
 * @param oformat (type: std::string) (default: GTiff) Output image format (see also gdal_translate).
 * @param co (type: std::string) Creation option for output file. Multiple options can be specified.
 * @param a_srs (type: std::string) Override the spatial reference for the output file (leave blank to copy from input file, use epsg:3035 to use European projection and force to European grid
 * @param mem (type: unsigned long) (default: 0) Buffer size (in MB) to read image data blocks in memory
 * @param a_srs (type: std::string) Override the projection for the output file (leave blank to copy from input file, use epsg:3035 to use European projection and force to European grid
 * @param ulx (type: double) (default: 0) Upper left x value bounding box
 * @param uly (type: double) (default: 0) Upper left y value bounding box
 * @param lrx (type: double) (default: 0) Lower right x value bounding box
 * @param lry (type: double) (default: 0) Lower right y value bounding box
 * @param band (type: unsigned int) band index to stack (leave empty to retain all bands)
 * @param startband (type: unsigned int) Start band sequence number
 * @param endband (type: unsigned int) End band sequence number
 * @param otype (type: std::string) Data type for output image ({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}). Empty string: inherit type from input image
 * @param oformat (type: std::string) (default: GTiff) Output image format (see also gdal_translate).
 * @param ct (type: std::string) color table (file with 5 columns: id R G B ALFA (0: transparent, 255: solid)
 * @param dx (type: double) Output resolution in x (in meter) (empty: keep original resolution)
 * @param dy (type: double) Output resolution in y (in meter) (empty: keep original resolution)
 * @param resampling-method (type: std::string) (default: near) Resampling method (near: nearest neighbor, bilinear: bi-linear interpolation).
 * @param extent (type: std::string) get boundary from extent from polygons in vector file
 * @param crop_to_cutline (type: bool) (default: 0) Crop the extent of the target dataset to the extent of the cutline.
 * @param eo (type: std::string) special extent options controlling rasterization: ATTRIBUTE|CHUNKYSIZE|ALL_TOUCHED|BURN_VALUE_FROM|MERGE_ALG, e.g., -eo ATTRIBUTE=fieldname
 * @param mask (type: std::string) Use the the specified file as a validity mask (0 is nodata).
 * @param msknodata (type: double) (default: 0) Mask value not to consider for crop.
 * @param mskband (type: unsigned int) (default: 0) Mask band to read (0 indexed)
 * @param x (type: double) x-coordinate of image center to crop (in meter)
 * @param y (type: double) y-coordinate of image center to crop (in meter)
 * @param nx (type: double) image size in x to crop (in meter)
 * @param ny (type: double) image size in y to crop (in meter)
 * @param ns (type: unsigned int) number of samples  to crop (in pixels)
 * @param nl (type: unsigned int) number of lines to crop (in pixels)
 * @param scale (type: double) output=scale*input+offset
 * @param offset (type: double) output=scale*input+offset
 * @param nodata (type: double) Nodata value to put in image if out of bounds.
 * @param description (type: std::string) Set image description
 * @param align (type: bool) (default: 0) Align output bounding box to input image
 * @return output image
 **/
std::shared_ptr<jiplib::Jim> JimList::stack(app::AppFactory& app){return(crop(app));};
///stack all images in collection to multiband image (alias for crop)
/**
 * @param input (type: std::string) Input image file(s). If input contains multiple images, a multi-band output is created
 * @param output (type: std::string) Output image file
 * @param oformat (type: std::string) (default: GTiff) Output image format (see also gdal_translate).
 * @param co (type: std::string) Creation option for output file. Multiple options can be specified.
 * @param a_srs (type: std::string) Override the spatial reference for the output file (leave blank to copy from input file, use epsg:3035 to use European projection and force to European grid
 * @param mem (type: unsigned long) (default: 0) Buffer size (in MB) to read image data blocks in memory
 * @param a_srs (type: std::string) Override the projection for the output file (leave blank to copy from input file, use epsg:3035 to use European projection and force to European grid
 * @param ulx (type: double) (default: 0) Upper left x value bounding box
 * @param uly (type: double) (default: 0) Upper left y value bounding box
 * @param lrx (type: double) (default: 0) Lower right x value bounding box
 * @param lry (type: double) (default: 0) Lower right y value bounding box
 * @param band (type: unsigned int) band index to stack (leave empty to retain all bands)
 * @param startband (type: unsigned int) Start band sequence number
 * @param endband (type: unsigned int) End band sequence number
 * @param otype (type: std::string) Data type for output image ({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}). Empty string: inherit type from input image
 * @param oformat (type: std::string) (default: GTiff) Output image format (see also gdal_translate).
 * @param ct (type: std::string) color table (file with 5 columns: id R G B ALFA (0: transparent, 255: solid)
 * @param dx (type: double) Output resolution in x (in meter) (empty: keep original resolution)
 * @param dy (type: double) Output resolution in y (in meter) (empty: keep original resolution)
 * @param resampling-method (type: std::string) (default: near) Resampling method (near: nearest neighbor, bilinear: bi-linear interpolation).
 * @param extent (type: std::string) get boundary from extent from polygons in vector file
 * @param crop_to_cutline (type: bool) (default: 0) Crop the extent of the target dataset to the extent of the cutline.
 * @param eo (type: std::string) special extent options controlling rasterization: ATTRIBUTE|CHUNKYSIZE|ALL_TOUCHED|BURN_VALUE_FROM|MERGE_ALG, e.g., -eo ATTRIBUTE=fieldname
 * @param mask (type: std::string) Use the the specified file as a validity mask (0 is nodata).
 * @param msknodata (type: double) (default: 0) Mask value not to consider for crop.
 * @param mskband (type: unsigned int) (default: 0) Mask band to read (0 indexed)
 * @param x (type: double) x-coordinate of image center to crop (in meter)
 * @param y (type: double) y-coordinate of image center to crop (in meter)
 * @param nx (type: double) image size in x to crop (in meter)
 * @param ny (type: double) image size in y to crop (in meter)
 * @param ns (type: unsigned int) number of samples  to crop (in pixels)
 * @param nl (type: unsigned int) number of lines to crop (in pixels)
 * @param scale (type: double) output=scale*input+offset
 * @param offset (type: double) output=scale*input+offset
 * @param nodata (type: double) Nodata value to put in image if out of bounds.
 * @param description (type: std::string) Set image description
 * @param align (type: bool) (default: 0) Align output bounding box to input image
 * @return output image
 **/
std::shared_ptr<jiplib::Jim> JimList::stack(){app::AppFactory app;return(stack(app));};
///create statistical profile from a collection
/**
 * @param input (type: std::string) input image file
 * @param output (type: std::string) Output image file
 * @param oformat (type: std::string) (default: GTiff) Output image format (see also gdal_translate)
 * @param mem (type: unsigned long) (default: 0) Buffer size (in MB) to read image data blocks in memory
 * @param function (type: std::string) Statistics function (mean, median, var, stdev, min, max, sum, mode (provide classes), ismin, ismax, proportion (provide classes), percentile, nvalid
 * @param perc (type: double) Percentile value(s) used for rule percentile
 * @param nodata (type: double) nodata value(s)
 * @param otype (type: std::string) (default: GDT_Unknown) Data type for output image ({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}). Empty string: inherit type from input image
 * @return output image
 **/
std::shared_ptr<jiplib::Jim> JimList::statProfile(app::AppFactory& app){
  /* std::shared_ptr<jiplib::Jim> imgWriter=Jim::createImg(); */
  std::shared_ptr<jiplib::Jim> imgWriter=std::make_shared<jiplib::Jim>();
  ImgCollection::statProfile(*imgWriter, app);
  return(imgWriter);
}

///functions from mialib
JimList JimList::imrgb2hsx(int x){
  int ninput=3;
  int noutput=3;
  JimList listout;
  try{
    if(size()!=ninput){
      std::ostringstream ess;
      ess << "Error: input image list should have " << ninput << " images (got " << size() << ")";
      throw(ess.str());
    }
    IMAGE ** imout;
    imout=::imrgb2hsx(this->getImage(0)->getMIA(),this->getImage(1)->getMIA(),this->getImage(2)->getMIA(),x);
    if(imout){
      for(int iim=0;iim<noutput;++iim){
        std::shared_ptr<Jim> imgWriter=std::make_shared<Jim>(imout[iim]);
        imgWriter->copyGeoTransform(*front());
        imgWriter->setProjection(front()->getProjectionRef());
        listout.pushImage(imgWriter);
      }
      return(listout);
    }
    else{
      std::string errorString="Error: imrgb2hsx() function in MIA failed, returning empty list";
      throw(errorString);
    }
  }
  catch(std::string errorString){
    std::cerr << errorString << std::endl;
    return(listout);
  }
  catch(...){
    return(listout);
  }
}

JimList JimList::alphaTree(int alphaMax){
  int ninput=2;
  int noutput=5;
  JimList listout;
  if(size()!=ninput){
    std::ostringstream ess;
    ess << "Error: input image list should have " << ninput << " images (got " << size() << ")";
    throw(ess.str());
  }
  try{
    IMAGE ** imout;
    imout=::alphatree(this->getImage(0)->getMIA(),this->getImage(1)->getMIA(),alphaMax);
    if(imout){
      for(int iim=0;iim<noutput;++iim){
        std::shared_ptr<Jim> imgWriter=std::make_shared<Jim>(imout[iim]);
        imgWriter->copyGeoTransform(*front());
        imgWriter->setProjection(front()->getProjectionRef());
        listout.pushImage(imgWriter);
      }
      return(listout);
    }
    else{
      std::string errorString="Error: alphatree() function in MIA failed, returning empty list";
      throw(errorString);
    }
  }
  catch(std::string errorString){
    std::cerr << errorString << std::endl;
    return(listout);
  }
  catch(...){
    return(listout);
  }
}

JimList JimList::histrgbmatch(){
  int ninput=4;
  int noutput=3;
  JimList listout;
  if(size()!=ninput){
    std::ostringstream ess;
    ess << "Error: input image list should have " << ninput << " images (got " << size() << ")";
    throw(ess.str());
  }
  try{
    IMAGE ** imout;
    imout=::histrgbmatch(this->getImage(0)->getMIA(),this->getImage(1)->getMIA(),this->getImage(2)->getMIA(),this->getImage(3)->getMIA());
    if(imout){
      for(int iim=0;iim<noutput;++iim){
        std::shared_ptr<Jim> imgWriter=std::make_shared<Jim>(imout[iim]);
        imgWriter->copyGeoTransform(*front());
        imgWriter->setProjection(front()->getProjectionRef());
        listout.pushImage(imgWriter);
      }
      return(listout);
    }
    else{
      std::string errorString="Error: histrgbmatch() function in MIA failed, returning empty list";
      throw(errorString);
    }
  }
  catch(std::string errorString){
    std::cerr << errorString << std::endl;
    return(listout);
  }
  catch(...){
    return(listout);
  }
}

JimList JimList::histrgb3dmatch (){
  int ninput=4;
  int noutput=3;
  JimList listout;
  if(size()!=ninput){
    std::ostringstream ess;
    ess << "Error: input image list should have " << ninput << " images (got " << size() << ")";
    throw(ess.str());
  }
  try{
    IMAGE ** imout;
    imout=::histrgb3dmatch(this->getImage(0)->getMIA(),this->getImage(1)->getMIA(),this->getImage(2)->getMIA(),this->getImage(3)->getMIA());
    if(imout){
      for(int iim=0;iim<noutput;++iim){
        std::shared_ptr<Jim> imgWriter=std::make_shared<Jim>(imout[iim]);
        imgWriter->copyGeoTransform(*front());
        imgWriter->setProjection(front()->getProjectionRef());
        listout.pushImage(imgWriter);
      }
      return(listout);
    }
    else{
      std::string errorString="Error: histrgb3dmatch() function in MIA failed, returning empty list";
      throw(errorString);
    }
  }
  catch(std::string errorString){
    std::cerr << errorString << std::endl;
    return(listout);
  }
  catch(...){
    return(listout);
  }
}
