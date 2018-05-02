/**********************************************************************
jim.h: class to read raster files
History
2016/06/24 - Created by Pieter Kempeneers
Change log
***********************************************************************/
/*! \file jim.h
  \brief A Documented file.

  Global functions are not documented unless the file itself is documented. The only other way to get this working is EXTRACT_ALL.
  Details: https://www.stack.nl/%7Edimitri/doxygen/manual/docblocks.html
  To document global objects (functions, typedefs, enum, macros, etc), you must document the file in which they are defined. In other words, there must at least be a
*/
#ifndef _JIM_H_
#define _JIM_H_

#include <string>
#include <vector>
#include <memory>
#include "pktools/imageclasses/ImgRaster.h"
#include "pktools/imageclasses/VectorOgr.h"
#include "pktools/apps/AppFactory.h"
#include "pktools/base/Optionpk.h"
#include "jimlist.h"
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

/**
   Name space jiplib
**/
namespace jiplib{
  enum JIPLIBDataType {JDT_Int64=14, JDT_UInt64=15};
  class Jim;
  class JimList;
  class Jim : public ImgRaster
  {
  public:
    ///default constructor
  Jim();
    ///constructor opening an image in memory using an external data pointer
  Jim(void* dataPointer, int ncol, int nrow, int nplane, const GDALDataType& dataType);
  Jim(std::vector<void*> dataPointers, int ncol, int nrow, int nplane, const GDALDataType& dataType);
    ///constructor input image
  Jim(IMAGE *mia);
    ///constructor input image
  Jim(const std::string& filename, bool readData=true, unsigned int memory=0);
    ///constructor input image
  Jim(const std::string& filename, const Jim& imgSrc, unsigned int memory=0, const std::vector<std::string>& options=std::vector<std::string>());
    ///constructor input image
    /* Jim(std::shared_ptr<ImgRaster> imgSrc, bool copyData=true) : m_nplane(1), ImgRaster(imgSrc, copyData){}; */
    ///constructor input image
  Jim(Jim& imgSrc, bool copyData=true);
    ///constructor output image
  Jim(const std::string& filename, int ncol, int nrow, int nband, const GDALDataType& dataType, const std::string& imageType, unsigned int memory=0, const std::vector<std::string>& options=std::vector<std::string>());
    ///constructor output image
  Jim(int ncol, int nrow, int nband, const GDALDataType& dataType);
    ///constructor from app
    /* Jim(app::AppFactory &theApp): m_nplane(1), ImgRaster(theApp){}; */
    //test
  Jim(app::AppFactory &theApp);
    ///destructor
    ~Jim(void);
    ///Create new shared pointer to Jim object using app
    static std::shared_ptr<Jim> createImg(app::AppFactory &theApp);
    ///Create new shared pointer to Jim object
    static std::shared_ptr<Jim> createImg();
    ///Create new shared pointer to Jim object using existing image object
    static std::shared_ptr<Jim> createImg(const std::string filename, bool readData=true, unsigned int memory=0);
    ///Create new shared pointer to Jim object using existing image object
    static std::shared_ptr<Jim> createImg(const std::shared_ptr<Jim> pSrc, bool copyData=true);
    ///create shared pointer to Jim using an external data pointer
    /* static std::shared_ptr<Jim> createImg(void* dataPointer, int ncol, int nrow, int nplane, const GDALDataType& dataType){ */
    /*   std::shared_ptr<Jim> pJim=std::make_shared<Jim>(dataPointer,ncol,nrow,nplane,dataType); */
    /*   return(pJim); */
    /* } */
    /* ///create shared pointer to multi-band Jim using external data pointers */
    /* static std::shared_ptr<Jim> createImg(std::vector<void*> dataPointers, int ncol, int nrow, int nplane, const GDALDataType& dataType){ */
    /*   std::shared_ptr<Jim> pJim=std::make_shared<Jim>(dataPointers,ncol,nrow,nplane,dataType); */
    /*   return(pJim); */
    /* } */
    ///Create a JSON string from a Jim image
    std::string jim2json();
    ///Create a custom collection from a Jim image
    /* std::string jim2custom(); */
    ///Clone as new shared pointer to ImgRaster object
    /**
     * @param copyData (type: bool) value set to True if data needs to be copied
     * @return shared pointer to new ImgRaster object alllowing polymorphism
     */
    std::shared_ptr<Jim> clone(bool copyData=true);
    ///get size in Bytes of the current data type
    size_t getDataTypeSizeBytes(int band=0) const;
    /* --------------------- */
    /* Access Jim attributes */
    /* --------------------- */
    ///Get the number of columns of this dataset
#ifdef SWIG
    %pythonprepend nrOfCol()  "\"\"\"HELP.METHOD.Jim.nrOfCol()\"\"\""
#endif
       int nrOfCol() const { return ImgRaster::nrOfCol();};
    ///Get the number of rows of this dataset
#ifdef SWIG
    %pythonprepend nrOfRow()  "\"\"\"HELP.METHOD.Jim.nrOfRow()\"\"\""
#endif
       int nrOfRow() const { return ImgRaster::nrOfRow();};
    ///Get the number of planes of this dataset
#ifdef SWIG
    %pythonprepend nrOfPlane()  "\"\"\"HELP.METHOD.Jim.nrOfPlane()\"\"\""
#endif
       int nrOfPlane() const { return m_nplane;};
    ///printNoDataValues
#ifdef SWIG
    %pythonprepend printNoDataValues()  "\"\"\"HELP.METHOD.Jim.printNoDataValues()\"\"\""
#endif
       CPLErr printNoDataValues() const { return ImgRaster::printNoDataValues();};
    //needed in order not to hide this base class function
    using ImgRaster::getNoDataValues;
    ///getNoDataValues
#ifdef SWIG
    %pythonprepend getNoDataValues()  "\"\"\"HELP.METHOD.Jim.getNoDataValues()\"\"\""
#endif
       std::vector<double> getNoDataValues() const { return ImgRaster::getNoDataValues();};
    ///Set the single no data values of this dataset
#ifdef SWIG
    %pythonprepend setDataValue(double)  "\"\"\"HELP.METHOD.Jim.setNoDataValue(*args)\"\"\""
#endif
       CPLErr setNoDataValue(double nodata){return ImgRaster::setNoDataValue(nodata);};
    ///pushNoDataValue
#ifdef SWIG
    %pythonprepend pushNoDataValue(double)  "\"\"\"HELP.METHOD.Jim.pushNoDataValue(*args)\"\"\""
#endif
       CPLErr pushNoDataValue(double noDataValue) { return ImgRaster::pushNoDataValue(noDataValue);};
    ///setNoData
#ifdef SWIG
    %pythonprepend setNoData(const std::vector<double>&) "\"\"\"HELP.METHOD.Jim.setNoData(*args)\"\"\""
#endif
       CPLErr setNoData(const std::vector<double>& nodata) { return ImgRaster::setNoData(nodata);};
    ///Clear all no data values, including the one in GDAL dataset if it is set
#ifdef SWIG
    %pythonprepend clearNoData(int)  "\"\"\"HELP.METHOD.Jim.clearNoData(band)\"\"\""
#endif
       CPLErr clearNoData(int band=0){return(ImgRaster::clearNoData(band));}
    ///Get the internal datatype for this raster dataset
#ifdef SWIG
    %pythonprepend getDataType(int)  "\"\"\"HELP.METHOD.Jim.getDataType(*args)\"\"\""
#endif
       int getDataType(int band=0) const { return ImgRaster::getDataType(band);};
    /* -------------------------- */
    /* Get geospatial information */
    /* -------------------------- */
    ///Check if a geolocation is covered by this dataset. Only the bounding box is checked, irrespective of no data values.
#ifdef SWIG
    %pythonprepend covers(double, double, OGRCoordinateTransformation*)  "\"\"\"HELP.METHOD.Jim.covers(*args)\"\"\""
#endif
       bool covers(double x, double y, OGRCoordinateTransformation *poCT=NULL) const{return ImgRaster::covers(x,y,poCT);};
    ///Check if a region of interest is (partially or all if all is set) covered by this dataset. Only the bounding box is checked, irrespective of no data values.
#ifdef SWIG
    %pythonprepend covers(double, double, double, double, bool, OGRCoordinateTransformation*)  "\"\"\"HELP.METHOD.Jim.covers(*args)\"\"\""
#endif
       bool covers(double ulx, double  uly, double lrx, double lry, bool all=false, OGRCoordinateTransformation *poCT=NULL) const{return ImgRaster::covers(ulx,uly,lrx,lry,all,poCT);};
    ///Get the geotransform data for this dataset
#ifdef SWIG
    %pythonprepend getGeoTransform(double&, double&, double&, double&, double&, double&)  "\"\"\"HELP.METHOD.Jim.getGeoTransform()\"\"\""
#endif
       void getGeoTransform(double& gt0, double& gt1, double& gt2, double& gt3, double& gt4, double& gt5) const{return ImgRaster::getGeoTransform(gt0, gt1, gt2, gt3, gt4, gt5);};
#ifdef SWIG
    %pythonprepend getGeoTransform(double*)  "\"\"\"HELP.METHOD.Jim.getGeoTransform()\"\"\""
#endif
       CPLErr getGeoTransform(double* gt) const{return ImgRaster::getGeoTransform(gt);};
    ///Set the geotransform data for this dataset
#ifdef SWIG
    %pythonprepend setGeoTransform(double*)  "\"\"\"HELP.METHOD.Jim.setGeoTransform()\"\"\""
#endif
       CPLErr setGeoTransform(double* gt){return ImgRaster::setGeoTransform(gt);};
    CPLErr setGeoTransform(std::vector<double> gt){return ImgRaster::setGeoTransform(gt);};
    ///Copy geotransform information from another georeferenced image
#ifdef SWIG
    %pythonprepend copyGeoTransform(const ImgRaster&)  "\"\"\"HELP.METHOD.Jim.copyGeoTransform(*args)\"\"\""
#endif
       CPLErr copyGeoTransform(const ImgRaster& imgSrc){return ImgRaster::copyGeoTransform(imgSrc);};
    ///Get the projection for this dataget in well known text (wkt) format
#ifdef SWIG
    %pythonprepend getProjection()  "\"\"\"HELP.METHOD.Jim.getProjection()\"\"\""
#endif
       std::string getProjection(){return ImgRaster::getProjection();};
    ///Set the projection for this dataset in well known text (wkt) format
#ifdef SWIG
    %pythonprepend setProjection(const std::string&)  "\"\"\"HELP.METHOD.Jim.setProjection(*args)\"\"\""
#endif
       CPLErr setProjection(const std::string& projection){return ImgRaster::setProjection(projection);};
    ///Get the bounding box of this dataset in georeferenced coordinates with coordinate transform
#ifdef SWIG
    %pythonprepend getBoundingBox(double&, double&, double&, double&, OGRCoordinateTransformation *)  "\"\"\"HELP.METHOD.Jim.getBoundingBox()\"\"\""
#endif
       void getBoundingBox(double& ulx, double& uly, double& lrx, double& lry, OGRCoordinateTransformation *poCT=NULL) const{return ImgRaster::getBoundingBox(ulx,uly,lrx,lry,poCT);};
    ///Get the bounding box of this dataset in georeferenced coordinates with coordinate transform
    void getBoundingBox(std::vector<double> &bbvector, OGRCoordinateTransformation *poCT=NULL) const{return ImgRaster::getBoundingBox(bbvector,poCT);};
    ///Get the center position of the image in georeferenced coordinates with coordinate transform
#ifdef SWIG
    %pythonprepend getCenterPos(double&, double&)  "\"\"\"HELP.METHOD.Jim.getCenterPos()\"\"\""
#endif
       void getCenterPos(double& centerX, double& centerY){return ImgRaster::getCenterPos(centerX, centerY);};
#ifdef SWIG
    %pythonprepend getUlx()  "\"\"\"HELP.METHOD.Jim.getUlx()\"\"\""
#endif
       double getUlx(){return ImgRaster::getUlx();};
#ifdef SWIG
    %pythonprepend getUly()  "\"\"\"HELP.METHOD.Jim.getUly()\"\"\""
#endif
       double getUly(){return ImgRaster::getUly();};
#ifdef SWIG
    %pythonprepend getLrx()  "\"\"\"HELP.METHOD.Jim.getLrx()\"\"\""
#endif
       double getLrx(){return ImgRaster::getLrx();};
#ifdef SWIG
    %pythonprepend getLry()  "\"\"\"HELP.METHOD.Jim.getLry()\"\"\""
#endif
       double getLry(){return ImgRaster::getLry();};
#ifdef SWIG
    %pythonprepend getDeltaX()  "\"\"\"HELP.METHOD.Jim.getDeltaX()\"\"\""
#endif
       double getDeltaX(){return ImgRaster::getDeltaX();};
#ifdef SWIG
    %pythonprepend getDeltaY()  "\"\"\"HELP.METHOD.Jim.getDeltaY()\"\"\""
#endif
       double getDeltaY(){return ImgRaster::getDeltaY();};
#ifdef SWIG
    %pythonprepend getRefPix(double&, double &, int)  "\"\"\"HELP.METHOD.Jim.getRefPix(\"\"\""
#endif
       void getRefPix(double& centerX, double &centerY, int band=0){return ImgRaster::getRefPix(centerX, centerY, band);};
    /* -------------------- */
    /* Input/Output methods */
    /* -------------------- */
    ///Open an image for writing using an external data pointer
    CPLErr open(void* dataPointer, int ncol, int nrow, int nplane, const GDALDataType& dataType);
    ///Open a multiband image for writing using a external data pointers
    CPLErr open(std::vector<void*> dataPointers, int ncol, int nrow, int nplane, const GDALDataType& dataType);
    ///Open an image for writing in memory, defining image attributes.
    /* void open(int ncol, int nrow, int nband, int dataType); */
    ///Open an image for writing, based on an existing image object
    CPLErr open(Jim& imgSrc, bool copyData=true);
    ///Open dataset
    CPLErr open(const std::string& filename, bool readData=true, unsigned int memory=0);
    //open dataset
#ifdef SWIG
    %pythonprepend open(app::AppFactory &theApp)  "\"\"\"HELP.METHOD.Jim.open(dict)\"\"\""
#endif
    CPLErr open(app::AppFactory &app);
    ///Close dataset (specialization of the close member function of ImgRaster, avoiding writing the data)
#ifdef SWIG
    %pythonprepend close()  "\"\"\"HELP.METHOD.Jim.close()\"\"\""
#endif
       CPLErr close();
    ///write to file previously set (eg., with setFile). Specialization of the writeData member function of ImgRaster, avoiding reset of the memory.
    CPLErr write();
    ///write to file Specialization of the writeData member function of ImgRaster, avoiding reset of the memory.
    /**
     * @param output (type: std::string) Output image file
     * @param oformat (type: std::string) (default: GTiff) Output image format (see also gdal_translate).
     * @param co (type: std::string) Creation option for output file. Multiple options can be specified.
     * @return CE_None if successful, CE_Failure if not.
     * @param nodata Nodata value to put in image.
     **/
#ifdef SWIG
    %pythonprepend write(app::AppFactory&)  "\"\"\"HELP.METHOD.Jim.write(dict)\"\"\""
#endif
    CPLErr write(app::AppFactory &theApp);
    ///dump raster dataset
#ifdef SWIG
    %pythonprepend dumpImg(app::AppFactory&)  "\"\"\"HELP.METHOD.Jim.dumpImg(dict)\"\"\""
#endif
       CPLErr dumpImg(app::AppFactory& app){return ImgRaster::dumpImg(app);};
    ///assignment operator
    Jim& operator=(Jim& imgSrc);
    /* ///relational == operator */
    /* bool operator==(Jim& refImg); */
    ///relational == operator
    /* bool operator==(std::shared_ptr<Jim> refImg); */
    ///test for equality (relational == operator)
    /* bool isEqual(Jim& refImg){return(*this==(refImg));}; */
    ///Test raster dataset for equality.
#ifdef SWIG
    %pythonprepend isEqual(std::shared_ptr<Jim>)  "\"\"\"HELP.METHOD.Jim.isEqual(*args)\"\"\""
#endif
    bool isEqual(std::shared_ptr<Jim> refImg);

    /* --------------- */
    /* Convert methods */
    /* --------------- */
    ///convert Jim image in memory returning Jim image
#ifdef SWIG
    %pythonprepend convert(app::AppFactory&)  "\"\"\"HELP.METHOD.Jim.convert(dict)\"\"\""
#endif
       std::shared_ptr<Jim> convert(app::AppFactory& app);
    /* ------------------------------------- */
    /* Subset methods and geometry operators */
    /* ------------------------------------- */
    ///crop Jim image in memory returning Jim image
#ifdef SWIG
    %pythonprepend crop(app::AppFactory&)  "\"\"\"HELP.METHOD.Jim.crop(dict)\"\"\""
#endif
       std::shared_ptr<Jim> crop(app::AppFactory& app);
    ///crop Jim image in memory based on VectorOgr returning Jim image
#ifdef SWIG
    %pythonprepend cropOgr(VectorOgr&, app::AppFactory&)  "\"\"\"HELP.METHOD.Jim.crop(*args)\"\"\""
#endif
    std::shared_ptr<Jim> cropOgr(VectorOgr& sampleReader, app::AppFactory& app);
    /* ----------------------------------------------- */
    /* Convolution filters and morphological operators */
    /* ----------------------------------------------- */
    /* ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ */
    /* spectral/temporal domain (1D) */
    /* ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ */
    ///filter Jim image in spectral/temporal domain
#ifdef SWIG
    %pythonprepend filter1d(app::AppFactory&)  "\"\"\"HELP.METHOD.filter1d(dict)\"\"\""
#endif
       std::shared_ptr<Jim> filter1d(app::AppFactory& theApp);
    /* ^^^^^^^^^^^^^^^^^^^ */
    /* spatial domain (2D) */
    /* ^^^^^^^^^^^^^^^^^^^ */
    ///filter Jim image in spatial domain
    std::shared_ptr<Jim> filter2d(const app::AppFactory& theApp);
    /* ---------------------- */
    /* Classification methods */
    /* ---------------------- */
    ///Supervised classification (train with extractImg/extractOgr)
#ifdef SWIG
    %pythonprepend classify(app::AppFactory&)  "\"\"\"HELP.METHOD.classify(dict)\"\"\""
#endif
    std::shared_ptr<Jim> classify(app::AppFactory& app);
    ///Supervised classification using Symbolic Machine Learning
#ifdef SWIG
    %pythonprepend classifySML(JimList&, app::AppFactory&)  "\"\"\"HELP.METHOD.classifySML(dict)\"\"\""
#endif
    std::shared_ptr<Jim> classifySML(JimList& referenceReader, app::AppFactory& app);
    std::shared_ptr<Jim> classifySML(app::AppFactory& app);
    ///replace categorical pixel values in raster dataset
#ifdef SWIG
    %pythonprepend reclass(app::AppFactory&)  "\"\"\"HELP.METHOD.reclass(dict)\"\"\""
#endif
    std::shared_ptr<Jim> reclass(app::AppFactory& app);
    ///validate classified image
#ifdef SWIG
    %pythonprepend validate(app::AppFactory&)  "\"\"\"HELP.METHOD.validate(dict)\"\"\""
#endif
    CPLErr validate(app::AppFactory& app);

    /* ------------------------ */
    /* Mask / Threshold methods */
    /* ------------------------ */
    ///Apply thresholds: set to no data if not within thresholds t1 and t2
#ifdef SWIG
    %pythonprepend setThreshold(app::AppFactory&)  "\"\"\"HELP.METHOD.setThreshold(dict)\"\"\""
#endif
       std::shared_ptr<Jim> setThreshold(app::AppFactory& theApp);
    ///get mask from a raster dataset
#ifdef SWIG
    %pythonprepend getMask(app::AppFactory&)  "\"\"\"HELP.METHOD.getMask(dict)\"\"\""
#endif
    std::shared_ptr<Jim> getMask(app::AppFactory& app);
    ///set mask to raster dataset
#ifdef SWIG
    %pythonprepend setMask(VectorOgr&, app::AppFactory&)  "\"\"\"HELP.METHOD.setMask(*args)\"\"\""
#endif
    std::shared_ptr<Jim> setMask(VectorOgr& ogrReader, app::AppFactory& app);
#ifdef SWIG
    %pythonprepend setMask(JimList&, app::AppFactory&)  "\"\"\"HELP.METHOD.setMask(*args)\"\"\""
#endif
    std::shared_ptr<Jim> setMask(JimList& maskList, app::AppFactory& app);

    /* -------------------------------------- */
    /* Statistical methods and interpolations */
    /* -------------------------------------- */
    ///create statistical profile
#ifdef SWIG
    %pythonprepend statProfile(app::AppFactory&)  "\"\"\"HELP.METHOD.statProfile(dict)\"\"\""
#endif
    std::shared_ptr<Jim> statProfile(app::AppFactory& theApp);
    ///stretch Jim image and return stretched image as shared pointer
#ifdef SWIG
    %pythonprepend stretch(app::AppFactory&)  "\"\"\"HELP.METHOD.stretch(dict)\"\"\""
#endif
    std::shared_ptr<Jim> stretch(app::AppFactory& app);
    ///get statistics on image list
#ifdef SWIG
    %pythonprepend getStats(app::AppFactory&)  "\"\"\"HELP.METHOD.getStats(dict)\"\"\""
#endif
    std::multimap<std::string,std::string> getStats(app::AppFactory& theApp);

    /* -------------------------------------------------- */
    /* Extracting pixel values from overlays and sampling */
    /* -------------------------------------------------- */
    ///extract pixel values from raster image from a vector sample
#ifdef SWIG
    %pythonprepend extractOgr(VectorOgr&, app::AppFactory&)  "\"\"\"HELP.METHOD.extractOgr(*args)\"\"\""
#endif
    std::shared_ptr<VectorOgr> extractOgr(VectorOgr& sampleReader, app::AppFactory& app){return ImgRaster::extractOgr(sampleReader,app);};
    ///extract pixel values from raster image with random or grid sampling
#ifdef SWIG
    %pythonprepend extractSample(app::AppFactory&)  "\"\"\"HELP.METHOD.extractSample(dict)\"\"\""
#endif
       std::shared_ptr<VectorOgr> extractSample(app::AppFactory& app){return ImgRaster::extractSample(app);};
    ///extract pixel values from raster image from a raster sample
#ifdef SWIG
    %pythonprepend extractImg(Jim&, app::AppFactory&)  "\"\"\"HELP.METHOD.extractImg(*args)\"\"\""
#endif
       std::shared_ptr<VectorOgr> extractImg(Jim& classReader, app::AppFactory& app){return ImgRaster::extractImg(classReader,app);};
    ///Initialize the memory for read/write image in cache
    CPLErr initMem(unsigned int memory);
    /// convert single plane multiband image to single band image with multiple planes
    CPLErr band2plane();
    ///read data bands into planes
    CPLErr readDataPlanes(std::vector<int> bands);
    /// convert single band multiple plane image to single plane multiband image
    /* CPLErr plane2band(){};//not implemented yet */
    ///get MIA representation for a particular band
    IMAGE* getMIA(int band=0);
    ///set memory from internal MIA representation for particular band
    CPLErr setMIA(int band=0);
    // ///set memory from MIA representation for particular band
    CPLErr setMIA(IMAGE* mia, int band=0);
    ///convert a GDAL data type to MIA data type
    int getMIADataType();
    ///convert a JIPLIB to MIA data type
    int JIPLIB2MIADataType(int aJIPLIBDataType);
    ///convert a GDAL to MIA data type
    int GDAL2MIADataType(GDALDataType aGDALDataType);
    ///convert a MIA data type to GDAL data type
    int MIA2JIPLIBDataType(int aMIADataType);

    //start insert from fun2method_imagetype
    //end insert from fun2method_imagetype
    //start insert from fun2method_imagetype_multi
    //end insert from fun2method_imagetype_multi
    //start insert from fun2method_imagelisttype
    //end insert from fun2method_imagelisttype
    //start insert from fun2method_errortype
    //end insert from fun2method_errortype
    //start insert from fun2method_errortype_d
    //end insert from fun2method_errortype_d
    //start insert from fun2method_errortype_nd
    //end insert from fun2method_errortype_nd

    /* ///read data from with reduced resolution */
    /* CPLErr GDALRead(std::string filename, int band, int nXOff, int nYOff, int nXSize, int nYSize, int nBufXSize=0, int nBufYSize=0); */

    ///get unique pixels
    /* unsigned int getUniquePixels(){ */
    /*   std::map<std::vector<char>,std::vector<std::pair<unsigned short,unsigned short> > > theMap; */
    /*   theMap=ImgRaster::getUniquePixels<char>(); */
    /*   return(theMap.size()); */
    /* }; */
    /* template<typename T> std::map<std::vector<T>,std::vector<std::pair<unsigned short,unsigned short> > > getUniquePixels(){ */
    /*   std::map<std::vector<T>,std::vector<std::pair<unsigned short,unsigned short> > > theMap; */
    /*   theMap=ImgRaster::getUniquePixels<T>(); */
    /*   return(theMap); */
    /* }; */
    ///check the difference between two images
    std::shared_ptr<Jim> diff(app::AppFactory& app);
    ///Check for difference with reference image
    CPLErr diff(std::shared_ptr<Jim> refImage,app::AppFactory& app);

    std::shared_ptr<Jim> getShared(){
      return(std::dynamic_pointer_cast<Jim>(shared_from_this()));
    }
    //todo: manual for now, but need to be done with Python script
    /* std::shared_ptr<jiplib::Jim> labelConstrainedCCsMultiband(Jim &imgRaster, int ox, int oy, int oz, int r1, int r2); */
  protected:
    ///reset all member variables
    void reset(void){
      ImgRaster::reset();
      m_nplane=1;
      for(int iband=0;iband<m_mia.size();++iband)
        delete(m_mia[iband]);
      m_mia.clear();
    }
    ///number of planes in this dataset
    int m_nplane;
  private:
    std::shared_ptr<Jim> cloneImpl(bool copyData) {
      return std::make_shared<Jim>(*this,copyData);
      /* return(std::make_shared<Jim>()); */
    };
    std::vector<IMAGE*> m_mia;
  };
  ///Create new image object
  /**
   * @param input (type: std::string) input filename
   * @param nodata (type: double) Nodata value to put in image if out of bounds.
   * @param band (type: int) Bands to open, index starts from 0
   * @param ulx (type: double) Upper left x value bounding box
   * @param uly (type: double) Upper left y value bounding box
   * @param lrx (type: double) Lower right x value bounding box
   * @param lry (type: double) Lower right y value bounding box
   * @param dx (type: double) Resolution in x
   * @param dy (type: double) Resolution in y
   * @param resample (type: std::string) (default: GRIORA_NearestNeighbour) resample: GRIORA_NearestNeighbour|GRIORA_Bilinear|GRIORA_Cubic|GRIORA_CubicSpline|GRIORA_Lanczos|GRIORA_Average|GRIORA_Average|GRIORA_Gauss (check http://www.gdal.org/gdal_8h.html#a640ada511cbddeefac67c548e009d5a)
   * @param extent (type: std::string) get boundary from extent from polygons in vector file
   * @param mem (type: unsigned long) (default: 0) Buffer size (in MB) to read image data blocks in memory
   * @param ncol (type: int) Number of columns
   * @param nrow (type: int) Number of rows
   * @param nband (type: int) (default: 1) Number of bands
   * @param otype (type: std::string) (default: Byte) Data type for output image ({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64})
   * @param seed (type: unsigned long) (default: 0) seed value for random generator
   * @param mean (type: double) (default: 0) Mean value for random generator
   * @param sigma (type: double) (default: 0) Sigma value for random generator
   * @param description (type: std::string) Set image description
   * @param a_srs (type: std::string) Assign the spatial reference for the output file, e.g., psg:3035 to use European projection and force to European grid
   * @return shared pointer to new Jim object
   **/
#ifdef SWIG
  %pythonprepend createJim()  "\"\"\"HELP.METHOD.createJim()\"\"\""
#endif
  static std::shared_ptr<Jim> createJim(){return Jim::createImg();};
#ifdef SWIG
  %pythonprepend createJim(app::AppFactory&)  "\"\"\"HELP.METHOD.createJim(dict)\"\"\""
#endif
     static std::shared_ptr<Jim> createJim(app::AppFactory &theApp){return(Jim::createImg(theApp));};
#ifdef SWIG
  %pythonprepend createJim(const std::shared_ptr<Jim>, bool)  "\"\"\"HELP.METHOD.createJim(*args)\"\"\""
#endif
     static std::shared_ptr<Jim> createJim(const std::shared_ptr<Jim> pSrc, bool copyData=true){return(Jim::createImg(pSrc, copyData));};
  static std::shared_ptr<Jim> createJim(const std::string& filename, bool readData=true){return(Jim::createImg(filename,readData));};
  /* static std::shared_ptr<Jim> createImg(void* dataPointer, int ncol, int nrow, int nplane, const GDALDataType& dataType); */
  /* static std::shared_ptr<Jim> createImg(std::vector<void*> dataPointers, int ncol, int nrow, int nplane, const GDALDataType& dataType); */

}

#endif // _JIM_H_
