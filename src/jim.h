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
/* #include "pktools/imageclasses/ImgCollection.h" */
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
    %pythonprepend write(app::AppFactory &theApp)  "\"\"\"HELP.METHOD.Jim.write(dict)\"\"\""
#endif
    CPLErr write(app::AppFactory &theApp);
    ///Close dataset (specialization of the close member function of ImgRaster, avoiding writing the data)
#ifdef SWIG
    %pythonprepend close()  "\"\"\"HELP.METHOD.Jim.close()\"\"\""
#endif
    CPLErr close();
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
    ///Get the number of planes of this dataset
    int nrOfPlane(void) const { return m_nplane;};
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
    ///assignment operator
    Jim& operator=(Jim& imgSrc);
    /* ///relational == operator */
    /* bool operator==(Jim& refImg); */
    ///relational == operator
    /* bool operator==(std::shared_ptr<Jim> refImg); */
    ///test for equality (relational == operator)
    /* bool isEqual(Jim& refImg){return(*this==(refImg));}; */
    ///relational == operator
    bool isEqual(std::shared_ptr<Jim> refImg);

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

    //in memory functions from ImgRaster using AppFactory

    ///crop Jim image in memory returning Jim image
    std::shared_ptr<Jim> crop(app::AppFactory& app);
    ///crop Jim image in memory returning Jim image
    std::shared_ptr<Jim> crop(VectorOgr& sampleReader, app::AppFactory& app);
    ///filter Jim image in spectral/temporal domain
    std::shared_ptr<Jim> filter1d(app::AppFactory& theApp);
    ///filter Jim image in spatial domain
    std::shared_ptr<Jim> filter2d(const app::AppFactory& theApp);
    ///get statistics on image list
    std::multimap<std::string,std::string> getStats(app::AppFactory& theApp);
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
    ///create statistical profile
    std::shared_ptr<Jim> statProfile(app::AppFactory& theApp);
    ///check the difference between two images
    std::shared_ptr<Jim> diff(app::AppFactory& app);
    ///Check for difference with reference image
    CPLErr diff(std::shared_ptr<Jim> refImage,app::AppFactory& app);
    ///check the difference between two images
    CPLErr validate(app::AppFactory& app);
    ///supervised classification (train with extractImg/extractOgr)
    std::shared_ptr<Jim> classify(app::AppFactory& app);
    ///supervised classification using Artificial Neural Network (train with trainANN)
    /* std::shared_ptr<Jim> classifyANN(app::AppFactory& app); */
    ///supervised classification using support vector machine (train with trainSVM)
    /* std::shared_ptr<Jim> classifySVM(app::AppFactory& app); */
    ///supervised classification using SML
    std::shared_ptr<Jim> classifySML(JimList& referenceReader, app::AppFactory& app);
    std::shared_ptr<Jim> classifySML(app::AppFactory& app);
    ///supervised classification using support vector machine (train with extractImg/extractOgr)
    /* std::shared_ptr<Jim> svm(app::AppFactory& app); */
    ///supervised classification using support artificial neural network (train with extractImg/extractOgr)
    /* std::shared_ptr<Jim> ann(app::AppFactory& app); */
    ///stretch Jim image and return stretched image as shared pointer
    std::shared_ptr<Jim> stretch(app::AppFactory& app);
    ///Apply thresholds: set to no data if not within thresholds t1 and t2
    std::shared_ptr<Jim> setThreshold(double t1, double t2);
    ///Apply absolute thresholds: set to no data if not within thresholds t1 and t2
    std::shared_ptr<Jim> setAbsThreshold(double t1, double t2);
    ///Apply thresholds: set to no data if not within thresholds t1 and t2, else set to value
    std::shared_ptr<Jim> setThreshold(double t1, double t2, double value);
    ///Apply absolute thresholds: set to no data if not within thresholds t1 and t2, else set to value
    std::shared_ptr<Jim> setAbsThreshold(double t1, double t2, double value);
    ///get mask from a raster dataset
    std::shared_ptr<Jim> getMask(app::AppFactory& app);
    ///set mask to raster dataset
    std::shared_ptr<Jim> setMask(VectorOgr& ogrReader, app::AppFactory& app);
    std::shared_ptr<Jim> setMask(JimList& maskList, app::AppFactory& app);
    ///Clear all no data values, including the one in GDAL dataset if it is set
    CPLErr clearNoData(int band=0);
    ///reclass raster dataset
    std::shared_ptr<Jim> reclass(app::AppFactory& app);

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
  %pythonprepend createJim(app::AppFactory &theApp)  "\"\"\"HELP.METHOD.createJim(dict)\"\"\""
#endif
     static std::shared_ptr<Jim> createJim(app::AppFactory &theApp){return(Jim::createImg(theApp));};
  static std::shared_ptr<Jim> createJim(const std::shared_ptr<Jim> pSrc, bool copyData=true){return(Jim::createImg(pSrc, copyData));};
  static std::shared_ptr<Jim> createJim(const std::string& filename, bool readData=true){return(Jim::createImg(filename,readData));};
  /* static std::shared_ptr<Jim> createImg(void* dataPointer, int ncol, int nrow, int nplane, const GDALDataType& dataType); */
  /* static std::shared_ptr<Jim> createImg(std::vector<void*> dataPointers, int ncol, int nrow, int nplane, const GDALDataType& dataType); */

}

#endif // _JIM_H_
