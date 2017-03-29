/**********************************************************************
jim.h: class to read raster files
History
2016/06/24 - Created by Pieter Kempeneers
Change log
***********************************************************************/
#ifndef _JIM_H_
#define _JIM_H_

#include <string>
#include <vector>
#include <memory>
#include "pktools/imageclasses/ImgRaster.h"
#include "pktools/imageclasses/ImgCollection.h"
#include "pktools/apps/AppFactory.h"
#include "jimlist.h"
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
  Jim() : m_nplane(1), ImgRaster(){};
    ///constructor opening an image in memory using an external data pointer (not tested yet)
  Jim(void* dataPointer, int ncol, int nrow, int nplane, const GDALDataType& dataType) : Jim() {open(dataPointer,ncol,nrow,nplane,dataType);};
    ///constructor input image
  Jim(IMAGE *mia) : Jim() {setMIA(mia,0);};
    ///constructor input image
  Jim(const std::string& filename, unsigned int memory=0) : m_nplane(1), ImgRaster(filename,memory){};
    ///constructor input image
  Jim(const std::string& filename, const Jim& imgSrc, unsigned int memory=0, const std::vector<std::string>& options=std::vector<std::string>()) : m_nplane(1), ImgRaster(filename,imgSrc,memory,options){};
    ///constructor input image
    /* Jim(std::shared_ptr<ImgRaster> imgSrc, bool copyData=true) : m_nplane(1), ImgRaster(imgSrc, copyData){}; */
    ///constructor input image
  Jim(Jim& imgSrc, bool copyData=true) : m_nplane(1), ImgRaster(imgSrc, copyData){};
    ///constructor output image
  Jim(const std::string& filename, int ncol, int nrow, int nband, const GDALDataType& dataType, const std::string& imageType, unsigned int memory=0, const std::vector<std::string>& options=std::vector<std::string>()) : m_nplane(1), ImgRaster(filename, ncol, nrow, nband, dataType, imageType, memory, options){};
    ///constructor output image
  Jim(int ncol, int nrow, int nband, const GDALDataType& dataType) : m_nplane(1), ImgRaster(ncol, nrow, nband, dataType){};
    ///constructor from app
  /* Jim(app::AppFactory &theApp): m_nplane(1), ImgRaster(theApp){}; */
    //test
  Jim(app::AppFactory &theApp): m_nplane(1), ImgRaster(theApp){};
    ///destructor
    ~Jim(void){
      if(m_mia.size()){
        for(int iband=0;iband<m_mia.size();++iband)
          if(m_mia[iband])
            delete(m_mia[iband]);
        m_mia.clear();
      }
      ImgRaster::reset();
    }
    static std::shared_ptr<Jim> createImg(app::AppFactory &theApp);
    static std::shared_ptr<Jim> createImg();
    static std::shared_ptr<Jim> createImg(const std::shared_ptr<Jim> pSrc, bool copyData=true);
    ///Open an image for writing using an external data pointer (not tested yet)
    CPLErr open(void* dataPointer, int ncol, int nrow, int nplane, const GDALDataType& dataType);
    ///Open an image for writing in memory, defining image attributes.
    /* void open(int ncol, int nrow, int nband, int dataType); */
    CPLErr open(Jim& imgSrc, bool copyData=true){
      m_ncol=imgSrc.nrOfCol();
      m_nrow=imgSrc.nrOfRow();
      m_nband=imgSrc.nrOfBand();
      m_dataType=imgSrc.getDataType();
      setProjection(imgSrc.getProjection());
      copyGeoTransform(imgSrc);
      imgSrc.getNoDataValues(m_noDataValues);
      imgSrc.getScale(m_scale);
      imgSrc.getOffset(m_offset);
      initMem(0);
      for(int iband=0;iband<m_nband;++iband){
        m_begin[iband]=0;
        m_end[iband]=m_begin[iband]+m_blockSize;
        if(copyData)
          imgSrc.copyData(m_data[iband],iband);
      }
      //todo: check if filename needs to be set, but as is it is used for writing, I don't think so.
      // if(imgSrc.getFileName()!=""){
      //   m_filename=imgSrc.getFileName();
      // std::cerr << "Warning: filename not set, dataset not defined yet" << std::endl;
      // }
      return(CE_None);
    }
    ///Open dataset (specialization of the open member function of ImgRaster, closing the dataset after reading in memory)
    ///open dataset, read data and close (keep data in memory)
    //CPLErr open(app::AppFactory &app);
    ///write to file previously set (eg., with setFile). Specialization of the writeData member function of ImgRaster, avoiding reset of the memory.
    CPLErr write();
    ///write to file Specialization of the writeData member function of ImgRaster, avoiding reset of the memory.
    CPLErr write(app::AppFactory &app);
    ///Close dataset (specialization of the close member function of ImgRaster, avoiding writing the data)
    CPLErr close(){ImgRaster::reset();};

    ///Create a JSON string from a Jim image
    std::string jim2json();
    ///Create a custom collection from a Jim image
    /* std::string jim2custom(); */
    ///Clone as new shared pointer to ImgRaster object
    /**
     *
     * @return shared pointer to new ImgRaster object alllowing polymorphism
     */
    std::shared_ptr<Jim> clone(bool copyData=true) {
      std::shared_ptr<Jim> pJim=std::dynamic_pointer_cast<Jim>(cloneImpl(copyData));
      if(pJim)
        return(pJim);
      else{
        std::cerr << "Warning: static pointer cast may slice object" << std::endl;
        return(std::static_pointer_cast<Jim>(cloneImpl(copyData)));
      }
    }
    size_t getDataTypeSizeBytes(int band=0) const {
      switch (getDataType()){
      case JDT_UInt64:
      case JDT_Int64:
        return(static_cast<size_t>(8));
      default:
        return(ImgRaster::getDataTypeSizeBytes());
      }
    }
    ///Get the number of planes of this dataset
    int nrOfPlane(void) const { return m_nplane;};
    /// convert single plane multiband image to single band image with multiple planes
    CPLErr band2plane(){};//not implemented yet
    /// convert single band multiple plane image to single plane multiband image
    CPLErr plane2band(){};//not implemented yet
    ///get MIA representation for a particular band
    IMAGE* getMIA(int band=0);
    ///set memory from internal MIA representation for particular band
    CPLErr setMIA(int band=0);
    // ///set memory from MIA representation for particular band
    CPLErr setMIA(IMAGE* mia, int band=0);
    ///convert a GDAL data type to MIA data type
    /**
     *
     *
     * @param aGDALDataType
     *
     * @return MIA data type
     */
    int getMIADataType(){
      switch (getDataType()){
      case GDT_Byte:
        return t_UCHAR;
      case GDT_UInt16:
        return t_USHORT;
      case GDT_Int16:
        return t_SHORT;
      case GDT_UInt32:
        return t_UINT32;
      case GDT_Int32:
        return t_INT32;
      case GDT_Float32:
        return t_FLOAT;
      case GDT_Float64:
        return t_DOUBLE;
      case JDT_UInt64:
        return t_UINT64;
      case JDT_Int64:
        return t_INT64;
      case t_UNSUPPORTED:
        return GDT_Unknown;
      default:
        return GDT_Unknown;
      }
    }
    int JIPLIB2MIADataType(int aJIPLIBDataType){
      //function exists, but introduced for naming consistency
      if(aJIPLIBDataType==JDT_UInt64)
        return(t_UINT64);
      else if(aJIPLIBDataType==JDT_Int64)
        return(t_INT64);
      else
        return(GDAL2MIALDataType(aJIPLIBDataType));
    };
    int GDAL2MIADataType(GDALDataType aGDALDataType){
      //function exists, but introduced for naming consistency
      return(GDAL2MIALDataType(aGDALDataType));
    };
    ///convert a MIA data type to GDAL data type
    /**
     *
     *
     * @param aMIADataType the MIA data type to be converted
     *
     * @return GDAL data type (GDT_Byte, GDT_UInt16, GDT_Int16, GDT_UInt32, GDT_Int32, GDT_Float32, GDT_Float64)
     */
    /* GDALDataType MIA2JIPLIBDataType(int aMIADataType) */
    int MIA2JIPLIBDataType(int aMIADataType)
    {
      switch (aMIADataType){
      case t_UCHAR:
        return GDT_Byte;
      case t_USHORT:
        return GDT_UInt16;
      case t_SHORT:
        return GDT_Int16;
      case t_UINT32:
        return GDT_UInt32;
      case t_INT32:
        return GDT_Int32;
      case t_FLOAT:
        return GDT_Float32;
      case t_DOUBLE:
        return GDT_Float64;
      case t_UINT64:
        return JDT_UInt64;
      case t_INT64:
        return JDT_Int64;
      case t_UNSUPPORTED:
        return GDT_Unknown;
      default:
        return GDT_Unknown;
      }
    };
    ///assignment operator
    Jim& operator=(Jim& imgSrc);
    /* ///relational == operator */
    /* bool operator==(Jim& refImg); */
    ///relational == operator
    bool operator==(std::shared_ptr<Jim> refImg);
    ///test for equality (relational == operator)
    /* bool isEqual(Jim& refImg){return(*this==(refImg));}; */
    ///relational == operator
    bool isEqual(std::shared_ptr<Jim> refImg){return(this->operator==(refImg));};
    /* ///relational != operator */
    /* bool operator!=(Jim& refImg){ return !(this->operator==(refImg)); }; */
    /* ///relational != operator */
    /* bool operator!=(std::shared_ptr<Jim> refImg){ return !(this->operator==(refImg)); }; */
    /* /// perform bitwise shift for a particular band */
    /* CPLErr shift(int value, int iband=0); */
    /* CPLErr magnify(int value, int iband=0); */
    ///crop Jim image in memory returning Jim image
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
    std::shared_ptr<Jim> crop(app::AppFactory& app){
      /* ImgRaster::crop(*this,app); */
      /* return(std::dynamic_pointer_cast<Jim>(shared_from_this())); */
      std::shared_ptr<Jim> imgWriter=Jim::createImg();
      /* std::shared_ptr<Jim> imgWriter=std::make_shared<Jim>(); */
      ImgRaster::crop(*imgWriter, app);
      return(imgWriter);
    }

    std::shared_ptr<Jim> getShared(){
      return(std::dynamic_pointer_cast<Jim>(shared_from_this()));
    }
    //used as a template for functions returning IMAGE* for destructive function
    /* std::shared_ptr<Jim> d_arith(Jim& imRaster_im2, int op, int iband=0); */
    //start insert from fun2method_imagetype
    //end insert from fun2method_imagetype
    //start insert from fun2method_imagelisttype
    //end insert from fun2method_imagelisttype
    //start insert from fun2method_errortype
    //end insert from fun2method_errortype
    //start insert from fun2method_errortype_d
    //end insert from fun2method_errortype_d
    //start insert from fun2method_errortype_nd
    //end insert from fun2method_errortype_nd
    /* std::shared_ptr<Jim> arith(Jim& imRaster_im2, int op, int iband=0); */
    /* CPLErr d_arith(Jim& imRaster_im2, int op, int iband=0); */

///functions from mialib returning an image (example only must be wrapped automated via Python script)
// std::shared_ptr<Jim> mean2d(int width, int iband=0);
// std::shared_ptr<Jim> copy_image(int iband=0);
 //functions from mialib returning image list (manually wrapped)
 /* JimList rotatecoor(double theta, int iband=0); */
 /* JimList imgc(int iband=0); */
//
    /* /\* CPLErr arith(std::shared_ptr<Jim> imgRaster, int theOperation, int band=0); *\/ */
    /* /// perform arithmetic operation for a particular band (non-destructive version) */
    /* std::shared_ptr<jiplib::Jim> getArith(Jim& imgRaster, int theOperation, int iband=0); */
    /* std::shared_ptr<jiplib::Jim> getArith(std::shared_ptr<Jim> imgRaster, int theOperation, int iband=0){return(getArith(*imgRaster,theOperation,iband));}; */
    /* /// perform arithmetic operation with a cst argument for a particular band */
    //CPLErr arithcst(double dcst, int theOperation, int band=0);
    /* /// perform arithmetic operation with a cst argument for a particular band (non-destructive version) */
    /* std::shared_ptr<jiplib::Jim> getArithcst(double dcst, int theOperation, int iband=0); */
    /* /// perform a morphological reconstruction by dilation for a particular band */
    /* CPLErr rdil(std::shared_ptr<Jim> mask, int graph, int flag, int band=0); */
    /* /// perform a morphological reconstruction by dilation for a particular band (non-destructive version) */
    /* std::shared_ptr<jiplib::Jim> getRdil(std::shared_ptr<Jim> mask, int graph, int flag, int iband=0); */
    /* /// perform a morphological reconstruction by erosion for a particular band */
    /* CPLErr rero(std::shared_ptr<Jim> mask, int graph, int flag, int band=0); */
    /* /// perform a morphological reconstruction by erosion for a particular band (non-destructive version) */
    /* std::shared_ptr<jiplib::Jim> getRero(std::shared_ptr<Jim> mask, int graph, int flag, int iband=0); */
    ///get volume (from mialib)
    double getVolume(int iband=0) {
      IMAGE *mia=getMIA(iband);
      ::volume(mia);
      return(mia->vol);
    };
    /* ///read data from with reduced resolution */
    /* CPLErr GDALRead(std::string filename, int band, int nXOff, int nYOff, int nXSize, int nYSize, int nBufXSize=0, int nBufYSize=0); */

    //in memory functions from ImgRaster using AppFactory
    ///filter Jim image and return filtered image as shared pointer
    /**
     * @param input  (type: std::string)Input image file(s). If input contains multiple images, a multi-band output is created
     * @param output  (type: std::string)Output image file
     * @param oformat  (type: std::string) (default: GTiff) Output image format (see also gdal_translate).
     * @param co  (type: std::string)Creation option for output file. Multiple options can be specified.
     * @param a_srs  (type: std::string)Override the spatial reference for the output file (leave blank to copy from input file, use epsg:3035 to use European projection and force to European grid
     * @param mem  (type: unsigned long) (default: 0) Buffer size (in MB) to read image data blocks in memory
     * @param a_srs  (type: std::string)Override the projection for the output file (leave blank to copy from input file, use epsg:3035 to use European projection and force to European grid
     * @param ulx  (type: double) (default: 0) Upper left x value bounding box
     * @param uly  (type: double) (default: 0) Upper left y value bounding box
     * @param lrx  (type: double) (default: 0) Lower right x value bounding box
     * @param lry  (type: double) (default: 0) Lower right y value bounding box
     * @param band  (type: unsigned int)band index to crop (leave empty to retain all bands)
     * @param startband  (type: unsigned int)Start band sequence number
     * @param endband  (type: unsigned int)End band sequence number
     * @param autoscale  (type: double)scale output to min and max, e.g., --autoscale 0 --autoscale 255
     * @param otype  (type: std::string)Data type for output image ({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}). Empty string: inherit type from input image
     * @param oformat  (type: std::string) (default: GTiff) Output image format (see also gdal_translate).
     * @param ct  (type: std::string)color table (file with 5 columns: id R G B ALFA (0: transparent, 255: solid)
     * @param dx  (type: double)Output resolution in x (in meter) (empty: keep original resolution)
     * @param dy  (type: double)Output resolution in y (in meter) (empty: keep original resolution)
     * @param resampling-method  (type: std::string) (default: near) Resampling method (near: nearest neighbor, bilinear: bi-linear interpolation).
     * @param extent  (type: std::string)get boundary from extent from polygons in vector file
     * @param crop_to_cutline  (type: bool) (default: 0) Crop the extent of the target dataset to the extent of the cutline.
     * @param eo  (type: std::string)special extent options controlling rasterization: ATTRIBUTE|CHUNKYSIZE|ALL_TOUCHED|BURN_VALUE_FROM|MERGE_ALG, e.g., -eo ATTRIBUTE=fieldname
     * @param mask  (type: std::string)Use the the specified file as a validity mask (0 is nodata).
     * @param msknodata  (type: double) (default: 0) Mask value not to consider for crop.
     * @param mskband  (type: unsigned int) (default: 0) Mask band to read (0 indexed)
     * @param x  (type: double)x-coordinate of image center to crop (in meter)
     * @param y  (type: double)y-coordinate of image center to crop (in meter)
     * @param nx  (type: double)image size in x to crop (in meter)
     * @param ny  (type: double)image size in y to crop (in meter)
     * @param ns  (type: unsigned int)number of samples  to crop (in pixels)
     * @param nl  (type: unsigned int)number of lines to crop (in pixels)
     * @param scale  (type: double)output=scale*input+offset
     * @param offset  (type: double)output=scale*input+offset
     * @param nodata  (type: double)Nodata value to put in image if out of bounds.
     * @param description  (type: std::string)Set image description
     * @param align  (type: bool) (default: 0) Align output bounding box to input image
     **/
    std::shared_ptr<Jim> filter1d(app::AppFactory& theApp){
      std::shared_ptr<Jim> imgWriter=Jim::createImg();
      /* std::shared_ptr<Jim> imgWriter=std::make_shared<Jim>(); */
      ImgRaster::filter1d(*imgWriter,theApp);
      return(imgWriter);
    }
    std::shared_ptr<Jim> filter2d(app::AppFactory& theApp){
      std::shared_ptr<Jim> imgWriter=Jim::createImg();
      /* std::shared_ptr<Jim> imgWriter=std::make_shared<Jim>(); */
      ImgRaster::filter2d(*imgWriter,theApp);
      return(imgWriter);
    }
    ///create statistical profile
    std::shared_ptr<Jim> statProfile(app::AppFactory& theApp){
      std::shared_ptr<Jim> imgWriter=Jim::createImg();
      /* std::shared_ptr<Jim> imgWriter=std::make_shared<Jim>(); */
      ImgRaster::statProfile(*imgWriter,theApp);
      return(imgWriter);
    }
    /**
     * @param input (type: std::string) input image
     * @param reference (type: std::string) Reference (raster or vector) dataset
     * @param mem (type: unsigned long) (default: 0) Buffer size (in MB) to read image data blocks in memory
     * @param band (type: unsigned int) (default: 0) Input (reference) raster band. Optionally, you can define different bands for input and reference bands respectively: -b 1 -b 0.
     * @param rmse (type: bool) (default: 0) Report root mean squared error
     * @param reg (type: bool) (default: 0) Report linear regression (Input = c0+c1*Reference)
     * @param confusion (type: bool) (default: 0) Create confusion matrix (to std out)
     * @param class (type: std::string) List of class names.
     * @param reclass (type: short) List of class values (use same order as in classname option).
     * @param nodata (type: double) No data value(s) in input or reference dataset are ignored
     * @param mask (type: std::string) Use the first band of the specified file as a validity mask. Nodata values can be set with the option msknodata.
     * @param msknodata (type: double) (default: 0) Mask value(s) where image is invalid. Use negative value for valid data (example: use -t -1: if only -1 is valid value)
     * @param output (type: std::string) Output dataset (optional)
     * @param cmf (type: std::string) (default: ascii) Format for confusion matrix (ascii or latex)
     * @param cmo (type: std::string) Output file for confusion matrix
     * @param se95 (type: bool) (default: 0) Report standard error for 95 confidence interval
     * @param ct (type: std::string) Color table in ASCII format having 5 columns: id R G B ALFA (0: transparent, 255: solid).
     * @param commission (type: short) (default: 2) Value for commission errors: input label < reference label
     * @return output image
     **/
    std::shared_ptr<Jim> diff(app::AppFactory& app){
      std::shared_ptr<Jim> imgWriter=Jim::createImg();
      /* std::shared_ptr<Jim> imgWriter=std::make_shared<Jim>(); */
      ImgRaster::diff(*imgWriter, app);
      return(imgWriter);
    }
    ///supervised classification using support vector machine (train with extractImg/extractOgr)
    /**
     * @param input (type: std::string) input image
     * @param output (type: std::string) Output classification image
     * @param oformat (type: std::string) (default: GTiff) Output image format (see also gdal_translate).
     * @param co (type: std::string) Creation option for output file. Multiple options can be specified.
     * @param mem (type: unsigned long) (default: 0) Buffer size (in MB) to read image data blocks in memory
     * @param training (type: std::string) Training vector file. A single vector file contains all training features (must be set as: b0, b1, b2,...) for all classes (class numbers identified by label option). Use multiple training files for bootstrap aggregation (alternative to the bag and bsize options, where a random subset is taken from a single training file)
     * @param cv (type: unsigned short) (default: 0) N-fold cross validation mode
     * @param cmf (type: std::string) (default: ascii) Format for confusion matrix (ascii or latex)
     * @param tln (type: std::string) Training layer name(s)
     * @param class (type: std::string) List of class names.
     * @param reclass (type: short) List of class values (use same order as in class opt).
     * @param f (type: std::string) (default: SQLite) Output ogr format for active training sample
     * @param ct (type: std::string) Color table in ASCII format having 5 columns: id R G B ALFA (0: transparent, 255: solid)
     * @param label (type: std::string) (default: label) Attribute name for class label in training vector file.
     * @param prior (type: double) (default: 0) Prior probabilities for each class (e.g., -p 0.3 -p 0.3 -p 0.2 ). Used for input only (ignored for cross validation)
     * @param gamma (type: float) (default: 1) Gamma in kernel function
     * @param ccost (type: float) (default: 1000) The parameter C of C_SVC, epsilon_SVR, and nu_SVR
     * @param extent (type: std::string) Only classify within extent from polygons in vector file
     * @param eo (type: std::string) special extent options controlling rasterization: ATTRIBUTE|CHUNKYSIZE|ALL_TOUCHED|BURN_VALUE_FROM|MERGE_ALG, e.g., -eo ATTRIBUTE=fieldname
     * @param mask (type: std::string) Only classify within specified mask. For raster mask, set nodata values with the option msknodata.
     * @param msknodata (type: short) (default: 0) Mask value(s) not to consider for classification. Values will be taken over in classification image.
     * @param nodata (type: unsigned short) (default: 0) Nodata value to put where image is masked as nodata
     * @param band (type: unsigned int) Band index (starting from 0, either use band option or use start to end)
     * @param startband (type: unsigned int) Start band sequence number
     * @param endband (type: unsigned int) End band sequence number
     * @param balance (type: unsigned int) (default: 0) Balance the input data to this number of samples for each class
     * @param min (type: unsigned int) (default: 0) If number of training pixels is less then min, do not take this class into account (0: consider all classes)
     * @param bag (type: unsigned short) (default: 1) Number of bootstrap aggregations
     * @param bagsize (type: int) (default: 100) Percentage of features used from available training features for each bootstrap aggregation (one size for all classes, or a different size for each class respectively
     * @param comb (type: unsigned short) (default: 0) How to combine bootstrap aggregation classifiers (0: sum rule, 1: product rule, 2: max rule). Also used to aggregate classes with rc option.
     * @param classbag (type: std::string) Output for each individual bootstrap aggregation
     * @param prob (type: std::string) Probability image.
     * @param priorimg (type: std::string) (default: ) Prior probability image (multi-band img with band for each class
     * @param offset (type: double) (default: 0) Offset value for each spectral band input features: refl[band]=(DN[band]-offset[band])/scale[band]
     * @param scale (type: double) (default: 0) Scale value for each spectral band input features: refl=(DN[band]-offset[band])/scale[band] (use 0 if scale min and max in each band to -1.0 and 1.0)
     * @param svmtype (type: std::string) (default: C_SVC) Type of SVM (C_SVC, nu_SVC,one_class, epsilon_SVR, nu_SVR)
     * @param kerneltype (type: std::string) (default: radial) Type of kernel function (linear,polynomial,radial,sigmoid)
     * @param kd (type: unsigned short) (default: 3) Degree in kernel function
     * @param coef0 (type: float) (default: 0) Coef0 in kernel function
     * @param nu (type: float) (default: 0.5) The parameter nu of nu_SVC, one_class SVM, and nu_SVR
     * @param eloss (type: float) (default: 0.1) The epsilon in loss function of epsilon_SVR
     * @param cache (type: int) (default: 100) Cache memory size in MB
     * @param etol (type: float) (default: 0.001) The tolerance of termination criterion
     * @param shrink (type: bool) (default: 0) Whether to use the shrinking heuristics
     * @param probest (type: bool) (default: 1) Whether to train a SVC or SVR model for probability estimates
     * @param entropy (type: std::string) (default: ) Entropy image (measure for uncertainty of classifier output
     * @param active (type: std::string) (default: ) Ogr output for active training sample.
     * @param nactive (type: unsigned int) (default: 1) Number of active training points
     * @param random (type: bool) (default: 1) Randomize training data for balancing and bagging
     * @return output image
     **/
    std::shared_ptr<Jim> svm(app::AppFactory& app){
      std::shared_ptr<Jim> imgWriter=Jim::createImg();
      /* std::shared_ptr<Jim> imgWriter=std::make_shared<Jim>(); */
      ImgRaster::svm(*imgWriter, app);
      return(imgWriter);
    }
    std::shared_ptr<Jim> ann(app::AppFactory& app){
      std::shared_ptr<Jim> imgWriter=Jim::createImg();
      /* std::shared_ptr<Jim> imgWriter=std::make_shared<Jim>(); */
      ImgRaster::ann(*imgWriter, app);
      return(imgWriter);
    }
    std::shared_ptr<Jim> classify(app::AppFactory& app){
      std::shared_ptr<Jim> imgWriter=Jim::createImg();
      /* std::shared_ptr<Jim> imgWriter=std::make_shared<Jim>(); */
      ImgRaster::classify(*imgWriter, app);
      return(imgWriter);
    }
    ///stretch Jim image and return stretched image as shared pointer
    std::shared_ptr<Jim> stretch(app::AppFactory& app){
      std::shared_ptr<Jim> imgWriter=Jim::createImg();
      /* std::shared_ptr<Jim> imgWriter=std::make_shared<Jim>(); */
      ImgRaster::stretch(*imgWriter, app);
      return(imgWriter);
    }
    ///Apply thresholds: set to no data if not within thresholds t1 and t2
    std::shared_ptr<Jim> setThreshold(double t1, double t2){
      std::shared_ptr<Jim> imgWriter=std::make_shared<Jim>();
      ImgRaster::setThreshold(*imgWriter,t1,t2);
      return(imgWriter);
    }
    ///Apply thresholds: set to no data if not within thresholds t1 and t2, else set to value
    std::shared_ptr<Jim> setThreshold(double t1, double t2, double value){
      std::shared_ptr<Jim> imgWriter=std::make_shared<Jim>();
      ImgRaster::setThreshold(*imgWriter,t1,t2,value);
      return(imgWriter);
    }
    ///Get mask
    std::shared_ptr<Jim> getMask(app::AppFactory& app){
      std::shared_ptr<Jim> imgWriter=std::make_shared<Jim>();
      ImgRaster::getMask(*imgWriter,app);
      return(imgWriter);
    }
    ///set mask to raster dataset (needs to be implemented in jim.cc because of JimList)
    std::shared_ptr<Jim> setMask(app::AppFactory& app);
    ///set mask to raster dataset (needs to be implemented in jim.cc because of JimList)
    std::shared_ptr<Jim> setMask(JimList& maskList, app::AppFactory& app);
    ///Check for difference with reference image
    CPLErr diff(std::shared_ptr<Jim> refImage,app::AppFactory& app){
      return(ImgRaster::diff(*refImage,app));
    }
    ///Clear all no data values, including the one in GDAL dataset if it is set
    CPLErr clearNoData(int band=0){return(ImgRaster::clearNoData(band));}
    ///reclass raster dataset
    std::shared_ptr<Jim> reclass(app::AppFactory& app){
      std::shared_ptr<Jim> imgWriter=std::make_shared<Jim>();
      ImgRaster::reclass(*imgWriter, app);
      return(imgWriter);
    }


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
      //test
      std::cout << "clone Jim object" << std::endl;
      return std::make_shared<Jim>(*this,copyData);
      /* return(std::make_shared<Jim>()); */
    };
    std::vector<IMAGE*> m_mia;
  };
  static std::shared_ptr<Jim> createJim(app::AppFactory &theApp){return(Jim::createImg(theApp));};
  static std::shared_ptr<Jim> createJim(){return Jim::createImg();};
  static std::shared_ptr<Jim> createJim(const std::shared_ptr<Jim> pSrc, bool copyData=true){return(Jim::createImg(pSrc, copyData));};
}
#endif // _JIM_H_
