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
    }
    ///Open an image for writing using an external data pointer (not tested yet)
    CPLErr open(void* dataPointer, int ncol, int nrow, int nplane, const GDALDataType& dataType);
    ///Open an image for writing in memory, defining image attributes.
    /* void open(int ncol, int nrow, int nband, int dataType); */

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
    ///Create new shared pointer to Jim object
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
    static std::shared_ptr<Jim> createImg(app::AppFactory &theApp){
      std::shared_ptr<Jim> pJim=std::make_shared<Jim>(theApp);
      return(pJim);
    }
    ///this is a testFunction
    static void testFunction(){}
    /* ///Create new shared pointer to Jim object */
    /* /\** */
    /*  * @param input (type: std::string) input filename */
    /*  * @param nodata (type: double) Nodata value to put in image if out of bounds. */
    /*  * @param band (type: int) Bands to open, index starts from 0 */
    /*  * @param ulx (type: double) Upper left x value bounding box */
    /*  * @param uly (type: double) Upper left y value bounding box */
    /*  * @param lrx (type: double) Lower right x value bounding box */
    /*  * @param lry (type: double) Lower right y value bounding box */
    /*  * @param dx (type: double) Resolution in x */
    /*  * @param dy (type: double) Resolution in y */
    /*  * @param resample (type: std::string) (default: GRIORA_NearestNeighbour) resample: GRIORA_NearestNeighbour|GRIORA_Bilinear|GRIORA_Cubic|GRIORA_CubicSpline|GRIORA_Lanczos|GRIORA_Average|GRIORA_Average|GRIORA_Gauss (check http://www.gdal.org/gdal_8h.html#a640ada511cbddeefac67c548e009d5a) */
    /*  * @param extent (type: std::string) get boundary from extent from polygons in vector file */
    /*  * @param mem (type: unsigned long) (default: 0) Buffer size (in MB) to read image data blocks in memory */
    /*  * @param ncol (type: int) Number of columns */
    /*  * @param nrow (type: int) Number of rows */
    /*  * @param nband (type: int) (default: 1) Number of bands */
    /*  * @param otype (type: std::string) (default: Byte) Data type for output image ({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}) */
    /*  * @param seed (type: unsigned long) (default: 0) seed value for random generator */
    /*  * @param mean (type: double) (default: 0) Mean value for random generator */
    /*  * @param sigma (type: double) (default: 0) Sigma value for random generator */
    /*  * @param description (type: std::string) Set image description */
    /*  * @param a_srs (type: std::string) Assign the spatial reference for the output file, e.g., psg:3035 to use European projection and force to European grid */
    /*  * @return shared pointer to new Jim object */
    /*  **\/ */
    static std::shared_ptr<Jim> createImg() {
      return(std::make_shared<Jim>());
    };
    /* ///Create new shared pointer to Jim object */
    /* /\** */
    /*  * @param input (type: std::string) input filename */
    /*  * @param nodata (type: double) Nodata value to put in image if out of bounds. */
    /*  * @param band (type: int) Bands to open, index starts from 0 */
    /*  * @param ulx (type: double) Upper left x value bounding box */
    /*  * @param uly (type: double) Upper left y value bounding box */
    /*  * @param lrx (type: double) Lower right x value bounding box */
    /*  * @param lry (type: double) Lower right y value bounding box */
    /*  * @param dx (type: double) Resolution in x */
    /*  * @param dy (type: double) Resolution in y */
    /*  * @param resample (type: std::string) (default: GRIORA_NearestNeighbour) resample: GRIORA_NearestNeighbour|GRIORA_Bilinear|GRIORA_Cubic|GRIORA_CubicSpline|GRIORA_Lanczos|GRIORA_Average|GRIORA_Average|GRIORA_Gauss (check http://www.gdal.org/gdal_8h.html#a640ada511cbddeefac67c548e009d5a) */
    /*  * @param extent (type: std::string) get boundary from extent from polygons in vector file */
    /*  * @param mem (type: unsigned long) (default: 0) Buffer size (in MB) to read image data blocks in memory */
    /*  * @param ncol (type: int) Number of columns */
    /*  * @param nrow (type: int) Number of rows */
    /*  * @param nband (type: int) (default: 1) Number of bands */
    /*  * @param otype (type: std::string) (default: Byte) Data type for output image ({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}) */
    /*  * @param seed (type: unsigned long) (default: 0) seed value for random generator */
    /*  * @param mean (type: double) (default: 0) Mean value for random generator */
    /*  * @param sigma (type: double) (default: 0) Sigma value for random generator */
    /*  * @param description (type: std::string) Set image description */
    /*  * @param a_srs (type: std::string) Assign the spatial reference for the output file, e.g., psg:3035 to use European projection and force to European grid */
    /*  * @return shared pointer to new Jim object */
    /*  **\/ */
    static std::shared_ptr<Jim> createImg(const std::shared_ptr<Jim> pSrc, bool copyData=true){
      std::shared_ptr<Jim> pJim=std::make_shared<Jim>(*pSrc,copyData);
      return(pJim);
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
    GDALDataType MIA2GDALDataType(int aMIADataType)
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
        // case t_UINT64:
        //   return GDT_UInt64;
        // case t_INT64:
        //   return GDT_Int64;
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
    //used as a template for functions returning IMAGE* with destructive flag
    /* std::shared_ptr<Jim> arith(Jim& imRaster_im2, int op, int iband=0, bool destructive=false); */
    //start insert from fun2method_imagetype
std::shared_ptr<Jim> attribute(int  type, int  oporclo, double  lambdaVal, int  graph, int iband=0);
std::shared_ptr<Jim> GreyAreaOpening(int  lambdaVal, int  graph, int iband=0);
std::shared_ptr<Jim> GreyAreaClosing(int  lambdaVal, int  graph, int iband=0);
std::shared_ptr<Jim> GreyAreaOpeningROI(int  lambdaVal, int  graph, int iband=0);
std::shared_ptr<Jim> GreyAreaClosingROI(int  lambdaVal, int  graph, int iband=0);
std::shared_ptr<Jim> chull(int  graph, int iband=0);
std::shared_ptr<Jim> hpclose(int  dx, int  dy, int iband=0);
std::shared_ptr<Jim> hpcloseti(int  dx, int  dy, int iband=0);
std::shared_ptr<Jim> sqedt(int iband=0);
std::shared_ptr<Jim> iz(int iband=0);
std::shared_ptr<Jim> ced(Jim& imRaster_mask, int iband=0);
std::shared_ptr<Jim> convolve(Jim& imRaster_imse, Jim& imRaster_imweight, int  ox, int  oy, int  oz, int iband=0);
std::shared_ptr<Jim> convolvedownsample(Jim& imRaster_imse, Jim& imRaster_imweight, int  w, int  ox, int  oy, int  oz, int iband=0);
std::shared_ptr<Jim> rsum2d(int iband=0);
std::shared_ptr<Jim> rsum3d(int iband=0);
std::shared_ptr<Jim> rsumsq2d(int iband=0);
std::shared_ptr<Jim> mean2d(int  width, int iband=0);
std::shared_ptr<Jim> mean2dse(Jim& imRaster_imse, int  ox, int  oy, int iband=0);
std::shared_ptr<Jim> variance2dse(Jim& imRaster_imse, int  ox, int  oy, int iband=0);
std::shared_ptr<Jim> phase_correlation(Jim& imRaster_im_template, int iband=0);
std::shared_ptr<Jim> dirmean(Jim& imRaster_imy, Jim& imRaster_imse, int  ox, int  oy, int  oz, int iband=0);
std::shared_ptr<Jim> coherence(Jim& imRaster_imy, Jim& imRaster_imse, int  ox, int  oy, int  oz, int iband=0);
std::shared_ptr<Jim> coor_extrema_paraboloid(int iband=0);
std::shared_ptr<Jim> fitlinear(IMAGE  * yarray, int iband=0);
std::shared_ptr<Jim> transgrad(int  graph, int iband=0);
std::shared_ptr<Jim> region_lut(int  graph, int  type, int  param1, int  param2, int iband=0);
std::shared_ptr<Jim> region_lut_seq(int  graph, int  type, int iband=0);
std::shared_ptr<Jim> region_im_lut(Jim& imRaster_im, int  graph, int  type, float  aval, int iband=0);
std::shared_ptr<Jim> contortion_lut(int  graph, int iband=0);
std::shared_ptr<Jim> alphacc(Jim& imRaster_dissy, int  alpha, int iband=0);
std::shared_ptr<Jim> labelvertex(int  alpha, int  graph, int iband=0);
std::shared_ptr<Jim> vertexseparation(int  graph, int  type, int iband=0);
std::shared_ptr<Jim> labelvertexconnectedness(int  alpha, int  graph, int  deg, int iband=0);
std::shared_ptr<Jim> labelcc(Jim& imRaster_imse, int  ox, int  oy, int  oz, int  rg, int  rl, int iband=0);
std::shared_ptr<Jim> labelccmi(Jim& imRaster_immi, Jim& imRaster_imse, int  ox, int  oy, int  oz, int  rg, int  rl, int iband=0);
std::shared_ptr<Jim> labelci(Jim& imRaster_imse, int  ox, int  oy, int  oz, int  rl, int iband=0);
std::shared_ptr<Jim> labelccdissim(Jim& imRaster_imh, Jim& imRaster_imv, int  rg, int  rl, int iband=0);
std::shared_ptr<Jim> labelccvar(Jim& imRaster_imse, int  ox, int  oy, int  oz, int  rg, int  rl, double  varmax, int iband=0);
std::shared_ptr<Jim> labelccattr(int  graph, int  rg, int  rl, int iband=0);
std::shared_ptr<Jim> edgeweight(int  dir, int  type, int iband=0);
std::shared_ptr<Jim> dbscan(double  eps, int  MinPts, int iband=0);
std::shared_ptr<Jim> outeredgelut(Jim& imRaster_iedgelbl, int iband=0);
std::shared_ptr<Jim> outeredge(int  graph, int iband=0);
std::shared_ptr<Jim> outercontour(int  graph, int iband=0);
std::shared_ptr<Jim> erode(Jim& imRaster_imse, int  ox, int  oy, int  oz, int  trflag, int iband=0);
std::shared_ptr<Jim> dilate(Jim& imRaster_imse, int  ox, int  oy, int  oz, int  trflag, int iband=0);
std::shared_ptr<Jim> volerode(Jim& imRaster_imse, Jim& imRaster_imweight, int  ox, int  oy, int  oz, int iband=0);
std::shared_ptr<Jim> rank(Jim& imRaster_imse, int  rank, int  ox, int  oy, int  oz, int  trflag, int iband=0);
std::shared_ptr<Jim> squarerank(int  k, int  rank, int  ox, int  oy, int iband=0);
std::shared_ptr<Jim> squarevol(int  k, int  ox, int  oy, int iband=0);
std::shared_ptr<Jim> lrankti(int  dx, int  dy, int  k, int  rank, int  o, int  t, int  tr, int iband=0);
std::shared_ptr<Jim> erodelabel(int  graph, int iband=0);
std::shared_ptr<Jim> to_tiff1bitpp(int iband=0);
std::shared_ptr<Jim> to_tiff4bitpp(int iband=0);
std::shared_ptr<Jim> to_ushort(int iband=0);
std::shared_ptr<Jim> to_int32(int iband=0);
std::shared_ptr<Jim> to_float(int iband=0);
std::shared_ptr<Jim> to_double(int iband=0);
std::shared_ptr<Jim> deinterleave(int iband=0);
std::shared_ptr<Jim> imhsi2rgb(Jim& imRaster_ims, Jim& imRaster_imi, int iband=0);
std::shared_ptr<Jim> imhls2rgb(Jim& imRaster_ims, Jim& imRaster_imi, int iband=0);
std::shared_ptr<Jim> crgb2rgb(Jim& imRaster_ims, Jim& imRaster_imi, int iband=0);
std::shared_ptr<Jim> histo1d(int iband=0);
std::shared_ptr<Jim> histo2d(Jim& imRaster_im2, int iband=0);
std::shared_ptr<Jim> histo3d(Jim& imRaster_im2, Jim& imRaster_im3, int iband=0);
std::shared_ptr<Jim> rsum(int iband=0);
std::shared_ptr<Jim> lookuprgb(Jim& imRaster_img, Jim& imRaster_imb, Jim& imRaster_imlut, int iband=0);
std::shared_ptr<Jim> class2d(Jim& imRaster_im2, Jim& imRaster_imlut, int iband=0);
std::shared_ptr<Jim> area(int  r, int  type, int iband=0);
std::shared_ptr<Jim> dirsum(int  dir, int iband=0);
std::shared_ptr<Jim> sortindex(int iband=0);
std::shared_ptr<Jim> ssda(Jim& imRaster_imt, int  xi, int  yi, int  w, int iband=0);
std::shared_ptr<Jim> ncclewis(Jim& imRaster_imt, Jim& imRaster_sim, Jim& imRaster_ssqim, int  xi, int  yi, int  w, int iband=0);
std::shared_ptr<Jim> ncc(Jim& imRaster_imt, int  xi, int  yi, int  w, int iband=0);
std::shared_ptr<Jim> copy_image(int iband=0);
std::shared_ptr<Jim> imtoarray(Jim& imRaster_imroi, int iband=0);
std::shared_ptr<Jim> arraytoim(Jim& imRaster_imroi, int iband=0);
std::shared_ptr<Jim> minima(int  graph, int iband=0);
std::shared_ptr<Jim> sqtgsym(Jim& imRaster_im_r, int  graph, int iband=0);
std::shared_ptr<Jim> d8(int iband=0);
std::shared_ptr<Jim> slope8(int iband=0);
std::shared_ptr<Jim> flow(int  graph, int iband=0);
std::shared_ptr<Jim> flownew(Jim& imRaster_imdir, int  graph, int iband=0);
std::shared_ptr<Jim> cda(int  graph, int iband=0);
std::shared_ptr<Jim> stratify(Jim& imRaster_thresh, Jim& imRaster_dir, int iband=0);
std::shared_ptr<Jim> dinf(int iband=0);
std::shared_ptr<Jim> cdainf(int iband=0);
std::shared_ptr<Jim> slopeinf(int iband=0);
std::shared_ptr<Jim> aflood(Jim& imRaster_imr, int  graph, int  maxfl, int iband=0);
std::shared_ptr<Jim> fillocarve(Jim& imRaster_imr, int  graph, int  maxfl, int  flag, int iband=0);
std::shared_ptr<Jim> FlatDir(Jim& imRaster_im, int  graph, int iband=0);
std::shared_ptr<Jim> htop(Jim& imRaster_d8, int iband=0);
std::shared_ptr<Jim> shade(int  dir, int iband=0);
std::shared_ptr<Jim> LineDilate3D(float  dh, int iband=0);
std::shared_ptr<Jim> grid(Jim& imRaster_roi, Jim& imRaster_imx, Jim& imRaster_imy, float  alpha, int iband=0);
std::shared_ptr<Jim> epc(Jim& imRaster_lut, int iband=0);
std::shared_ptr<Jim> epcgrey(Jim& imRaster_lut, int iband=0);
std::shared_ptr<Jim> switchop(Jim& imRaster_imse, int  ox, int  oy, int  oz, int iband=0);
std::shared_ptr<Jim> ws(int  graph, int iband=0);
std::shared_ptr<Jim> imcut(int  x1, int  y1, int  z1, int  x2, int  y2, int  z2, int iband=0);
std::shared_ptr<Jim> getboundingbox(int iband=0);
std::shared_ptr<Jim> magnify(int  n, int iband=0);
    //end insert from fun2method_imagetype
    //start insert from fun2method_errortype
std::shared_ptr<Jim> dst2d4(int iband=0, bool destructive=false);
std::shared_ptr<Jim> dst2dchamfer(int iband=0, bool destructive=false);
std::shared_ptr<Jim> chamfer2d(int  type, int iband=0, bool destructive=false);
std::shared_ptr<Jim> oiiz(int iband=0, bool destructive=false);
std::shared_ptr<Jim> geodist(Jim& imRaster_im_r, int  graph, int iband=0, bool destructive=false);
std::shared_ptr<Jim> azimuth(Jim& imRaster_iy, int iband=0, bool destructive=false);
std::shared_ptr<Jim> mapori(int  ox, int  oy, int iband=0, bool destructive=false);
std::shared_ptr<Jim> label(Jim& imRaster_im2, int  ox, int  oy, int  oz, int iband=0, bool destructive=false);
std::shared_ptr<Jim> labelpixngb(Jim& imRaster_im2, int  ox, int  oy, int  oz, int iband=0, bool destructive=false);
std::shared_ptr<Jim> labelplat(Jim& imRaster_im2, int  ox, int  oy, int  oz, int iband=0, bool destructive=false);
std::shared_ptr<Jim> seededlabelplat(Jim& imRaster_im2, Jim& imRaster_im3, int  ox, int  oy, int  oz, int iband=0, bool destructive=false);
std::shared_ptr<Jim> seededplat(Jim& imRaster_im2, Jim& imRaster_im3, int  ox, int  oy, int  oz, int iband=0, bool destructive=false);
std::shared_ptr<Jim> labelpix(int iband=0, bool destructive=false);
std::shared_ptr<Jim> resolveLabels(Jim& imRaster_imlut, Jim& imRaster_imlutback, int  graph, int iband=0, bool destructive=false);
std::shared_ptr<Jim> gorder(Jim& imRaster_g, int  n, int iband=0, bool destructive=false);
std::shared_ptr<Jim> propagate(Jim& imRaster_dst, IMAGE ** imap, int  nc, int  graph, int iband=0, bool destructive=false);
std::shared_ptr<Jim> set_regions(Jim& imRaster_ival, int  indic, int iband=0, bool destructive=false);
std::shared_ptr<Jim> setregionsgraph(Jim& imRaster_ival, int  indic, int  graph, int iband=0, bool destructive=false);
std::shared_ptr<Jim> tessel_surface(int iband=0, bool destructive=false);
std::shared_ptr<Jim> relabel(Jim& imRaster_ilbl2, Jim& imRaster_iarea2, int iband=0, bool destructive=false);
std::shared_ptr<Jim> linero(int  dx, int  dy, int  n, int  line_type, int iband=0, bool destructive=false);
std::shared_ptr<Jim> lindil(int  dx, int  dy, int  n, int  line_type, int iband=0, bool destructive=false);
std::shared_ptr<Jim> herkpldil(int  dx, int  dy, int  k, int  o, int  t, int iband=0, bool destructive=false);
std::shared_ptr<Jim> herkplero(int  dx, int  dy, int  k, int  o, int  t, int iband=0, bool destructive=false);
std::shared_ptr<Jim> linerank(int  dx, int  dy, int  k, int  rank, int  o, int iband=0, bool destructive=false);
std::shared_ptr<Jim> write_ColorMap_tiff(char * fn, int iband=0, bool destructive=false);
std::shared_ptr<Jim> write_tiff(char * fn, int iband=0, bool destructive=false);
std::shared_ptr<Jim> writeTiffOneStripPerLine(char * fn, char * desc, int iband=0, bool destructive=false);
std::shared_ptr<Jim> writeGeoTiffOneStripPerLine(char * fn, int  PCSCode, double  xoff, double  yoff, double  scale, unsigned short  RasterType, int  nodata_flag, int  nodata_val, int  metadata_flag, char * metadata_str, int iband=0, bool destructive=false);
std::shared_ptr<Jim> to_uchar(int iband=0, bool destructive=false);
std::shared_ptr<Jim> dbltofloat(int iband=0, bool destructive=false);
std::shared_ptr<Jim> uint32_to_float(int iband=0, bool destructive=false);
std::shared_ptr<Jim> swap(int iband=0, bool destructive=false);
std::shared_ptr<Jim> getfirstmaxpos(unsigned long int * pos, int iband=0, bool destructive=false);
std::shared_ptr<Jim> histcompress(int iband=0, bool destructive=false);
std::shared_ptr<Jim> lookup(Jim& imRaster_imlut, int iband=0, bool destructive=false);
std::shared_ptr<Jim> lookuptypematch(Jim& imRaster_imlut, int iband=0, bool destructive=false);
std::shared_ptr<Jim> volume(int iband=0, bool destructive=false);
std::shared_ptr<Jim> dirmax(int  dir, int iband=0, bool destructive=false);
std::shared_ptr<Jim> imequalp(Jim& imRaster_im2, int iband=0, bool destructive=false);
std::shared_ptr<Jim> getmax(double * maxval, int iband=0, bool destructive=false);
std::shared_ptr<Jim> getminmax(double * minval, double * maxval, int iband=0, bool destructive=false);
std::shared_ptr<Jim> classstatsinfo(Jim& imRaster_imin, int iband=0, bool destructive=false);
std::shared_ptr<Jim> clmindist(Jim& imRaster_imin, int  bklabel, int  mode, double  thr, int iband=0, bool destructive=false);
std::shared_ptr<Jim> clparpip(Jim& imRaster_imin, int  bklabel, int  mode, double  mult, int iband=0, bool destructive=false);
std::shared_ptr<Jim> clmaha(Jim& imRaster_imin, int  bklabel, int  mode, double  thr, int iband=0, bool destructive=false);
std::shared_ptr<Jim> clmaxlike(Jim& imRaster_imin, int  bklabel, int  type, double  thr, int iband=0, bool destructive=false);
std::shared_ptr<Jim> iminfo(int iband=0, bool destructive=false);
std::shared_ptr<Jim> copy_lut(Jim& imRaster_im2, int iband=0, bool destructive=false);
std::shared_ptr<Jim> create_lut(int iband=0, bool destructive=false);
std::shared_ptr<Jim> setpixval(unsigned long  offset, double d_g, int iband=0, bool destructive=false);
std::shared_ptr<Jim> rdil(Jim& imRaster_mask, int  graph, int  flag, int iband=0, bool destructive=false);
std::shared_ptr<Jim> rero(Jim& imRaster_mask, int  graph, int  flag, int iband=0, bool destructive=false);
std::shared_ptr<Jim> rerodilp(Jim& imRaster_mask, int  graph, int  flag, int  version, int iband=0, bool destructive=false);
std::shared_ptr<Jim> complete(Jim& imRaster_im_rmin, int  graph, int iband=0, bool destructive=false);
std::shared_ptr<Jim> sqtgpla(Jim& imRaster_im_r, int  graph, int iband=0, bool destructive=false);
std::shared_ptr<Jim> sqtg(Jim& imRaster_im_r, int  graph, int iband=0, bool destructive=false);
std::shared_ptr<Jim> dir(int  graph, int iband=0, bool destructive=false);
std::shared_ptr<Jim> cboutlet(Jim& imRaster_d8, int iband=0, bool destructive=false);
std::shared_ptr<Jim> cbconfluence(Jim& imRaster_d8, int iband=0, bool destructive=false);
std::shared_ptr<Jim> strahler(int iband=0, bool destructive=false);
std::shared_ptr<Jim> FlatIGeodAFAB(Jim& imRaster_im, int  graph, int iband=0, bool destructive=false);
std::shared_ptr<Jim> skeleton(int iband=0, bool destructive=false);
std::shared_ptr<Jim> bprune(int  occa, int  graph, int iband=0, bool destructive=false);
std::shared_ptr<Jim> oiskeleton(Jim& imRaster_imanchor, int iband=0, bool destructive=false);
std::shared_ptr<Jim> oiask(Jim& imRaster_imanchor, int iband=0, bool destructive=false);
std::shared_ptr<Jim> binODthin_noqueue(int  stype, int  atype, Jim& imRaster_imanchor, int iband=0, bool destructive=false);
std::shared_ptr<Jim> binODthin_FIFO(int  stype, int  atype, Jim& imRaster_imanchor, int iband=0, bool destructive=false);
std::shared_ptr<Jim> binOIthin_noqueue(int  stype, int  atype, Jim& imRaster_imanchor, int iband=0, bool destructive=false);
std::shared_ptr<Jim> binOIthin_FIFO(int  stype, int  atype, Jim& imRaster_imanchor, int iband=0, bool destructive=false);
std::shared_ptr<Jim> wsfah(Jim& imRaster_imr, int  graph, int  maxfl, int iband=0, bool destructive=false);
std::shared_ptr<Jim> skelfah(Jim& imRaster_imr, Jim& imRaster_imdir, int  graph, int  maxfl, int iband=0, bool destructive=false);
std::shared_ptr<Jim> skelfah2(Jim& imRaster_impskp, int  n, int  graph, int iband=0, bool destructive=false);
std::shared_ptr<Jim> compose(Jim& imRaster_mask, Jim& imRaster_g, Jim& imRaster_lbl, int  graph, int iband=0, bool destructive=false);
std::shared_ptr<Jim> oiws(int iband=0, bool destructive=false);
std::shared_ptr<Jim> srg(Jim& imRaster_im2, Jim& imRaster_im3, int  ox, int  oy, int  oz, int iband=0, bool destructive=false);
std::shared_ptr<Jim> IsPartitionEqual(Jim& imRaster_im2, int * result, int iband=0, bool destructive=false);
std::shared_ptr<Jim> IsPartitionFiner(Jim& imRaster_im2, int  graph, unsigned long int * res, int iband=0, bool destructive=false);
std::shared_ptr<Jim> framebox(int * box, double d_gval, int iband=0, bool destructive=false);
std::shared_ptr<Jim> addframebox(int * box, double d_gval, int iband=0, bool destructive=false);
std::shared_ptr<Jim> subframebox(int * box, int iband=0, bool destructive=false);
std::shared_ptr<Jim> dumpxyz(int  x, int  y, int  z, int  dx, int  dy, int iband=0, bool destructive=false);
std::shared_ptr<Jim> imputop(Jim& imRaster_im2, int  x, int  y, int  z, int  op, int iband=0, bool destructive=false);
std::shared_ptr<Jim> imputcompose(Jim& imRaster_imlbl, Jim& imRaster_im2, int  x, int  y, int  z, int  val, int iband=0, bool destructive=false);
std::shared_ptr<Jim> szcompat(Jim& imRaster_im2, int iband=0, bool destructive=false);
std::shared_ptr<Jim> szgeocompat(Jim& imRaster_im2, int iband=0, bool destructive=false);
std::shared_ptr<Jim> plotline(int  x1, int  y1, int  x2, int  y2, int  val, int iband=0, bool destructive=false);
std::shared_ptr<Jim> ovlmatrix(Jim& imRaster_maxg_array, char * odir, int iband=0, bool destructive=false);
std::shared_ptr<Jim> bitwise_op(Jim& imRaster_im2, int  op, int iband=0, bool destructive=false);
std::shared_ptr<Jim> negation(int iband=0, bool destructive=false);
std::shared_ptr<Jim> arith(Jim& imRaster_im2, int  op, int iband=0, bool destructive=false);
std::shared_ptr<Jim> arithcst(double d_gt, int  op, int iband=0, bool destructive=false);
std::shared_ptr<Jim> imabs(int iband=0, bool destructive=false);
std::shared_ptr<Jim> imsqrt(int iband=0, bool destructive=false);
std::shared_ptr<Jim> imlog(int iband=0, bool destructive=false);
std::shared_ptr<Jim> imatan(int iband=0, bool destructive=false);
std::shared_ptr<Jim> imcos(int iband=0, bool destructive=false);
std::shared_ptr<Jim> imacos(int iband=0, bool destructive=false);
std::shared_ptr<Jim> imsin(int iband=0, bool destructive=false);
std::shared_ptr<Jim> imasin(int iband=0, bool destructive=false);
std::shared_ptr<Jim> thresh(double d_gt1, double d_gt2, double d_gbg, double d_gfg, int iband=0, bool destructive=false);
std::shared_ptr<Jim> setlevel(double d_gt1, double d_gt2, double d_gval, int iband=0, bool destructive=false);
std::shared_ptr<Jim> modulo(int  val, int iband=0, bool destructive=false);
std::shared_ptr<Jim> complement(int iband=0, bool destructive=false);
std::shared_ptr<Jim> power2p(int iband=0, bool destructive=false);
std::shared_ptr<Jim> blank(double d_gval, int iband=0, bool destructive=false);
std::shared_ptr<Jim> shift(int  val, int iband=0, bool destructive=false);
std::shared_ptr<Jim> setrange(double d_gt1, double d_gt2, int iband=0, bool destructive=false);
std::shared_ptr<Jim> FindPixWithVal(double d_gval, unsigned long int * ofs, int iband=0, bool destructive=false);
    //end insert from fun2method_errortype
    //start insert from fun2method_errortype_d
CPLErr d_dst2d4(int iband=0);
CPLErr d_dst2dchamfer(int iband=0);
CPLErr d_chamfer2d(int  type, int iband=0);
CPLErr d_oiiz(int iband=0);
CPLErr d_geodist(Jim& imRaster_im_r, int  graph, int iband=0);
CPLErr d_azimuth(Jim& imRaster_iy, int iband=0);
CPLErr d_mapori(int  ox, int  oy, int iband=0);
CPLErr d_label(Jim& imRaster_im2, int  ox, int  oy, int  oz, int iband=0);
CPLErr d_labelpixngb(Jim& imRaster_im2, int  ox, int  oy, int  oz, int iband=0);
CPLErr d_labelplat(Jim& imRaster_im2, int  ox, int  oy, int  oz, int iband=0);
CPLErr d_seededlabelplat(Jim& imRaster_im2, Jim& imRaster_im3, int  ox, int  oy, int  oz, int iband=0);
CPLErr d_seededplat(Jim& imRaster_im2, Jim& imRaster_im3, int  ox, int  oy, int  oz, int iband=0);
CPLErr d_labelpix(int iband=0);
CPLErr d_resolveLabels(Jim& imRaster_imlut, Jim& imRaster_imlutback, int  graph, int iband=0);
CPLErr d_gorder(Jim& imRaster_g, int  n, int iband=0);
CPLErr d_propagate(Jim& imRaster_dst, IMAGE ** imap, int  nc, int  graph, int iband=0);
CPLErr d_set_regions(Jim& imRaster_ival, int  indic, int iband=0);
CPLErr d_setregionsgraph(Jim& imRaster_ival, int  indic, int  graph, int iband=0);
CPLErr d_tessel_surface(int iband=0);
CPLErr d_relabel(Jim& imRaster_ilbl2, Jim& imRaster_iarea2, int iband=0);
CPLErr d_linero(int  dx, int  dy, int  n, int  line_type, int iband=0);
CPLErr d_lindil(int  dx, int  dy, int  n, int  line_type, int iband=0);
CPLErr d_herkpldil(int  dx, int  dy, int  k, int  o, int  t, int iband=0);
CPLErr d_herkplero(int  dx, int  dy, int  k, int  o, int  t, int iband=0);
CPLErr d_linerank(int  dx, int  dy, int  k, int  rank, int  o, int iband=0);
CPLErr d_write_ColorMap_tiff(char * fn, int iband=0);
CPLErr d_write_tiff(char * fn, int iband=0);
CPLErr d_writeTiffOneStripPerLine(char * fn, char * desc, int iband=0);
CPLErr d_writeGeoTiffOneStripPerLine(char * fn, int  PCSCode, double  xoff, double  yoff, double  scale, unsigned short  RasterType, int  nodata_flag, int  nodata_val, int  metadata_flag, char * metadata_str, int iband=0);
CPLErr d_to_uchar(int iband=0);
CPLErr d_dbltofloat(int iband=0);
CPLErr d_uint32_to_float(int iband=0);
CPLErr d_swap(int iband=0);
CPLErr d_getfirstmaxpos(unsigned long int * pos, int iband=0);
CPLErr d_histcompress(int iband=0);
CPLErr d_lookup(Jim& imRaster_imlut, int iband=0);
CPLErr d_lookuptypematch(Jim& imRaster_imlut, int iband=0);
CPLErr d_volume(int iband=0);
CPLErr d_dirmax(int  dir, int iband=0);
CPLErr d_imequalp(Jim& imRaster_im2, int iband=0);
CPLErr d_getmax(double * maxval, int iband=0);
CPLErr d_getminmax(double * minval, double * maxval, int iband=0);
CPLErr d_classstatsinfo(Jim& imRaster_imin, int iband=0);
CPLErr d_clmindist(Jim& imRaster_imin, int  bklabel, int  mode, double  thr, int iband=0);
CPLErr d_clparpip(Jim& imRaster_imin, int  bklabel, int  mode, double  mult, int iband=0);
CPLErr d_clmaha(Jim& imRaster_imin, int  bklabel, int  mode, double  thr, int iband=0);
CPLErr d_clmaxlike(Jim& imRaster_imin, int  bklabel, int  type, double  thr, int iband=0);
CPLErr d_iminfo(int iband=0);
CPLErr d_copy_lut(Jim& imRaster_im2, int iband=0);
CPLErr d_create_lut(int iband=0);
CPLErr d_setpixval(unsigned long  offset, double d_g, int iband=0);
CPLErr d_rdil(Jim& imRaster_mask, int  graph, int  flag, int iband=0);
CPLErr d_rero(Jim& imRaster_mask, int  graph, int  flag, int iband=0);
CPLErr d_rerodilp(Jim& imRaster_mask, int  graph, int  flag, int  version, int iband=0);
CPLErr d_complete(Jim& imRaster_im_rmin, int  graph, int iband=0);
CPLErr d_sqtgpla(Jim& imRaster_im_r, int  graph, int iband=0);
CPLErr d_sqtg(Jim& imRaster_im_r, int  graph, int iband=0);
CPLErr d_dir(int  graph, int iband=0);
CPLErr d_cboutlet(Jim& imRaster_d8, int iband=0);
CPLErr d_cbconfluence(Jim& imRaster_d8, int iband=0);
CPLErr d_strahler(int iband=0);
CPLErr d_FlatIGeodAFAB(Jim& imRaster_im, int  graph, int iband=0);
CPLErr d_skeleton(int iband=0);
CPLErr d_bprune(int  occa, int  graph, int iband=0);
CPLErr d_oiskeleton(Jim& imRaster_imanchor, int iband=0);
CPLErr d_oiask(Jim& imRaster_imanchor, int iband=0);
CPLErr d_binODthin_noqueue(int  stype, int  atype, Jim& imRaster_imanchor, int iband=0);
CPLErr d_binODthin_FIFO(int  stype, int  atype, Jim& imRaster_imanchor, int iband=0);
CPLErr d_binOIthin_noqueue(int  stype, int  atype, Jim& imRaster_imanchor, int iband=0);
CPLErr d_binOIthin_FIFO(int  stype, int  atype, Jim& imRaster_imanchor, int iband=0);
CPLErr d_wsfah(Jim& imRaster_imr, int  graph, int  maxfl, int iband=0);
CPLErr d_skelfah(Jim& imRaster_imr, Jim& imRaster_imdir, int  graph, int  maxfl, int iband=0);
CPLErr d_skelfah2(Jim& imRaster_impskp, int  n, int  graph, int iband=0);
CPLErr d_compose(Jim& imRaster_mask, Jim& imRaster_g, Jim& imRaster_lbl, int  graph, int iband=0);
CPLErr d_oiws(int iband=0);
CPLErr d_srg(Jim& imRaster_im2, Jim& imRaster_im3, int  ox, int  oy, int  oz, int iband=0);
CPLErr d_IsPartitionEqual(Jim& imRaster_im2, int * result, int iband=0);
CPLErr d_IsPartitionFiner(Jim& imRaster_im2, int  graph, unsigned long int * res, int iband=0);
CPLErr d_framebox(int * box, double d_gval, int iband=0);
CPLErr d_addframebox(int * box, double d_gval, int iband=0);
CPLErr d_subframebox(int * box, int iband=0);
CPLErr d_dumpxyz(int  x, int  y, int  z, int  dx, int  dy, int iband=0);
CPLErr d_imputop(Jim& imRaster_im2, int  x, int  y, int  z, int  op, int iband=0);
CPLErr d_imputcompose(Jim& imRaster_imlbl, Jim& imRaster_im2, int  x, int  y, int  z, int  val, int iband=0);
CPLErr d_szcompat(Jim& imRaster_im2, int iband=0);
CPLErr d_szgeocompat(Jim& imRaster_im2, int iband=0);
CPLErr d_plotline(int  x1, int  y1, int  x2, int  y2, int  val, int iband=0);
CPLErr d_ovlmatrix(Jim& imRaster_maxg_array, char * odir, int iband=0);
CPLErr d_bitwise_op(Jim& imRaster_im2, int  op, int iband=0);
CPLErr d_negation(int iband=0);
CPLErr d_arith(Jim& imRaster_im2, int  op, int iband=0);
CPLErr d_arithcst(double d_gt, int  op, int iband=0);
CPLErr d_imabs(int iband=0);
CPLErr d_imsqrt(int iband=0);
CPLErr d_imlog(int iband=0);
CPLErr d_imatan(int iband=0);
CPLErr d_imcos(int iband=0);
CPLErr d_imacos(int iband=0);
CPLErr d_imsin(int iband=0);
CPLErr d_imasin(int iband=0);
CPLErr d_thresh(double d_gt1, double d_gt2, double d_gbg, double d_gfg, int iband=0);
CPLErr d_setlevel(double d_gt1, double d_gt2, double d_gval, int iband=0);
CPLErr d_modulo(int  val, int iband=0);
CPLErr d_complement(int iband=0);
CPLErr d_power2p(int iband=0);
CPLErr d_blank(double d_gval, int iband=0);
CPLErr d_shift(int  val, int iband=0);
CPLErr d_setrange(double d_gt1, double d_gt2, int iband=0);
CPLErr d_FindPixWithVal(double d_gval, unsigned long int * ofs, int iband=0);
    //end insert from fun2method_errortype_d
    /* std::shared_ptr<Jim> arith(Jim& imRaster_im2, int op, int iband=0, bool destructive=false); */
    /* CPLErr d_arith(Jim& imRaster_im2, int op, int iband=0); */

///functions from mialib returning an image (example only must be wrapped automated via Python script)
// std::shared_ptr<Jim> mean2d(int width, int iband=0);
// std::shared_ptr<Jim> copy_image(int iband=0);
 //functions from mialib returning image list (manually wrapped)
 JimList rotatecoor(double theta, int iband=0);
 JimList imgc(int iband=0);
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
    std::shared_ptr<Jim> filter(app::AppFactory& theApp){
      std::shared_ptr<Jim> imgWriter=createImg();
      /* std::shared_ptr<Jim> imgWriter=std::make_shared<Jim>(); */
      ImgRaster::filter(*imgWriter,theApp);
      return(imgWriter);
    }
    ///create statistical profile from a collection
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
      std::shared_ptr<Jim> imgWriter=createImg();
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
    ///Check for difference with reference image
    CPLErr diff(std::shared_ptr<Jim> refImage,app::AppFactory& app){
      return(ImgRaster::diff(*refImage,app));
    }
    ///Clear all no data values, including the one in GDAL dataset if it is set
    CPLErr clearNoData(int band=0){return(ImgRaster::clearNoData(band));}
    ///set mask to raster dataset
    std::shared_ptr<Jim> setMask(JimList& maskList, app::AppFactory& app);

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
}
#endif // _JIM_H_

