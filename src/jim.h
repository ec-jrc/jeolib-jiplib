/**********************************************************************
jim.h: class to read raster files
History
2016/06/24 - Created by Pieter Kempeneers
Change log
***********************************************************************/
#ifndef _JIM_H_
#define _JIM_H_

#include "pktools/imageclasses/ImgRaster.h"
#include <string>
#include <vector>
extern "C" {
#include "mialib_swig.h"
#include "op.h"
}

/**
   Name space jiplib
**/
namespace jiplib{

  /** @brief class for raster dataset (read and write).

      Jim is a class that enables the integration of functionalities from both pktools and mia image processing libraries
      @author Pierre Soille, Pieter Kempeneers
      @date 2016
  */
  class Jim : public ImgRaster
  {
  public:
    ///default constructor
  Jim(void) : ImgRaster(){m_nplane=1;m_mia=0;
    };
    ///constructor input image
  Jim(const std::string& filename, unsigned int memory=0) : ImgRaster(filename,memory){
      m_nplane=1;
      m_mia=0;
    };
    ///constructor input image
  Jim(const std::string& filename, const Jim& imgSrc, unsigned int memory=0, const std::vector<std::string>& options=std::vector<std::string>()) : ImgRaster(filename,imgSrc,memory,options){m_nplane=1;m_mia=0;
    };
    ///constructor input image
  Jim(std::shared_ptr<ImgRaster> imgSrc, bool copyData=true) : ImgRaster(imgSrc, copyData){m_nplane=1;m_mia=0;
    };
    ///constructor input image
  Jim(Jim& imgSrc, bool copyData=true) : ImgRaster(imgSrc, copyData){m_nplane=1;m_mia=0;
    };
    ///constructor output image
  Jim(const std::string& filename, unsigned int ncol, unsigned int nrow, int nband, const GDALDataType& dataType, const std::string& imageType, unsigned int memory=0, const std::vector<std::string>& options=std::vector<std::string>()) : ImgRaster(filename, ncol, nrow, nband, dataType, imageType, memory, options){m_nplane=1;m_mia=0;
    };
    ///constructor output image
  Jim(unsigned int ncol, unsigned int nrow, unsigned int nband, const GDALDataType& dataType) : ImgRaster(ncol, nrow, nband, dataType){m_nplane=1;m_mia=0;
    };
    ///destructor
    ~Jim(void){if(m_mia) delete(m_mia);m_mia=0;};

    ///Clone as new shared pointer to ImgRaster object
    /**
     *
     * @return shared pointer to new ImgRaster object alllowing polymorphism
     */
    std::shared_ptr<ImgRaster> clone() { return std::make_shared<Jim>(*this,false); };
    // std::shared_ptr<Jim> clone() { return std::shared_ptr<Jim>(new Jim(*this,false) ); };
    // std::shared_ptr<ImgRaster> clone() { return std::shared_ptr<ImgRaster>(new Jim(*this,false) ); };

    ///reset all member variables
    void reset(void){ImgRaster::reset();m_nplane=1;m_mia=0;};
    ///Get the number of planes of this dataset
    unsigned int nrOfPlane(void) const { return m_nplane;};
    /// convert single plane multiband image to single band image with multiple planes
    CPLErr band2plane(){};//not implemented yet
    /// convert single band multiple plane image to single plane multiband image
    CPLErr plane2band(){};//not implemented yet
    ///get MIA representation for a particular band
    IMAGE* getMIA(unsigned int band);
    ///set memory from internal MIA representation for particular band
    CPLErr setMIA(unsigned int band);
    // ///set memory from MIA representation for particular band
    CPLErr setMIA(IMAGE* mia, unsigned int band);
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
    ///relational == operator
    bool operator==(Jim& refImg);
    ///relational == operator
    bool operator==(std::shared_ptr<Jim> refImg);
    ///test for equality (relational == operator)
    bool isEqual(Jim& refImg){return(*this==(refImg));};
    ///relational == operator
    bool isEqual(std::shared_ptr<Jim> refImg){return(this->operator==(refImg));};
    ///relational != operator
    bool operator!=(Jim& refImg){ return !(this->operator==(refImg)); };
    ///relational != operator
    bool operator!=(std::shared_ptr<Jim> refImg){ return !(this->operator==(refImg)); };
    /// perform arithmetic operation for a particular band
    CPLErr arith(std::shared_ptr<Jim> imgRaster, int theOperation, unsigned int band=0);
    /// perform arithmetic operation for a particular band (non-destructive version)
    std::shared_ptr<jiplib::Jim> getArith(std::shared_ptr<Jim> imgRaster, int theOperation, unsigned int iband=0);
    /// perform a morphological reconstruction by dilation for a particular band
    CPLErr rdil(std::shared_ptr<Jim> mask, int graph, int flag, unsigned int band=0);
    /// perform a morphological reconstruction by dilation for a particular band (non-destructive version)
    std::shared_ptr<jiplib::Jim> getRdil(std::shared_ptr<Jim> mask, int graph, int flag, unsigned int iband=0);
    /// perform a morphological reconstruction by erosion for a particular band
    CPLErr rero(std::shared_ptr<Jim> mask, int graph, int flag, unsigned int band=0);
    /// perform a morphological reconstruction by erosion for a particular band (non-destructive version)
    std::shared_ptr<jiplib::Jim> getRero(std::shared_ptr<Jim> mask, int graph, int flag, unsigned int iband=0);

  protected:
    ///number of planes in this dataset
    unsigned int m_nplane;
  private:
    IMAGE* m_mia;
  };
}
#endif // _JIM_H_
