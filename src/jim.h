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
#include "jipl_glue.h"
#include "op.h"
}

//extern ERROR_TYPE rdil(IMAGE *, IMAGE *, int, int);

namespace jiplib{
/**
   Base class for raster dataset (read and write) in a format supported by GDAL. This general raster class is used to store e.g., filename, number of columns, rows and bands of the dataset. 
**/
class Jim : public ImgRaster
{
public:
  ///default constructor
  Jim(void) : ImgRaster(){};
  ///constructor input image
  Jim(const std::string& filename, unsigned long int memory=0) : ImgRaster(filename,memory){};
  ///constructor input image
  Jim(const std::string& filename, const ImgRaster& imgSrc, unsigned int memory=0, const std::vector<std::string>& options=std::vector<std::string>()) : ImgRaster(filename,imgSrc,memory,options){};
  ///constructor input image
  Jim(ImgRaster& imgSrc, bool copyData=true) : ImgRaster(imgSrc, copyData){};
  ///constructor output image
  Jim(const std::string& filename, unsigned int ncol, unsigned int nrow, int nband, const GDALDataType& dataType, const std::string& imageType, unsigned int memory=0, const std::vector<std::string>& options=std::vector<std::string>()) : ImgRaster(filename, ncol, nrow, nband, dataType, imageType, memory, options){};
  ///constructor output image
  Jim(unsigned int ncol, unsigned int nrow, unsigned int nband, const GDALDataType& dataType) : ImgRaster(ncol, nrow, nband, dataType){};
  ///destructor
  ~Jim(void){};
  ///get MIA representation of single band image
  IMAGE getMIA(unsigned int band);
  ///get pointer to MIA representation of single band image
  // IMAGE* getMIA(unsigned int band);
  ///get MIA representation of multiband image: we need to allocate and duplicate the memory
  ///use only for mia operations that operate on 3D input images
  ///for band wise operations, better loop over getMIA(band) to avoid duplicate memory
  IMAGE getMIA();
  // IMAGE* getMIA();
  ///set memory from MIA representation
  CPLErr setMIA(IMAGE& mia);
  ///set memory from MIA representation for particular band
  CPLErr setMIA(IMAGE& mia, unsigned int band=0);

  GDALDataType LIIAR2GDALDataType(int aLIIARDataType)
  {
    switch (aLIIARDataType){
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
      /*   case t_UINT64; *:
           /*     return GDT_UInt64; */
      /*   case t_INT64; *:
           /*     return GDT_Int64; */
    case t_UNSUPPORTED:
      return GDT_Unknown;
    default:
      return GDT_Unknown;
    }
  };

  CPLErr arith(Jim& imgRaster, int theOperation);
  CPLErr rdil(Jim& mask, int graph, int flag);
  CPLErr imequalp(Jim& ref);

private:
};
}
#endif // _JIM_H_
