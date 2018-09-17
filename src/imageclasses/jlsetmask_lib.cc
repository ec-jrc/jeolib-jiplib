/**********************************************************************
jlsetmask_lib.cc: program to apply mask image (set invalid values) to raster image
Copyright (C) 2008-2016 Pieter Kempeneers

This file is part of pktools

pktools is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

pktools is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with pktools.  If not, see <http://www.gnu.org/licenses/>.
***********************************************************************/
#include <assert.h>

#include "imageclasses/Jim.h"
#include "imageclasses/VectorOgr.h"
#include "base/Optionjl.h"
#include "apps/AppFactory.h"

using namespace std;
using namespace app;

/**
 * @param app application specific option arguments
 * @return output image
 **/
shared_ptr<Jim> Jim::setMask(JimList& maskReader, app::AppFactory& app){
  shared_ptr<Jim> imgWriter=createImg();
  setMask(maskReader, *imgWriter, app);
  return(imgWriter);
}

/**
 * @param app application specific option arguments
 * @return output image
 **/
shared_ptr<Jim> Jim::setMask(VectorOgr& ogrReader, app::AppFactory& app){
  shared_ptr<Jim> imgWriter=createImg();
  setMask(ogrReader, *imgWriter, app);
  return(imgWriter);
}

/**
 * @param imgWriter output raster setmask dataset
 * @return CE_None if successful, CE_Failure if failed
 **/
CPLErr Jim::setMask(JimList& maskReader, Jim& imgWriter, app::AppFactory& app){
  //command line options
  // Optionjl<string> mask_opt("m", "mask", "Mask image(s)");
  Optionjl<string> vectorMask_opt("vm", "vectormask", "Vector mask dataset(s)");
  Optionjl<string> otype_opt("ot", "otype", "Data type for output image ({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}). Empty string: inherit type from input image");
  Optionjl<int> msknodata_opt("msknodata", "msknodata", "Mask value(s) where image has nodata. Use one value for each mask, or multiple values for a single mask.", 1);
  Optionjl<short> mskband_opt("mskband", "mskband", "Mask band to read (0 indexed). Provide band for each mask.", 0);
  Optionjl<char> operator_opt("p", "operator", "Operator: < = > !. Use operator for each msknodata option", '=');
  Optionjl<double> nodata_opt("nodata", "nodata", "nodata value to put in image if not valid", 0);
  Optionjl<string> eoption_opt("eo","eo", "special extent options controlling rasterization: ATTRIBUTE|CHUNKYSIZE|ALL_TOUCHED|BURN_VALUE_FROM|MERGE_ALG, e.g., -eo ALL_TOUCHED=TRUE");
  Optionjl<string> layernames_opt("ln", "ln", "Layer names");
  Optionjl<short> verbose_opt("v", "verbose", "verbose", 0,2);

  otype_opt.setHide(1);
  mskband_opt.setHide(1);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    // doProcess=mask_opt.retrieveOption(app);
    doProcess=vectorMask_opt.retrieveOption(app);
    msknodata_opt.retrieveOption(app);
    mskband_opt.retrieveOption(app);
    nodata_opt.retrieveOption(app);
    operator_opt.retrieveOption(app);
    otype_opt.retrieveOption(app);
    eoption_opt.retrieveOption(app);
    layernames_opt.retrieveOption(app);
    verbose_opt.retrieveOption(app);

    if(!doProcess){
      cout << endl;
      std::ostringstream helpStream;
      helpStream << "short option -h shows basic options only, use long option --help to show all options" << std::endl;
      throw(helpStream.str());//help was invoked, stop processing
    }

    std::vector<std::string> badKeys;
    app.badKeys(badKeys);
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
    // Jim
    // open(input_opt[0],memory_opt[0]);
    GDALDataType theType=GDT_Unknown;
    if(otype_opt.size()){
      theType=string2GDAL(otype_opt[0]);
      if(theType==GDT_Unknown)
        std::cout << "Warning: unknown output pixel type: " << otype_opt[0] << ", using input type as default" << std::endl;
    }
    //if output type not set, get type from input image
    if(theType==GDT_Unknown){
      theType=getGDALDataType();
      if(verbose_opt[0])
        cout << "Using data type from input image: " << GDALGetDataTypeName(theType) << endl;
    }
    if(verbose_opt[0])
      cout << "Output pixel type:  " << GDALGetDataTypeName(theType) << endl;
    // Jim imgWriter;
    try{
      imgWriter.open(nrOfCol(),nrOfRow(),nrOfBand(),theType);
      // for(unsigned int iband=0;iband<nrOfBand();++iband)
      //   imgWriter.GDALSetNoDataValue(nodata_opt[0],iband);
      imgWriter.setProjection(getProjection());
      imgWriter.copyGeoTransform(*this);
    }
    catch(string errorstring){
      cout << errorstring << endl;
      return(CE_Failure);
    }
    // if(verbose_opt[0])
    //   cout << "opening output image file " << output_opt[0] << endl;
    // imgWriter.open(output_opt[0],;
    if (getColorTable()!=NULL)//copy colorTable from input image
      imgWriter.setColorTable(getColorTable());
    while(nodata_opt.size()<msknodata_opt.size())
      nodata_opt.push_back(nodata_opt.back());
    imgWriter.setNoData(nodata_opt);
    if(operator_opt.size()!=msknodata_opt.size()&&operator_opt.size()!=1){
      std::string errorString="Error: number of operators and masks do not match";
      throw(errorString);
    }

    // int nmask=vectorMask_opt.size()? vectorMask_opt.size() : mask_opt.size();
    int nmask=vectorMask_opt.size()? vectorMask_opt.size() : maskReader.size();
    if(verbose_opt[0]){
      cout << " mask files selected: " << nmask << endl;
      for(int iv=0;iv<msknodata_opt.size();++iv){
        char op=(operator_opt.size()==msknodata_opt.size())?operator_opt[iv]:operator_opt[0];
        cout << op << " " << msknodata_opt[iv] << "->" << nodata_opt[iv] << endl;
      }
    }

    // vector<Jim> maskReader(nmask);
    if(vectorMask_opt.size()){
      // if(mask_opt.size()){
      if(maskReader.size()){
        string errorString="Error: either raster mask or vector mask can be set, not both";
        throw(errorString);
      }
      try{
        imgWriter.open(nrOfCol(),nrOfRow(),nrOfBand(),theType);
        for(int imask=0;imask<vectorMask_opt.size();++imask){
          shared_ptr<Jim> imgMask=createImg();
          imgMask->open(nrOfCol(),nrOfRow(),1,GDT_Float64);
          double gt[6];
          gt[0]=getUlx();
          gt[1]=getDeltaX();
          gt[2]=0;
          gt[3]=getUly();
          gt[4]=0;
          gt[5]=-getDeltaY();
          imgMask->setGeoTransform(gt);
          imgMask->setProjection(getProjectionRef());
          // imgMask->rasterizeBuf(vectorMask_opt[imask]);
          VectorOgr ogrReader;
          int msknodata=(msknodata_opt.size()>=imask)? msknodata_opt[imask] : msknodata_opt[0];
          ogrReader.open(vectorMask_opt[imask],layernames_opt,true);
          imgMask->rasterizeBuf(ogrReader,msknodata,eoption_opt,layernames_opt);
          maskReader.pushImage(imgMask);
        }
      }
      catch(string error){
        cerr << error << std::endl;
        return(CE_Failure);
      }
    }
    // else{
    //   for(int imask=0;imask<mask_opt.size();++imask){
    //     if(verbose_opt[0])
    //       cout << "opening mask image file " << mask_opt[imask] << endl;
    //     maskReader[imask].open(mask_opt[imask]);
    //   }
    // }

    //duplicate band used for mask if not explicitly provided
    while(mskband_opt.size()<maskReader.size())
      mskband_opt.push_back(mskband_opt[0]);
    while(msknodata_opt.size()<maskReader.size())
      msknodata_opt.push_back(msknodata_opt[0]);

    Vector2d<double> lineInput(nrOfBand(),nrOfCol());
    Vector2d<double> lineOutput(imgWriter.nrOfBand(),imgWriter.nrOfCol());
    assert(lineOutput.size()==lineInput.size());
    assert(nrOfCol()==imgWriter.nrOfCol());
    Vector2d<double> lineMask(nmask);
    for(int imask=0;imask<nmask;++imask){
      if(verbose_opt[0])
        cout << "mask " << imask << " has " << maskReader.getImage(imask)->nrOfCol() << " columns and " << maskReader.getImage(imask)->nrOfRow() << " rows" << endl;
      lineMask[imask].resize(maskReader.getImage(imask)->nrOfCol());
    }
    unsigned int irow=0;
    unsigned int icol=0;
    const char* pszMessage;
    void* pProgressArg=NULL;
    GDALProgressFunc pfnProgress=GDALTermProgress;
    float progress=0;
    if(!verbose_opt[0])
      MyProgressFunc(progress,pszMessage,pProgressArg);
    vector<double> oldRowMask(nmask);
    for(int imask=0;imask<nmask;++imask)
      oldRowMask[imask]=-1;
    for(irow=0;irow<nrOfRow();++irow){
      //read line in lineInput buffer
      for(unsigned int iband=0;iband<nrOfBand();++iband){
        try{
          readData(lineInput[iband],irow,iband);
        }
        catch(string errorstring){
          cerr << errorstring << endl;
          exit(1);
        }
      }
      //todo: support projection difference in mask and input raster
      double x,y;//geo coordinates
      double colMask,rowMask;//image coordinates in mask image
      for(icol=0;icol<nrOfCol();++icol){
        if(nmask>1){//multiple masks
          for(int imask=0;imask<nmask;++imask){
            image2geo(icol,irow,x,y);
            maskReader.getImage(imask)->geo2image(x,y,colMask,rowMask);
            colMask=static_cast<unsigned int>(colMask);
            rowMask=static_cast<unsigned int>(rowMask);
            bool masked=false;
            if(rowMask>=0&&rowMask<maskReader.getImage(imask)->nrOfRow()&&colMask>=0&&colMask<maskReader.getImage(imask)->nrOfCol()){
              if(static_cast<unsigned int>(rowMask)!=static_cast<unsigned int>(oldRowMask[imask])){
                assert(rowMask>=0&&rowMask<maskReader.getImage(imask)->nrOfRow());
                try{
                  maskReader.getImage(imask)->readData(lineMask[imask],static_cast<unsigned int>(rowMask),mskband_opt[imask]);
                }
                catch(string errorstring){
                  cerr << errorstring << endl;
                  exit(1);
                }
                oldRowMask[imask]=rowMask;
              }
            }
            else
              continue;//no coverage in this mask
            int ivalue=0;
            if(nmask==msknodata_opt.size())//one invalid value for each mask
              ivalue=msknodata_opt[imask];
            else//use same invalid value for each mask
              ivalue=msknodata_opt[0];
            char op=(operator_opt.size()==nmask)?operator_opt[imask]:operator_opt[0];
            switch(op){
            case('='):
            default:
              if(lineMask[imask][colMask]==ivalue)
                masked=true;
            break;
            case('<'):
              if(lineMask[imask][colMask]<ivalue)
                masked=true;
              break;
            case('>'):
              if(lineMask[imask][colMask]>ivalue)
                masked=true;
              break;
            case('!'):
              if(lineMask[imask][colMask]!=ivalue)
                masked=true;
              break;
            }
            if(masked){
              if(verbose_opt[0]>1)
                cout << "image masked at (col=" << icol << ",row=" << irow <<") and value " << ivalue << endl;
              for(unsigned int iband=0;iband<nrOfBand();++iband){
                if(nmask==nodata_opt.size())//one flag value for each mask
                  lineInput[iband][icol]=nodata_opt[imask];
                else
                  lineInput[iband][icol]=nodata_opt[0];
              }
              masked=false;
              break;
            }
          }
        }
        else{//potentially more invalid values for single mask
          image2geo(icol,irow,x,y);
          maskReader.getImage(0)->geo2image(x,y,colMask,rowMask);
          colMask=static_cast<unsigned int>(colMask);
          rowMask=static_cast<unsigned int>(rowMask);
          bool masked=false;
          if(rowMask>=0&&rowMask<maskReader.getImage(0)->nrOfRow()&&colMask>=0&&colMask<maskReader.getImage(0)->nrOfCol()){
            if(static_cast<unsigned int>(rowMask)!=static_cast<unsigned int>(oldRowMask[0])){
              assert(rowMask>=0&&rowMask<maskReader.getImage(0)->nrOfRow());
              try{
                // maskReader.getImage(0)->readData(lineMask[0],static_cast<unsigned int>(rowMask));
                maskReader.getImage(0)->readData(lineMask[0],static_cast<unsigned int>(rowMask),mskband_opt[0]);
              }
              catch(string errorstring){
                cerr << errorstring << endl;
                exit(1);
              }
              oldRowMask[0]=rowMask;
            }
            for(int ivalue=0;ivalue<msknodata_opt.size();++ivalue){
              assert(msknodata_opt.size()==nodata_opt.size());
              char op=(operator_opt.size()==msknodata_opt.size())?operator_opt[ivalue]:operator_opt[0];
              switch(op){
              case('='):
              default:
                if(lineMask[0][colMask]==msknodata_opt[ivalue])
                  masked=true;
              break;
              case('<'):
                if(lineMask[0][colMask]<msknodata_opt[ivalue])
                  masked=true;
                break;
              case('>'):
                if(lineMask[0][colMask]>msknodata_opt[ivalue])
                  masked=true;
                break;
              case('!'):
                if(lineMask[0][colMask]!=msknodata_opt[ivalue])
                  masked=true;
                break;
              }
              if(masked){
                for(unsigned int iband=0;iband<nrOfBand();++iband)
                  lineInput[iband][icol]=nodata_opt[ivalue];
                masked=false;
                break;
              }
            }
          }
        }
        for(unsigned int iband=0;iband<lineOutput.size();++iband)
          lineOutput[iband][icol]=lineInput[iband][icol];
      }
      //write buffer lineOutput to output file
      for(unsigned int iband=0;iband<imgWriter.nrOfBand();++iband){
        try{
          imgWriter.writeData(lineOutput[iband],irow,iband);
        }
        catch(string errorstring){
          cerr << errorstring << endl;
          exit(1);
        }
      }
      //progress bar
      progress=static_cast<float>(irow+1.0)/imgWriter.nrOfRow();
      MyProgressFunc(progress,pszMessage,pProgressArg);
    }
    // for(int imask=0;imask<nmask;++imask)
    //   maskReader[imask].close();
    return(CE_None);
  }
  catch(string predefinedString){
    std::cout << predefinedString << std::endl;
    return(CE_Failure);
  }
}

/**
 * @param imgWriter output raster setmask dataset
 * @return CE_None if successful, CE_Failure if failed
 **/
CPLErr Jim::setMask(VectorOgr& ogrReader, Jim& imgWriter, app::AppFactory& app){
  //command line options
  Optionjl<string> otype_opt("ot", "otype", "Data type for output image ({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}). Empty string: inherit type from input image");
  Optionjl<double> nodata_opt("nodata", "nodata", "nodata value to put in image if not valid", 0);
  Optionjl<string> eoption_opt("eo","eo", "special extent options controlling rasterization: ATTRIBUTE|CHUNKYSIZE|ALL_TOUCHED|BURN_VALUE_FROM|MERGE_ALG, e.g., -eo ALL_TOUCHED=TRUE");
  Optionjl<string> layernames_opt("ln", "ln", "Layer names");
  Optionjl<short> verbose_opt("v", "verbose", "verbose", 0,2);

  otype_opt.setHide(1);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    // doProcess=mask_opt.retrieveOption(app);
    doProcess=nodata_opt.retrieveOption(app);
    otype_opt.retrieveOption(app);
    eoption_opt.retrieveOption(app);
    layernames_opt.retrieveOption(app);
    verbose_opt.retrieveOption(app);

    if(!doProcess){
      cout << endl;
      std::ostringstream helpStream;
      helpStream << "short option -h shows basic options only, use long option --help to show all options" << std::endl;
      throw(helpStream.str());//help was invoked, stop processing
    }

    std::vector<std::string> badKeys;
    app.badKeys(badKeys);
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
    GDALDataType theType=GDT_Unknown;
    if(otype_opt.size()){
      theType=string2GDAL(otype_opt[0]);
      if(theType==GDT_Unknown)
        std::cout << "Warning: unknown output pixel type: " << otype_opt[0] << ", using input type as default" << std::endl;
    }
    //if output type not set, get type from input image
    if(theType==GDT_Unknown){
      theType=getGDALDataType();
      if(verbose_opt[0])
        cout << "Using data type from input image: " << GDALGetDataTypeName(theType) << endl;
    }
    if(verbose_opt[0])
      cout << "Output pixel type:  " << GDALGetDataTypeName(theType) << endl;
    try{
      imgWriter.open(nrOfCol(),nrOfRow(),nrOfBand(),theType);
      imgWriter.setProjection(getProjection());
      imgWriter.copyGeoTransform(*this);
    }
    catch(string errorstring){
      cout << errorstring << endl;
      return(CE_Failure);
    }
    if (getColorTable()!=NULL)//copy colorTable from input image
      imgWriter.setColorTable(getColorTable());
    imgWriter.setNoData(nodata_opt);

    JimList maskReader;

    try{
      imgWriter.open(nrOfCol(),nrOfRow(),nrOfBand(),theType);
      shared_ptr<Jim> imgMask=createImg();
      //todo: check if GDT_Float64 is needed (Float32 might be sufficient)
      imgMask->open(nrOfCol(),nrOfRow(),1,GDT_Float64);
      double gt[6];
      gt[0]=getUlx();
      gt[1]=getDeltaX();
      gt[2]=0;
      gt[3]=getUly();
      gt[4]=0;
      gt[5]=-getDeltaY();
      imgMask->setGeoTransform(gt);
      imgMask->setProjection(getProjectionRef());
      imgMask->rasterizeBuf(ogrReader,1,eoption_opt,layernames_opt);
      maskReader.pushImage(imgMask);
    }
    catch(string error){
      cerr << error << std::endl;
      return(CE_Failure);
    }

    int nmask=maskReader.size();
    Vector2d<double> lineInput(nrOfBand(),nrOfCol());
    Vector2d<double> lineOutput(imgWriter.nrOfBand(),imgWriter.nrOfCol());
    assert(lineOutput.size()==lineInput.size());
    assert(nrOfCol()==imgWriter.nrOfCol());
    Vector2d<double> lineMask(nmask);
    for(int imask=0;imask<nmask;++imask){
      if(verbose_opt[0])
        cout << "mask " << imask << " has " << maskReader.getImage(imask)->nrOfCol() << " columns and " << maskReader.getImage(imask)->nrOfRow() << " rows" << endl;
      lineMask[imask].resize(maskReader.getImage(imask)->nrOfCol());
    }
    unsigned int irow=0;
    unsigned int icol=0;
    const char* pszMessage;
    void* pProgressArg=NULL;
    GDALProgressFunc pfnProgress=GDALTermProgress;
    float progress=0;
    if(!verbose_opt[0])
      MyProgressFunc(progress,pszMessage,pProgressArg);
    vector<double> oldRowMask(nmask);
    for(int imask=0;imask<nmask;++imask)
      oldRowMask[imask]=-1;
    for(irow=0;irow<nrOfRow();++irow){
      //read line in lineInput buffer
      for(unsigned int iband=0;iband<nrOfBand();++iband){
        try{
          readData(lineInput[iband],irow,iband);
        }
        catch(string errorstring){
          cerr << errorstring << endl;
          exit(1);
        }
      }
      //todo: support projection difference in mask and input raster
      double x,y;//geo coordinates
      double colMask,rowMask;//image coordinates in mask image
      for(icol=0;icol<nrOfCol();++icol){
        for(int imask=0;imask<nmask;++imask){
          image2geo(icol,irow,x,y);
          maskReader.getImage(imask)->geo2image(x,y,colMask,rowMask);
          colMask=static_cast<unsigned int>(colMask);
          rowMask=static_cast<unsigned int>(rowMask);
          bool masked=false;
          if(rowMask>=0&&rowMask<maskReader.getImage(imask)->nrOfRow()&&colMask>=0&&colMask<maskReader.getImage(imask)->nrOfCol()){
            if(static_cast<unsigned int>(rowMask)!=static_cast<unsigned int>(oldRowMask[imask])){
              assert(rowMask>=0&&rowMask<maskReader.getImage(imask)->nrOfRow());
              try{
                maskReader.getImage(imask)->readData(lineMask[imask],static_cast<unsigned int>(rowMask));
              }
              catch(string errorstring){
                cerr << errorstring << endl;
                exit(1);
              }
              oldRowMask[imask]=rowMask;
            }
          }
          else{
            if(verbose_opt[0])
              std::cerr << "Warning: no coverage in mask " << imask << std::endl;
            continue;//no coverage in this mask
          }
          if(lineMask[imask][colMask]==1){
            masked=true;
          }
          if(masked){
            if(verbose_opt[0]>1)
              cout << "image masked at (col=" << icol << ",row=" << irow << ")" << endl;
            for(unsigned int iband=0;iband<nrOfBand();++iband){
              lineInput[iband][icol]=nodata_opt[0];
            }
            masked=false;
            break;
          }
        }
        for(unsigned int iband=0;iband<lineOutput.size();++iband)
          lineOutput[iband][icol]=lineInput[iband][icol];
      }
      //write buffer lineOutput to output file
      for(unsigned int iband=0;iband<imgWriter.nrOfBand();++iband){
        try{
          imgWriter.writeData(lineOutput[iband],irow,iband);
        }
        catch(string errorstring){
          cerr << errorstring << endl;
          exit(1);
        }
      }
      //progress bar
      progress=static_cast<float>(irow+1.0)/imgWriter.nrOfRow();
      MyProgressFunc(progress,pszMessage,pProgressArg);
    }
    maskReader.close();
    return(CE_None);
  }
  catch(string predefinedString){
    std::cout << predefinedString << std::endl;
    return(CE_Failure);
  }
}
