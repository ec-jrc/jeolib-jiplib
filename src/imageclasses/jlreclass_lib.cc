/**********************************************************************
jlreclass_lib.cc: program to replace categorical pixel values in raster dataset
Author(s): Pieter.Kempeneers@ec.europa.eu
Copyright (C) 2016-2020 European Union (Joint Research Centre)

This file is part of jiplib.

jiplib is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

jiplib is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with jiplib.  If not, see <https://www.gnu.org/licenses/>.
***********************************************************************/
#include <assert.h>
#include <map>
#include "base/Optionjl.h"
#include "imageclasses/Jim.h"
#include "apps/AppFactory.h"

using namespace std;
using namespace app;

shared_ptr<Jim> Jim::reclass(app::AppFactory& app){
  shared_ptr<Jim> imgWriter=createImg();
  reclass(*imgWriter, app);
  return(imgWriter);
}

void Jim::reclass(Jim& imgWriter, app::AppFactory& app){
  Optionjl<string> mask_opt("m", "mask", "Mask image(s)");
  Optionjl<unsigned short> masknodata_opt("msknodata", "msknodata", "Mask value(s) where image has nodata. Use one value for each mask, or multiple values for a single mask.", 1);
  Optionjl<int> nodata_opt("nodata", "nodata", "nodata value to put in image if not valid (0)", 0);
  Optionjl<string> colorTable_opt("ct", "ct", "color table (file with 5 columns: id R G B ALFA (0: transparent, 255: solid)");
  Optionjl<unsigned short>  band_opt("b", "band", "band index(es) to replace (other bands are copied to output)", 0);
  Optionjl<string> otype_opt("ot", "otype", "Data type for output image ({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}). Empty string: inherit type from input image", "");
  Optionjl<string> code_opt("code", "code", "Recode text file (2 columns: from to)");
  Optionjl<string> class_opt("c", "class", "list of classes to reclass (in combination with reclass option)");
  Optionjl<string> reclass_opt("r", "reclass", "list of recoded classes (in combination with class option)");
  // Optionjl<string> fieldname_opt("n", "fname", "field name of the shape file to be replaced", "label");
  // Optionjl<string> description_opt("d", "description", "Set image description");
  Optionjl<short> verbose_opt("v", "verbose", "verbose", 0);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=mask_opt.retrieveOption(app);
    masknodata_opt.retrieveOption(app);
    nodata_opt.retrieveOption(app);
    code_opt.retrieveOption(app);
    class_opt.retrieveOption(app);
    reclass_opt.retrieveOption(app);
    colorTable_opt.retrieveOption(app);
    otype_opt.retrieveOption(app);
    band_opt.retrieveOption(app);
    // fieldname_opt.retrieveOption(app);
    // description_opt.retrieveOption(app);
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

    map<string,string> codemapString;//map with codes: codemapString[theKey(from)]=theValue(to)
    map<double,double> codemap;//map with codes: codemap[theKey(from)]=theValue(to)
    if(code_opt.size()){
      if(verbose_opt[0])
        cout << "opening code text file " << code_opt[0] << endl;
      ifstream codefile;
      codefile.open(code_opt[0].c_str());
      string theKey;
      string theValue;
      while(codefile>>theKey){
        codefile >> theValue;
        codemapString[theKey]=theValue;
        codemap[string2type<double>(theKey)]=string2type<double>(theValue);
      }
      codefile.close();
    }
    else{//use combination of class_opt and reclass_opt
      assert(class_opt.size()==reclass_opt.size());
      for(int iclass=0;iclass<class_opt.size();++iclass){
        codemapString[class_opt[iclass]]=reclass_opt[iclass];
        codemap[string2type<double>(class_opt[iclass])]=string2type<double>(reclass_opt[iclass]);
      }
    }
    assert(codemapString.size());
    assert(codemap.size());
    //if verbose true, print the codes to screen
    if(verbose_opt[0]){
      map<string,string>::iterator mit;
      cout << codemapString.size() << " codes used: " << endl;
      for(mit=codemapString.begin();mit!=codemapString.end();++mit)
        cout << (*mit).first << " " << (*mit).second << endl;
    }
    bool refIsRaster=true;
    // if(input_opt[0].find(".shp")!=string::npos){//shape file
    if(!refIsRaster){
      std::string errorString="Error: input must be raster dataset";
      throw(errorString);
    }
    else{//image file
      vector<Jim> maskReader(mask_opt.size());
      for(int imask=0;imask<mask_opt.size();++imask){
        if(verbose_opt[0])
          cout << "opening mask image file " << mask_opt[imask] << endl;
        maskReader[imask].open(mask_opt[imask]);
      }
      if(verbose_opt[0]){
        cout << "data type: " << otype_opt[0] << endl;
      }
      GDALDataType theType=GDT_Unknown;
      if(otype_opt.size()){
        theType=string2GDAL(otype_opt[0]);
        if(theType==GDT_Unknown)
          std::cout << "Warning: unknown output pixel type: " << otype_opt[0] << ", using input type as default" << std::endl;
      }
      if(theType==GDT_Unknown){
        theType=getGDALDataType();
        if(verbose_opt[0])
          cout << "Using data type from input image: " << GDALGetDataTypeName(theType) << endl;
      }
      if(verbose_opt[0])
        cout << endl << "Output pixel type:  " << GDALGetDataTypeName(theType) << endl;
      imgWriter.open(nrOfCol(),nrOfRow(),nrOfBand(),theType);
      for(int iband=0;iband<nrOfBand();++iband)
        imgWriter.GDALSetNoDataValue(nodata_opt[0],iband);
      // if(description_opt.size())
      //   imgWriter.setImageDescription(description_opt[0]);

      if(colorTable_opt.size()){
        if(colorTable_opt[0]!="none")
          imgWriter.setColorTable(colorTable_opt[0]);
      }
      else if (getColorTable()!=NULL)//copy colorTable from input image
        imgWriter.setColorTable(getColorTable());

      //if input image is georeferenced, copy projection info to output image
      if(isGeoRef()){
        for(int imask=0;imask<mask_opt.size();++imask)
          assert(maskReader[imask].isGeoRef());
      }
      imgWriter.copyGeoTransform(*this);
      imgWriter.setProjection(getProjection());
      double ulx,uly,lrx,lry;
      getBoundingBox(ulx,uly,lrx,lry);
      imgWriter.copyGeoTransform(*this);
      assert(nodata_opt.size()==masknodata_opt.size());
      if(verbose_opt[0]&&mask_opt.size()){
        for(int iv=0;iv<masknodata_opt.size();++iv)
          cout << masknodata_opt[iv] << "->" << nodata_opt[iv] << endl;
      }

      assert(imgWriter.nrOfCol()==nrOfCol());
      // Vector2d<int> lineInput(nrOfBand(),nrOfCol());
      Vector2d<double> lineInput(nrOfBand(),nrOfCol());
      Vector2d<short> lineMask(mask_opt.size());
      for(int imask=0;imask<mask_opt.size();++imask)
        lineMask[imask].resize(maskReader[imask].nrOfCol());
      Vector2d<double> lineOutput(imgWriter.nrOfBand(),imgWriter.nrOfCol());
      unsigned int irow=0;
      unsigned int icol=0;
      const char* pszMessage;
      void* pProgressArg=NULL;
      GDALProgressFunc pfnProgress=GDALTermProgress;
      double progress=0;
      MyProgressFunc(progress,pszMessage,pProgressArg);
      double oldRowMask=-1;
      for(irow=0;irow<nrOfRow();++irow){
        //read line in lineInput buffer
        for(unsigned int iband=0;iband<nrOfBand();++iband){
          try{
            // readData(lineInput[iband],GDT_Int32,irow,iband);
            readData(lineInput[iband],irow,iband);
          }
          catch(string errorstring){
            cerr << errorstring << endl;
            throw;
          }
        }
        double x,y;//geo coordinates
        double colMask,rowMask;//image coordinates in mask image
        for(icol=0;icol<nrOfCol();++icol){
          bool masked=false;
          if(mask_opt.size()>1){//multiple masks
            for(int imask=0;imask<mask_opt.size();++imask){
              image2geo(icol,irow,x,y);
              maskReader[imask].geo2image(x,y,colMask,rowMask);
              if(static_cast<unsigned int>(rowMask)!=static_cast<unsigned int>(oldRowMask)){
                assert(rowMask>=0&&rowMask<maskReader[imask].nrOfRow());
                try{
                  maskReader[imask].readData(lineMask[imask],static_cast<unsigned int>(rowMask));
                }
                catch(string errorstring){
                  cerr << errorstring << endl;
                  throw;
                }
                oldRowMask=rowMask;
              }
              short ivalue=0;
              if(mask_opt.size()==masknodata_opt.size())//one invalid value for each mask
                ivalue=masknodata_opt[imask];
              else//use same invalid value for each mask
                ivalue=masknodata_opt[0];
              if(lineMask[imask][colMask]==ivalue){
                for(unsigned int iband=0;iband<nrOfBand();++iband)
                  lineInput[iband][icol]=nodata_opt[imask];
                masked=true;
                break;
              }
            }
          }
          else if(mask_opt.size()){//potentially more invalid values for single mask
            image2geo(icol,irow,x,y);
            maskReader[0].geo2image(x,y,colMask,rowMask);
            if(static_cast<unsigned int>(rowMask)!=static_cast<unsigned int>(oldRowMask)){
              assert(rowMask>=0&&rowMask<maskReader[0].nrOfRow());
              try{
                maskReader[0].readData(lineMask[0],static_cast<unsigned int>(rowMask));
              }
              catch(string errorstring){
                cerr << errorstring << endl;
                throw;
              }
              oldRowMask=rowMask;
            }
            for(int ivalue=0;ivalue<masknodata_opt.size();++ivalue){
              assert(masknodata_opt.size()==nodata_opt.size());
              if(lineMask[0][colMask]==masknodata_opt[ivalue]){
                for(int iband=0;iband<nrOfBand();++iband)
                  lineInput[iband][icol]=nodata_opt[ivalue];
                masked=true;
                break;
              }
            }
          }
          for(unsigned int iband=0;iband<lineOutput.size();++iband){
            lineOutput[iband][icol]=lineInput[iband][icol];
            if(find(band_opt.begin(),band_opt.end(),iband)!=band_opt.end()){
              if(!masked && codemap.find(lineInput[iband][icol])!=codemap.end()){
                double toValue=codemap[lineInput[iband][icol]];
                lineOutput[iband][icol]=toValue;
              }
            }
          }
        }
        //write buffer lineOutput to output file
        try{
          for(unsigned int iband=0;iband<imgWriter.nrOfBand();++iband)
            imgWriter.writeData(lineOutput[iband],irow,iband);
        }
        catch(string errorstring){
          cerr << errorstring << endl;
          throw;
        }
        //progress bar
        progress=static_cast<float>((irow+1.0)/imgWriter.nrOfRow());
        MyProgressFunc(progress,pszMessage,pProgressArg);
      }
    }
  }
  catch(string predefinedString){
    std::cout << predefinedString << std::endl;
    throw;
  }
}

void Jim::d_reclass(app::AppFactory& app){
  Optionjl<string> class_opt("c", "class", "list of classes to reclass (in combination with reclass option)");
  Optionjl<string> reclass_opt("r", "reclass", "list of recoded classes (in combination with class option)");
  Optionjl<short> verbose_opt("v", "verbose", "verbose", 0);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=class_opt.retrieveOption(app);
    reclass_opt.retrieveOption(app);
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

    if(class_opt.size()!=reclass_opt.size()){
      std::ostringstream errorStream;
      errorStream << "Error: size of class and reclass arguments are not equal" << std::endl;
      throw(errorStream.str());
    }
    map<string,string> codemapString;//map with codes: codemapString[theKey(from)]=theValue(to)
    map<double,double> codemap;//map with codes: codemap[theKey(from)]=theValue(to)
    for(int iclass=0;iclass<class_opt.size();++iclass){
      codemapString[class_opt[iclass]]=reclass_opt[iclass];
      codemap[string2type<double>(class_opt[iclass])]=string2type<double>(reclass_opt[iclass]);
    }
    //if verbose true, print the codes to screen
    if(verbose_opt[0]){
      map<string,string>::iterator mit;
      cout << codemapString.size() << " codes used: " << endl;
      for(mit=codemapString.begin();mit!=codemapString.end();++mit)
        cout << (*mit).first << " " << (*mit).second << endl;
    }
    for(int iband=0;iband<nrOfBand();++iband){
#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
      for(int irow=0;irow<nrOfRow();++irow){
        for(int icol=0;icol<nrOfCol();++icol){
          double value=readData(icol,irow,iband);
          if(codemap.find(value)!=codemap.end())
              value=codemap[value];
          writeData(value,icol,irow,iband);
        }
      }
    }
  }
  catch(string predefinedString){
    std::cout << predefinedString << std::endl;
    throw;
  }
}
