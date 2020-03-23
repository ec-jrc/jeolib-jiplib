/**********************************************************************
jlextractimg_lib.h: extract pixel values from raster image using a raster sample
Author(s): Pieter.Kempeneers@ec.europa.eu
Copyright (c) 2016-2020 European Union (Joint Research Centre)
License EUPLv1.2

This file is part of jiplib
***********************************************************************/
#ifndef _JLEXTRACTIMG_LIB_H_
#define _JLEXTRACTIMG_LIB_H_

#include "imageclasses/Jim.h"
#include "imageclasses/VectorOgr.h"
#include "apps/AppFactory.h"

template<typename T> void Jim::extractImg_t(Jim& classReader, VectorOgr& ogrWriter, app::AppFactory& app){
  Optionjl<std::string> output_opt("o", "output", "Output sample dataset");
  Optionjl<std::string> layer_opt("ln", "ln", "output layer name","sample");
  Optionjl<int> class_opt("c", "class", "Class(es) to extract from input sample image. Leave empty to extract all valid data pixels from sample dataset");
  Optionjl<float> threshold_opt("t", "threshold", "Probability threshold for selecting samples (randomly). Provide probability in percentage (>0) or absolute (<0). If using raster land cover maps as a sample dataset, you can provide a threshold value for each class (e.g. -t 80 -t 60). Use value 100 to select all pixels for selected class(es)", 100);
  Optionjl<std::string> ogrformat_opt("f", "oformat", "Output sample dataset format","SQLite");
  Optionjl<std::string> option_opt("co", "co", "Creation option for output file. Multiple options can be specified.");
  Optionjl<std::string> ftype_opt("ft", "ftype", "Field type (only Real or Integer)", "Real");
  Optionjl<std::string> ltype_opt("lt", "ltype", "Label type: In16 or String", "Integer");
  Optionjl<std::string> bandNames_opt("bn", "bandname", "Band name(s) corresponding to band index(es)","b");
  Optionjl<std::string> planeNames_opt("bn", "planename", "Plane name(s) corresponding to plane index(es).");
  Optionjl<double> srcnodata_opt("srcnodata", "srcnodata", "Invalid value for input image");
  Optionjl<unsigned int> bndnodata_opt("bndnodata", "bndnodata", "Band in input image to check if pixel is valid (used for srcnodata)", 0);
  Optionjl<std::string> attribute_opt("attribute", "attribute", "Name of the class label in the output vector dataset", "label");
  Optionjl<std::string> fid_opt("fid", "fid", "Create extra field with field identifier (sequence in which the features have been read");
  Optionjl<short> verbose_opt("v", "verbose", "Verbose mode if > 0", 0,2);

  bndnodata_opt.setHide(1);
  srcnodata_opt.setHide(1);
  attribute_opt.setHide(1);
  option_opt.setHide(1);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    // doProcess=sample_opt.retrieveOption(app);
    doProcess=output_opt.retrieveOption(app);
    layer_opt.retrieveOption(app);
    class_opt.retrieveOption(app);
    threshold_opt.retrieveOption(app);
    ogrformat_opt.retrieveOption(app);
    option_opt.retrieveOption(app);
    ftype_opt.retrieveOption(app);
    ltype_opt.retrieveOption(app);
    planeNames_opt.retrieveOption(app);
    bandNames_opt.retrieveOption(app);
    bndnodata_opt.retrieveOption(app);
    srcnodata_opt.retrieveOption(app);
    attribute_opt.retrieveOption(app);
    fid_opt.retrieveOption(app);
    verbose_opt.retrieveOption(app);
    if(!doProcess){
      std::cout << std::endl;
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
    statfactory::StatFactory stat;
    if(srcnodata_opt.size()){
      while(srcnodata_opt.size()<bndnodata_opt.size())
        srcnodata_opt.push_back(srcnodata_opt[0]);
      stat.setNoDataValues(srcnodata_opt);
    }
    Vector2d<unsigned int> posdata;
    /* unsigned long int nsample=0; */
    /* unsigned long int ntotalvalid=0; */
    /* unsigned long int ntotalinvalid=0; */

    /* std::map<int,unsigned long int> nvalid; */
    /* map<int,unsigned long int> ninvalid; */

    /* map <int,short> classmap;//class->index */
    /* for(int iclass=0;iclass<class_opt.size();++iclass){ */
    /*   nvalid[class_opt[iclass]]=0; */
    /*   ninvalid[class_opt[iclass]]=0; */
    /*   classmap[class_opt[iclass]]=iclass; */
    /* } */

    if(output_opt.empty()){
      std::string errorString="Error: No output dataset provided (use option -o). Use --help for help information";
      throw(errorString);
    }

    int nplane=nrOfPlane();
    if(nplane>1){
      if(planeNames_opt.size()<nplane){
        planeNames_opt.clear();
        for(size_t iplane=0;iplane<nplane;++iplane){
          int thePlane=iplane;
          std::ostringstream planestream;
          planestream << "t" << thePlane;
          planeNames_opt.push_back(planestream.str());
        }
      }
    }
    int nband=nrOfBand();

    if(bandNames_opt.size()<nband){
      std::string bandString=bandNames_opt[0];
      bandNames_opt.clear();
      bandNames_opt.resize(nband);
      for(int iband=0;iband<nband;++iband){
        std::ostringstream fs;
        fs << bandString << iband;
        bandNames_opt[iband]=fs.str();
      }
    }

    if(verbose_opt[0]){
      std::cout << planeNames_opt << std::endl;
      std::cout << bandNames_opt << std::endl;
    }
    if(verbose_opt[0]>1){
      std::cout << "Number of planes in input image: " << this->nrOfPlane() << std::endl;
      std::cout << "Number of bands in input image: " << this->nrOfBand() << std::endl;
    }

    OGRFieldType fieldType;
    OGRFieldType labelType;
    int ogr_typecount=11;//hard coded for now!
    if(verbose_opt[0]>1)
      std::cout << "field and label types can be: ";
    for(int iType = 0; iType < ogr_typecount; ++iType){
      if(verbose_opt[0]>1)
        std::cout << " " << OGRFieldDefn::GetFieldTypeName((OGRFieldType)iType);
      if( OGRFieldDefn::GetFieldTypeName((OGRFieldType)iType) != NULL
          && EQUAL(OGRFieldDefn::GetFieldTypeName((OGRFieldType)iType),
                   ftype_opt[0].c_str()))
        fieldType=(OGRFieldType) iType;
      if( OGRFieldDefn::GetFieldTypeName((OGRFieldType)iType) != NULL
          && EQUAL(OGRFieldDefn::GetFieldTypeName((OGRFieldType)iType),
                   ltype_opt[0].c_str()))
        labelType=(OGRFieldType) iType;
    }
    switch( fieldType ){
    case OFTInteger:
    case OFTReal:
    case OFTRealList:
    case OFTString:
      if(verbose_opt[0]>1)
        std::cout << std::endl << "field type is: " << OGRFieldDefn::GetFieldTypeName(fieldType) << std::endl;
      break;
    default:
      std::ostringstream errorStream;
      errorStream << "Error: field type " << OGRFieldDefn::GetFieldTypeName(fieldType) << " not supported";
      throw(errorStream.str());
      break;
    }
    switch( labelType ){
    case OFTInteger:
    case OFTReal:
    case OFTRealList:
    case OFTString:
      if(verbose_opt[0]>1)
        std::cout << std::endl << "label type is: " << OGRFieldDefn::GetFieldTypeName(labelType) << std::endl;
      break;
    default:
      std::ostringstream errorStream;
      errorStream << "Error: label type " << OGRFieldDefn::GetFieldTypeName(labelType) << " not supported";
      throw(errorStream.str());
      break;
    }

    srand(time(NULL));

    size_t index=0;
    unsigned short nclass=256;
    std::vector< std::vector<size_t> > sample(nclass);//[class][index]
    if(verbose_opt[0]>1)
      std::cout << "extracting sample from image..." << std::endl;


    std::vector<size_t> nvalid(nclass);
    std::vector<size_t> ninvalid(nclass);
    double x=0;//geo x coordinate
    double y=0;//geo y coordinate
    T* pim=0;

    unsigned char* pmask=0;
    for(size_t irow=0;irow<nrOfRow();++irow){
      for(size_t icol=0;icol<nrOfCol();++icol){
        bool validData=true;
        bool validClass=class_opt.empty();//if no classes are defined, process all classes
        index=icol+irow*nrOfCol();
        if(srcnodata_opt.size()){
          pim=static_cast<T*>(getDataPointer(bndnodata_opt[0]));
          if(pim[index]==srcnodata_opt[0])
            validData=false;
        }
        pmask=static_cast<unsigned char*>(classReader.getDataPointer(0));
        unsigned short readClass=pmask[index];
        if(validData){
          if(class_opt.size()){
            for(int iclass=0;iclass<class_opt.size();++iclass){
              if(readClass==class_opt[iclass]){
                validClass=true;//process this class
                break;
              }
            }
          }
        }
        if(validData&&validClass){
          sample[readClass].push_back(index);
          ++(nvalid[readClass]);
        }
        else
          ++(ninvalid[readClass]);
      }
    }
    char **papszOptions=NULL;
    for(std::vector<std::string>::const_iterator optionIt=option_opt.begin();optionIt!=option_opt.end();++optionIt)
      papszOptions=CSLAddString(papszOptions,optionIt->c_str());

    if(ogrWriter.open(output_opt[0], layer_opt, ogrformat_opt[0], VectorOgr::string2geotype("wkbPoint"), this->getProjection(),papszOptions)!=OGRERR_NONE){
      std::ostringstream fs;
      fs << "open ogrWriter failed ";
      fs << "output name: " << output_opt[0] << ", ";
      fs << "format: "<< ogrformat_opt[0] << std::endl;
      throw(fs.str());
    }
    std::map<std::string,double> pointAttributes;
    ogrWriter.createField(attribute_opt[0],labelType);
    if(fid_opt.size())
      ogrWriter.createField(fid_opt[0],OFTInteger64);
    for(size_t iplane=0;iplane<nplane;++iplane){
      for(int iband=0;iband<nband;++iband){
        int theBand=iband;
        std::string fieldname;
        std::ostringstream fs;
        if(bandNames_opt.size()){
          if(planeNames_opt.size())
            fs << planeNames_opt[iplane];
          fs << bandNames_opt[iband];
        }
        else{
          if(planeNames_opt.size())
            fs << planeNames_opt[iplane];
          if(nband>1)
            fs << "b" << theBand;
        }
        fieldname=fs.str();
        ogrWriter.createField(fieldname,fieldType);
      }
    }
    if(threshold_opt[0]!=100){
      //todo (optimization): if selection is small, better select random sample than shuffle all...
#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
      for(unsigned short iclass=0;iclass<nclass;++iclass){
        if(verbose_opt[0]){
          if(nvalid[iclass]){
            std::cout << "nvalid[" << iclass << "]: " << nvalid[iclass] << std::endl;
            std::cout << "ninvalid[" << iclass << "]: " << ninvalid[iclass] << std::endl;
          }
        }
        if(sample[iclass].size()){
          if(verbose_opt[0]){
            std::cout << "random shuffle class " << iclass << " with size " << sample[iclass].size() << std::endl;
          }
          std::random_shuffle(sample[iclass].begin(), sample[iclass].end());
        }
      }
    }

    size_t fid=0;//unique field identifier
    for(unsigned short iclass=0;iclass<nclass;++iclass){
      if(sample[iclass].empty())
        continue;
      /* unsigned short theClass=(class_opt.size()) ? class_opt[iclass] : iclass; */
      //select random indices for iclass
      size_t nsample=sample[iclass].size();
      double absThreshold=(threshold_opt.size()>iclass)?threshold_opt[iclass]:threshold_opt[0];
      if(absThreshold<0){
        absThreshold=-absThreshold;
        if(absThreshold > sample[iclass].size())
          absThreshold=sample[iclass].size();
      }
      else{
        if(absThreshold > 100)
          absThreshold=100;
        absThreshold/=100;
        absThreshold*=nsample;
      }
      for(size_t isample=0;isample<absThreshold;++isample){
        size_t index=sample[iclass][isample];
        int icol=static_cast<int>(index%nrOfCol());
        int irow=static_cast<int>(index/nrOfCol());
        image2geo(icol,irow,x,y);
        pointAttributes[attribute_opt[0]]=iclass;
        if(fid_opt.size())
          pointAttributes[fid_opt[0]]=fid++;

        for(size_t iplane=0;iplane<nplane;++iplane){
          for(int iband=0;iband<nband;++iband){
            int theBand=iband;
            std::string fieldname;
            std::ostringstream fs;
            if(bandNames_opt.size()){
              if(planeNames_opt.size())
                fs << planeNames_opt[iplane];
              fs << bandNames_opt[iband];
            }
            else{
              if(planeNames_opt.size())
                fs << planeNames_opt[iplane];
              if(nband>1)
                fs << "b" << theBand;
            }
            fieldname=fs.str();
            size_t imindex=index+iplane*nrOfCol()*nrOfRow();
            T* pim=static_cast<T*>(getDataPointer(iband));
            pointAttributes[fieldname]=pim[imindex];
          }
        }
        ogrWriter.addPoint(x,y,pointAttributes);
      }
    }
  }
  catch(std::string predefinedString){
    std::cerr << predefinedString << std::endl;
    throw;
  }
}

#endif // _JLEXTRACTIMG_LIB_H_
