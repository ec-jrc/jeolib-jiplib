/**********************************************************************
jlvalidate_lib.cc: program to validate classified raster image based on reference vector dataset
Author(s): Pieter.Kempeneers@ec.europa.eu
Copyright (c) 2016-2019 European Union (Joint Research Centre)
License EUPLv1.2

This file is part of jiplib
***********************************************************************/
#include <assert.h>
#include "Jim.h"
#include "JimList.h"
#include "base/Optionjl.h"
#include "apps/AppFactory.h"
#include "algorithms/ConfusionMatrix.h"
#include "imageclasses/VectorOgr.h"

using namespace std;
using namespace app;

/**
 * @param reference (type: std::string) Reference vector dataset
 * @param ln (type: std::string) Layer name(s) in sample. Leave empty to select all (for vector reference datasets only)
 * @param band (type: unsigned int) (default: 0) Input (reference) raster band. Optionally, you can define different bands for input and reference bands respectively: -b 1 -b 0.
 * @param confusion (type: bool) (default: 1) Create confusion matrix (to std out)
 * @param lref (type: std::string) (default: label) Attribute name of the reference label (for vector reference datasets only)
 * @param class (type: std::string) List of class names.
 * @param reclass (type: short) List of class values (use same order as in classname option).
 * @param nodata (type: double) No data value(s) in input or reference dataset are ignored
 * @param mask (type: std::string) Use the first band of the specified file as a validity mask. Nodata values can be set with the option msknodata.
 * @param msknodata (type: double) (default: 0) Mask value(s) where image is invalid. Use negative value for valid data (example: use -t -1: if only -1 is valid value)
 * @param output (type: std::string) Output dataset (optional)
 * @param f (type: std::string) (default: SQLite) OGR format for output vector
 * @param lclass (type: std::string) (default: class) Attribute name of the classified label
 * @param cmf (type: std::string) (default: ascii) Format for confusion matrix (ascii or latex)
 * @param cmo (type: std::string) Output file for confusion matrix
 * @param se95 (type: bool) (default: 0) Report standard error for 95 confidence interval
 * @param boundary (type: short) (default: 1) Boundary for selecting the sample
 * @param homogeneous (type: bool) (default: 0) Only take regions with homogeneous boundary into account (for reference datasets only)
 * @param circular (type: bool) (default: 0) Use circular boundary
 * @return CE_None if successful, CE_Failure if not
 **/
CPLErr Jim::validate(AppFactory& app){
  JimList singleList;
  std::shared_ptr<Jim> imgReader=shared_from_this();
  singleList.pushImage(imgReader);
  if(singleList.validate(app).size())
    return(CE_None);
  else
    return(CE_Failure);
}

/**
 * @param reference (type: std::string) Reference vector dataset
 * @param ln (type: std::string) Layer name(s) in sample. Leave empty to select all (for vector reference datasets only)
 * @param band (type: unsigned int) (default: 0) Input (reference) raster band. Optionally, you can define different bands for input and reference bands respectively: -b 1 -b 0.
 * @param confusion (type: bool) (default: 1) Create confusion matrix (to std out)
 * @param lref (type: std::string) (default: label) Attribute name of the reference label (for vector reference datasets only)
 * @param class (type: std::string) List of class names.
 * @param reclass (type: short) List of class values (use same order as in classname option).
 * @param nodata (type: double) No data value(s) in input or reference dataset are ignored
 * @param mask (type: std::string) Use the first band of the specified file as a validity mask. Nodata values can be set with the option msknodata.
 * @param msknodata (type: double) (default: 0) Mask value(s) where image is invalid. Use negative value for valid data (example: use -t -1: if only -1 is valid value)
 * @param output (type: std::string) Output dataset (optional)
 * @param f (type: std::string) (default: SQLite) OGR format for output vector
 * @param lclass (type: std::string) (default: class) Attribute name of the classified label
 * @param cmf (type: std::string) (default: ascii) Format for confusion matrix (ascii or latex)
 * @param cmo (type: std::string) Output file for confusion matrix
 * @param se95 (type: bool) (default: 0) Report standard error for 95 confidence interval
 * @param boundary (type: short) (default: 1) Boundary for selecting the sample
 * @param homogeneous (type: bool) (default: 0) Only take regions with homogeneous boundary into account (for reference datasets only)
 * @param circular (type: bool) (default: 0) Use circular boundary
 * @return reference to the image collection
 **/
JimList& JimList::validate(app::AppFactory& app){
  Optionjl<string> reference_opt("ref", "reference", "Reference vector dataset");
  Optionjl<string> layer_opt("ln", "ln", "Layer name(s) in sample. Leave empty to select all (for vector reference datasets only)");
  Optionjl<string> mask_opt("m", "mask", "Use the first band of the specified file as a validity mask. Nodata values can be set with the option msknodata.");
  Optionjl<double> msknodata_opt("msknodata", "msknodata", "Mask value(s) where image is invalid. Use negative value for valid data (example: use -t -1: if only -1 is valid value)", 0);
  Optionjl<double> nodata_opt("nodata", "nodata", "No data value(s) in input or reference dataset are ignored");
  Optionjl<unsigned int> band_opt("b", "band", "Input (reference) raster band. Optionally, you can define different bands for input and reference bands respectively: -b 1 -b 0.", 0);
  Optionjl<bool> confusion_opt("cm", "confusion", "Create confusion matrix (to std out)", true);
  Optionjl<string> cmformat_opt("cmf","cmf","Format for confusion matrix (ascii or latex)","ascii");
  Optionjl<string> cmoutput_opt("cmo","cmo","Output file for confusion matrix");
  Optionjl<bool> se95_opt("se95","se95","Report standard error for 95 confidence interval",false);
  Optionjl<string> labelref_opt("lr", "lref", "Attribute name of the reference label (for vector reference datasets only)", "label");
  Optionjl<string> classname_opt("c", "class", "List of class names.");
  Optionjl<short> classvalue_opt("r", "reclass", "List of class values (use same order as in classname option).");
  Optionjl<string> output_opt("o", "output", "Output dataset (optional)");
  Optionjl<string> ogrformat_opt("f", "oformat", "OGR format for output vector","SQLite");
  Optionjl<string> labelclass_opt("lc", "lclass", "Attribute name of the classified label", "class");
  Optionjl<short> boundary_opt("bnd", "boundary", "Boundary for selecting the sample", 1,1);
  Optionjl<bool> homogeneous_opt("hom", "homogeneous", "Only take regions with homogeneous boundary into account (for reference datasets only)", false,1);
  Optionjl<bool> disc_opt("circ", "circular", "Use circular boundary", false,1);
  Optionjl<short> verbose_opt("v", "verbose", "Verbose level", 0,2);

  output_opt.setHide(1);
  ogrformat_opt.setHide(1);
  labelclass_opt.setHide(1);
  boundary_opt.setHide(1);
  homogeneous_opt.setHide(1);
  disc_opt.setHide(1);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=reference_opt.retrieveOption(app.getArgc(),app.getArgv());
    layer_opt.retrieveOption(app.getArgc(),app.getArgv());
    band_opt.retrieveOption(app.getArgc(),app.getArgv());
    confusion_opt.retrieveOption(app.getArgc(),app.getArgv());
    labelref_opt.retrieveOption(app.getArgc(),app.getArgv());
    classname_opt.retrieveOption(app.getArgc(),app.getArgv());
    classvalue_opt.retrieveOption(app.getArgc(),app.getArgv());
    nodata_opt.retrieveOption(app.getArgc(),app.getArgv());
    mask_opt.retrieveOption(app.getArgc(),app.getArgv());
    msknodata_opt.retrieveOption(app.getArgc(),app.getArgv());
    output_opt.retrieveOption(app.getArgc(),app.getArgv());
    ogrformat_opt.retrieveOption(app.getArgc(),app.getArgv());
    labelclass_opt.retrieveOption(app.getArgc(),app.getArgv());
    cmformat_opt.retrieveOption(app.getArgc(),app.getArgv());
    cmoutput_opt.retrieveOption(app.getArgc(),app.getArgv());
    se95_opt.retrieveOption(app.getArgc(),app.getArgv());
    boundary_opt.retrieveOption(app.getArgc(),app.getArgv());
    homogeneous_opt.retrieveOption(app.getArgc(),app.getArgv());
    disc_opt.retrieveOption(app.getArgc(),app.getArgv());
    verbose_opt.retrieveOption(app.getArgc(),app.getArgv());
    if(!doProcess){
      cout << endl;
      std::ostringstream helpStream;
      helpStream << "short option -h shows basic options only, use long option --help to show all options" << std::endl;
      throw(helpStream.str());//help was invoked, stop processing
    }

    Jim inputReader;
    Jim maskReader;
    VectorOgr referenceReaderOgr;
    if(verbose_opt[0]){
      cout << "no data flag(s) set to";
      for(int iflag=0;iflag<nodata_opt.size();++iflag)
        cout << " " << nodata_opt[iflag];
      cout << endl;
    }

    if(empty()){
      std::ostringstream errorStream;
      errorStream << "Input collection is empty. Use --help for more help information" << std::endl;
      throw(errorStream.str());
    }
    if(reference_opt.empty()){
      std::ostringstream errorStream;
      errorStream << "No reference file provided (use option -ref). Use --help for help information" << std::endl;
      throw(errorStream.str());
    }

    //band_opt[0] is for input
    //band_opt[1] is for reference
    if(band_opt.size()<2)
      band_opt.push_back(band_opt[0]);

    if(mask_opt.size())
      while(mask_opt.size()<this->size())
        mask_opt.push_back(mask_opt[0]);
    vector<short> inputRange;
    vector<short> referenceRange;
    confusionmatrix::ConfusionMatrix cm;
    int nclass=0;
    map<string,short> classValueMap;
    vector<std::string> nameVector(255);//the inverse of the classValueMap
    vector<string> classNames;

    unsigned int ntotalValidation=0;
    unsigned int nflagged=0;
    Vector2d<unsigned int> resultClass;
    vector<float> user;
    vector<float> producer;
    vector<unsigned int> nvalidation;

    const char* pszMessage;
    void* pProgressArg=NULL;
    GDALProgressFunc pfnProgress=GDALTermProgress;
    float progress=0;
    // if(reference_opt[0].find(".shp")!=string::npos){
    unsigned int iinput=0;
    std::list<std::shared_ptr<Jim> >::const_iterator pimit=begin();
    for(pimit=begin();pimit!=end();++pimit){
    // for(int iinput=0;iinput<this->size();++iinput){
      // if(verbose_opt[0])
      //   cout << "Processing input " << (*pimit)->getFilename() << endl;
      if(output_opt.size())
        assert(reference_opt.size()==output_opt.size());
      for(int iref=0;iref<reference_opt.size();++iref){
        cout << "reference " << reference_opt[iref] << endl;
        // assert(reference_opt[iref].find(".shp")!=string::npos);
        // inputReader.open(input_opt[iinput],memory_opt[0]);//,imagicX_opt[0],imagicY_opt[0]);
        if(mask_opt.size()){
          maskReader.open(mask_opt[iinput]);
          // maskReader.open(mask_opt[iinput],memory_opt[0]);
          // assert(this->at(iinput)->nrOfCol()==maskReader.nrOfCol());
          // assert(this->at(iinput)->nrOfRow()==maskReader.nrOfRow());
        }
        referenceReaderOgr.open(reference_opt[iref]);
        if(confusion_opt[0])
          referenceRange=inputRange;

        VectorOgr ogrWriter;
        if(output_opt.size()){
          try{
            ogrWriter.open(output_opt[iref],ogrformat_opt[0]);
          }
          catch(string error){
            cerr << error << endl;
            throw;
          }
        }
        int nlayer=referenceReaderOgr.getLayerCount();
        for(int ilayer=0;ilayer<nlayer;++ilayer){
          progress=0;
          OGRLayer *readLayer=referenceReaderOgr.getLayer(ilayer);
          //    readLayer = referenceReaderOgr.getDataSource()->GetLayer(ilayer);
          string currentLayername=readLayer->GetName();
          if(layer_opt.size())
            if(find(layer_opt.begin(),layer_opt.end(),currentLayername)==layer_opt.end())
              continue;
          MyProgressFunc(progress,pszMessage,pProgressArg);

          readLayer->ResetReading();
          OGRLayer *writeLayer;
          if(output_opt.size()){
            if(verbose_opt[0])
              cout << "creating output vector file " << output_opt[0] << endl;
            // assert(output_opt[0].find(".shp")!=string::npos);
            char     **papszOptions=NULL;
            if(verbose_opt[0])
              cout << "creating layer: " << readLayer->GetName() << endl;
            // if(ogrWriter.createLayer(layername, referenceReaderOgr.getProjection(ilayer), referenceReaderOgr.getGeometryType(ilayer), papszOptions)==NULL)
            ogrWriter.pushLayer(readLayer->GetName(), referenceReaderOgr.getProjection(ilayer), wkbPoint, papszOptions);
            writeLayer=ogrWriter.getLayer(readLayer->GetName());
            assert(writeLayer);
            if(verbose_opt[0]){
              cout << "created layer" << endl;
              cout << "copy fields from " << reference_opt[iref] << endl;
            }
            ogrWriter.copyFields(referenceReaderOgr,std::vector<std::string>(),ilayer);
            //create extra field for classified label
            short theDim=boundary_opt[0];
            for(int windowJ=-theDim/2;windowJ<(theDim+1)/2;++windowJ){
              for(int windowI=-theDim/2;windowI<(theDim+1)/2;++windowI){
                if(disc_opt[0]&&(windowI*windowI+windowJ*windowJ>(theDim/2)*(theDim/2)))
                  continue;
                ostringstream fs;
                if(theDim>1)
                  fs << labelclass_opt[0] << "_" << windowJ << "_" << windowI;
                else
                  fs << labelclass_opt[0];
                if(verbose_opt[0])
                  cout << "creating field " << fs.str() << endl;
                ogrWriter.createField(fs.str(),OFTInteger,ilayer);
              }
            }
          }
          OGRFeature *readFeature;
          OGRFeature *writeFeature;
          int isample=0;
          unsigned int nfeatureInLayer=readLayer->GetFeatureCount();
          unsigned int ifeature=0;
          while( (readFeature = readLayer->GetNextFeature()) != NULL ){
            if(verbose_opt[0])
              cout << "sample " << ++isample << endl;
            //get x and y from readFeature
            double x,y;
            OGRGeometry *poGeometry;
            OGRPoint centroidPoint;
            OGRPoint *poPoint;
            poGeometry = readFeature->GetGeometryRef();
            // assert( poGeometry != NULL && wkbFlatten(poGeometry->getGeometryType()) == wkbPoint );
            if(poGeometry==NULL)
              continue;
            else if(wkbFlatten(poGeometry->getGeometryType()) == wkbMultiPolygon){
              OGRMultiPolygon readPolygon = *((OGRMultiPolygon *) poGeometry);
              readPolygon = *((OGRMultiPolygon *) poGeometry);
              readPolygon.Centroid(&centroidPoint);
              poPoint=&centroidPoint;
            }
            else if(wkbFlatten(poGeometry->getGeometryType()) == wkbPolygon){
              OGRPolygon readPolygon=*((OGRPolygon *) poGeometry);
              readPolygon.Centroid(&centroidPoint);
              poPoint=&centroidPoint;
            }
            else if(wkbFlatten(poGeometry->getGeometryType()) == wkbPoint )
              poPoint = (OGRPoint *) poGeometry;
            else{
              std::cerr << "Warning: skipping feature (not of type point or polygon)" << std::endl;
              continue;
            }
            x=poPoint->getX();
            y=poPoint->getY();
            double inputValue;
            vector<double> inputValues;
            bool isHomogeneous=true;
            short maskValue;
            //read referenceValue from feature
            unsigned short referenceValue;
            string referenceClassName;
            if(classValueMap.size()){
              referenceClassName=readFeature->GetFieldAsString(readFeature->GetFieldIndex(labelref_opt[0].c_str()));
              referenceValue=classValueMap[referenceClassName];
            }
            else
              referenceValue=readFeature->GetFieldAsInteger(readFeature->GetFieldIndex(labelref_opt[0].c_str()));
            if(verbose_opt[0])
              cout << "reference value: " << referenceValue << endl;

            bool pixelFlagged=false;
            bool maskFlagged=false;
            for(int iflag=0;iflag<nodata_opt.size();++iflag){
              if(referenceValue==nodata_opt[iflag])
                pixelFlagged=true;
            }
            if(pixelFlagged)
              continue;
            double i_centre,j_centre;
            //input reader is georeferenced!
            // this->at(iinput)->geo2image(x,y,i_centre,j_centre);
            (*pimit)->geo2image(x,y,i_centre,j_centre);
            //       else{
            //         i_centre=x;
            //         j_centre=y;
            //       }
            //nearest neighbour
            j_centre=static_cast<unsigned int>(j_centre);
            i_centre=static_cast<unsigned int>(i_centre);
            //check if j_centre is out of bounds
            // if(static_cast<unsigned int>(j_centre)<0||static_cast<unsigned int>(j_centre)>=this->at(iinput)->nrOfRow())
            if(static_cast<unsigned int>(j_centre)<0||static_cast<unsigned int>(j_centre)>=(*pimit)->nrOfRow())
              continue;
            //check if i_centre is out of bounds
            // if(static_cast<unsigned int>(i_centre)<0||static_cast<unsigned int>(i_centre)>=this->at(iinput)->nrOfCol())
            if(static_cast<unsigned int>(i_centre)<0||static_cast<unsigned int>(i_centre)>=(*pimit)->nrOfCol())
              continue;

            if(output_opt.size()){
              writeFeature = OGRFeature::CreateFeature(writeLayer->GetLayerDefn());
              assert(readFeature);
              int nfield=readFeature->GetFieldCount();
              writeFeature->SetGeometry(poPoint);
              if(verbose_opt[0])
                cout << "copying fields from " << reference_opt[0] << endl;
              assert(readFeature);
              assert(writeFeature);
              vector<int> panMap(nfield);
              vector< int>::iterator panit=panMap.begin();
              for(unsigned int ifield=0;ifield<nfield;++ifield)
                panMap[ifield]=ifield;
              writeFeature->SetFieldsFrom(readFeature,&(panMap[0]));
              // if(writeFeature->SetFrom(readFeature)!= OGRERR_NONE)
              //  cerr << "writing feature failed" << endl;
              // if(verbose_opt[0])
              //  cout << "feature written" << endl;
            }
            bool windowAllFlagged=true;
            bool windowHasFlag=false;
            short theDim=boundary_opt[0];
            for(int windowJ=-theDim/2;windowJ<(theDim+1)/2;++windowJ){
              for(int windowI=-theDim/2;windowI<(theDim+1)/2;++windowI){
                if(disc_opt[0]&&(windowI*windowI+windowJ*windowJ>(theDim/2)*(theDim/2)))
                  continue;
                int j=j_centre+windowJ;
                //check if j is out of bounds
                // if(static_cast<unsigned int>(j)<0||static_cast<unsigned int>(j)>=this->at(iinput)->nrOfRow())
                if(static_cast<unsigned int>(j)<0||static_cast<unsigned int>(j)>=(*pimit)->nrOfRow())
                  continue;
                int i=i_centre+windowI;
                //check if i is out of bounds
                // if(static_cast<unsigned int>(i)<0||static_cast<unsigned int>(i)>=this->at(iinput)->nrOfCol())
                if(static_cast<unsigned int>(i)<0||static_cast<unsigned int>(i)>=(*pimit)->nrOfCol())
                  continue;
                if(verbose_opt[0])
                  cout << setprecision(12) << "reading image value at x,y " << x << "," << y << " (" << i << "," << j << "), ";
                // this->at(iinput)->readData(inputValue,i,j,band_opt[0]);
                (*pimit)->readData(inputValue,i,j,band_opt[0]);
                inputValues.push_back(inputValue);
                if(inputValues.back()!=*(inputValues.begin()))
                  isHomogeneous=false;
                if(verbose_opt[0])
                  cout << "input value: " << inputValue << endl;
                pixelFlagged=false;
                for(int iflag=0;iflag<nodata_opt.size();++iflag){
                  if(inputValue==nodata_opt[iflag]){
                    pixelFlagged=true;
                    break;
                  }
                }
                maskFlagged=false;//(msknodata_opt[ivalue]>=0)?false:true;
                if(mask_opt.size()){
                  maskReader.readData(maskValue,i,j,0);
                  for(int ivalue=0;ivalue<msknodata_opt.size();++ivalue){
                    if(msknodata_opt[ivalue]>=0){//values set in msknodata_opt are invalid
                      if(maskValue==msknodata_opt[ivalue]){
                        maskFlagged=true;
                        break;
                      }
                    }
                    else{//only values set in msknodata_opt are valid
                      if(maskValue!=-msknodata_opt[ivalue])
                        maskFlagged=true;
                      else{
                        maskFlagged=false;
                        break;
                      }
                    }
                  }
                }
                pixelFlagged=pixelFlagged||maskFlagged;
                if(pixelFlagged)
                  windowHasFlag=true;
                else
                  windowAllFlagged=false;//at least one good pixel in neighborhood
              }
            }
            //at this point we know the values for the entire window

            if(homogeneous_opt[0]){//only centre pixel
              //flag if not all pixels are homogeneous or if at least one pixel flagged

              if(!windowHasFlag&&isHomogeneous){
                if(output_opt.size())
                  writeFeature->SetField(labelclass_opt[0].c_str(),static_cast<int>(inputValue));
                if(confusion_opt[0]){
                  ++ntotalValidation;
                  if(classValueMap.size()){
                    assert(inputValue<nameVector.size());
                    string className=nameVector[static_cast<unsigned short>(inputValue)];
                    cm.incrementResult(type2string<short>(classValueMap[referenceClassName]),type2string<short>(classValueMap[className]),1);
                  }
                  else{
                    int rc=distance(referenceRange.begin(),find(referenceRange.begin(),referenceRange.end(),static_cast<unsigned short>(referenceValue)));
                    int ic=distance(inputRange.begin(),find(inputRange.begin(),inputRange.end(),static_cast<unsigned short>(inputValue)));
                    assert(rc<nclass);
                    assert(ic<nclass);
                    ++nvalidation[rc];
                    ++resultClass[rc][ic];
                    if(verbose_opt[0]>1)
                      cout << "increment: " << rc << " " << referenceRange[rc] << " " << ic << " " << inputRange[ic] << endl;
                    cm.incrementResult(cm.getClass(rc),cm.getClass(ic),1);
                  }
                }
              }
            }
            else{
              for(int windowJ=-theDim/2;windowJ<(theDim+1)/2;++windowJ){
                for(int windowI=-theDim/2;windowI<(theDim+1)/2;++windowI){
                  if(disc_opt[0]&&(windowI*windowI+windowJ*windowJ>(theDim/2)*(theDim/2)))
                    continue;
                  int j=j_centre+windowJ;
                  //check if j is out of bounds
                  if(static_cast<unsigned int>(j)<0||static_cast<unsigned int>(j)>=(*pimit)->nrOfRow())
                    continue;
                  int i=i_centre+windowI;
                  //check if i is out of bounds
                  if(static_cast<unsigned int>(i)<0||static_cast<unsigned int>(i)>=(*pimit)->nrOfCol())
                    continue;
                  if(!windowAllFlagged){
                    ostringstream fs;
                    if(theDim>1)
                      fs << labelclass_opt[0] << "_" << windowJ << "_" << windowI;
                    else
                      fs << labelclass_opt[0];
                    if(output_opt.size())
                      writeFeature->SetField(fs.str().c_str(),inputValue);
                    if(!windowJ&&!windowI){//centre pixel
                      if(confusion_opt[0]){
                        ++ntotalValidation;
                        if(classValueMap.size()){
                          assert(inputValue<nameVector.size());
                          string className=nameVector[static_cast<unsigned short>(inputValue)];
                          cm.incrementResult(type2string<short>(classValueMap[referenceClassName]),type2string<short>(classValueMap[className]),1);
                        }
                        else{
                          int rc=distance(referenceRange.begin(),find(referenceRange.begin(),referenceRange.end(),static_cast<unsigned short>(referenceValue)));
                          int ic=distance(inputRange.begin(),find(inputRange.begin(),inputRange.end(),static_cast<unsigned short>(inputValue)));
                          if(rc>=nclass)
                            continue;
                          if(ic>=nclass)
                            continue;
                          // assert(rc<nclass);
                          // assert(ic<nclass);
                          ++nvalidation[rc];
                          ++resultClass[rc][ic];
                          if(verbose_opt[0]>1)
                            cout << "increment: " << rc << " " << referenceRange[rc] << " " << ic << " " << inputRange[ic] << endl;
                          cm.incrementResult(cm.getClass(rc),cm.getClass(ic),1);
                        }
                      }
                    }
                  }
                }
              }
            }
            if(output_opt.size()){
              if(!windowAllFlagged){
                if(verbose_opt[0])
                  cout << "creating feature" << endl;
                if(writeLayer->CreateFeature( writeFeature ) != OGRERR_NONE ){
                  string errorString="Failed to create feature in OGR vector file";
                  throw(errorString);
                }
              }
              OGRFeature::DestroyFeature( writeFeature );
            }
            ++ifeature;
            progress=static_cast<float>(ifeature+1)/nfeatureInLayer;
            MyProgressFunc(progress,pszMessage,pProgressArg);
          }//next feature
        }//next layer
        if(output_opt.size())
          ogrWriter.close();
        referenceReaderOgr.close();
        // (*pimit)->close();
        if(mask_opt.size())
          maskReader.close();
      }//next reference
      ++iinput;
    }//next input
    MyProgressFunc(1.0,pszMessage,pProgressArg);

    if(confusion_opt[0]){
      cm.setFormat(cmformat_opt[0]);
      cm.reportSE95(se95_opt[0]);
      ofstream outputFile;
      if(cmoutput_opt.size()){
        outputFile.open(cmoutput_opt[0].c_str(),ios::out);
        outputFile << cm << endl;
      }
      else
        cout << cm << endl;
    }
    // return(CE_None);
  }
  catch(string predefinedString){
    std::cout << predefinedString << std::endl;
    throw;
  }
  return(*this);
}
