/**********************************************************************
VectorOgr.cc: class to hold OGR features, typically read with readNextFeature
Copyright (c) 2016-2019 European Union (Joint Research Centre)
License EUPLv1.2

This file is part of jiplib
***********************************************************************/
#include <unordered_map>
#include "VectorOgr.h"
#include "base/Optionjl.h"
#include "imageclasses/Jim.h"

using namespace std;
using namespace statfactory;

namespace svm{
  enum SVM_TYPE {C_SVC=0, nu_SVC=1,one_class=2, epsilon_SVR=3, nu_SVR=4};
  enum KERNEL_TYPE {linear=0,polynomial=1,radial=2,sigmoid=3};
}

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

///Default constructor
// VectorOgr::VectorOgr(void) : m_gds(NULL), m_layer(NULL), m_access(GDAL_OF_READONLY), m_projection(std::string()){}
VectorOgr::VectorOgr(void) : m_gds(NULL), m_access(GDAL_OF_READONLY){m_filename.clear();}

///Copy constructor
VectorOgr::VectorOgr(VectorOgr& other, app::AppFactory &app) : m_gds(NULL){
  Optionjl<std::string> filename_opt("fn", "filename", "filename");
  Optionjl<std::string> ogrformat_opt("f", "oformat", "Output sample dataset format","SQLite");
  Optionjl<short> verbose_opt("v", "verbose", "Verbose mode if > 0", 0,2);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=filename_opt.retrieveOption(app);
    ogrformat_opt.retrieveOption(app);
    verbose_opt.retrieveOption(app);
  }
  catch(std::string predefinedString){
    std::cout << predefinedString << std::endl;
  }
  if(!doProcess){
    std::cout << std::endl;
    std::ostringstream helpStream;
    helpStream << "exception thrown due to help info";
    throw(helpStream.str());//help was invoked, stop processing
  }

  if(filename_opt.empty()){
    std::ostringstream helpStream;
    helpStream << "Error: VectorOgr constructor needs filename key";
    throw(helpStream.str());//help was invoked, stop processing
  }

  open(filename_opt[0], ogrformat_opt[0], 1);
  copy(other,app);
}

///Copy constructor
// VectorOgr::VectorOgr(VectorOgr& other, const std::string& filename, const std::string& imageType, char** options){
//   open(filename, imageType);
//   copy(other);
// }

///Destructor
VectorOgr::~VectorOgr(void){
  destroyAll();
  close();
  m_filename.clear();
};

///assignment operator
// VectorOgr& VectorOgr::operator=(VectorOgr& other)
// {
//   //check for assignment to self (of the form v=v)
//   if(this==&other)
//     return *this;
//   else{
//     copy(other,copyData);
//     return *this;
//   }
// }

// ///reset all features, releasing memory from heap
// void VectorOgr::destroyFeatures(size_t ilayer){
//   for(std::vector<OGRFeature*>::iterator fit=m_features[ilayer].begin();fit!=m_features[ilayer].end();++fit){
//     if(*fit){
//       OGRFeature::DestroyFeature( *fit );
//       *fit=NULL;
//     }
//     else
//       std::cerr << "Error: cannot destroy NULL feature" << std::endl;
//   }
//   m_features[ilayer].clear();
//   m_features.erase(m_features.begin()+ilayer);
//   m_layer.erase(m_layer.begin()+ilayer);
// }

///reset all features, releasing memory from heap
void VectorOgr::destroyAll(){
  for(size_t ilayer=0;ilayer<getLayerCount();++ilayer){
    destroyFeatures(ilayer);
  }
  m_features.clear();
  m_layer.clear();
}

bool VectorOgr::isEmpty(size_t ilayer){
  if(m_features.size()>ilayer){
    std::vector<OGRFeature*>::iterator fit=m_features[ilayer].begin();
    while(fit!=m_features[ilayer].end()){
      if(*fit){
        return false;
      }
      ++fit;
    }
  }
  return true;
}

void VectorOgr::destroyEmptyFeatures(size_t ilayer){
  if(m_features.size()>ilayer){
    std::vector<OGRFeature*>::iterator fit=m_features[ilayer].begin();
    while(fit!=m_features[ilayer].end()){
      if(*fit){
        ++fit;
        continue;
      }
      else{
        m_features[ilayer].erase(fit);
      }
    }
  }
}

void VectorOgr::destroyFeatures(size_t ilayer){
  if(m_features.size()>ilayer){
    for(std::vector<OGRFeature*>::iterator fit=m_features[ilayer].begin();fit!=m_features[ilayer].end();++fit){
      if(*fit){
        OGRFeature::DestroyFeature( *fit );
        *fit=NULL;
      }
      // else
      //   std::cerr << "Error: cannot destroy NULL feature" << std::endl;
    }
    m_features[ilayer].clear();
  }
}

// ///Assignment operator
// VectorOgr& VectorOgr::operator=(VectorOgr& other){
//   if(this!=&other){
//     destroyFeatures();
//     for(unsigned int index=0;index<other.getFeatureCount();++index)
//       m_features.push_back(other.cloneFeature(index));
//   }
//   return(*this);
// }

///open a GDAL vector dataset for reading
OGRErr VectorOgr::open(const std::string& ogrFilename, const std::vector<std::string>& layernames, bool noread){
  try{
    m_filename = ogrFilename;
#if GDAL_VERSION_MAJOR < 2
    //register the drivers
    OGRRegisterAll();
    //open the input OGR datasource. Datasources can be files, RDBMSes, directories full of files, or even remote web services depending on the driver being used. However, the datasource name is always a single string.
    m_gds = OGRSFDriverRegistrar::Open(ogrFilename, FALSE);//FALSE: do not update
#else
    //register the drivers
    GDALAllRegister();
    //open the input OGR datasource. Datasources can be files, RDBMSes, directories full of files, or even remote web services depending on the driver being used. However, the datasource name is always a single string.
    m_gds = (GDALDataset*) GDALOpenEx(ogrFilename.c_str(), GDAL_OF_VECTOR||GDAL_OF_READONLY, NULL, NULL, NULL);
#endif
    if( m_gds == NULL ){
#if GDAL_VERSION_MAJOR < 2
      std::string errorString="Open failed";
      throw(errorString);
#else
      m_gds = (GDALDataset*) GDALOpenEx(ogrFilename.c_str(), GDAL_OF_VECTOR, NULL, NULL, NULL);
      if( m_gds == NULL ){
        ostringstream errorStream;
        errorStream << "Open failed for file " << ogrFilename << std::endl;
        throw(errorStream.str());
      }
#endif
    }
    for(size_t ilayer=0;ilayer<getGDSLayerCount();++ilayer){
      OGRLayer *readLayer=m_gds->GetLayer(ilayer);
      string currentLayername=readLayer->GetName();
      if(layernames.size()){
        vector<string>::const_iterator it=find(layernames.begin(),layernames.end(),currentLayername);
        if(it==layernames.end()){
          continue;
        }
      }
      m_layer.push_back(readLayer);
      m_features.resize(m_layer.size());
      if(!noread)
        readFeatures(m_layer.size()-1);
    }
  }
  catch(std::string errorString){
    std::cerr << errorString << std::endl;
    throw;
  }
  return(OGRERR_NONE);
  // return(setProjection(m_gds->GetProjectionRef()));
}

///open a GDAL vector dataset for writing with layers to be pushed later
OGRErr VectorOgr::open(const std::string& filename, const std::string& imageType, unsigned int access){
  try{
    setAccess(access);
    m_filename = filename;
    setCodec(filename,imageType);
    if( m_gds == NULL ){
      std::string errorString="Open failed (1)";
      throw(errorString);
    }
    return(OGRERR_NONE);
  }
  catch(std::string errorString){
    std::cerr << errorString << std::endl;
    throw;
  }
}

///open a GDAL vector dataset for writing
OGRErr VectorOgr::open(const std::string& filename, const std::vector<std::string>& layernames, const std::string& imageType, const OGRwkbGeometryType& geometryType, OGRSpatialReference* theSRS, char** options){
  try{
    if(open(filename,imageType)!=OGRERR_NONE)
      return(OGRERR_FAILURE);
    for(size_t ilayer=0;ilayer<layernames.size();++ilayer){
      if(pushLayer(layernames[ilayer],theSRS,geometryType,options)!=OGRERR_NONE){
        std::string errorString="Open failed";
        throw(errorString);
      }
    }
    return(OGRERR_NONE);
  }
  catch(std::string errorString){
    std::cerr << errorString << std::endl;
    throw;
  }
}

///open a GDAL vector dataset for writing
OGRErr VectorOgr::open(const std::string& filename, const std::vector<std::string>& layernames, const std::string& imageType, const OGRwkbGeometryType& geometryType, const std::string& theProjection, char** options){
  try{
    if(open(filename,imageType)!=OGRERR_NONE)
      return(OGRERR_FAILURE);
    for(size_t ilayer=0;ilayer<layernames.size();++ilayer){
      if(pushLayer(layernames[ilayer],theProjection,geometryType,options)!=OGRERR_NONE){
        std::string errorString="Open failed";
        throw(errorString);
      }
    }
    return(OGRERR_NONE);
  }
  catch(std::string errorString){
    std::cerr << errorString << std::endl;
    throw;
  }
}

///open a GDAL vector dataset for writing
OGRErr VectorOgr::open(const std::string& filename, const std::vector<std::string>& layernames, const std::string& imageType, const std::string& geometryType, const std::string& theProjection, char** options){
  try{
    if(open(filename,imageType)!=OGRERR_NONE)
      return(OGRERR_FAILURE);
    for(size_t ilayer=0;ilayer<layernames.size();++ilayer){
      if(pushLayer(layernames[ilayer],theProjection,geometryType,options)!=OGRERR_NONE){
        std::string errorString="Open failed";
        throw(errorString);
      }
    }
    return(OGRERR_NONE);
  }
  catch(std::string errorString){
    std::cerr << errorString << std::endl;
    throw;
  }
}

///open vector dataset for reading/writing
OGRErr VectorOgr::open(app::AppFactory& app){
  Optionjl<std::string> filename_opt("fn", "filename", "filename");
  Optionjl<std::string> layer_opt("ln", "ln", "Layer name");
  Optionjl<std::string> projection_opt("a_srs", "a_srs", "Assign projection");
  Optionjl<std::string> geometryType_opt("gtype", "gtype", "Geometry type","wkbUnknown");
  Optionjl<std::string> options_opt("co", "co", "format dependent options controlling creation of the output file");
  Optionjl<std::string> ogrformat_opt("f", "oformat", "Output sample dataset format","SQLite");
  Optionjl<unsigned int> access_opt("access", "access", "Access (0: GDAL_OF_READ_ONLY, 1: GDAL_OF_UPDATE)",0);
  Optionjl<bool> noread_opt("noread", "noread", "do not read features when opening)",false);
  Optionjl<std::string> attributeFilter_opt("af", "attributeFilter", "attribute filter");
  Optionjl<double> ulx_opt("ulx", "ulx", "Upper left x value bounding box");
  Optionjl<double> uly_opt("uly", "uly", "Upper left y value bounding box");
  Optionjl<double> lrx_opt("lrx", "lrx", "Lower right x value bounding box");
  Optionjl<double> lry_opt("lry", "lry", "Lower right y value bounding box");
  Optionjl<short> verbose_opt("v", "verbose", "Verbose mode if > 0", 0,2);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=filename_opt.retrieveOption(app);
    layer_opt.retrieveOption(app);
    projection_opt.retrieveOption(app);
    geometryType_opt.retrieveOption(app);
    options_opt.retrieveOption(app);
    ogrformat_opt.retrieveOption(app);
    access_opt.retrieveOption(app);
    noread_opt.retrieveOption(app);
    attributeFilter_opt.retrieveOption(app);
    ulx_opt.retrieveOption(app);
    uly_opt.retrieveOption(app);
    lrx_opt.retrieveOption(app);
    lry_opt.retrieveOption(app);
    verbose_opt.retrieveOption(app);
  }
  catch(std::string predefinedString){
    std::cout << predefinedString << std::endl;
  }
  if(!doProcess){
    std::cout << std::endl;
    std::ostringstream helpStream;
    helpStream << "exception thrown due to help info";
    throw(helpStream.str());//help was invoked, stop processing
  }

  // std::vector<std::string> badKeys;
  // app.badKeys(badKeys);
  // if(badKeys.size()){
  //   std::ostringstream errorStream;
  //   if(badKeys.size()>1)
  //     errorStream << "Error: unknown keys: ";
  //   else
  //     errorStream << "Error: unknown key: ";
  //   for(int ikey=0;ikey<badKeys.size();++ikey){
  //     errorStream << badKeys[ikey] << " ";
  //   }
  //   errorStream << std::endl;
  //   throw(errorStream.str());
  // }
  if(filename_opt.empty()){
    std::ostringstream helpStream;
    helpStream << "Error: VectorOgr constructor needs filename key";
    throw(helpStream.str());//help was invoked, stop processing
  }
  setAccess(access_opt[0]);
  if(getAccess()==GDAL_OF_READONLY){
    if(verbose_opt[0])
      std::cout << "open in read access mode" << std::endl;
    bool noread=true;
    //do not read features yet, only initialize layers
    open(filename_opt[0],layer_opt,noread);
    if(m_gds){
      // for(size_t ilayer=0;ilayer<getGDSLayerCount();++ilayer){
      for(size_t ilayer=0;ilayer<getLayerCount();++ilayer){
        // OGRLayer *readLayer=m_gds->GetLayer(ilayer);
        OGRLayer *readLayer=getLayer(ilayer);
        // string currentLayername=readLayer->GetName();
        // if(layer_opt.size()){
        //   vector<string>::const_iterator it=find(layer_opt.begin(),layer_opt.end(),currentLayername);
        //   if(it==layer_opt.end())
        //     continue;
        // }
        //we already initialized the layers in the open above
        // if(ilayer<getLayerCount())
        //   m_layer[ilayer]=readLayer;
        // else
        //   m_layer.push_back(readLayer);
        // m_features.resize(m_layer.size());
        unsigned int nfeatures=0;
        if(attributeFilter_opt.size())
          setAttributeFilter(attributeFilter_opt[0],ilayer);
        if(ulx_opt.size()&&uly_opt.size()&&lrx_opt.size()&&lry_opt.size())
          setSpatialFilterRect(ulx_opt[0],uly_opt[0],lrx_opt[0],lry_opt[0],ilayer);
        if(!noread_opt[0])
          nfeatures=readFeatures(ilayer);
        if(verbose_opt[0])
          std::cout << "read " << nfeatures << " features in layer " << getLayerName(ilayer) << std::endl;
        // unsigned int nfeatures=0;
        // if(attributeFilter_opt.size())
        //   setAttributeFilter(attributeFilter_opt[0],ilayer);
        // if(!noread_opt[0])
        //   nfeatures=readFeatures(ilayer);
        // if(verbose_opt[0])
        //   std::cout << "read " << nfeatures << " features in layer " << ilayer << std::endl;
      }
    }
  }
  //open for writing
  if(!m_gds){
    setAccess(GDAL_OF_UPDATE);
  }
  if(getAccess()==GDAL_OF_UPDATE){
    if(verbose_opt[0])
      std::cout << "open in update mode" << std::endl;
    if(layer_opt.size()){
      if(projection_opt.size()){
        return(open(filename_opt[0], layer_opt, ogrformat_opt[0], geometryType_opt[0], projection_opt[0]));
      }
      else{
        OGRwkbGeometryType gType=string2geotype(geometryType_opt[0]);
        return(open(filename_opt[0], layer_opt, ogrformat_opt[0], gType, NULL));
      }
    }
    else{
      return(open(filename_opt[0],ogrformat_opt[0]),getAccess());
    }
  }
  if(verbose_opt[0])
    std::cout << "Opened vector dataset" << std::endl;
}

///prepare driver for writing vector dataset
void VectorOgr::setCodec(const std::string& filename, const std::string& imageType){
  if(m_gds){
    std::string errorString="Warning: vector dataset is already open";
    throw(errorString);
  }
#if GDAL_VERSION_MAJOR < 2
  //register the drivers
  OGRRegisterAll();
  //fetch the OGR file driver
  OGRSFDriver *poDriver;
  poDriver = OGRSFDriverRegistrar::GetRegistrar()->GetDriverByName(imageType.c_str());
#else
  //register the drivers
  GDALAllRegister();
  GDALDriver *poDriver;
  poDriver = GetGDALDriverManager()->GetDriverByName(imageType.c_str());
#endif
  if( poDriver == NULL ){
    std::string errorString="Error: FileOpenError";
    throw(errorString);
  }
  if(getAccess()==GDAL_OF_UPDATE){
    //check if file already exists
#if GDAL_VERSION_MAJOR < 2
    m_gds = OGRSFDriverRegistrar::Open( filename.c_str(), TRUE );
#else
    m_gds = (GDALDataset*) GDALOpenEx(filename.c_str(), GDAL_OF_UPDATE||GDAL_OF_VECTOR, NULL, NULL, NULL);
    if(!m_gds)
      m_gds=poDriver->Create(filename.c_str(),0,0,0,GDT_Unknown,NULL);
#endif
  }
  else{//(overwrite existing layer)
#if GDAL_VERSION_MAJOR < 2
    m_gds = OGRSFDriverRegistrar::Open( filename.c_str(), FALSE );
#else
    m_gds = (GDALDataset*) GDALOpenEx(filename.c_str(), GDAL_OF_READONLY||GDAL_OF_VECTOR, NULL, NULL, NULL);
#endif
    if(m_gds){
#if GDAL_VERSION_MAJOR < 2
      OGRDataSource::DestroyDataSource(m_gds);
#else
      GDALClose(m_gds);
#endif
    }
#if GDAL_VERSION_MAJOR < 2
    m_gds=poDriver->CreateDataSource(filename.c_str(),NULL);
#else
    m_gds=poDriver->Create(filename.c_str(),0,0,0,GDT_Unknown,NULL);
#endif
  }
  if(!m_gds){
    std::string errorString="m_gds is 0";
    throw(errorString);
  }
}

//close the vector dataset
void VectorOgr::close(void)
{
  // destroyFeatures();//already done in destructor
  if(m_gds){
#if GDAL_VERSION_MAJOR < 2
    OGRDataSource::DestroyDataSource(m_gds);
#else
    GDALClose(m_gds);
#endif
    m_gds=NULL;
  }
}

// OGRErr VectorOgr::setProjection(const std::string& theProjection, size_t ilayer){
//   OGRSpatialReference theRef;
//   if(theProjection.empty())
//     return(CE_None);
//   theRef.SetFromUserInput(theProjection.c_str());
//   char *wktString;
//   theRef.exportToWkt(&wktString);
//   m_projection=wktString;
//   if(m_gds)
//     return(m_gds->SetProjection(wktString));
//   else
//     return(CE_Warning);
// }

///Create a layer
OGRErr VectorOgr::pushLayer(const std::string& layername, OGRSpatialReference* theSRS, const OGRwkbGeometryType& geometryType, char** papszOptions){
  if( !m_gds->TestCapability( ODsCCreateLayer ) ){
    // std::ostringstream errorStream;
    // errorStream << "Error: Test capability to create layer " << layername << " failed (1)" << std::endl;
    // throw(errorStream.str());
    std::cerr << "Error: Test capability to create layer " << layername << " failed (1)" << std::endl;
    return(OGRERR_FAILURE);
  }
  //if no constraints on the types geometry to be written: use wkbUnknown
  m_layer.push_back(m_gds->CreateLayer(layername.c_str(), theSRS, geometryType ,papszOptions));
  m_features.resize(m_layer.size());
  if(!m_layer.back()){
    std::string errorString="Open failed";
    throw(errorString);
  }
  return(OGRERR_NONE);
}

///Create a layer
OGRErr VectorOgr::pushLayer(const std::string& layername, const std::string& theProjection, const OGRwkbGeometryType& geometryType, char** papszOptions){
  OGRErr result=OGRERR_NONE;
  if( !m_gds->TestCapability( ODsCCreateLayer ) ){
    // std::ostringstream errorStream;
    // errorStream << "Error: Test capability to create layer " << layername << " failed (2)" << std::endl;
    // throw(errorStream.str());
    std::cerr << "Error: Test capability to create layer " << layername << " failed (2)" << std::endl;
    return(OGRERR_FAILURE);
  }
  //if no constraints on the types geometry to be written: use wkbUnknown
  if(theProjection.size()){
    OGRSpatialReference* poSRS = new OGRSpatialReference();
    // OGRSpatialReference oSRS;
    poSRS->SetFromUserInput(theProjection.c_str());
    result=pushLayer(layername.c_str(), poSRS, geometryType, papszOptions);
    poSRS->Release();
  }
  else
    result=pushLayer(layername.c_str(), NULL, geometryType, papszOptions);
  return(result);
}

///Create a layer
OGRErr VectorOgr::pushLayer(const std::string& layername, const std::string& theProjection, const std::string& geometryType, char** papszOptions){
  OGRErr result=OGRERR_NONE;
  if( !m_gds->TestCapability( ODsCCreateLayer ) ){
    // std::ostringstream errorStream;
    // errorStream << "Error: Test capability to create layer " << layername << " failed (3)" << std::endl;
    // throw(errorStream.str());
    std::cerr << "Error: Test capability to create layer " << layername << " failed (3)" << std::endl;
    return(OGRERR_FAILURE);
  }
  //if no constraints on the types geometry to be written: use wkbUnknown
  OGRwkbGeometryType eGType=wkbUnknown;
  if(geometryType.size())
    eGType=string2geotype(geometryType);

  if(theProjection.size()){
    OGRSpatialReference* poSRS = new OGRSpatialReference();
    // OGRSpatialReference oSRS;
    poSRS->SetFromUserInput(theProjection.c_str());
    result=pushLayer(layername.c_str(), poSRS, eGType, papszOptions);
    poSRS->Release();
  }
  else
    result=pushLayer(layername.c_str(), NULL, eGType, papszOptions);
  //check if destroy is needed?!
  // CSLDestroy( papszOptions );
  return(result);
}

std::shared_ptr<VectorOgr> VectorOgr::intersect(OGRPolygon *pGeom, app::AppFactory& app){
  shared_ptr<VectorOgr> ogrWriter=VectorOgr::createVector();
  if(intersect(pGeom, *ogrWriter, app)!=OGRERR_NONE){
    std::ostringstream errorStream;
    errorStream << "Error: failed to intersect" << std::endl;
    std::cerr << errorStream.str() << std::endl;
    throw(errorStream);
  }
  return(ogrWriter);
}

std::shared_ptr<VectorOgr> VectorOgr::intersect(const Jim& aJim, app::AppFactory& app){
  shared_ptr<VectorOgr> ogrWriter=VectorOgr::createVector();
  if(intersect(aJim, *ogrWriter, app)!=OGRERR_NONE){
    std::ostringstream errorStream;
    errorStream << "Error: failed to intersect" << std::endl;
    std::cerr << errorStream.str() << std::endl;
    throw(errorStream);
  }
  return(ogrWriter);
}

OGRErr VectorOgr::intersect(const Jim& aJim, VectorOgr& ogrWriter, app::AppFactory& app){
  OGRErr result=OGRERR_NONE;
  OGRPolygon *pGeom = (OGRPolygon*) OGRGeometryFactory::createGeometry(wkbPolygon);
  OGRSpatialReference imgSpatialRef(aJim.getProjectionRef().c_str());
  OGRSpatialReference *thisSpatialRef=getLayer()->GetSpatialRef();
  OGRCoordinateTransformation *img2vector = OGRCreateCoordinateTransformation(&imgSpatialRef, thisSpatialRef);
  aJim.getBoundingBox(pGeom,img2vector);
  result=intersect(pGeom,ogrWriter,app);
  OGRGeometryFactory::destroyGeometry(pGeom );
  return(result);
}

OGRErr VectorOgr::intersect(OGRPolygon *pGeom, VectorOgr& ogrWriter, app::AppFactory& app){
  Optionjl<string> output_opt("o", "output", "Output sample dataset");
  Optionjl<string> ogrformat_opt("f", "oformat", "Output vector dataset format","SQLite");
  Optionjl<unsigned int> access_opt("access", "access", "Access (0: GDAL_OF_READ_ONLY, 1: GDAL_OF_UPDATE)",1);
  Optionjl<std::string> options_opt("co", "co", "format dependent options controlling creation of the output file");
  // Optionjl<bool> allCovered_opt("ac", "all_covered", "Set this flag to include only those polygons that are entirely covered by the raster", false);
  Optionjl<short> verbose_opt("v", "verbose", "Verbose mode if > 0", 0,2);

  // allCovered_opt.setHide(1);
  options_opt.setHide(1);


  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=output_opt.retrieveOption(app);
    ogrformat_opt.retrieveOption(app);
    access_opt.retrieveOption(app);
    options_opt.retrieveOption(app);
    // allCovered_opt.retrieveOption(app);
    verbose_opt.retrieveOption(app);
    if(!doProcess){
      cout << endl;
      std::ostringstream helpStream;
      helpStream << "short option -h shows basic options only, use long option --help to show all options" << std::endl;
      throw(helpStream.str());//help was invoked, stop processing
    }
    char **papszOptions=NULL;
    for(std::vector<std::string>::const_iterator optionIt=options_opt.begin();optionIt!=options_opt.end();++optionIt){
      papszOptions=CSLAddString(papszOptions,optionIt->c_str());
    }

    if(output_opt.empty()){
      std::ostringstream errorStream;
      errorStream << "Error: no output file provided" << std::endl;
      throw(errorStream.str());
    }

    ogrWriter.open(output_opt[0],ogrformat_opt[0]);

    for(int ilayer=0;ilayer<getLayerCount();++ilayer){
      ogrWriter.pushLayer(getLayerName(ilayer),getProjection(ilayer),getGeometryType(ilayer),papszOptions);
      ogrWriter.copyFields(*this,std::vector<std::string>(),ilayer);
      ogrWriter.resize(getFeatureCount(ilayer),ilayer);

#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
      for(size_t ifeature=0;ifeature<getFeatureCount(ilayer);++ifeature){
        if(verbose_opt[0]>1)
          std::cout << "feature " << ifeature << endl;
        OGRFeature *readFeature=getFeatureRef(ifeature,ilayer);
        if(readFeature){
          if(readFeature->GetGeometryRef()->Intersects(pGeom)){
            if(verbose_opt[0]>1)
              std::cout << "write valid feature " << ifeature << endl;
            OGRFeature *writeFeature=ogrWriter.createFeature(ilayer);
            writeFeature->SetFrom(readFeature);
            //todo: only set intersected features. check if NULL features are a problem when writing
            // ogrWriter.pushFeature(writeFeature,ilayer);
            ogrWriter.setFeature(ifeature,writeFeature,ilayer);
          }
          else{
            if(verbose_opt[0]>1)
              std::cerr << "Warning: " << ifeature << " is not intersecting" << std::endl;
          }
        }
      }
      //todo: check if needed?
      // ogrWriter.destroyEmptyFeatures(ilayer);
    }
    return(OGRERR_NONE);
  }
  catch(std::string errorString){
    std::cerr << "Error: " << errorString << std::endl;
    throw;
  }
  catch(...){
    std::cerr << "Error: undefined" << std::endl;
    throw;
  }
}

std::shared_ptr<VectorOgr> VectorOgr::convexHull(app::AppFactory& app){
  shared_ptr<VectorOgr> ogrWriter=VectorOgr::createVector();
  if(convexHull(*ogrWriter, app)!=OGRERR_NONE){
    std::cerr << "Failed to convexHull" << std::endl;
  }
  return(ogrWriter);
}

OGRErr VectorOgr::convexHull(VectorOgr& ogrWriter, app::AppFactory& app){
  Optionjl<string> output_opt("o", "output", "Output sample dataset");
  Optionjl<string> ogrformat_opt("f", "oformat", "Output vector dataset format","SQLite");
  Optionjl<unsigned int> access_opt("access", "access", "Access (0: GDAL_OF_READ_ONLY, 1: GDAL_OF_UPDATE)",1);
  Optionjl<std::string> options_opt("co", "co", "format dependent options controlling creation of the output file");
  // Optionjl<bool> allCovered_opt("ac", "all_covered", "Set this flag to include only those polygons that are entirely covered by the raster", false);
  Optionjl<short> verbose_opt("v", "verbose", "Verbose mode if > 0", 0,2);

  // allCovered_opt.setHide(1);
  options_opt.setHide(1);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=output_opt.retrieveOption(app);
    ogrformat_opt.retrieveOption(app);
    access_opt.retrieveOption(app);
    options_opt.retrieveOption(app);
    // allCovered_opt.retrieveOption(app);
    verbose_opt.retrieveOption(app);
    if(!doProcess){
      cout << endl;
      std::ostringstream helpStream;
      helpStream << "short option -h shows basic options only, use long option --help to show all options" << std::endl;
      throw(helpStream.str());//help was invoked, stop processing
    }
    char **papszOptions=NULL;
    for(std::vector<std::string>::const_iterator optionIt=options_opt.begin();optionIt!=options_opt.end();++optionIt){
      papszOptions=CSLAddString(papszOptions,optionIt->c_str());
    }

    if(output_opt.empty()){
      std::ostringstream errorStream;
      errorStream << "Error: no output file provided" << std::endl;
      throw(errorStream.str());
    }

    ogrWriter.open(output_opt[0],ogrformat_opt[0]);
    for(int ilayer=0;ilayer<getLayerCount();++ilayer){
      ogrWriter.pushLayer(getLayerName(ilayer),getLayer(ilayer)->GetSpatialRef(),wkbPolygon,papszOptions);
// #if JIPLIB_PROCESS_IN_PARALLEL == 1
// #pragma omp parallel for
// #else
// #endif
      ogrWriter.createField("id",OFTInteger,ilayer);
      OGRGeometryCollection geomColl;
      for(size_t ifeature=0;ifeature<getFeatureCount(ilayer);++ifeature){
        OGRFeature *readFeature=getFeatureRef(ifeature,ilayer);
        if(readFeature)
          geomColl.addGeometry(readFeature->GetGeometryRef());
      }

      OGRFeature *writeFeature=ogrWriter.createFeature(ilayer);
      //todo: only set convexHulled features. check if NULL features are a problem when writing
      // ogrWriter.pushFeature(writeFeature,ilayer);
      OGRGeometry *pGeom=geomColl.ConvexHull();
      OGRFeature *poFeature=ogrWriter.createFeature(ilayer);
      poFeature->SetField("id",1);
      poFeature->SetGeometry(pGeom);
      ogrWriter.pushFeature(poFeature,ilayer);
      //todo:check if need to destroy feature?
      // destroyEmptyFeatures(ilayer);
    }
    return(OGRERR_NONE);
  }
  catch(std::string errorString){
    std::cerr << "Error: " << errorString << std::endl;
    throw;
  }
  catch(...){
    std::cerr << "Error: undefined" << std::endl;
    throw;
  }
}

  OGRErr VectorOgr::createField(const std::string& fieldname, const OGRFieldType& fieldType, size_t ilayer){
  OGRFieldDefn oField( fieldname.c_str(), fieldType );
  if(fieldType==OFTString)
    oField.SetWidth(32);
  return(getLayer(ilayer)->CreateField( &oField ));
}

  ///create field
 OGRErr VectorOgr::createField(OGRFieldDefn*	poField, size_t ilayer){
    if(m_gds)
      return(m_gds->GetLayer(ilayer)->CreateField(poField));
    else
      return(OGRERR_FAILURE);
  }

///copy fields from other VectorOgr instance
 OGRErr VectorOgr::copyFields(const VectorOgr& vectorOgr,const vector<std::string>& fieldnames, size_t ilayer){
  try{
    if(!m_gds){
      std::string errorString="Error: no GDAL dataset";
      throw(errorString);
    }
    //get fields from vectorOgr
    std::vector<OGRFieldDefn*> fields;
    vectorOgr.getFields(fields,ilayer);

    for(unsigned int iField=0;iField<fields.size();++iField){
      if(fieldnames.size()){
        std::string fieldname=fields[iField]->GetNameRef();
        std::vector<std::string>::const_iterator fit = std::find(fieldnames.begin(), fieldnames.end(), fieldname);
        if(fit!=fieldnames.end()){
          if(!m_gds->GetLayer(ilayer))
            std::cerr << "Warning: could not get layer" << std::endl;
          if(createField(fields[iField],ilayer)!=OGRERR_NONE){
            std::string errorString="Error: could not create field";
            throw(errorString);
          }
        }
        else
          continue;
      }
      else{
        if(createField(fields[iField],ilayer)!=OGRERR_NONE){
          std::string errorString="Error: could not create field";
          throw(errorString);
        }
      }
    }
    return(OGRERR_NONE);
  }
  catch(std::string errorString){
    std::cerr << errorString << std::endl;
    throw;
  }
}

///perform a deep copy, including layers and features
OGRErr VectorOgr::copy(VectorOgr& other, app::AppFactory &app){
  char **papszOptions=NULL;
  Optionjl<std::string> options_opt("co", "co", "format dependent options controlling creation of the output file");
  options_opt.retrieveOption(app);
  for(std::vector<std::string>::const_iterator optionIt=options_opt.begin();optionIt!=options_opt.end();++optionIt)
    papszOptions=CSLAddString(papszOptions,optionIt->c_str());
  for(size_t ilayer=0;ilayer<other.getLayerCount();++ilayer){
    pushLayer(other.getLayerName(ilayer),other.getProjection(ilayer),other.getGeometryType(ilayer),papszOptions);
    destroyFeatures(ilayer);
    copyFields(other,std::vector<std::string>(),ilayer);
    m_features[ilayer].resize(other.getFeatureCount(ilayer));
#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
    for(size_t ifeature=0;ifeature<other.getFeatureCount(ilayer);++ifeature){
      OGRFeature *writeFeature=createFeature(ilayer);
      OGRFeature *otherFeature=other.getFeatureRef(ifeature,ilayer);
      if(otherFeature)
        writeFeature->SetFrom(otherFeature);
      // else
      //   std::cerr << "Warning: " << ifeature << " is NULL" << std::endl;
      m_features[ilayer][ifeature]=writeFeature;
    }
  }
  return(OGRERR_NONE);
}

///merge another vector
//todo: handle different definitions or layers
 //todo: handle multi-layers
// OGRErr VectorOgr::merge(VectorOgr& theVector){
//   //better not to parallellize here?
//   // #if JIPLIB_PROCESS_IN_PARALLEL == 1
//   // #pragma omp parallel for
//   // #else
//   // #endif
//   for(unsigned int index=0;index<theVector.getFeatureCount();++index){
//     OGRFeature *poFeature;
//     poFeature=theVector.cloneFeature(index);
//     pushFeature(poFeature);
//   }
// }

 ///write features to the vector dataset
OGRErr VectorOgr::write(const std::string& filename){
  if(filename.size()){
    vector<unsigned char> vbytes;
    size_t nbytes=serialize(vbytes);
    std::ofstream outputStream(filename, std::ios::out|std::ios::binary);
    outputStream.write(nbytes ? (char*)&vbytes[0] : 0, std::streamsize(nbytes));
    outputStream.close();
    // FILE *file = fopen(filename.c_str(),"wb");
    // if(file){
    //   fwrite(vbytes.data(),vbytes.size(),1,file);
    //   fclose(file);
    // }
  }
  else if(m_gds){
    for(size_t ilayer=0;ilayer<getLayerCount();++ilayer){
      auto fit=m_features[ilayer].begin();
      size_t ifeature=0;
      while(fit!=m_features[ilayer].end()){
        if(*fit){
          if(getLayer(ilayer)->CreateFeature(*fit)!=OGRERR_NONE){
            std::ostringstream errorStream;
            errorStream << "Warning: could not create feature " << ifeature << std::endl;
            std::cerr << errorStream.str();
            // throw(errorStream.str());
          }
          ++ifeature;
          ++fit;
        }
        else{
          // std::string errorString="Warning: NULL feature in m_feature";
          // std::cout << errorString << std::endl;
          m_features[ilayer].erase(fit);
          ++ifeature;
        }
      }
    }
  }
  return(OGRERR_NONE);
}

 ///copy all features from existing VectorOgr and write to vector dataset
//not tested yet
 // void VectorOgr::write(VectorOgr& vectorOgr, const std::string& filename, const std::vector<std::string>& layernames, const::std::string& imageType, char** papszOptions){
 //   m_layer.clear();
 //   setCodec(filename,imageType);
 //   if( !m_gds->TestCapability( ODsCCreateLayer ) ){
 //     std::string errorString="Error: Test capability to create layer failed";
 //     throw(errorString);
 //   }
 //   for(size_t ilayer=0; ilayer<vectorOgr.getLayerCount();++ilayer){
 //     //if no constraints on the types geometry to be written: use wkbUnknown
 //     OGRLayer* poLayer;
 //     m_layer.push_back(m_gds->CreateLayer(vectorogr.getLayer(ilayer)->GetName(), vectorOgr.getLayer(ilayer)->GetSpatialRef(), vectorOgr.getLayer(ilayer)->GetGeomType() ,papszOptions));
 //     copyFields(vectorOgr,ilayer);
 //     for(unsigned int index=0;index<vectorOgr.getFeatureCount(ilayer);++index){
 //       OGRFeature *poFeature;
 //       poFeature=OGRFeature::CreateFeature(getLayer(ilayer)->GetLayerDefn());
 //       poFeature=vectorOgr.cloneFeature(index,ilayer);
 //       if(getLayer(ilayer)->CreateFeature(poFeature)!=OGRERR_NONE){
 //         std::string errorString="Error: could not create feature";
 //         throw(errorString);
 //       }
 //       OGRFeature::DestroyFeature(poFeature);
 //     }
 //   }
 // }

///get projection
std::string VectorOgr::getProjection(size_t ilayer) const{
  char *wktString;
  OGRLayer* thisLayer=getLayer(ilayer);
  if(thisLayer){
    thisLayer->GetSpatialRef()->exportToWkt(&wktString);
  }
  else{
    std::string errorString("Error: could not getLayer");
    throw(errorString);
  }
  std::string projectionString(wktString);
  CPLFree(wktString);
  return(projectionString);
};

///get extent of the layer
bool VectorOgr::getExtent(double& ulx, double& uly, double& lrx, double& lry, OGRCoordinateTransformation *poCT) const{
  bool result=true;
  for(size_t ilayer=0;ilayer<getLayerCount();++ilayer){
    double layer_ulx=0;
    double layer_uly=0;
    double layer_lrx=0;
    double layer_lry=0;
    if(!getExtent(layer_ulx,layer_uly,layer_lrx,layer_lry, ilayer, poCT)){
      result=false;
    }
    if(!ilayer){
      ulx=layer_ulx;
      uly=layer_uly;
      lrx=layer_lrx;
      lry=layer_lry;
    }
    else{
      ulx=std::min(ulx,layer_ulx);
      uly=std::max(uly,layer_uly);
      lrx=std::max(lrx,layer_lrx);
      lry=std::min(lry,layer_lry);
    }
  }
  return result;
}

///get extent of the layer
bool VectorOgr::getExtent(double& ulx, double& uly, double& lrx, double& lry, size_t ilayer, OGRCoordinateTransformation *poCT) const{
  // try{
    OGREnvelope oExt;
    OGRLayer* thisLayer=getLayer(ilayer);
    if(thisLayer){
      int nGeomFieldCount = thisLayer->GetLayerDefn()->GetGeomFieldCount();
      if(nGeomFieldCount<1){
        std::ostringstream errorStream;
        errorStream << "Error: layer does not contain geometry" << std::endl;
        throw(errorStream.str());
      }
      else if(nGeomFieldCount>1){
        for(int iGeom = 0;iGeom < nGeomFieldCount; ++iGeom){
          OGRGeomFieldDefn* poGFldDefn = thisLayer->GetLayerDefn()->GetGeomFieldDefn(iGeom);
          if (thisLayer->GetExtent(iGeom, &oExt, TRUE) == OGRERR_NONE){
            ulx=oExt.MinX;
            uly=oExt.MaxY;
            lrx=oExt.MaxX;
            lry=oExt.MinY;
          }
          else{
            std::ostringstream errorStream;
            errorStream << "Error: could not get extent from layer with multiple geometry field count" << std::endl;
            throw(errorStream.str());
          }
        }
      }
      else{
        if(thisLayer->GetExtent(&oExt, true) == OGRERR_NONE){
          ulx=oExt.MinX;
          uly=oExt.MaxY;
          lrx=oExt.MaxX;
          lry=oExt.MinY;
        }
        else{
          std::ostringstream errorStream;
          errorStream << "Error: could not get extent from layer with 1 geometry field count" << std::endl;
          throw(errorStream.str());
        }
      }
    }
    if(poCT){
      std::vector<double> xvector(4);//ulx,urx,llx,lrx
      std::vector<double> yvector(4);//uly,ury,lly,lry
      xvector[0]=ulx;
      xvector[1]=lrx;
      xvector[2]=ulx;
      xvector[3]=lrx;
      yvector[0]=uly;
      yvector[1]=uly;
      yvector[2]=lry;
      yvector[3]=lry;
      if(!poCT->Transform(xvector.size(),&xvector[0],&yvector[0])){
        std::ostringstream errorStream;
        errorStream << "Error: cannot apply OGRCoordinateTransformation in VectorOgr::getExtent" << std::endl;
        throw(errorStream.str());
      }
      ulx=xvector[0];
      lrx=xvector[1];
      ulx=xvector[2];
      lrx=xvector[3];
      uly=yvector[0];
      uly=yvector[1];
      lry=yvector[2];
      lry=yvector[3];
    }
    return true;
}

// bool VectorOgr::getExtent(std::vector<double> &bbvector, size_t ilayer, OGRCoordinateTransformation *poCT) const{
//   bbvector.resize(4);
//   return getExtent(bbvector[0],bbvector[1],bbvector[2],bbvector[3],ilayer,poCT);
// }

///set feature to the object
 OGRErr VectorOgr::setFeature(unsigned int index, OGRFeature *poFeature, size_t ilayer){
   if(index>=0&&index<m_features[ilayer].size()){
     m_features[ilayer][index]=poFeature;
   }
   else
     return(OGRERR_FAILURE);
   return(OGRERR_NONE);
}

///push feature to the object
 OGRErr VectorOgr::pushFeature(OGRFeature *poFeature, size_t ilayer){
   if(ilayer<m_features.size())
     m_features[ilayer].push_back(poFeature);
   else
     return(OGRERR_FAILURE);
   return(OGRERR_NONE);
 }

///read all features from an OGR dataset, specifying layer, attribute filter and spatial filter optionally
unsigned int VectorOgr::readFeatures(){
  unsigned int nfeatures=0;
  destroyAll();
  m_layer.resize(getGDSLayerCount());
  m_features.resize(m_layer.size());
  for(size_t ilayer=0;ilayer<getLayerCount();++ilayer){
    OGRFeature *poFeature;
    //start reading features from the layer
    m_layer[ilayer]->ResetReading();
    while( (poFeature = m_layer[ilayer]->GetNextFeature()) != NULL ){
      m_features[ilayer].push_back(poFeature);
      ++nfeatures;
    }
  }
  return(nfeatures);
}

///read all features from an OGR dataset, specifying layer, attribute filter and spatial filter optionally
unsigned int VectorOgr::readFeatures(size_t ilayer){
  if(ilayer>=m_features.size()){
    std::cout << "Warning: resize m_features" << std::endl;
    m_features.resize(ilayer+1);
  }
  unsigned int nfeatures=0;
  OGRFeature *poFeature;
  //start reading features from the layer
  m_layer[ilayer]->ResetReading();
  while( (poFeature = m_layer[ilayer]->GetNextFeature()) != NULL ){
    m_features[ilayer].push_back(poFeature);
    ++nfeatures;
  }
  return(nfeatures);
}

///clone feature. The newly created feature is owned by the caller, and will have it's own reference to the OGRFeatureDefn.
 OGRFeature* VectorOgr::cloneFeature(unsigned int index, size_t ilayer){
  OGRFeature* poFeature=NULL;
  if(index>=0&&index<m_features[ilayer].size()){
    poFeature=(m_features[ilayer][index])->Clone();
  }
  return(poFeature);
}

///get feature reference (feature should not be deleted)
 OGRFeature* VectorOgr::getFeatureRef(unsigned int index, size_t ilayer){
  OGRFeature* poFeature=NULL;
  if(m_features.size()<=ilayer){
    std::ostringstream errorStream;
    errorStream << "Error: m_features not initialized for layer " << ilayer << std::endl;
    std::cerr << errorStream.str() << std::endl;
    throw(errorStream.str());
  }
  if(index>=0&&index<m_features[ilayer].size()){
    poFeature=m_features[ilayer][index];
  }
  return(poFeature);
}

 OGRErr VectorOgr::getFields(std::vector<OGRFieldDefn*>& fields, size_t ilayer) const{
  OGRFeatureDefn *poFeatureDefn = getLayer(ilayer)->GetLayerDefn();
  fields.clear();
  fields.resize(poFeatureDefn->GetFieldCount());
  for(int iField=0;iField<poFeatureDefn->GetFieldCount();++iField)
    fields[iField]=poFeatureDefn->GetFieldDefn(iField);
  return(OGRERR_NONE);
}

void VectorOgr::getFieldNames(std::vector<std::string>& fieldnames, size_t ilayer) const{
  OGRFeatureDefn *poFeatureDefn = getLayer(ilayer)->GetLayerDefn();
  fieldnames.clear();
  fieldnames.resize(poFeatureDefn->GetFieldCount());
  for(int iField=0;iField<poFeatureDefn->GetFieldCount();++iField)
    fieldnames[iField]=poFeatureDefn->GetFieldDefn(iField)->GetNameRef();
}

OGRErr VectorOgr::addPoint(double x, double y, const std::map<std::string,double>& pointAttributes, std::string fieldName, int theId, size_t ilayer){
  OGRFeature *poFeature=createFeature(ilayer);
  OGRPoint pt;
  if(pointAttributes.size()+1!=poFeature->GetFieldCount()){
    std::ostringstream ess;
    ess << "Failed to add feature: " << pointAttributes.size() << " != " << poFeature->GetFieldCount() << std::endl;
    throw(ess.str());
  }
  poFeature->SetField( fieldName.c_str(), theId);
  for(std::map<std::string,double>::const_iterator mit=pointAttributes.begin();mit!=pointAttributes.end();++mit){
    poFeature->SetField((mit->first).c_str(),mit->second);
  }
  pt.setX(x);
  pt.setY(y);
  poFeature->SetGeometry( &pt );
  return(pushFeature(poFeature,ilayer));
}

 OGRErr VectorOgr::addPoint(double x, double y, const std::map<std::string,double>& pointAttributes, size_t ilayer){
  OGRFeature *poFeature=createFeature(ilayer);
  OGRPoint pt;
  if(pointAttributes.size()!=poFeature->GetFieldCount()){
    std::ostringstream ess;
    ess << "Failed to add feature: " << pointAttributes.size() << " != " << poFeature->GetFieldCount() << std::endl;
    throw(ess.str());
  }
  for(std::map<std::string,double>::const_iterator mit=pointAttributes.begin();mit!=pointAttributes.end();++mit){
    poFeature->SetField((mit->first).c_str(),mit->second);
  }
  pt.setX(x);
  pt.setY(y);
  poFeature->SetGeometry( &pt );
  return(pushFeature(poFeature,ilayer));
}


/**
 * @param app application specific option arguments
 * @return output Vector
 **/
shared_ptr<VectorOgr> VectorOgr::join(VectorOgr &ogrReader, app::AppFactory& app){
  std::shared_ptr<VectorOgr> ogrWriter=VectorOgr::createVector();
  join(ogrReader, *ogrWriter, app);
  return(ogrWriter);
}

///serialize vector in vector of unsigned chars
size_t VectorOgr::serialize(vector<unsigned char> &vbytes){
  VSIStatBufL statbuf;
  if( VSIStatL(getFileName().c_str(), &statbuf) == 0 ){
    size_t filesize = static_cast<size_t>(statbuf.st_size);
    if( filesize > 0 ){
      vbytes.resize(filesize);
      VSILFILE *file = VSIFOpenL(getFileName().c_str(),"rb");
      if( file ){
        VSIFReadL(vbytes.data(),filesize,1,file);
        VSIFCloseL(file);
      }
    }
    return(filesize);
  }
  else{
    std::cerr << "Error: VSIStatL failed" << std::endl;
    return(0);
  }
}

void VectorOgr::dumpOgr(app::AppFactory& app){
  Optionjl<std::string> fname_opt("n", "name", "the field name to dump");
  Optionjl<string> output_opt("o", "output", "Output ascii file (Default is empty: dump to standard output)");
  Optionjl<short> verbose_opt("v", "verbose", "Verbose mode if > 0", 0,2);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=fname_opt.retrieveOption(app);
    output_opt.retrieveOption(app);
    verbose_opt.retrieveOption(app);

    if(!doProcess){
      cout << endl;
      std::ostringstream helpStream;
      helpStream << "short option -h shows basic options only, use long option --help to show all options" << std::endl;
      throw(helpStream.str());//help was invoked, stop processing
    }

    ofstream outputStream;
    if(output_opt.size())
      outputStream.open(output_opt[0].c_str());

    for(size_t ilayer=0;ilayer<getLayerCount();++ilayer){
      std::vector<OGRFieldDefn*> fields;
      std::vector<size_t> fieldindexes;
      getFields(fields,ilayer);
      for(unsigned int ifield=0;ifield<fields.size();++ifield){
        if(fname_opt.size()){
          if(std::find(fname_opt.begin(),fname_opt.end(),fields[ifield]->GetNameRef())!=fname_opt.end())
            fieldindexes.push_back(ifield);
        }
        else
          fieldindexes.push_back(ifield);
      }
      if(verbose_opt[0])
        std::cout << "ilayer is: " << ilayer << std::endl;
      for(size_t ifeature = 0; ifeature < getFeatureCount(ilayer); ++ifeature) {
        OGRFeature *thisFeature=getFeatureRef(ifeature,ilayer);
        if(!thisFeature){
          // std::cerr << "Warning: " << ifeature << " is NULL" << std::endl;
          continue;
        }
        for(std::vector<size_t>::const_iterator fit=fieldindexes.begin();fit!=fieldindexes.end();++fit){
          if(output_opt.empty())
            std::cout << thisFeature->GetFieldAsString(*fit) << " ";
          else
            outputStream << thisFeature->GetFieldAsString(*fit) << " ";
        }
        if(output_opt.empty())
          std::cout << std::endl;
        else
          outputStream << std::endl;
      }
    }
    if(!output_opt.empty())
      outputStream.close();
  }
  catch(std::string errorString){
    std::cerr << errorString << std::endl;
    throw;
  }
}

///joins two VectorOgr based on key value
OGRErr VectorOgr::join(VectorOgr &ogrReader, VectorOgr &ogrWriter, app::AppFactory& app){
  Optionjl<string> output_opt("o", "output", "Filename of joined vector dataset");
  Optionjl<string> ogrformat_opt("f", "oformat", "Output ogr format for joined vector dataset","SQLite");
  Optionjl<std::string> option_opt("co", "co", "Creation option for output file. Multiple options can be specified.");
  Optionjl<unsigned int> access_opt("access", "access", "Access (0: GDAL_OF_READ_ONLY, 1: GDAL_OF_UPDATE)",1);
  Optionjl<std::string> key_opt("key", "key", "Key(s) used to join", "fid");
  Optionjl<std::string> method_opt("method", "method", "Join method (INNER, OUTER_LEFT, OUTER_RIGHT, OUTER_FULL)", "INNER");
  Optionjl<short> verbose_opt("v", "verbose", "Verbose mode if > 0", 0,2);

  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=output_opt.retrieveOption(app);
    ogrformat_opt.retrieveOption(app);
    access_opt.retrieveOption(app);
    option_opt.retrieveOption(app);
    key_opt.retrieveOption(app);
    method_opt.retrieveOption(app);
    verbose_opt.retrieveOption(app);

    if(!doProcess||output_opt.empty()){
      cout << endl;
      std::ostringstream helpStream;
      helpStream << "short option -h shows basic options only, use long option --help to show all options" << std::endl;
      throw(helpStream.str());//help was invoked, stop processing
    }
    OGRErr result=OGRERR_NONE;
    char **papszOptions=NULL;
    for(std::vector<std::string>::const_iterator optionIt=option_opt.begin();optionIt!=option_opt.end();++optionIt)
      papszOptions=CSLAddString(papszOptions,optionIt->c_str());
    bool initWriter=true;

    // if(ogrWriter.open(output_opt[0],ogrformat_opt[0],access_opt[0])!=OGRERR_NONE)
    if(ogrWriter.open(output_opt[0],ogrformat_opt[0])!=OGRERR_NONE)
      initWriter=false;
    if(isEmpty() || ogrReader.isEmpty()){
      ostringstream errorStream;
      errorStream << "Error: features are empty";
      throw(errorStream.str());
    }
    if(verbose_opt[0]){
      std::cout << "join this vector containing " << getFeatureCount() << " features with vector containing " << ogrReader.getFeatureCount() << std::endl;

      if(initWriter)
        std::cout << "initWriter is true" << std::endl;
      else
        std::cout << "initWriter is false" << std::endl;
    }
    // if(output_opt.size()){
    //   if(verbose_opt[0])
    //     std::cout << "opening ogrWriter " << output_opt[0] << std::endl;
    //   if(ogrWriter.open(output_opt[0],ogrformat_opt[0])!=OGRERR_NONE){
    //     if(verbose_opt[0])
    //       std::cout << "Warning: could not open ogrWriter " << output_opt[0] << std::endl;
    //   }
    // }
    // if(getLayerCount()!=ogrReader.getLayerCount()){
    //   ostringstream errorStream;
    //   errorStream << "Warning: number of layers in this layer and ogrReader are not equal: " << getLayerCount() << " != " << ogrReader.getLayerCount() << std::endl;
    //   throw(errorStream.str());
    // }
    for(size_t ilayer=0;ilayer<getLayerCount();++ilayer){
      if(verbose_opt[0])
        std::cout << "ilayer is: " << ilayer << std::endl;
      // if(output_opt.size()){
      if(initWriter){
        if(verbose_opt[0])
          std::cout << "push layer " << getLayerName(ilayer) << std::endl;
        if(ogrWriter.pushLayer(getLayerName(ilayer),getProjection(ilayer),getGeometryType(ilayer),papszOptions)!=OGRERR_NONE){
          ostringstream fs;
          fs << "push layer to ogrWriter with polygons failed ";
          fs << "layer name: "<< getLayerName(ilayer) << std::endl;
          throw(fs.str());
        }
        ogrWriter.destroyFeatures(ilayer);
        // ogrWriter.pushLayer(getLayer()->GetName(),getProjection(),getGeometryType(),NULL);
        std::vector<std::string> thisfields;
        this->getFieldNames(thisfields,ilayer);
        if(verbose_opt[0]){
          cout << "copy thisfields: ";
          for(unsigned int iField=0;iField<thisfields.size();++iField)
            cout << " " << thisfields[iField];
          cout  << std::endl;
        }
        if(verbose_opt[0]){
          cout << "Fields in ogrWriter before copy are: ";
          std::vector<OGRFieldDefn*> fields;
          ogrWriter.getFields(fields,ilayer);
          for(unsigned int iField=0;iField<fields.size();++iField)
            cout << " " << fields[iField]->GetNameRef();
          cout  << std::endl;
        }
        if(verbose_opt[0])
          std::cout << "copyFields from this" << std::endl;

        ogrWriter.copyFields(*this,thisfields,ilayer);
        if(verbose_opt[0]){
          cout << "Fields in ogrWriter after copy from this are: ";
          std::vector<OGRFieldDefn*> fields;
          ogrWriter.getFields(fields,ilayer);
          for(unsigned int iField=0;iField<fields.size();++iField)
            cout << " " << fields[iField]->GetNameRef();
          cout  << std::endl;
        }
        std::vector<std::string> thatfields;
        ogrReader.getFieldNames(thatfields,ilayer);
        for(auto itr=thisfields.begin();itr!=thisfields.end();++itr){
          auto thatitr = std::find(thatfields.begin(), thatfields.end(), *itr);
          if (thatitr != thatfields.end()) thatfields.erase(thatitr);
        }
        if(verbose_opt[0]){
          std::cout << "copy new fields from that" << std::endl;
          for(unsigned int iField=0;iField<thatfields.size();++iField)
            std::cout << " " << thatfields[iField];
          std::cout << std::endl;
        }
        if(thatfields.size()){
          if(verbose_opt[0]){
            std::cout << "copy new fields from that" << std::endl;
            for(unsigned int iField=0;iField<thatfields.size();++iField)
              std::cout << " " << thatfields[iField];
            std::cout << std::endl;
          }
          ogrWriter.copyFields(ogrReader,thatfields,ilayer);
        }
      }
      if(verbose_opt[0]){
        cout << "Fields are: ";
        std::vector<OGRFieldDefn*> fields;
        ogrWriter.getFields(fields,ilayer);
        for(unsigned int iField=0;iField<fields.size();++iField)
          cout << " " << fields[iField]->GetNameRef();
        cout  << std::endl;
      }
      //make sure to use resize and setFeature instead of pushFeature when in processing in parallel!!!
      std::string theKey1=key_opt[0];
      std::string theKey2=(key_opt.size()>1)? key_opt[1] : key_opt[0];
      switch(string2method(method_opt[0])){
      case(OUTER_FULL):{
        if(verbose_opt[0])
          std::cout << "in OUTER_FULL method" << std::endl;
        std::unordered_multimap<std::string, size_t> hashmap;
        // hash
        for(size_t ifeature = 0; ifeature < ogrReader.getFeatureCount(ilayer); ++ifeature) {
          OGRFeature *thatFeature=ogrReader.getFeatureRef(ifeature,ilayer);
          if(!thatFeature){
            // std::cerr << "Warning: " << ifeature << " is NULL" << std::endl;
            continue;
          }
          int iField=thatFeature->GetFieldIndex(theKey2.c_str());
          if(iField<0){
            std::ostringstream errorStream;
            errorStream << "Error: iField not found for " << theKey2 << std::endl;
            errorStream << "fields are: " << std::endl;
            std::vector<OGRFieldDefn*> fields;
            ogrReader.getFields(fields,ilayer);
            for(unsigned int iField=0;iField<fields.size();++iField)
              errorStream << " " << fields[iField]->GetNameRef();
            errorStream << std::endl;
            throw(errorStream.str());
          }
          hashmap.insert(std::make_pair(thatFeature->GetFieldAsString(iField), ifeature));
          if(verbose_opt[0]>1)
            std::cout << "hash ifeature: " << ifeature << ": " << thatFeature->GetFieldAsString(iField) << "," << ifeature << std::endl;
        }
        // map
        if(verbose_opt[0]>1)
          std::cout << "map" << std::endl;
        std::vector<size_t> thatFeaturesAdded;
        for(size_t ifeature = 0; ifeature < getFeatureCount(ilayer); ++ifeature) {
          OGRFeature *thisFeature=getFeatureRef(ifeature,ilayer);
          if(!thisFeature){
            // std::cerr << "Warning: " << ifeature << " is NULL" << std::endl;
            continue;
          }
          OGRFeature *writeFeature=ogrWriter.createFeature(ilayer);
          if(verbose_opt[0]>1)
            std::cout << "set from this" << std::endl;
          writeFeature->SetFrom(thisFeature);
          if(verbose_opt[0]>1)
            std::cout << "map ifeature: " << ifeature << std::endl;
          int iField=thisFeature->GetFieldIndex(theKey1.c_str());
          if(iField<0){
            std::ostringstream errorStream;
            errorStream << "Error: iField not found for " << theKey1 << std::endl;
            errorStream << "fields are: " << std::endl;
            std::vector<OGRFieldDefn*> fields;
            getFields(fields,ilayer);
            for(unsigned int iField=0;iField<fields.size();++iField)
              errorStream << " " << fields[iField]->GetNameRef();
            errorStream << std::endl;
            throw(errorStream.str());
          }
          auto range = hashmap.equal_range(thisFeature->GetFieldAsString(iField));
          bool thatFeatureAdded=false;
          for(auto it = range.first; it != range.second; ++it) {
            if(verbose_opt[0]>1)
              std::cout << "set from this " << it->first << " " << it->second << std::endl;
            writeFeature->SetFrom(ogrReader.getFeatureRef(it->second,ilayer));
            thatFeaturesAdded.push_back(it->second);
          }
          if(verbose_opt[0]>1)
            std::cout << "pushFeature" << std::endl;
          ogrWriter.pushFeature(writeFeature,ilayer);
        }
        for(size_t ifeature = 0; ifeature < ogrReader.getFeatureCount(); ++ifeature) {
          if(find(thatFeaturesAdded.begin(),thatFeaturesAdded.end(),ifeature)!=thatFeaturesAdded.end()){
            if(verbose_opt[0]>1)
              std::cout << "skip feature " << ifeature << "already added" << std::endl;
            continue;
          }
          else if(verbose_opt[0]>1)
            std::cout << "adding feature " << ifeature << std::endl;

          OGRFeature *thatFeature=ogrReader.getFeatureRef(ifeature,ilayer);
          if(!thatFeature){
            // std::cerr << "Warning: " << ifeature << " is NULL" << std::endl;
            continue;
          }
          int iField=thatFeature->GetFieldIndex(theKey2.c_str());
          if(iField<0){
            std::ostringstream errorStream;
            errorStream << "Error: iField not found for " << theKey2 << std::endl;
            errorStream << "fields are: " << std::endl;
            std::vector<OGRFieldDefn*> fields;
            ogrReader.getFields(fields,ilayer);
            for(unsigned int iField=0;iField<fields.size();++iField)
              errorStream << " " << fields[iField]->GetNameRef();
            errorStream << std::endl;
            throw(errorStream.str());
          }
          // auto range = hashmap.equal_range(thatFeature->GetFieldAsString(iField));
          // bool skip=false;//check if we already added this feature
          // for(auto it = range.first; it != range.second; ++it) {
          //   if(verbose_opt[0]>1)
          //     std::cout << "feature " << ifeature << " with key (" << it->first << ", " << it->second << ") was already added" << std::endl;
          //   skip=true;//feature was already added
          //   break;
          // }
          // if(skip)
          //   continue;
          OGRFeature *writeFeature=ogrWriter.createFeature(ilayer);
          if(verbose_opt[0]>1)
            std::cout << "set from ogrReader" << std::endl;
          writeFeature->SetFrom(thatFeature);
          if(verbose_opt[0]>1)
            std::cout << "map ifeature: " << ifeature << std::endl;
          if(verbose_opt[0]>1)
            std::cout << "pushFeature" << std::endl;
          ogrWriter.pushFeature(writeFeature,ilayer);
        }
        break;
      }
      case(OUTER_LEFT):{
        if(verbose_opt[0])
          std::cout << "in OUTER_LEFT method" << std::endl;
        std::unordered_multimap<std::string, size_t> hashmap;
        // hash
        for(size_t ifeature = 0; ifeature < ogrReader.getFeatureCount(ilayer); ++ifeature) {
          if(verbose_opt[0]>1)
            std::cout << "hash ifeature: " << ifeature << std::endl;
          OGRFeature *thatFeature=ogrReader.getFeatureRef(ifeature,ilayer);
          if(!thatFeature){
            // std::cerr << "Warning: " << ifeature << " is NULL" << std::endl;
            continue;
          }
          int iField=thatFeature->GetFieldIndex(theKey2.c_str());
          if(iField<0){
            std::ostringstream errorStream;
            errorStream << "Error: iField not found for " << theKey2 << std::endl;
            errorStream << "fields are: " << std::endl;
            std::vector<OGRFieldDefn*> fields;
            ogrReader.getFields(fields,ilayer);
            for(unsigned int iField=0;iField<fields.size();++iField)
              errorStream << " " << fields[iField]->GetNameRef();
            errorStream << std::endl;
            throw(errorStream.str());
          }
          hashmap.insert(std::make_pair(thatFeature->GetFieldAsString(iField), ifeature));
        }
        // map
        if(verbose_opt[0]>1)
          std::cout << "map" << std::endl;
        for(size_t ifeature = 0; ifeature < getFeatureCount(ilayer); ++ifeature) {
          OGRFeature *writeFeature=ogrWriter.createFeature(ilayer);
          OGRFeature *thisFeature=getFeatureRef(ifeature,ilayer);
          if(!thisFeature){
            // std::cerr << "Warning: " << ifeature << " is NULL" << std::endl;
            continue;
          }
          if(verbose_opt[0]>1)
            std::cout << "set from this" << std::endl;
          writeFeature->SetFrom(thisFeature);
          if(verbose_opt[0]>1)
            std::cout << "map ifeature: " << ifeature << std::endl;
          int iField=thisFeature->GetFieldIndex(theKey1.c_str());
          if(iField<0){
            std::ostringstream errorStream;
            errorStream << "Error: iField not found for " << theKey1 << std::endl;
            errorStream << "fields are: " << std::endl;
            std::vector<OGRFieldDefn*> fields;
            getFields(fields,ilayer);
            for(unsigned int iField=0;iField<fields.size();++iField)
              errorStream << " " << fields[iField]->GetNameRef();
            errorStream << std::endl;
            throw(errorStream.str());
          }
          auto range = hashmap.equal_range(thisFeature->GetFieldAsString(iField));
          for(auto it = range.first; it != range.second; ++it) {
            if(verbose_opt[0]>1)
              std::cout << "set from this" << std::endl;
            writeFeature->SetFrom(ogrReader.getFeatureRef(it->second,ilayer));
          }
          if(verbose_opt[0]>1)
            std::cout << "pushFeature" << std::endl;
          ogrWriter.pushFeature(writeFeature,ilayer);
        }
        break;
      }
      case(OUTER_RIGHT):{
        if(verbose_opt[0])
          std::cout << "in OUTER_RIGHT method" << std::endl;
        std::unordered_multimap<std::string, size_t> hashmap;
        // hash
        for(size_t ifeature = 0; ifeature < getFeatureCount(ilayer); ++ifeature) {
          if(verbose_opt[0]>1)
            std::cout << "hash ifeature: " << ifeature << std::endl;
          OGRFeature *thisFeature=getFeatureRef(ifeature,ilayer);
          if(!thisFeature){
            // std::cerr << "Warning: " << ifeature << " is NULL" << std::endl;
            continue;
          }
          int iField=thisFeature->GetFieldIndex(theKey1.c_str());
          if(iField<0){
            std::ostringstream errorStream;
            errorStream << "Error: iField not found for " << theKey1 << std::endl;
            errorStream << "fields are: " << std::endl;
            std::vector<OGRFieldDefn*> fields;
            this->getFields(fields,ilayer);
            for(unsigned int iField=0;iField<fields.size();++iField)
              errorStream << " " << fields[iField]->GetNameRef();
            errorStream << std::endl;
            throw(errorStream.str());
          }
          hashmap.insert(std::make_pair(thisFeature->GetFieldAsString(iField), ifeature));
        }
        // map
        if(verbose_opt[0]>1)
          std::cout << "map" << std::endl;
        for(size_t ifeature = 0; ifeature < ogrReader.getFeatureCount(ilayer); ++ifeature) {
          OGRFeature *writeFeature=ogrWriter.createFeature(ilayer);
          OGRFeature *thatFeature=ogrReader.getFeatureRef(ifeature,ilayer);
          if(!thatFeature){
            // std::cerr << "Warning: " << ifeature << " is NULL" << std::endl;
            continue;
          }
          if(verbose_opt[0]>1)
            std::cout << "set from ogrReader" << std::endl;
          writeFeature->SetFrom(thatFeature);
          if(verbose_opt[0]>1)
            std::cout << "map ifeature: " << ifeature << std::endl;
          int iField=thatFeature->GetFieldIndex(theKey2.c_str());
          if(iField<0){
            std::ostringstream errorStream;
            errorStream << "Error: iField not found for " << theKey2 << std::endl;
            errorStream << "fields are: " << std::endl;
            std::vector<OGRFieldDefn*> fields;
            ogrReader.getFields(fields,ilayer);
            for(unsigned int iField=0;iField<fields.size();++iField)
              errorStream << " " << fields[iField]->GetNameRef();
            errorStream << std::endl;
            throw(errorStream.str());
          }
          auto range = hashmap.equal_range(thatFeature->GetFieldAsString(iField));
          for(auto it = range.first; it != range.second; ++it) {
            if(verbose_opt[0]>1)
              std::cout << "set from this" << std::endl;
            writeFeature->SetFrom(this->getFeatureRef(it->second,ilayer));
          }
          if(verbose_opt[0]>1)
            std::cout << "pushFeature" << std::endl;
          ogrWriter.pushFeature(writeFeature,ilayer);
        }
        break;
      }
      case(INNER):{
        if(verbose_opt[0])
          std::cout << "in INNER method" << std::endl;
        std::unordered_multimap<std::string, size_t> hashmap;
        // hash
        for(size_t ifeature = 0; ifeature < getFeatureCount(ilayer); ++ifeature) {
          if(verbose_opt[0]>1)
            std::cout << "hash ifeature: " << ifeature << std::endl;
          OGRFeature *thisFeature=getFeatureRef(ifeature,ilayer);
          if(!thisFeature){
            // std::cerr << "Warning: " << ifeature << " is NULL" << std::endl;
            continue;
          }
          int iField=thisFeature->GetFieldIndex(theKey1.c_str());
          if(verbose_opt[0]>1){
            std::cout << "theKey1: " << theKey1 << std::endl;
            std::cout << "theKey2: " << theKey2 << std::endl;
            std::cout << "key for this feature: " << thisFeature->GetFieldAsString(iField) << std::endl;
          }
          if(iField<0){
            std::ostringstream errorStream;
            errorStream << "Error: iField not found for " << theKey1 << std::endl;
            errorStream << "fields are: " << std::endl;
            std::vector<OGRFieldDefn*> fields;
            this->getFields(fields,ilayer);
            for(unsigned int iField=0;iField<fields.size();++iField)
              errorStream << " " << fields[iField]->GetNameRef();
            errorStream << std::endl;
            throw(errorStream.str());
          }
          else if(verbose_opt[0]>1){
            std::vector<OGRFieldDefn*> fields;
            this->getFields(fields,ilayer);
            std::cout  << "fields are: " << std::endl;
            for(unsigned int iField=0;iField<fields.size();++iField)
              std::cout << " " << fields[iField]->GetNameRef();
            std::cout << std::endl;
          }
          hashmap.insert(std::make_pair(thisFeature->GetFieldAsString(iField), ifeature));
        }
        // map
        if(verbose_opt[0]>1)
          std::cout << "map" << std::endl;
        for(size_t ifeature = 0; ifeature < ogrReader.getFeatureCount(ilayer); ++ifeature) {
          if(verbose_opt[0]>1)
            std::cout << "map ifeature: " << ifeature << std::endl;
          OGRFeature *thatFeature=ogrReader.getFeatureRef(ifeature,ilayer);
          if(!thatFeature){
            // std::cerr << "Warning: " << ifeature << " is NULL" << std::endl;
            continue;
          }
          int iField=thatFeature->GetFieldIndex(theKey2.c_str());
          if(iField<0){
            std::ostringstream errorStream;
            errorStream << "Error: iField not found for " << theKey2 << std::endl;
            throw(errorStream.str());
          }
          if(verbose_opt[0]>1)
            std::cout << "calculate range" << std::endl;
          auto range = hashmap.equal_range(thatFeature->GetFieldAsString(iField));
          for(auto it = range.first; it != range.second; ++it) {
            OGRLayer *writeLayer=ogrWriter.getLayer(ilayer);
            if(!writeLayer){
              std::ostringstream errorStream;
              errorStream << "Error: could not get layer " << ilayer << std::endl;
              throw(errorStream.str());
            }
            // OGRFeatureDefn *poFDefn = writeLayer->GetLayerDefn();
            // if(!poFDefn){
            //   std::ostringstream errorStream;
            //   errorStream << "Error: could not get layer definition" << std::endl;
            //   throw(errorStream.str());
            // }
            OGRFeature *writeFeature=ogrWriter.createFeature(ilayer);
            OGRFeature *thisFeature=getFeatureRef(it->second,ilayer);
            if(thisFeature){
              if(verbose_opt[0]>1)
                std::cout << "setFrom thisFeature" << std::endl;
              writeFeature->SetFrom(thisFeature);
            }
            if(thatFeature){
              if(verbose_opt[0]>1)
                std::cout << "setFrom thatFeature" << std::endl;
              writeFeature->SetFrom(thatFeature);
            }
            if(verbose_opt[0]>1)
              std::cout << "pushing feature" << std::endl;
            if(ogrWriter.pushFeature(writeFeature,ilayer)!=OGRERR_NONE){
              std::ostringstream errorStream;
              errorStream << "Error: could not pushFeature, OGRERR_FAILURE " << std::endl;
              throw(errorStream.str());
            }
            if(verbose_opt[0]>1)
              std::cout << "pushed feature" << std::endl;
          }
        }
        break;
      }
      default:
        std::ostringstream errorStream;
        errorStream << "Error: join method " << method_opt[0] << " not implemented " << std::endl;
        throw(errorStream.str());
        break;
      }
      return(OGRERR_NONE);
    }
  }
  catch(std::string errorString){
    std::cerr << errorString << std::endl;
    throw;
  }
}
///append two VectorOgr objects (append to first layer)
// void VectorOgr::append(VectorOgr &ogrReader){
//   size_t ilayer=0;
//   size_t currentSize=getFeatureCount(ilayer);
//   resize(currentSize+ogrReader.getFeatureCount(ilayer),ilayer);
// #if JIPLIB_PROCESS_IN_PARALLEL == 1
// #pragma omp parallel for
// #else
// #endif
//   for(size_t ifeature = 0; ifeature < ogrReader.getFeatureCount(ilayer); ++ifeature) {
//     try{
//       OGRFeature *thatFeature=ogrReader.getFeatureRef(ifeature,ilayer);
//       if(!thatFeature){
//         // std::cerr << "Warning: " << ifeature << " is NULL" << std::endl;
//         continue;
//       }
//       OGRFeature *writeFeature=createFeature(ilayer);
//       writeFeature->SetFrom(thatFeature);
//       setFeature(currentSize+ifeature,writeFeature,ilayer);
//     }
//     catch(std::string errorString){
//       std::cerr << errorString << std::endl;
//       continue;
//     }
//   }
// }

// OGRErr VectorOgr::sortByLabel(std::map<std::string,Vector2d<float> > &mapPixels, const std::string& label, std::vector<std::string>& bandNames){
//   //[classNr][pixelNr][bandNr]
//   try{
//     mapPixels.clear();
//     int nsample=0;
//     int nband=0;
//     std::vector<OGRFieldDefn*> fields;
//     getFields(fields);
//     int fieldLabel=-1;
//     int fieldId=-1;
//     for(int iField=0;iField<fields.size();++iField){
//       std::string fieldname=fields[iField]->GetNameRef();
//       if(fieldname==label){
//         fieldLabel=iField;
//       }
//       if(fieldname=="fid"){
//         fieldId=iField;
//       }
//       if(fieldLabel>0&&fieldId>0)
//         break;
//     }
//     if(fieldLabel<0||fieldLabel>=fields.size()){
//       std::string errorString="Error: label not found";
//       throw(errorString);
//     }

//     for(unsigned int index=0;index<getFeatureCount();++index){
//       std::vector<float> theFeature;
//       std::string theClass;
//       for(int iField=0;iField<fields.size();++iField){
//         if(iField==fieldLabel){
//           theClass=m_features[index]->GetFieldAsString(iField);
//         }
//         else if(iField!=fieldId){
//           double theValue=m_features[index]->GetFieldAsDouble(iField);
//           theFeature.push_back(theValue);
//         }
//       }
//       mapPixels[theClass].push_back(theFeature);
//     }
//     return(OGRERR_NONE);
//   }
//   catch(std::string errorString){
//     std::cerr << errorString << std::endl;
//     return(OGRERR_FAILURE);
//   }
// }

OGRErr VectorOgr::sortByLabel(std::map<std::string,Vector2d<float> > &mapPixels, const std::string& label, const std::vector<std::string>& bandNames){
  //[classNr][pixelNr][bandNr]
  ///Warning: bands got ordered as they have been stored in this vector and not according to order bandNames have been provided in argument
  try{
    mapPixels.clear();
    for(size_t ilayer=0;ilayer<getLayerCount();++ilayer){
      for(unsigned int index=0;index<getFeatureCount(ilayer);++index){
        std::vector<float> theFeature;
        std::string theClass;
        for(int iField=0;iField<m_features[ilayer][index]->GetFieldCount();++iField){
          std::string fieldname=m_features[ilayer][index]->GetFieldDefnRef(iField)->GetNameRef();
          if(fieldname==label){
            theClass=m_features[ilayer][index]->GetFieldAsString(iField);
          }
          else if(bandNames.size()){
            if(find(bandNames.begin(),bandNames.end(),fieldname)!=bandNames.end()){
              double theValue=m_features[ilayer][index]->GetFieldAsDouble(iField);
              theFeature.push_back(theValue);
            }
          }
          else if(fieldname!="fid"){
            double theValue=m_features[ilayer][index]->GetFieldAsDouble(iField);
            theFeature.push_back(theValue);
          }
        }
        mapPixels[theClass].push_back(theFeature);
      }
    }
    return(OGRERR_NONE);
  }
  catch(std::string errorString){
    std::cerr << errorString << std::endl;
    throw;
  }
}

// ///static function for coordinate transform based on proj4 parameters
//  bool VectorOgr::transform(OGRGeometry *pGeom, const std::string& outputProj4){
//    try{
//      OGRSpatialReference targetSRS;
//      if( targetSRS.importFromProj4(outputProj4.c_str()) != OGRERR_NONE ){
//      std::ostringstream errorStream;
//      errorStream << "Error: cannot import SRS from Proj4 string: " << outputProj4 << std::endl;
//      throw(errorStream.str());
//      }
//      return(VectorOgr::transform(pGeom,&targetSRS));
//    }
//    catch(std::string errorString){
//      std::cerr << errorString << std::endl;
//      return false;
//    }
//  }

//  ///static function for coordinate transform based on EPSG codes
//  bool VectorOgr::transform(OGRGeometry *pGeom, int outputEPSG){
//    try{
//      OGRSpatialReference targetSRS;
//      if( targetSRS.importFromEPSG(outputEPSG) != OGRERR_NONE ){
//        std::ostringstream errorStream;
//        errorStream << "Error: cannot import SRS from EPSG code: " << outputEPSG << std::endl;
//        throw(errorStream.str());
//      }
//      return(transform(pGeom,&targetSRS));
//    }
//    catch(std::string errorString){
//      std::cerr << errorString << std::endl;
//      return false;
//    }
//  }

///static function for coordinate transform based on OGRCoordinateTransformation
bool VectorOgr::transform(OGRGeometry *pGeom, OGRCoordinateTransformation *poCT){
  if(poCT)
    return pGeom->transform(poCT) == OGRERR_NONE;
  else
    return true;
}

///static function for coordinate transform based on OGRCoordinateTransformation
bool VectorOgr::transform(OGREnvelope *pEnv, OGRCoordinateTransformation *poCT){
  if(poCT){
    try{
      std::vector<double> xvector(2);//ulx,lrx
      std::vector<double> yvector(2);//uly,lry
      xvector[0]=pEnv->MinX;
      xvector[1]=pEnv->MaxX;
      yvector[0]=pEnv->MaxY;
      yvector[1]=pEnv->MinY;
      if(!poCT->Transform(xvector.size(),&xvector[0],&yvector[0])){
        std::ostringstream errorStream;
        errorStream << "Error: cannot apply OGRCoordinateTransformation in VectorOgr::transform" << std::endl;
        throw(errorStream.str());
      }
      pEnv->MinX=xvector[0];
      pEnv->MaxX=xvector[1];
      pEnv->MaxY=yvector[0];
      pEnv->MinY=yvector[1];
      return true;
    }
    catch(std::string errorString){
      std::cerr << errorString << std::endl;
      throw;
    }
  }
  else
    return true;
}
///static function for coordinate transform of a vector of points based on OGRSpatialReference
// bool VectorOgr::transform(std::vector<double> &xvector, std::vector<double> &yvector, OGRSpatialReference *sourceSRS, OGRSpatialReference *targetSRS){
//   try{
//     if(sourceSRS->IsSame(targetSRS))
//       return true;
//     OGRCoordinateTransformation *poCT = OGRCreateCoordinateTransformation(sourceSRS, targetSRS);
//     if( !poCT ){
//       std::ostringstream errorStream;
//       errorStream << "Error: cannot create OGRCoordinateTransformation" << std::endl;
//       throw(errorStream.str());
//     }
//     if(!poCT->Transform(xvector.size(),&xvector[0],&yvector[0])){
//       std::ostringstream errorStream;
//       errorStream << "Error: cannot apply OGRCoordinateTransformation" << std::endl;
//       throw(errorStream.str());
//     }
//   }
//   catch(std::string errorString){
//     std::cerr << errorString << std::endl;
//     return false;
//   }
// }

///static function to join
// OGRErr VectorOgr::join(VectorOgr &ogrReader1, VectorOgr &ogrReader2, VectorOgr &ogrWriter, app::AppFactory& app){
//   Optionjl<string> output_opt("o", "output", "Filename of joined vector dataset");
//   Optionjl<string> ogrformat_opt("f", "f", "Output ogr format for joined vector dataset","SQLite");
//   Optionjl<std::string> option_opt("co", "co", "Creation option for output file. Multiple options can be specified.");
//   Optionjl<std::string> key_opt("key", "key", "Key(s) used to join", "fid");
//   Optionjl<std::string> method_opt("method", "method", "Join method (INNER, OUTER_LEFT, OUTER_RIGHT, OUTER_FULL)", "INNER");
//   Optionjl<short> verbose_opt("v", "verbose", "Verbose mode if > 0", 0,2);

//   bool doProcess;//stop process when program was invoked with help option (-h --help)
//   try{
//     doProcess=output_opt.retrieveOption(app);
//     ogrformat_opt.retrieveOption(app);
//     option_opt.retrieveOption(app);
//     key_opt.retrieveOption(app);
//     method_opt.retrieveOption(app);
//     verbose_opt.retrieveOption(app);

//     if(!doProcess){
//       cout << endl;
//       std::ostringstream helpStream;
//       helpStream << "short option -h shows basic options only, use long option --help to show all options" << std::endl;
//       throw(helpStream.str());//help was invoked, stop processing
//     }
//     OGRErr result=OGRERR_NONE;
//     char **papszOptions=NULL;
//     for(std::vector<std::string>::const_iterator optionIt=option_opt.begin();optionIt!=option_opt.end();++optionIt)
//       papszOptions=CSLAddString(papszOptions,optionIt->c_str());
//     if(output_opt.size()){
//       if(verbose_opt[0])
//         std::cout << "opening ogrWriter " << output_opt[0] << std::endl;
//       if(ogrWriter.open(output_opt[0],ogrformat_opt[0])!=OGRERR_NONE){
//         if(verbose_opt[0])
//           std::cout << "Warning: could not open ogrWriter " << output_opt[0] << std::endl;
//       }
//     }
//     std::cout << "number of layers in ogrReader1: " << ogrReader1.getLayerCount() << std::endl;
//     std::cout << "number of layers in ogrReader2: " << ogrReader2.getLayerCount() << std::endl;
//     std::cout << "number of layers in ogrWriter: " << ogrWriter.getLayerCount() << std::endl;
//     if(ogrReader1.getLayerCount()!=ogrReader2.getLayerCount()){
//       ostringstream errorStream;
//       errorStream << "Warning: number of layers in ogrReader1 and ogrReader2 are not equal: " << ogrReader1.getLayerCount() << " != " << ogrReader2.getLayerCount() << std::endl;
//       throw(errorStream.str());
//     }
//     for(size_t ilayer=0;ilayer<ogrReader1.getLayerCount();++ilayer){
//       //test
//       std::cout << "ilayer: " << ilayer << std::endl;
//       //test
//       std::cout << "number of layers in ogrWriter: " << ogrWriter.getLayerCount() << std::endl;
//       if(output_opt.size()){
//         if(ogrWriter.getLayerCount()<=ilayer){
//           if(verbose_opt[0])
//             std::cout << "push layer" << ogrReader1.getLayerName(ilayer) << std::endl;
//           if(ogrWriter.pushLayer(ogrReader1.getLayerName(ilayer),ogrReader1.getProjection(ilayer),ogrReader1.getGeometryType(),papszOptions)!=OGRERR_NONE){
//             ostringstream fs;
//             fs << "push layer to ogrWriter with polygons failed ";
//             fs << "layer name: "<< ogrReader1.getLayerName(ilayer) << std::endl;
//             throw(fs.str());
//           }
//         }
//         // ogrWriter.pushLayer(getLayer()->GetName(),getProjection(),getGeometryType(),NULL);
//         std::vector<std::string> thisfields;
//         ogrReader1.getFieldNames(thisfields,ilayer);
//         if(verbose_opt[0])
//           std::cout << "copyFields from ogrReader1" << std::endl;

//         //test
//         // if(verbose_opt[0]){
//         //   std::vector<std::string> writefields;
//         //   std::cout << "debug0" << std::endl;
//         //   ogrWriter.getFieldNames(writefields,ilayer);
//         //   std::cout << "debug1" << std::endl;
//         //   for(auto fit=writefields.begin();fit!=writefields.end();++fit)
//         //     std::cout << *fit << std::endl;
//         // }
//         // std::cout << "debug2" << std::endl;

//         ogrWriter.copyFields(ogrReader1,thisfields,ilayer);
//         std::cout << "debug3" << std::endl;
//         std::vector<std::string> thatfields;
//         ogrReader2.getFieldNames(thatfields,ilayer);
//         for(auto itr=thisfields.begin();itr!=thisfields.end();++itr){
//           auto thatitr = std::find(thatfields.begin(), thatfields.end(), *itr);
//           if (thatitr != thatfields.end()) thatfields.erase(thatitr);
//         }
//         if(verbose_opt[0]){
//           std::cout << "copyFields from ogrReader2" << std::endl;
//           for(unsigned int iField=0;iField<thatfields.size();++iField)
//             std::cout << " " << thatfields[iField];
//           std::cout << std::endl;
//         }
//         ogrWriter.copyFields(ogrReader2,thatfields,ilayer);
//       }
//       //make sure to use resize and setFeature instead of pushFeature when in processing in parallel!!!
//       std::string theKey1=key_opt[0];
//       std::string theKey2=(key_opt.size()>1)? key_opt[1] : key_opt[0];
//       switch(string2method(method_opt[0])){
//       case(OUTER_FULL):{
//         if(verbose_opt[0])
//           std::cout << "in OUTER_FULL method" << std::endl;
//         std::unordered_multimap<std::string, size_t> hashmap;
//         // hash
//         for(size_t ifeature = 0; ifeature < ogrReader2.getFeatureCount(ilayer); ++ifeature) {
//           if(verbose_opt[0]>1)
//             std::cout << "hash ifeature: " << ifeature << std::endl;
//           OGRFeature *thatFeature=ogrReader2.getFeatureRef(ifeature,ilayer);
//           if(!thatFeature){
//             std::cerr << "Warning: " << ifeature << " is NULL" << std::endl;
//             continue;
//           }
//           int iField=thatFeature->GetFieldIndex(theKey2.c_str());
//           if(iField<0){
//             std::ostringstream errorStream;
//             errorStream << "Error: iField not found for " << theKey2 << std::endl;
//             errorStream << "fields are: " << std::endl;
//             std::vector<OGRFieldDefn*> fields;
//             ogrReader2.getFields(fields,ilayer);
//             for(unsigned int iField=0;iField<fields.size();++iField)
//               errorStream << " " << fields[iField]->GetNameRef();
//             errorStream << std::endl;
//             throw(errorStream.str());
//           }
//           hashmap.insert(std::make_pair(thatFeature->GetFieldAsString(iField), ifeature));
//         }
//         // map
//         if(verbose_opt[0]>1)
//           std::cout << "map" << std::endl;
//         for(size_t ifeature = 0; ifeature < ogrReader1.getFeatureCount(ilayer); ++ifeature) {
//           OGRFeature *thisFeature=ogrReader1.getFeatureRef(ifeature,ilayer);
//           if(!thisFeature){
//             std::cerr << "Warning: " << ifeature << " is NULL" << std::endl;
//             continue;
//           }
//           OGRFeature *writeFeature=ogrWriter.createFeature(ilayer);
//           if(verbose_opt[0]>1)
//             std::cout << "set from this" << std::endl;
//           writeFeature->SetFrom(thisFeature);
//           if(verbose_opt[0]>1)
//             std::cout << "map ifeature: " << ifeature << std::endl;
//           int iField=thisFeature->GetFieldIndex(theKey1.c_str());
//           if(iField<0){
//             std::ostringstream errorStream;
//             errorStream << "Error: iField not found for " << theKey1 << std::endl;
//             errorStream << "fields are: " << std::endl;
//             std::vector<OGRFieldDefn*> fields;
//             ogrReader1.getFields(fields,ilayer);
//             for(unsigned int iField=0;iField<fields.size();++iField)
//               errorStream << " " << fields[iField]->GetNameRef();
//             errorStream << std::endl;
//             throw(errorStream.str());
//           }
//           auto range = hashmap.equal_range(thisFeature->GetFieldAsString(iField));
//           for(auto it = range.first; it != range.second; ++it) {
//             if(verbose_opt[0]>1)
//               std::cout << "set from this" << std::endl;
//             writeFeature->SetFrom(ogrReader2.getFeatureRef(it->second,ilayer));
//           }
//           if(verbose_opt[0]>1)
//             std::cout << "pushFeature" << std::endl;
//           ogrWriter.pushFeature(writeFeature);
//         }
//         for(size_t ifeature = 0; ifeature < ogrReader2.getFeatureCount(ilayer); ++ifeature) {
//           OGRFeature *thatFeature=ogrReader2.getFeatureRef(ifeature,ilayer);
//           if(!thatFeature){
//             std::cerr << "Warning: " << ifeature << " is NULL" << std::endl;
//             continue;
//           }
//           int iField=thatFeature->GetFieldIndex(theKey2.c_str());
//           if(iField<0){
//             std::ostringstream errorStream;
//             errorStream << "Error: iField not found for " << theKey2 << std::endl;
//             errorStream << "fields are: " << std::endl;
//             std::vector<OGRFieldDefn*> fields;
//             ogrReader2.getFields(fields,ilayer);
//             for(unsigned int iField=0;iField<fields.size();++iField)
//               errorStream << " " << fields[iField]->GetNameRef();
//             errorStream << std::endl;
//             throw(errorStream.str());
//           }
//           auto range = hashmap.equal_range(thatFeature->GetFieldAsString(iField));
//           bool skip=false;//check if we already added this feature
//           for(auto it = range.first; it != range.second; ++it) {
//             skip=true;//feature was already added
//             break;
//           }
//           if(skip)
//             continue;
//           OGRFeature *writeFeature=ogrWriter.createFeature(ilayer);
//           if(verbose_opt[0]>1)
//             std::cout << "set from ogrReader2" << std::endl;
//           writeFeature->SetFrom(thatFeature);
//           if(verbose_opt[0]>1)
//             std::cout << "map ifeature: " << ifeature << std::endl;
//           if(verbose_opt[0]>1)
//             std::cout << "pushFeature" << std::endl;
//           ogrWriter.pushFeature(writeFeature);
//         }
//         break;
//       }
//       case(OUTER_LEFT):{
//         if(verbose_opt[0])
//           std::cout << "in OUTER_LEFT method" << std::endl;
//         std::unordered_multimap<std::string, size_t> hashmap;
//         // hash
//         for(size_t ifeature = 0; ifeature < ogrReader2.getFeatureCount(ilayer); ++ifeature) {
//           if(verbose_opt[0]>1)
//             std::cout << "hash ifeature: " << ifeature << std::endl;
//           OGRFeature *thatFeature=ogrReader2.getFeatureRef(ifeature,ilayer);
//           if(!thatFeature){
//             std::cerr << "Warning: " << ifeature << " is NULL" << std::endl;
//             continue;
//           }
//           int iField=thatFeature->GetFieldIndex(theKey2.c_str());
//           if(iField<0){
//             std::ostringstream errorStream;
//             errorStream << "Error: iField not found for " << theKey2 << std::endl;
//             errorStream << "fields are: " << std::endl;
//             std::vector<OGRFieldDefn*> fields;
//             ogrReader2.getFields(fields,ilayer);
//             for(unsigned int iField=0;iField<fields.size();++iField)
//               errorStream << " " << fields[iField]->GetNameRef();
//             errorStream << std::endl;
//             throw(errorStream.str());
//           }
//           hashmap.insert(std::make_pair(thatFeature->GetFieldAsString(iField), ifeature));
//         }
//         // map
//         if(verbose_opt[0]>1)
//           std::cout << "map" << std::endl;
//         for(size_t ifeature = 0; ifeature < ogrReader1.getFeatureCount(ilayer); ++ifeature) {
//           OGRFeature *writeFeature=ogrWriter.createFeature(ilayer);
//           OGRFeature *thisFeature=ogrReader1.getFeatureRef(ifeature,ilayer);
//           if(!thisFeature){
//             std::cerr << "Warning: " << ifeature << " is NULL" << std::endl;
//             continue;
//           }
//           if(verbose_opt[0]>1)
//             std::cout << "set from this" << std::endl;
//           writeFeature->SetFrom(thisFeature);
//           if(verbose_opt[0]>1)
//             std::cout << "map ifeature: " << ifeature << std::endl;
//           int iField=thisFeature->GetFieldIndex(theKey1.c_str());
//           if(iField<0){
//             std::ostringstream errorStream;
//             errorStream << "Error: iField not found for " << theKey1 << std::endl;
//             errorStream << "fields are: " << std::endl;
//             std::vector<OGRFieldDefn*> fields;
//             ogrReader1.getFields(fields,ilayer);
//             for(unsigned int iField=0;iField<fields.size();++iField)
//               errorStream << " " << fields[iField]->GetNameRef();
//             errorStream << std::endl;
//             throw(errorStream.str());
//           }
//           auto range = hashmap.equal_range(thisFeature->GetFieldAsString(iField));
//           for(auto it = range.first; it != range.second; ++it) {
//             if(verbose_opt[0]>1)
//               std::cout << "set from this" << std::endl;
//             writeFeature->SetFrom(ogrReader2.getFeatureRef(it->second,ilayer));
//           }
//           if(verbose_opt[0]>1)
//             std::cout << "pushFeature" << std::endl;
//           ogrWriter.pushFeature(writeFeature);
//         }
//         break;
//       }
//       case(OUTER_RIGHT):{
//         if(verbose_opt[0])
//           std::cout << "in OUTER_RIGHT method" << std::endl;
//         std::unordered_multimap<std::string, size_t> hashmap;
//         // hash
//         for(size_t ifeature = 0; ifeature < ogrReader1.getFeatureCount(ilayer); ++ifeature) {
//           if(verbose_opt[0]>1)
//             std::cout << "hash ifeature: " << ifeature << std::endl;
//           OGRFeature *thisFeature=ogrReader1.getFeatureRef(ifeature,ilayer);
//           if(!thisFeature){
//             std::cerr << "Warning: " << ifeature << " is NULL" << std::endl;
//             continue;
//           }
//           int iField=thisFeature->GetFieldIndex(theKey1.c_str());
//           if(iField<0){
//             std::ostringstream errorStream;
//             errorStream << "Error: iField not found for " << theKey1 << std::endl;
//             errorStream << "fields are: " << std::endl;
//             std::vector<OGRFieldDefn*> fields;
//             ogrReader1.getFields(fields,ilayer);
//             for(unsigned int iField=0;iField<fields.size();++iField)
//               errorStream << " " << fields[iField]->GetNameRef();
//             errorStream << std::endl;
//             throw(errorStream.str());
//           }
//           hashmap.insert(std::make_pair(thisFeature->GetFieldAsString(iField), ifeature));
//         }
//         // map
//         if(verbose_opt[0]>1)
//           std::cout << "map" << std::endl;
//         for(size_t ifeature = 0; ifeature < ogrReader2.getFeatureCount(ilayer); ++ifeature) {
//           OGRFeature *writeFeature=ogrWriter.createFeature(ilayer);
//           OGRFeature *thatFeature=ogrReader2.getFeatureRef(ifeature,ilayer);
//           if(!thatFeature){
//             std::cerr << "Warning: " << ifeature << " is NULL" << std::endl;
//             continue;
//           }
//           if(verbose_opt[0]>1)
//             std::cout << "set from ogrReader2" << std::endl;
//           writeFeature->SetFrom(thatFeature);
//           if(verbose_opt[0]>1)
//             std::cout << "map ifeature: " << ifeature << std::endl;
//           int iField=thatFeature->GetFieldIndex(theKey2.c_str());
//           if(iField<0){
//             std::ostringstream errorStream;
//             errorStream << "Error: iField not found for " << theKey2 << std::endl;
//             errorStream << "fields are: " << std::endl;
//             std::vector<OGRFieldDefn*> fields;
//             ogrReader2.getFields(fields,ilayer);
//             for(unsigned int iField=0;iField<fields.size();++iField)
//               errorStream << " " << fields[iField]->GetNameRef();
//             errorStream << std::endl;
//             throw(errorStream.str());
//           }
//           auto range = hashmap.equal_range(thatFeature->GetFieldAsString(iField));
//           for(auto it = range.first; it != range.second; ++it) {
//             if(verbose_opt[0]>1)
//               std::cout << "set from this" << std::endl;
//             writeFeature->SetFrom(ogrReader1.getFeatureRef(it->second,ilayer));
//           }
//           if(verbose_opt[0]>1)
//             std::cout << "pushFeature" << std::endl;
//           ogrWriter.pushFeature(writeFeature);
//         }
//         break;
//       }
//       case(INNER):{
//         if(verbose_opt[0])
//           std::cout << "in INNER method" << std::endl;
//         std::unordered_multimap<std::string, size_t> hashmap;
//         // hash
//         for(size_t ifeature = 0; ifeature < ogrReader1.getFeatureCount(ilayer); ++ifeature) {
//           if(verbose_opt[0]>1)
//             std::cout << "hash ifeature: " << ifeature << std::endl;
//           OGRFeature *thisFeature=ogrReader1.getFeatureRef(ifeature,ilayer);
//           if(!thisFeature){
//             std::cerr << "Warning: " << ifeature << " is NULL" << std::endl;
//             continue;
//           }
//           int iField=thisFeature->GetFieldIndex(theKey1.c_str());
//           if(iField<0){
//             std::ostringstream errorStream;
//             errorStream << "Error: iField not found for " << theKey1 << std::endl;
//             errorStream << "fields are: " << std::endl;
//             std::vector<OGRFieldDefn*> fields;
//             ogrReader1.getFields(fields,ilayer);
//             for(unsigned int iField=0;iField<fields.size();++iField)
//               errorStream << " " << fields[iField]->GetNameRef();
//             errorStream << std::endl;
//             throw(errorStream.str());
//           }
//           hashmap.insert(std::make_pair(thisFeature->GetFieldAsString(iField), ifeature));
//         }
//         // map
//         if(verbose_opt[0]>1)
//           std::cout << "map" << std::endl;
//         for(size_t ifeature = 0; ifeature < ogrReader2.getFeatureCount(ilayer); ++ifeature) {
//           if(verbose_opt[0]>1)
//             std::cout << "map ifeature: " << ifeature << std::endl;
//           OGRFeature *thatFeature=ogrReader2.getFeatureRef(ifeature,ilayer);
//           //test
//           std::cout << "that feature: " << thatFeature << std::endl;
//           if(!thatFeature){
//             std::cerr << "Warning: " << ifeature << " is NULL" << std::endl;
//             continue;
//           }
//           int iField=thatFeature->GetFieldIndex(theKey2.c_str());
//           //test
//           std::cout << "iField: " << iField << std::endl;
//           if(iField<0){
//             std::ostringstream errorStream;
//             errorStream << "Error: iField not found for " << theKey2 << std::endl;
//             errorStream << "fields are: " << std::endl;
//             std::vector<OGRFieldDefn*> fields;
//             ogrReader2.getFields(fields,ilayer);
//             for(unsigned int iField=0;iField<fields.size();++iField)
//               errorStream << " " << fields[iField]->GetNameRef();
//             errorStream << std::endl;
//             throw(errorStream.str());
//           }
//           //test
//           std::cout << "getting range" << std::endl;
//           auto range = hashmap.equal_range(thatFeature->GetFieldAsString(iField));
//           for(auto it = range.first; it != range.second; ++it) {
//             //test
//             std::cout << "getLayer" << ilayer << std::endl;
//             OGRLayer *writeLayer=ogrWriter.getLayer(ilayer);
//             if(!writeLayer){
//               std::ostringstream errorStream;
//               errorStream << "Error: could not get layer " << ilayer << std::endl;
//               throw(errorStream.str());
//             }
//             //test
//             std::cout << "getLayerDefn" << std::endl;
//             OGRFeatureDefn *poFDefn = writeLayer->GetLayerDefn();
//             if(!poFDefn){
//               std::ostringstream errorStream;
//               errorStream << "Error: could not get layer definition" << std::endl;
//               throw(errorStream.str());
//             }
//             OGRFeature *writeFeature=ogrWriter.createFeature(ilayer);
//             if(verbose_opt[0]>1)
//               std::cout << "set from this" << std::endl;
//             writeFeature->SetFrom(ogrReader1.getFeatureRef(it->second,ilayer));
//             if(verbose_opt[0]>1)
//               std::cout << "set from ogrReader2" << std::endl;
//             writeFeature->SetFrom(thatFeature);
//             if(verbose_opt[0]>1)
//               std::cout << "pushFeature" << std::endl;
//             ogrWriter.pushFeature(writeFeature);
//           }
//         }
//         break;
//       }
//       default:
//         std::ostringstream errorStream;
//         errorStream << "Error: join method " << method_opt[0] << " not implemented " << std::endl;
//         throw(errorStream.str());
//         break;
//       }
//       return(OGRERR_NONE);
//     }
//   }
//   catch(std::string errorString){
//     std::cerr << errorString << std::endl;
//     return(OGRERR_FAILURE);
//   }
// }

std::ostream& operator<<(std::ostream& theOstream, VectorOgr& theVector){
  for(size_t ilayer=0;ilayer<theVector.getLayerCount();++ilayer){
    theOstream << "#" << theVector.getLayer(ilayer)->GetName() << std::endl;
    OGRFeatureDefn *poFDefn = theVector.getLayer(ilayer)->GetLayerDefn();
    theOstream << "#";
    for(int iField=0;iField<poFDefn->GetFieldCount();++iField){
      OGRFieldDefn *poFieldDefn = poFDefn->GetFieldDefn(iField);
      std::string fieldname=poFieldDefn->GetNameRef();
      theOstream << fieldname << " ";
    }
    theOstream << std::endl;

#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
    for(size_t index=0;index<theVector.getFeatureCount(ilayer);++index){
      OGRFeature *poFeature;
      poFeature=theVector.getFeatureRef(index,ilayer);
      for(int iField=0;iField<poFDefn->GetFieldCount();++iField)
        theOstream << poFeature->GetFieldAsString(iField) << " ";
      theOstream << std::endl;
    }
    theOstream << std::endl;
  }
  return(theOstream);
}
