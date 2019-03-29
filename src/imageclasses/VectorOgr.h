/**********************************************************************
VectorOgr.h: class to hold OGR features, typically read with readNextFeature
Author(s): Pieter.Kempeneers@ec.europa.eu
Copyright (c) 2016-2018 European Union (Joint Research Centre)
License EUPLv1.2

This file is part of jiplib
***********************************************************************/
#ifndef _VECTOROGR_H_
#define _VECTOROGR_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include "algorithms/ConfusionMatrix.h"
#include "algorithms/StatFactory.h"
#include "apps/AppFactory.h"
#include "ogr_feature.h"
#include "ogrsf_frmts.h"
#include "imageclasses/Jim.h"
#include "cpl_string.h"

/* enum OGR_DATA_ACCESS { READ_ONLY = 0, UPDATE = 1, WRITE = 3}; */
enum JOIN_METHOD { INNER = 0, OUTER_LEFT = 1, OUTER_RIGHT = 2, OUTER_FULL = 3};

class Jim;
class VectorOgr : public std::enable_shared_from_this<VectorOgr>
{
 public:
  ///Default constructor
  VectorOgr(void);
  ///constructor from filename (for reading)
  VectorOgr(const std::string& filename, const std::vector<std::string>& layernames=std::vector<std::string>(), bool noread=false){
    open(filename,layernames,noread);
  }
  ///constructor from filename (for reading or writing)
  VectorOgr(app::AppFactory &app) : m_gds(NULL), m_access(GDAL_OF_READONLY){m_filename.clear();open(app);}
  ///Copy constructor
  VectorOgr(VectorOgr& other, app::AppFactory &app);
  ///Copy constructor
  /* VectorOgr(VectorOgr& other, const std::string& filename, const std::string& imageType, char** options=NULL, bool copyData=true); */
  ///Destructor
  ~VectorOgr(void);
  ///perform a deep copy, including layers and features (if copyData is true)
  OGRErr copy(VectorOgr& other, app::AppFactory &app);
  ///assignment operator
  /* VectorOgr& operator=(VectorOgr& other); */
  ///Create new shared pointer to VectorOgr object
  /**
   *
   * @return shared pointer to new VectorOgr object
   */
  static std::shared_ptr<VectorOgr> createVector() {
    return(std::make_shared<VectorOgr>());
  };
  static std::shared_ptr<VectorOgr> createVector(const std::string& filename, const std::vector<std::string>& layernames=std::vector<std::string>()) {
    std::shared_ptr<VectorOgr> pVector=std::make_shared<VectorOgr>(filename,layernames);
    return(pVector);
  };
  static std::shared_ptr<VectorOgr> createVector(app::AppFactory &app){
    std::shared_ptr<VectorOgr> pVector=std::make_shared<VectorOgr>(app);
    return(pVector);
  };
  static std::shared_ptr<VectorOgr> createVector(VectorOgr& other,app::AppFactory &app){
  std::shared_ptr<VectorOgr> pVector=std::make_shared<VectorOgr>(other,app);
  return(pVector);
  };
  static OGRwkbGeometryType string2geotype(const std::string &typeString){
  //initialize selMap
  std::map<std::string,OGRwkbGeometryType> typeMap;
  typeMap["wkbUnknown"] = wkbUnknown;
  typeMap["wkbPoint"] = wkbPoint;
  typeMap["wkbLineString"] = wkbLineString;
  typeMap["wkbPolygon"] = wkbPolygon;
  typeMap["wkbMultiPoint"] = wkbMultiPoint;
  typeMap["wkbMultiLineString"] = wkbMultiLineString;
  typeMap["wkbMultiPolygon"] = wkbMultiPolygon;
  typeMap["wkbGeometryCollection"] = wkbGeometryCollection;
  typeMap["wkbCircularString"] = wkbCircularString;
  typeMap["wkbCompoundCurve"] = wkbCompoundCurve;
  typeMap["wkbCurvePolygon"] = wkbCurvePolygon;
  typeMap["wkbMultiCurve"] = wkbMultiCurve;
  typeMap["wkbMultiSurface"] = wkbMultiSurface;
  typeMap["wkbCurve"] = wkbCurve;
  typeMap["wkbSurface"] = wkbSurface;
  typeMap["wkbPolyhedralSurface"] = wkbPolyhedralSurface;
  typeMap["wkbTIN"] = wkbTIN;
  typeMap["wkbTriangle"] = wkbTriangle;
  typeMap["wkbNone"] = wkbNone;
  typeMap["wkbLinearRing"] = wkbLinearRing;
  typeMap["wkbCircularStringZ"] = wkbCircularStringZ;
  typeMap["wkbCompoundCurveZ"] = wkbCompoundCurveZ;
  typeMap["wkbCurvePolygonZ"] = wkbCurvePolygonZ;
  typeMap["wkbMultiCurveZ"] = wkbMultiCurveZ;
  typeMap["wkbMultiSurfaceZ"] = wkbMultiSurfaceZ;
  typeMap["wkbCurveZ"] = wkbCurveZ;
  typeMap["wkbSurfaceZ"] = wkbSurfaceZ;
  typeMap["wkbPolyhedralSurfaceZ"] = wkbPolyhedralSurfaceZ;
  typeMap["wkbTINZ"] = wkbTINZ;
  typeMap["wkbTriangleZ"] = wkbTriangleZ;
  typeMap["wkbPointM"] = wkbPointM;
  typeMap["wkbLineStringM"] = wkbLineStringM;
  typeMap["wkbPolygonM"] = wkbPolygonM;
  typeMap["wkbMultiPointM"] = wkbMultiPointM;
  typeMap["wkbMultiLineStringM"] = wkbMultiLineStringM;
  typeMap["wkbMultiPolygonM"] = wkbMultiPolygonM;
  typeMap["wkbGeometryCollectionM"] = wkbGeometryCollectionM;
  typeMap["wkbCircularStringM"] = wkbCircularStringM;
  typeMap["wkbCompoundCurveM"] = wkbCompoundCurveM;
  typeMap["wkbCurvePolygonM"] = wkbCurvePolygonM;
  typeMap["wkbMultiCurveM"] = wkbMultiCurveM;
  typeMap["wkbMultiSurfaceM"] = wkbMultiSurfaceM;
  typeMap["wkbCurveM"] = wkbCurveM;
  typeMap["wkbSurfaceM"] = wkbSurfaceM;
  typeMap["wkbPolyhedralSurfaceM"] = wkbPolyhedralSurfaceM;
  typeMap["wkbTINM"] = wkbTINM;
  typeMap["wkbTriangleM"] = wkbTriangleM;
  typeMap["wkbPointZM"] = wkbPointZM;
  typeMap["wkbLineStringZM"] = wkbLineStringZM;
  typeMap["wkbPolygonZM"] = wkbPolygonZM;
  typeMap["wkbMultiPointZM"] = wkbMultiPointZM;
  typeMap["wkbMultiLineStringZM"] = wkbMultiLineStringZM;
  typeMap["wkbMultiPolygonZM"] = wkbMultiPolygonZM;
  typeMap["wkbGeometryCollectionZM"] = wkbGeometryCollectionZM;
  typeMap["wkbCircularStringZM"] = wkbCircularStringZM;
  typeMap["wkbCompoundCurveZM"] = wkbCompoundCurveZM;
  typeMap["wkbCurvePolygonZM"] = wkbCurvePolygonZM;
  typeMap["wkbMultiCurveZM"] = wkbMultiCurveZM;
  typeMap["wkbMultiSurfaceZM"] = wkbMultiSurfaceZM;
  typeMap["wkbCurveZM"] = wkbCurveZM;
  typeMap["wkbSurfaceZM"] = wkbSurfaceZM;
  typeMap["wkbPolyhedralSurfaceZM"] = wkbPolyhedralSurfaceZM;
  typeMap["wkbTINZM"] = wkbTINZM;
  typeMap["wkbTriangleZM"] = wkbTriangleZM;
  typeMap["wkbPoint25D"] = wkbPoint25D;
  typeMap["wkbLineString25D"] = wkbLineString25D;
  typeMap["wkbPolygon25D"] = wkbPolygon25D;
  typeMap["wkbMultiPoint25D"] = wkbMultiPoint25D;
  typeMap["wkbMultiLineString25D"] = wkbMultiLineString25D;
  typeMap["wkbMultiPolygon25D"] = wkbMultiPolygon25D;
  typeMap["wkbGeometryCollection25D"] = wkbGeometryCollection25D;
  if(typeMap.count(typeString))
    return(typeMap[typeString]);
  else
    return(wkbUnknown);
  }

  static JOIN_METHOD string2method(const std::string &method){
  //initialize selMap
  std::map<std::string,JOIN_METHOD> methodMap;
  methodMap["INNER"] = INNER;
  methodMap["inner"] = INNER;
  methodMap["OUTER_LEFT"] = OUTER_LEFT;
  methodMap["outer_left"] = OUTER_LEFT;
  methodMap["OUTER_RIGHT"] = OUTER_RIGHT;
  methodMap["outer_right"] = OUTER_RIGHT;
  methodMap["OUTER_FULL"] = OUTER_FULL;
  methodMap["outer_full"] = OUTER_FULL;
  if(methodMap.count(method))
    return(methodMap[method]);
  else
    return(INNER);
  }
  ///destroy all features in object
  void destroyAll();
  void destroyEmptyFeatures(size_t ilayer=0);
  bool isEmpty(size_t ilayer=0);
  void destroyFeatures(size_t ilayer=0);
  ///open using a copy
  //OGRErr open(const VectorOgr& vectorOgr, const std::string& layername=std::string());
  ///open a GDAL vector dataset for writing with layers to be pushed later
  OGRErr open(const std::string& filename, const std::string& imageType, unsigned int access=GDAL_OF_UPDATE);
  ///open a GDAL vector dataset for reading
  OGRErr open(const std::string& filename, const std::vector<std::string>& layernames=std::vector<std::string>(), bool noread=false);
  //open a GDAL vector dataset for writing
  OGRErr open(const std::string& filename, const std::vector<std::string>& layernames, const std::string& imageType, const OGRwkbGeometryType& geometryType, OGRSpatialReference* theSRS=NULL, char** options=NULL);
  //open a GDAL vector dataset for writing
  OGRErr open(const std::string& filename, const std::vector<std::string>& layernames, const std::string& imageType, const OGRwkbGeometryType& geometryType, const std::string& theProjection, char** options=NULL);
  //open a GDAL vector dataset for writing
  OGRErr open(const std::string& filename, const std::vector<std::string>& layernames, const std::string& imageType, const std::string& geometryType=std::string(), const std::string& theProjection=std::string(), char** options=NULL);
  ///open a GDAL vector dataset using AppFactory argument, used for reading and writing
  OGRErr open(app::AppFactory& app);
  ///register driver and create GDAL dataset
  void setCodec(const std::string& filename, const std::string& imageType);
  ///close a GDAL vector dataset
  void close(void);
  ///get projection
  std::string getProjection(size_t ilayer=0) const;
  ///set projection
  /* OGRErr setProjection(const std::string& theProjection); */
  ///Get the filename of this dataset
  std::string getFileName() const {return m_filename;};
  ///Create a layer
  OGRErr pushLayer(const std::string& layername, OGRSpatialReference* theSRS, const OGRwkbGeometryType& geometryType=wkbUnknown, char** papszOptions=NULL);
  ///Create a layer
  OGRErr pushLayer(const std::string& layername, const std::string& theProjection=std::string(), const OGRwkbGeometryType& geometryType=wkbUnknown, char** papszOptions=NULL);
  ///Create a layer
  OGRErr pushLayer(const std::string& layername, const std::string& theProjection, const std::string& geometryType, char** papszOptions=NULL);
  ///clear all features, releasing memory from heap
  //OGRErr reset();
  std::shared_ptr<VectorOgr> intersect(OGRPolygon *pGeom, app::AppFactory& app);
  std::shared_ptr<VectorOgr> intersect(const Jim& aJim, app::AppFactory& app);
  OGRErr intersect(OGRPolygon *pGeom, VectorOgr& ogrWriter, app::AppFactory& app);
  OGRErr intersect(const Jim& aJim, VectorOgr& ogrWriter, app::AppFactory& app);
  std::shared_ptr<VectorOgr> convexHull(app::AppFactory& app);
  OGRErr convexHull(VectorOgr& ogrWriter, app::AppFactory& app);
  ///get access mode
  unsigned int getAccess(){return m_access;};
  ///set access mode
  OGRErr setAccess(unsigned int theAccess){m_access=theAccess;};
  ///set access mode using a string argument
  OGRErr setAccess(std::string accessString){
    if(accessString=="GDAL_OF_READONLY"){
      m_access=GDAL_OF_READONLY;
    }
    if(accessString=="GDAL_OF_UPDATE"){
      m_access=GDAL_OF_UPDATE;
    }
  }
  ///get number of layers
  size_t getGDSLayerCount() const {
    if(m_gds)
      return(m_gds->GetLayerCount());
    else
      return(0);
  };
  size_t getLayerCount() const {
    return(m_layer.size());
    /* if(m_gds) */
    /*   return(m_gds->GetLayerCount()); */
    /* else */
    /*   return(0); */
  };
  std::string getLayerName(size_t ilayer=0) const {if(getLayer(ilayer)) return(getLayer(ilayer)->GetName());else return(std::string());};
#if GDAL_VERSION_MAJOR < 2
  //Get a pointer to the GDAL dataset
  OGRDataSource* getDataset(){return m_gds;};
#else
  //Get a pointer to the GDAL dataset
  GDALDataset* getDataset(){return m_gds;};
#endif
  ///get layer
  OGRLayer* getLayer(size_t ilayer=0) {if(m_layer.size()>ilayer) return(m_layer[ilayer]);else return(0);};
  ///get layer const version
  OGRLayer* getLayer(size_t ilayer=0) const {if(m_layer.size()>ilayer) return(m_layer[ilayer]);else return(0);};
  ///get layer by name
  OGRLayer* getLayer(std::string layername) {
    for(size_t ilayer=0;ilayer<getGDSLayerCount();++ilayer){
      std::string currentLayername=getLayer(ilayer)->GetName();
      if(currentLayername==layername)
        return(m_layer[ilayer]);
      else
        continue;
    }
    return(0);
  }
  ///get layer by name const version
  OGRLayer* getLayer(std::string layername) const {
    for(size_t ilayer=0;ilayer<getGDSLayerCount();++ilayer){
      std::string currentLayername=getLayer(ilayer)->GetName();
      if(currentLayername==layername)
        return(m_layer[ilayer]);
      else
        continue;
    }
    return(0);
  }
  ///get layers
  std::vector<OGRLayer*> getLayers(const std::vector<std::string>& layernames=std::vector<std::string>()){
    if(layernames.size()){
      std::vector<OGRLayer*> layers;
      for(size_t ilayer=0;ilayer<getGDSLayerCount();++ilayer){
        std::string currentLayername=getLayer(ilayer)->GetName();
        std::vector<std::string>::const_iterator it=find(layernames.begin(),layernames.end(),currentLayername);
        if(it==layernames.end())
          continue;
        else
          layers.push_back(getLayer(ilayer));
      }
      return(layers);
    }
    else
      return(m_layer);
  };
  ///get layers const version
  std::vector<OGRLayer*> getLayers(const std::vector<std::string>& layernames=std::vector<std::string>()) const {
    if(layernames.size()){
      std::vector<OGRLayer*> layers;
      for(size_t ilayer=0;ilayer<getGDSLayerCount();++ilayer){
        std::string currentLayername=getLayer(ilayer)->GetName();
        std::vector<std::string>::const_iterator it=find(layernames.begin(),layernames.end(),currentLayername);
        if(it==layernames.end())
          continue;
        else
          layers.push_back(getLayer(ilayer));
      }
      return(layers);
    }
    else
      return(m_layer);
  };
  ///get geometry type
  OGRwkbGeometryType getGeometryType(size_t ilayer=0) {return(m_layer[ilayer]->GetGeomType());};
  ///get number of features over all layers
  size_t getFeatureCount() const {
    size_t nfeatures=0;
    for(size_t ilayer=0;ilayer<getLayerCount();++ilayer)
      nfeatures+=getFeatureCount(ilayer);
    return(nfeatures);
  };
  ///get number of features
  size_t getFeatureCount(size_t ilayer) const {
    if(m_features.size()<=ilayer){
      std::ostringstream errorStream;
      errorStream << "Error: m_features not initialized for layer " << ilayer << std::endl;
      std::cerr << errorStream.str() << std::endl;
      throw(errorStream.str());
    }
    return(m_features[ilayer].size());
  };
  ///create field
  OGRErr createField(const std::string& fieldname, const OGRFieldType& fieldType,size_t ilayer=0);
    ///create field
  OGRErr createField(OGRFieldDefn*	poField,size_t ilayer=0);
  ///copy fields from other VectorOgr instance
  OGRErr copyFields(const VectorOgr& vectorOgr,const std::vector<std::string>& fieldnames=std::vector<std::string>(),size_t ilayer=0);
  ///merge another vector
  /* OGRErr merge(VectorOgr& vectorOgr); */
  ///write all features (default to m_gds dataset already defined when opened, but optionally to another filename
  OGRErr write(const std::string& filename=std::string());
  ///write all features to a new vector dataset
  //void write(const std::string& filename, const std::string& layername, const::std::string& imageType="SQLite", char** papszOptions=NULL);
    ///copy all features from existing VectorOgr and write to vector dataset
  /* void write(VectorOgr& vectorOgr, const std::string& filename, const std::string& layername, const::std::string& imageType="SQLite", char** papszOptions=NULL); */
  ///Get geographical extent upper left and lower right corners over all layers
  bool getExtent(double& ulx, double& uly, double& lrx, double& lry, OGRCoordinateTransformation *poCT=0) const;
  ///Get geographical extent upper left and lower right corners for specific layer
  bool getExtent(double& ulx, double& uly, double& lrx, double& lry, size_t ilayer, OGRCoordinateTransformation *poCT) const;
  ///Get geographical extent upper left and lower right corners
  /* void getExtent(std::vector<double> &bbvector, size_t ilayer=0, OGRCoordinateTransformation *poCT=0) const; */
  /* void getExtent(OGRPolygon *bbPolygon, size_t ilayer=0, OGRCoordinateTransformation *poCT=0) const; */
  void getBoundingBox(double& ulx, double& uly, double& lrx, double& lry, OGRCoordinateTransformation *poCT=0) const{getExtent(ulx,uly,lrx,lry,poCT);};
  void getBoundingBox(double& ulx, double& uly, double& lrx, double& lry, size_t ilayer, OGRCoordinateTransformation *poCT) const{getExtent(ulx,uly,lrx,lry,ilayer,poCT);};
  /* void getBoundingBox(std::vector<double> &bbvector, size_t ilayer=0, OGRCoordinateTransformation *poCT=0) const{getExtent(bbvector,ilayer,poCT);}; */
  /* void getBoundingBox(OGRPolygon *bbPolygon, size_t ilayer=0, OGRCoordinateTransformation *poCT=0) const{getExtent(bbPolygon,ilayer,poCT);}; */
  ///Get Upper left corner in x over all layers
  double getUlx(OGRCoordinateTransformation *poCT=0) const{double ulx,uly,lrx,lry;getExtent(ulx,uly,lrx,lry,poCT);return(ulx);};
  ///Get Upper left corner in y over all layers
  double getUly(OGRCoordinateTransformation *poCT=0) const{double ulx,uly,lrx,lry;getExtent(ulx,uly,lrx,lry,poCT);return(uly);};
  ///Get lower right corner in x over all layers
  double getLrx(OGRCoordinateTransformation *poCT=0) const{double ulx,uly,lrx,lry;getExtent(ulx,uly,lrx,lry,poCT);return(lrx);};
  ///Get lower right corner in y over all layers
  double getLry(OGRCoordinateTransformation *poCT=0) const{double ulx,uly,lrx,lry;getExtent(ulx,uly,lrx,lry,poCT);return(lry);};
  ///Get Upper left corner in x
  double getUlx(size_t ilayer, OGRCoordinateTransformation *poCT) const{double ulx,uly,lrx,lry;getExtent(ulx,uly,lrx,lry,ilayer,poCT);return(ulx);};
  ///Get Upper left corner in y
  double getUly(size_t ilayer, OGRCoordinateTransformation *poCT) const{double ulx,uly,lrx,lry;getExtent(ulx,uly,lrx,lry,ilayer,poCT);return(uly);};
  ///Get lower right corner in x
  double getLrx(size_t ilayer, OGRCoordinateTransformation *poCT) const{double ulx,uly,lrx,lry;getExtent(ulx,uly,lrx,lry,ilayer,poCT);return(lrx);};
  ///Get lower right corner in y
  double getLry(size_t ilayer, OGRCoordinateTransformation *poCT) const{double ulx,uly,lrx,lry;getExtent(ulx,uly,lrx,lry,ilayer,poCT);return(lry);};
  ///resize features
  OGRErr resize(size_t theSize, size_t ilayer=0){
    m_features[ilayer].resize(theSize);
  };
  ///create a new feature
  OGRFeature* createFeature(size_t ilayer=0){OGRFeature* newFeature=OGRFeature::CreateFeature(getLayer(ilayer)->GetLayerDefn());return(newFeature);};
  ///push feature to the object
  OGRErr pushFeature(OGRFeature *poFeature,size_t ilayer=0);
  ///set feature to the object
  OGRErr setFeature(unsigned int index, OGRFeature *poFeature,size_t ilayer=0);
  ///Assignment operator
  /* VectorOgr& operator=(VectorOgr& other); */
  ///Set rectangular spatial filter (warning: mind order, unlike GDAL I do not use minX, minY, maxX, maxY!!!)
  OGRErr setSpatialFilterRect(double ulx, double uly, double lrx, double lry, size_t ilayer=0){m_layer[ilayer]->SetSpatialFilterRect(ulx,lry,lrx,uly);};
  ///Set spatial filter
  OGRErr setSpatialFilter(OGRGeometry* spatialFilter=NULL, size_t ilayer=0){m_layer[ilayer]->SetSpatialFilter(spatialFilter);};
  ///Set attribute filter
  OGRErr setAttributeFilter(const std::string& attributeFilter, size_t ilayer=0){if(attributeFilter.size()) m_layer[ilayer]->SetAttributeFilter(attributeFilter.c_str());else m_layer[ilayer]->SetAttributeFilter(NULL);};
  ///clone feature. The newly created feature is owned by the caller, and will have it's own reference to the OGRFeatureDefn.
  OGRFeature* cloneFeature(unsigned int index, size_t ilayer=0);
  ///get feature reference (feature should not be deleted)
  OGRFeature* getFeatureRef(unsigned int index,size_t ilayer=0);
  ///Get field definitions in vector
  OGRErr getFields(std::vector<OGRFieldDefn*>& fields, size_t ilayer=0) const;
  ///Get field names in vector
  void getFieldNames(std::vector<std::string>& fields, size_t layer=0) const;
  ///prepare dataset for writing, e.g., register driver
  //OGRErr createDS(const std::string& filename, const std::string& imageType, DATA_ACCESS theAccess=WRITE);
  OGRErr addPoint(double x, double y, const std::map<std::string,double>& pointAttributes, std::string fieldName, int theId, size_t ilayer=0);
  OGRErr addPoint(double x, double y, const std::map<std::string,double>& pointAttributes, size_t ilayer=0);
  size_t serialize(std::vector<unsigned char> &vbytes);
  void dumpOgr(app::AppFactory& app);
  ///append two VectorOgr
  void append(VectorOgr &ogrReader);
  ///joins two VectorOgr based on key value
  std::shared_ptr<VectorOgr> join(VectorOgr &ogrReader, app::AppFactory& app);
  ///joins two VectorOgr based on key value
  OGRErr join(VectorOgr &ogrReader, VectorOgr &ogrWriter, app::AppFactory& app);
  ///sort features by label and store in map
  OGRErr sortByLabel(std::map<std::string,Vector2d<float> > &mapPixels, const std::string& label="label", const std::vector<std::string>& bandNames=std::vector<std::string>());
  ///train classifier
#ifdef SWIG
  %pythonprepend train(app::AppFactory&)  "\"\"\"HELP.METHOD.train(dict)\"\"\""
#endif
  OGRErr train(app::AppFactory& app);
  ///train in memory without writing to file
  std::string trainMem(app::AppFactory& app);
  ///classify
  std::shared_ptr<VectorOgr> classify(app::AppFactory& app);
  ///classify
  void classify(VectorOgr& ogrWriter, app::AppFactory& app);
  //getShared from this
  std::shared_ptr<VectorOgr> getShared(){return(std::dynamic_pointer_cast<VectorOgr>(shared_from_this()));};
  ///overload output stream operator
  friend std::ostream& operator<<(std::ostream& theOstream, VectorOgr& theVector);
  ///static function to transform geometry based on OGRCoordinateTransformation
  static bool transform(OGRGeometry *pGeom, OGRCoordinateTransformation *poCT);
  ///static function to transform geometry based on OGRCoordinateTransformation
  static bool transform(OGREnvelope *pEnv, OGRCoordinateTransformation *poCT);
  /* bool transform(OGRGeometry *pGeom, const std::string& outputProj4); */
  /* ///function for coordinate transform based on EPSG codes */
  /* bool transform(OGRGeometry *pGeom, int outputEPSG); */
  ///function for coordinate transform based on OGRSpatialReference
  /* ///static function for coordinate transform based on proj4 parameters */
  /* bool transform(OGRGeometry *pGeom, const std::string& inputProj4, const std::string& outputProj4); */
  /* ///static function for coordinate transform based on EPSG codes */
  /* static bool transform(OGRGeometry *pGeom, int inputEPSG, int outputEPSG); */
  /* ///static function for coordinate transform based on OGRSpatialReference */
  /* static bool transform(OGRGeometry *pGeom, OGRSpatialReference *sourceSRS, OGRSpatialReference *targetSRS); */
  /* ///static function for coordinate transform of a vector of points based on OGRSpatialReference */
  /* static bool transform(std::vector<double> &xvector, std::vector<double> &yvector, OGRSpatialReference *sourceSRS, OGRSpatialReference *targetSRS); */
  /* static OGRErr join(VectorOgr &ogrReader1, VectorOgr &ogrReader2, VectorOgr &ogrWriter, app::AppFactory& app); */
  unsigned int readFeatures();
  unsigned int readFeatures(size_t ilayer);
 private:
  ///train SVM classifier
  std::string trainSVM(app::AppFactory& app);
  ///classify SVM
  std::shared_ptr<VectorOgr> classifySVM(app::AppFactory& app);
  ///classify SVM
  void classifySVM(VectorOgr& ogrWriter, app::AppFactory& app);
  ///train ANN classifier
  std::string trainANN(app::AppFactory& app);
  ///classify ANN
  std::shared_ptr<VectorOgr> classifyANN(app::AppFactory& app);
  ///classify ANN
  void classifyANN(VectorOgr& ogrWriter, app::AppFactory& app);
  ///read all features from an OGR dataset, attribute filter and spatial filter optionally
  std::string m_filename;
  unsigned int m_access;
  /* std::string m_projection; */
  std::vector<OGRLayer*> m_layer;
  ///instance of the GDAL dataset
#if GDAL_VERSION_MAJOR < 2
  OGRDataSource *m_gds;
#else
  GDALDataset *m_gds;
#endif

  ///vector containing all features in memory
  std::vector<std::vector<OGRFeature*> > m_features;
};

/* static std::shared_ptr<VectorOgr> createVector(){return VectorOgr::createVector();}; */
/* static std::shared_ptr<VectorOgr> createVector(const std::string& filename){return(VectorOgr::createVector(filename));}; */
/* static std::shared_ptr<VectorOgr> createVector(app::AppFactory &app){return(VectorOgr::createVector(app));}; */
/* static std::shared_ptr<VectorOgr> createVector(VectorOgr& other,app::AppFactory &app){return(VectorOgr::createVector(other,app));}; */
#endif // _VECTOROGR_H_
