/**********************************************************************
FileReaderLas.cc: class to read LAS files using liblas API library
Author(s): Pieter.Kempeneers@ec.europa.eu
Copyright (c) 2016-2018 European Union (Joint Research Centre)
License EUPLv1.2

This file is part of jiplib
***********************************************************************/
#include <string>
#include <iostream>
#include <fstream>
#include "FileReaderLas.h"
//---------------------------------------------------------------------------
LastReturnFilter::LastReturnFilter(  ) : liblas::FilterI(eInclusion) {}

bool LastReturnFilter::filter(const liblas::Point& p)
{

  // If the GetReturnNumber equals the GetNumberOfReturns,
  // we're a last return

  bool output = false;
  if (p.GetReturnNumber() == p.GetNumberOfReturns())
    {
      output = true;
    }

  // If the type is switched to eExclusion, we'll throw out all last returns.
  if (GetType() == eExclusion && output == true)
    {
      output = false;
    } else {
    output = true;
  }
  return output;
}

FileReaderLas::FileReaderLas(void)
{
  m_reader=NULL;
  m_ifstream=NULL;
}

FileReaderLas::FileReaderLas(const std::string& filename)
{
  open(filename);
}

FileReaderLas::~FileReaderLas(void)
{
  delete m_ifstream;
  delete m_reader;
}

//---------------------------------------------------------------------------

void FileReaderLas::open(const std::string& filename)
{
  m_filename = filename;
  setCodec(filename);
}

//---------------------------------------------------------------------------
void FileReaderLas::close(void)
{
  m_ifstream->close();
  m_ifstream=NULL;
  m_reader=NULL;
}

//---------------------------------------------------------------------------
void FileReaderLas::setCodec(const std::string& filename){
  m_ifstream = new(std::ifstream);
  m_ifstream->open(m_filename.c_str(),std::ios::in|std::ios::binary);
  liblas::ReaderFactory f;
  liblas::Reader reader = f.CreateWithStream(*m_ifstream);
  m_reader=new liblas::Reader(reader);
  // m_reader = new liblas::Reader(*m_ifstream);
  //Note: It is possible to use the basic liblas::Reader constructor that takes in a std::istream, but it will not be able to account for the fact that the file might be compressed. Using the ReaderFactory will take care of all of this for you.
  // m_reader=&rfactory.CreateWithStream(ifs);
}

liblas::Header const& FileReaderLas::getHeader() const{
  return(m_reader->GetHeader());
}

bool FileReaderLas::isCompressed() const{
  return getHeader().Compressed();
}

unsigned long int FileReaderLas::getPointCount() const{
  return getHeader().GetPointRecordsCount();
}

bool const& FileReaderLas::readNextPoint(liblas::Point& thePoint){
  bool returnValue=m_reader->ReadNextPoint();
  thePoint=m_reader->GetPoint();
  return(returnValue);
}

void FileReaderLas::las2ascii(const std::string& filename, bool verbose) const{
  std::ofstream fpoints(filename.c_str(),std::ios::out);
  fpoints << "#";
  fpoints << "X" << "," << "Y" << "," << "Z" << std::endl;
  if(verbose)
    std::cout << "reset reading" << std::endl;
  m_reader->Reset();
  if(verbose)
    std::cout << "going through points" << std::endl;
  while(m_reader->ReadNextPoint()){
    liblas::Point const& thePoint=m_reader->GetPoint();
    double x=thePoint.GetX();
    double y=thePoint.GetY();
    double z=thePoint.GetZ();
    fpoints.precision(12);
    fpoints << x << "," << y << "," << z << std::endl;
  }
  fpoints.close();
}

void FileReaderLas::getExtent(double& ulx, double& uly, double& lrx, double& lry) const{
  const liblas::Header& theHeader=getHeader();
  ulx=theHeader.GetMinX();
  uly=theHeader.GetMaxY();
  lrx=theHeader.GetMaxX();
  lry=theHeader.GetMinY();
}

double FileReaderLas::getMinZ() const{
  return(getHeader().GetMinZ());
}

double FileReaderLas::getMaxZ() const{
  return(getHeader().GetMaxZ());
}

//todo: does not work ??
// void FileReaderLas::addBoundsFilter(double ulx, double uly, double lrx, double lry){
//   liblas::Bounds<double> bounds = liblas::Bounds<double>(ulx,lry,lrx,uly);
//   typedef liblas::BoundsFilter filter;
//   filter* bounds_filter = new filter(bounds);
//   bounds_filter->SetType(liblas::FilterI::eInclusion);
//   m_filters.push_back(liblas::FilterPtr(bounds_filter));
// }

void FileReaderLas::addReturnsFilter(std::vector<unsigned short> const& returns){
  typedef liblas::ReturnFilter filter;
  filter* return_filter;
  std::vector<boost::uint16_t> returns_t;
  if(returns[0]<0)
    return_filter=new filter(returns_t,true);
  else{
    for(int index=0;index<returns.size();++index){
      assert(returns[index]>0);
      returns_t.push_back(returns[index]);
    }
    return_filter=new filter(returns_t,false);
  }
  m_filters.push_back(liblas::FilterPtr(return_filter));
}

void FileReaderLas::addClassFilter(std::vector<unsigned short> const& classes){

  std::vector<liblas::FilterPtr> filters;
  std::vector<liblas::Classification> theClasses;
  for(int iclass=0;iclass<classes.size();++iclass){
    liblas::Classification aClass(classes[iclass]);
    theClasses.push_back(aClass);
  }
  liblas::FilterPtr class_filter = liblas::FilterPtr(new liblas::ClassificationFilter(theClasses));
  // eInclusion means to keep the classes that match.  eExclusion would
  // throw out those that matched
  class_filter->SetType(liblas::FilterI::eInclusion);
  m_filters.push_back(class_filter);
}

