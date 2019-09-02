/**********************************************************************
FileReaderAscii.cc: class to read ASCII files using (colum based)
Author(s): Pieter.Kempeneers@ec.europa.eu
Copyright (c) 2016-2019 European Union (Joint Research Centre)
License EUPLv1.2

This file is part of jiplib
***********************************************************************/

#include <iostream>
#include "FileReaderAscii.h"

FileReaderAscii::FileReaderAscii()
  : m_min(0),m_max(0),m_minRow(0),m_maxRow(0),m_fs(' '),m_comment('#'){
}

FileReaderAscii::FileReaderAscii(const std::string& filename)
  : m_min(0),m_max(0),m_minRow(0),m_maxRow(0),m_fs(' '),m_comment('#'){
  open(filename);
}

FileReaderAscii::FileReaderAscii(const std::string& filename, const char& fieldseparator)
  : m_min(0),m_max(0),m_minRow(0),m_maxRow(0),m_fs(fieldseparator),m_comment('#'){
  open(filename);
}

FileReaderAscii::~FileReaderAscii()
{
}

void FileReaderAscii::open(const std::string& filename){
  m_filename=filename;
  m_ifstream.open(filename.c_str(),std::ios_base::in);
  if(!(m_ifstream)){
    std::string errorString;
    errorString="Error: could not open file ";
    errorString+=filename;
    throw(errorString);
  }
}

void FileReaderAscii::close(){
  m_ifstream.close();
  //  m_ifstream.clear();
}

unsigned int FileReaderAscii::nrOfCol(bool checkCols, bool verbose){
  reset();
  unsigned int totalCol=0;
  unsigned int nrow=0;
  if(m_fs>' '&&m_fs<='~'){//field separator is a regular character (minimum ASCII code is space, maximum ASCII code is tilde)
    if(verbose)
      std::cout << "reading csv file " << m_filename << std::endl;
    std::string csvRecord;
    while(getline(m_ifstream,csvRecord)){//read a line
      std::istringstream csvstream(csvRecord);
      std::string item;
      unsigned int ncol=0;
      bool isComment=false;
      while(getline(csvstream,item,m_fs)){//read a column
        if(verbose)
          std::cout << item << " ";
        size_t pos=item.find(m_comment);
        if(pos!=std::string::npos){
          if(pos>0)
            item=item.substr(0,pos-1);
          else
            break;
          if(verbose)
            std::cout << "comment found, string is " << item << std::endl;
          isComment=true;
        }
        ++ncol;
        if(isComment)
          break;
      }
      if(verbose)
        std::cout << std::endl << "number of columns: " << ncol << std::endl;
      if(!totalCol)
        totalCol=ncol;
      else if(checkCols){
        if(totalCol!=ncol){
          std::ostringstream ess;
          ess << "Error: different number of cols found in line " << nrow << " (" << ncol << "!=" << totalCol << ")" << std::endl;
          throw(ess.str());
        }
        ++nrow;
      }
      else
        break;
    }
  }
  else{//space or tab delimited fields
    if(verbose)
      std::cout << "space or tab delimited fields" << std::endl;
    std::string spaceRecord;
    while(!getline(m_ifstream, spaceRecord).eof()){
      if(verbose>1)
        std::cout << spaceRecord << std::endl;
      std::istringstream lineStream(spaceRecord);
      std::string item;
      unsigned int ncol=0;
      bool isComment=false;
      while(lineStream >> item){
        if(verbose)
          std::cout << item << " ";
        size_t pos=item.find(m_comment);
        if(pos!=std::string::npos){
          if(pos>0)
            item=item.substr(0,pos-1);
          else
            break;
          if(verbose)
            std::cout << "comment found, string is " << item << std::endl;
          isComment=true;
        }
        ++ncol;
        if(isComment)
          break;
      }
      if(verbose)
        std::cout << std::endl << "number of columns: " << ncol << std::endl;
      if(!totalCol)
        totalCol=ncol;
      else if(checkCols){
        if(totalCol!=ncol){
          std::ostringstream ess;
          ess << "Error: different number of cols found in line " << nrow << " (" << ncol << "!=" << totalCol << ")" << std::endl;
          throw(ess.str());
        }
      }
      else
        break;
      ++nrow;
    }
  }
  return totalCol;
}

unsigned int FileReaderAscii::nrOfRow(bool checkCols, bool verbose){
  reset();
  unsigned int totalCol=0;
  unsigned int nrow=0;
  unsigned int ncomment=0;
  if(m_fs>' '&&m_fs<='~'){//field separator is a regular character (minimum ASCII code is space, maximum ASCII code is tilde)
    if(verbose)
      std::cout << "reading csv file " << m_filename << std::endl;
    std::string csvRecord;
    while(getline(m_ifstream,csvRecord)){//read a line
      std::istringstream csvstream(csvRecord);
      std::string item;
      unsigned int ncol=0;
      bool isComment=false;
      while(getline(csvstream,item,m_fs)){//read a column
        if(verbose)
          std::cout << item << " ";
        size_t pos=item.find(m_comment);
        if(pos!=std::string::npos){
          if(pos>0){
            if(verbose)
              std::cout << "comment found, string is " << item << std::endl;
            isComment=true;
          }
          else{
            ++ncomment;
            break;
          }
        }
        ++ncol;
        if(isComment)
          break;
      }
      if(verbose)
        std::cout << std::endl;
      if(checkCols){
        if(totalCol!=ncol){
          std::ostringstream ess;
          ess << "Error: different number of cols found in line " << nrow << " (" << ncol << "!=" << totalCol << ")" << std::endl;
          throw(ess.str());
        }
      }
      ++nrow;
    }
  }
  else{//space or tab delimited fields
    if(verbose)
      std::cout << "space or tab delimited fields" << std::endl;
    std::string spaceRecord;
    int totalCol=0;
    while(!getline(m_ifstream, spaceRecord).eof()){
      if(verbose>1)
        std::cout << spaceRecord << std::endl;
      std::istringstream lineStream(spaceRecord);
      std::string item;
      unsigned int ncol=0;
      bool isComment=false;
      while(lineStream >> item){
        if(verbose)
          std::cout << item << " ";
        size_t pos=item.find(m_comment);
        if(pos!=std::string::npos){
          if(pos>0){
            if(verbose)
              std::cout << "comment found, string is " << item << std::endl;
            isComment=true;
          }
          else{
            ++ncomment;
            break;
          }
        }
        ++ncol;
        if(isComment)
          break;
      }
      if(verbose)
        std::cout << std::endl << "number of columns: " << ncol << std::endl;
      if(checkCols){
        if(totalCol!=ncol){
          std::ostringstream ess;
          ess << "Error: different number of cols found in line " << nrow << " (" << ncol << "!=" << totalCol << ")" << std::endl;
          throw(ess.str());
        }
      }
      ++nrow;
    }
  }
  return nrow;
}

