/**********************************************************************
Filter2d.cc: class for filtering images
Author(s): Pieter.Kempeneers@ec.europa.eu
Copyright (C) 2016-2022 European Union (Joint Research Centre)

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
#include <sstream>
#include <iomanip>
#include <iostream>
#include <cmath>
#include "Filter2d.h"
#include "StatFactory.h"

filter2d::Filter2d::Filter2d(void)
{
}

filter2d::Filter2d::Filter2d(const Vector2d<double> &taps, bool flip)
{
  setTaps(taps,flip);
}

int filter2d::Filter2d::pushNoDataValue(double noDataValue)
{
  if(find(m_noDataValues.begin(),m_noDataValues.end(),noDataValue)==m_noDataValues.end())
    m_noDataValues.push_back(noDataValue);
  return(m_noDataValues.size());
}

void filter2d::Filter2d::setTaps(const Vector2d<double> &taps, bool flip)
{
  size_t dimY=taps.size();
  if(dimY<=0){
    std::string errorString="Error: empty taps";
    throw(errorString);
  }
  size_t dimX=taps[0].size();
  m_taps.resize(dimY,dimX);
  if(flip){
    for(size_t j=0;j<taps.size();++j){
      for(size_t i=0;i<taps[j].size();++i){
        m_taps[taps.size()-j-1][taps[0].size()-i-1]=taps[j][i];
      }
    }
  }
  else
    m_taps=taps;
}

void filter2d::Filter2d::smoothNoData(Jim& input, Jim& output, int dim)
{
  smoothNoData(input, output,dim,dim);
}

void filter2d::Filter2d::smooth(Jim& input, Jim& output, int dim)
{
  smooth(input, output,dim,dim);
}

void filter2d::Filter2d::smoothNoData(Jim& input, Jim& output, int dimX, int dimY)
{
  m_taps.resize(dimY);
  for(int j=0;j<dimY;++j){
    m_taps[j].resize(dimX);
    for(int i=0;i<dimX;++i)
      m_taps[j][i]=1.0;
  }
  filterLB(input,output,false,true,true);
}

void filter2d::Filter2d::smooth(Jim& input, Jim& output, int dimX, int dimY)
{
  m_taps.resize(dimY);
  for(int j=0;j<dimY;++j){
    m_taps[j].resize(dimX);
    for(int i=0;i<dimX;++i)
      m_taps[j][i]=1.0;
  }
  filterLB(input,output,false,true,false);
}


void filter2d::Filter2d::filter(Jim& input, Jim& output, bool absolute, bool normalize){
  if(!output.isInit())
    output.open(input,false);
  output.setNoData(m_noDataValues);

  size_t dimx=input.nrOfCol();
  size_t dimy=input.nrOfRow();
#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
  for(size_t iband=0;iband<input.nrOfBand();++iband){
    switch(input.getDataType()){
    case(GDT_Byte):
      filter<unsigned char>(static_cast<unsigned char*>(input.getDataPointer(iband)),static_cast<unsigned char*>(output.getDataPointer(iband)), dimx, dimy, absolute,normalize);
      break;
    case(GDT_Int16):
      filter<short>(static_cast<short*>(input.getDataPointer(iband)),static_cast<short*>(output.getDataPointer(iband)), dimx, dimy, absolute,normalize);
      break;
    case(GDT_UInt16):
      filter<unsigned short>(static_cast<unsigned short*>(input.getDataPointer(iband)),static_cast<unsigned short*>(output.getDataPointer(iband)), dimx, dimy, absolute,normalize);
      break;
    case(GDT_Int32):
      filter<int>(static_cast<int*>(input.getDataPointer(iband)),static_cast<int*>(output.getDataPointer(iband)), dimx, dimy, absolute,normalize);
      break;
    case(GDT_UInt32):
      filter<unsigned int>(static_cast<unsigned int*>(input.getDataPointer(iband)),static_cast<unsigned int*>(output.getDataPointer(iband)), dimx, dimy, absolute,normalize);
      break;
    case(GDT_Float32):
      filter<float>(static_cast<float*>(input.getDataPointer(iband)),static_cast<float*>(output.getDataPointer(iband)), dimx, dimy, absolute,normalize);
      break;
    case(GDT_Float64):
      filter<double>(static_cast<double*>(input.getDataPointer(iband)),static_cast<double*>(output.getDataPointer(iband)), dimx, dimy, absolute,normalize);
      break;
    }
  }
}

//line based version
void filter2d::Filter2d::filterLB(Jim& input, Jim& output, bool absolute, bool normalize, bool noData)
{
  if(!output.isInit())
    output.open(input);
  output.setNoData(m_noDataValues);

  int dimX=m_taps[0].size();//horizontal!!!
  int dimY=m_taps.size();//vertical!!!
  const char* pszMessage;
  // void* pProgressArg=NULL;
  // GDALProgressFunc pfnProgress=GDALTermProgress;
  // double progress=0;
  // MyProgressFunc(progress,pszMessage,pProgressArg);
#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
  for(unsigned int iband=0;iband<input.nrOfBand();++iband){
    Vector2d<double> inBuffer(dimY,input.nrOfCol());
    std::vector<double> outBuffer(input.nrOfCol());
    int indexI=0;
    int indexJ=0;
    //initialize last half of inBuffer
    for(int j=-(dimY-1)/2;j<=dimY/2;++j){
      try{
        input.readData(inBuffer[indexJ],static_cast<unsigned int>(abs(j)),iband);
      }
      catch(std::string errorstring){
        std::cerr << errorstring << "in line " << indexJ << std::endl;
      }
      ++indexJ;
    }

    for(unsigned int y=0;y<input.nrOfRow();++y){
      if(y){//inBuffer already initialized for y=0
        //erase first line from inBuffer
        if(dimY>1)
          inBuffer.erase(inBuffer.begin());
        //read extra line and push back to inBuffer if not out of bounds
        if(y+dimY/2<input.nrOfRow()){
          //allocate buffer
          if(dimY>1)
            inBuffer.push_back(inBuffer.back());
          try{
            input.readData(inBuffer[inBuffer.size()-1],y+dimY/2,iband);
          }
          catch(std::string errorstring){
            std::cerr << errorstring << "in band " << iband << ", line " << y << std::endl;
            exit(1);
          }
        }
        else{
          int over=y+dimY/2-input.nrOfRow();
          int index=(inBuffer.size()-1)-over;
          assert(index>=0);
          assert(index<inBuffer.size());
          inBuffer.push_back(inBuffer[index]);
        }
      }
      for(int x=0;x<input.nrOfCol();++x){
        outBuffer[x]=0;
        double norm=0;
        bool masked=false;
        if(noData){//only filter noData values
          for(int imask=0;imask<m_noDataValues.size();++imask){
            if(inBuffer[(dimY-1)/2][x]==m_noDataValues[imask]){
              masked=true;
              break;
            }
          }
          if(!masked){
            outBuffer[x]=inBuffer[(dimY-1)/2][x];
            continue;
          }
        }
        assert(!noData||masked);
        for(int j=-(dimY-1)/2;j<=dimY/2;++j){
          for(int i=-(dimX-1)/2;i<=dimX/2;++i){
            indexI=x+i;
            indexJ=(dimY-1)/2+j;
            //check if out of bounds
            if(x<(dimX-1)/2)
              indexI=x+abs(i);
            else if(x>=input.nrOfCol()-(dimX-1)/2)
              indexI=x-abs(i);
            if(y<(dimY-1)/2)
              indexJ=(dimY-1)/2+abs(j);
            else if(y>=input.nrOfRow()-(dimY-1)/2)
              indexJ=(dimY-1)/2-abs(j);
            //do not take masked values into account
            masked=false;
            for(int imask=0;imask<m_noDataValues.size();++imask){
              if(inBuffer[indexJ][indexI]==m_noDataValues[imask]){
                masked=true;
                break;
              }
            }
            if(!masked){
              outBuffer[x]+=(m_taps[(dimY-1)/2+j][(dimX-1)/2+i]*inBuffer[indexJ][indexI]);
              norm+=m_taps[(dimY-1)/2+j][(dimX-1)/2+i];
            }
          }
        }
        if(absolute)
          outBuffer[x]=(normalize&&norm)? abs(outBuffer[x])/norm : abs(outBuffer[x]);
        else if(normalize&&norm!=0)
          outBuffer[x]=outBuffer[x]/norm;
      }
      //write outBuffer to file
      try{
        output.writeData(outBuffer,y,iband);
      }
      catch(std::string errorstring){
        std::cerr << errorstring << " in band " << iband << ", line " << y << std::endl;
        exit(1);
      }
      // progress=(1.0+y);
      // progress+=(output.nrOfRow()*iband);
      // progress/=output.nrOfBand()*output.nrOfRow();
      // MyProgressFunc(progress,pszMessage,pProgressArg);
    }
  }
}


void filter2d::Filter2d::majorVoting(Jim& input, Jim& output, int dim, const std::vector<int> &prior)
{
  if(!output.isInit())
    output.open(input);
  output.setNoData(m_noDataValues);

  const char* pszMessage;
  // void* pProgressArg=NULL;
  // GDALProgressFunc pfnProgress=GDALTermProgress;
  // double progress=0;
  // MyProgressFunc(progress,pszMessage,pProgressArg);

  bool usePriors=true;
  if(prior.empty()){
    std::cout << "no prior information" << std::endl;
    usePriors=false;
  }
  else{
    std::cout << "using priors ";
    for(int iclass=0;iclass<prior.size();++iclass)
      std::cout << " " << static_cast<short>(prior[iclass]);
    std::cout << std::endl;
  }

  // Jim input;
  // Jim output;
  // input.open(inputFilename);
  // output.open(outputFilename,input);
  int dimX=0;//horizontal!!!
  int dimY=0;//vertical!!!
  if(dim){
    dimX=dim;
    dimY=dim;
  }
  else{
    dimX=m_taps[0].size();
    dimY=m_taps.size();
  }

  assert(dimX);
  assert(dimY);

  Vector2d<double> inBuffer(dimY,input.nrOfCol());
  std::vector<double> outBuffer(input.nrOfCol());
  int indexI=0;
  int indexJ=0;
  //initialize last half of inBuffer
  for(int j=-(dimY-1)/2;j<=dimY/2;++j){
    try{
      input.readData(inBuffer[indexJ],static_cast<unsigned int>(abs(j)));
    }
    catch(std::string errorstring){
      std::cerr << errorstring << "in line " << indexJ << std::endl;
    }
    ++indexJ;
  }

  for(unsigned int y=0;y<input.nrOfRow();++y){
    if(y){//inBuffer already initialized for y=0
      //erase first line from inBuffer
      if(dimY>1)
        inBuffer.erase(inBuffer.begin());
      //read extra line and push back to inBuffer if not out of bounds
      if(y+dimY/2<input.nrOfRow()){
        //allocate buffer
        if(dimY>1)
          inBuffer.push_back(inBuffer.back());
        try{
          input.readData(inBuffer[inBuffer.size()-1],y+dimY/2);
        }
        catch(std::string errorstring){
          std::cerr << errorstring << "in line" << y << std::endl;
        }
      }
      else{
        int over=y+dimY/2-input.nrOfRow();
        int index=(inBuffer.size()-1)-over;
        assert(index>=0);
        assert(index<inBuffer.size());
        inBuffer.push_back(inBuffer[index]);
      }
    }
    for(unsigned int x=0;x<input.nrOfCol();++x){
      outBuffer[x]=0;
      std::map<int,int> occurrence;
      // int centre=dimX*(dimY-1)/2+(dimX-1)/2;
      for(int j=-(dimY-1)/2;j<=dimY/2;++j){
        for(int i=-(dimX-1)/2;i<=dimX/2;++i){
          indexI=x+i;
          //check if out of bounds
          if(indexI<0)
            indexI=-indexI;
          else if(indexI>=input.nrOfCol())
            indexI=input.nrOfCol()-i;
          if(y+j<0)
            indexJ=-j;
          else if(y+j>=input.nrOfRow())
            indexJ=(dimY>2) ? (dimY-1)/2-j : 0;
          else
            indexJ=(dimY-1)/2+j;

          // if(x<dimX/2)
          //   indexI=x+abs(i);
          // else if(x>=input.nrOfCol()-dimX/2)
          //   indexI=x-abs(i);
          // if(y<dimY/2)
          //   indexJ=dimY/2+abs(j);
          // else if(y>=input.nrOfRow()-dimY/2)
          //   indexJ=dimY/2-abs(j);
          if(usePriors){
            occurrence[inBuffer[indexJ][indexI]]+=prior[inBuffer[indexJ][indexI]-1];
          }
          else
            ++occurrence[inBuffer[indexJ][indexI]];
        }
      }
      std::map<int,int>::const_iterator maxit=occurrence.begin();
      for(std::map<int,int>::const_iterator mit=occurrence.begin();mit!=occurrence.end();++mit){
        if(mit->second>maxit->second)
          maxit=mit;
      }
      if(occurrence[inBuffer[(dimY-1)/2][x]]<maxit->second)//
        outBuffer[x]=maxit->first;
      else//favorize original value in case of ties
        outBuffer[x]=inBuffer[(dimY-1)/2][x];
    }
    //write outBuffer to file
    try{
      output.writeData(outBuffer,y);
    }
    catch(std::string errorstring){
      std::cerr << errorstring << "in line" << y << std::endl;
    }
    // progress=(1.0+y)/output.nrOfRow();
    // MyProgressFunc(progress,pszMessage,pProgressArg);
  }
  input.close();
  output.close();
}

void filter2d::Filter2d::median(Jim& input, Jim& output, int dim, bool disc)
{
  doit(input,output,"median",dim,disc);
}

void filter2d::Filter2d::var(Jim& input, Jim& output, int dim, bool disc)
{
  doit(input,output,"var",dim,disc);
}

void filter2d::Filter2d::doit(Jim& input, Jim& output, const std::string& method, int dim, short down, bool disc)
{
  doit(input,output,method,dim,dim,down,disc);
}

void filter2d::Filter2d::doit(Jim& input, Jim& output, const std::string& method, int dimX, int dimY, short down, bool disc)
{
  if(!output.isInit())
    output.open(input);
  output.setNoData(m_noDataValues);

  const char* pszMessage;
  // void* pProgressArg=NULL;
  // GDALProgressFunc pfnProgress=GDALTermProgress;
  // double progress=0;
  // MyProgressFunc(progress,pszMessage,pProgressArg);

  assert(dimX);
  assert(dimY);

  statfactory::StatFactory stat;
#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
  for(unsigned int iband=0;iband<input.nrOfBand();++iband){
    Vector2d<double> inBuffer(dimY,input.nrOfCol());
    std::vector<double> outBuffer((input.nrOfCol()+down-1)/down);
    int indexI=0;
    int indexJ=0;
    //initialize last half of inBuffer
    for(int j=-(dimY-1)/2;j<=dimY/2;++j){
      try{
        input.readData(inBuffer[indexJ],static_cast<unsigned int>(abs(j)),iband);
      }
      catch(std::string errorstring){
        std::cerr << errorstring << "in line " << indexJ << std::endl;
      }
      ++indexJ;
    }
    for(unsigned int y=0;y<input.nrOfRow();++y){
      if(y){//inBuffer already initialized for y=0
        //erase first line from inBuffer
        if(dimY>1)
          inBuffer.erase(inBuffer.begin());
        //read extra line and push back to inBuffer if not out of bounds
        if(y+dimY/2<input.nrOfRow()){
          //allocate buffer
          if(dimY>1)
            inBuffer.push_back(inBuffer.back());
          try{
            input.readData(inBuffer[inBuffer.size()-1],y+dimY/2,iband);
          }
          catch(std::string errorstring){
            std::cerr << errorstring << "in band " << iband << ", line " << y << std::endl;
          }
        }
        else{
          int over=y+dimY/2-input.nrOfRow();
          int index=(inBuffer.size()-1)-over;
          assert(index>=0);
          assert(index<inBuffer.size());
          inBuffer.push_back(inBuffer[index]);
        }
      }
      if((y+1+down/2)%down)
        continue;
      for(unsigned int x=0;x<input.nrOfCol();++x){
        if((x+1+down/2)%down)
          continue;
        outBuffer[x/down]=0;
        std::vector<double> windowBuffer;
        std::map<long int,int> occurrence;
        int centre=dimX*(dimY-1)/2+(dimX-1)/2;
        for(int j=-(dimY-1)/2;j<=dimY/2;++j){
          for(int i=-(dimX-1)/2;i<=dimX/2;++i){
            double d2=i*i+j*j;//square distance
            if(disc&&(d2>(dimX/2)*(dimY/2)))
              continue;
            indexI=x+i;
            //check if out of bounds
            if(indexI<0)
              indexI=-indexI;
            else if(indexI>=input.nrOfCol())
              indexI=input.nrOfCol()-i;
            if(y+j<0)
              indexJ=-j;
            else if(y+j>=input.nrOfRow())
              indexJ=(dimY>2) ? (dimY-1)/2-j : 0;
            else
              indexJ=(dimY-1)/2+j;
            bool masked=false;
            for(int imask=0;imask<m_noDataValues.size();++imask){
              if(inBuffer[indexJ][indexI]==m_noDataValues[imask]){
                masked=true;
                break;
              }
            }
            if(!masked){
              std::vector<short>::const_iterator vit=m_class.begin();
              if(!m_class.size())
                ++occurrence[inBuffer[indexJ][indexI]];
              else{
                while(vit!=m_class.end()){
                  if(inBuffer[indexJ][indexI]==*(vit++))
                    ++occurrence[inBuffer[indexJ][indexI]];
                }
              }
              windowBuffer.push_back(inBuffer[indexJ][indexI]);
            }
          }
        }
        switch(getFilterType(method)){
        case(filter2d::nvalid):
          outBuffer[x/down]=stat.nvalid(windowBuffer);
          break;
        case(filter2d::median):
          if(windowBuffer.empty())
            outBuffer[x/down]=(m_noDataValues.size())? m_noDataValues[0] : 0;
          else
            outBuffer[x/down]=stat.median(windowBuffer);
          break;
        case(filter2d::var):{
          if(windowBuffer.empty())
            outBuffer[x/down]=(m_noDataValues.size())? m_noDataValues[0] : 0;
          else
            outBuffer[x/down]=stat.var(windowBuffer);
          break;
        }
        case(filter2d::stdev):{
          if(windowBuffer.empty())
            outBuffer[x/down]=(m_noDataValues.size())? m_noDataValues[0] : 0;
          else
            outBuffer[x/down]=sqrt(stat.var(windowBuffer));
          break;
        }
        case(filter2d::mean):{
          if(windowBuffer.empty())
            outBuffer[x/down]=(m_noDataValues.size())? m_noDataValues[0] : 0;
          else
            outBuffer[x/down]=stat.mean(windowBuffer);
          break;
        }
        case(filter2d::min):{
          if(windowBuffer.empty())
            outBuffer[x/down]=(m_noDataValues.size())? m_noDataValues[0] : 0;
          else
            outBuffer[x/down]=stat.mymin(windowBuffer);
          break;
        }
        case(filter2d::ismin):{
          if(windowBuffer.empty())
            outBuffer[x/down]=(m_noDataValues.size())? m_noDataValues[0] : 0;
          else
            outBuffer[x/down]=(stat.mymin(windowBuffer)==windowBuffer[centre])? 1:0;
          break;
        }
        case(filter2d::minmax):{//is the same as homog?
          double min=0;
          double max=0;
          if(windowBuffer.empty())
            outBuffer[x/down]=(m_noDataValues.size())? m_noDataValues[0] : 0;
          else{
            stat.minmax(windowBuffer,windowBuffer.begin(),windowBuffer.end(),min,max);
            if(min!=max)
              outBuffer[x/down]=0;
            else
              outBuffer[x/down]=windowBuffer[centre];//centre pixels
          }
          break;
        }
        case(filter2d::max):{
          if(windowBuffer.empty())
            outBuffer[x/down]=(m_noDataValues.size())? m_noDataValues[0] : 0;
          else
            outBuffer[x/down]=stat.mymax(windowBuffer);
          break;
        }
        case(filter2d::ismax):{
          if(windowBuffer.empty())
            outBuffer[x/down]=(m_noDataValues.size())? m_noDataValues[0] : 0;
          else
            outBuffer[x/down]=(stat.mymax(windowBuffer)==windowBuffer[centre])? 1:0;
          break;
        }
        case(filter2d::order):{
          if(windowBuffer.empty())
            outBuffer[x/down]=(m_noDataValues.size())? m_noDataValues[0] : 0;
          else{
            double lbound=0;
            double ubound=dimX*dimY;
            double theMin=stat.mymin(windowBuffer);
            double theMax=stat.mymax(windowBuffer);
            double scale=(ubound-lbound)/(theMax-theMin);
            outBuffer[x/down]=static_cast<short>(scale*(windowBuffer[centre]-theMin)+lbound);
          }
          break;
        }
        case(filter2d::sum):{
          outBuffer[x/down]=stat.sum(windowBuffer);
          break;
        }
        case(filter2d::percentile):{
          assert(m_threshold.size());
          outBuffer[x/down]=stat.percentile(windowBuffer,windowBuffer.begin(),windowBuffer.end(),m_threshold[0]);
          break;
        }
        case(filter2d::proportion):{
          if(windowBuffer.size()){
            double sum=stat.sum(windowBuffer);
            if(sum)
              outBuffer[x/down]=100.0*windowBuffer[centre]/stat.sum(windowBuffer);
            else
              outBuffer[x/down]=(m_noDataValues.size())? m_noDataValues[0] : 0;
          }
          else
            outBuffer[x/down]=(m_noDataValues.size())? m_noDataValues[0] : 0;
          break;
        }
        case(filter2d::homog):
          if(occurrence.size()==1)//all values in window are the same
            outBuffer[x/down]=inBuffer[(dimY-1)/2][x];
          else
            outBuffer[x/down]=(m_noDataValues.size())? m_noDataValues[0] : 0;
          break;
        case(filter2d::heterog):{
          if(occurrence.size()==windowBuffer.size())
            outBuffer[x/down]=inBuffer[(dimY-1)/2][x];
          else
            outBuffer[x/down]=(m_noDataValues.size())? m_noDataValues[0] : 0;
          // if(occurrence.size()==1)//all values in window are the same
          //   outBuffer[x/down]=(m_noDataValues.size())? m_noDataValues[0] : 0;
          // else
          //   outBuffer[x/down]=inBuffer[(dimY-1)/2][x];
          // break;
          // for(std::vector<double>::const_iterator wit=windowBuffer.begin();wit!=windowBuffer.end();++wit){
          //   if(wit==windowBuffer.begin()+windowBuffer.size()/2)
          //     continue;
          //   else if(*wit!=inBuffer[(dimY-1)/2][x]){
          //     outBuffer[x/down]=1;
          //     break;
          //   }
          //   else if(*wit==inBuffer[(dimY-1)/2][x]){//todo:wit mag niet central pixel zijn
          //     outBuffer[x/down]=(m_noDataValues.size())? m_noDataValues[0] : 0;
          //     break;
          //   }
          // }
          // break;
        }
        case(filter2d::density):{
          if(windowBuffer.size()){
            std::vector<short>::const_iterator vit=m_class.begin();
            while(vit!=m_class.end())
              outBuffer[x/down]+=100.0*occurrence[*(vit++)]/windowBuffer.size();
          }
          else
            outBuffer[x/down]=(m_noDataValues.size())? m_noDataValues[0] : 0;
          break;
        }
        case(filter2d::countid):{
          if(windowBuffer.size())
            outBuffer[x/down]=occurrence.size();
          else
            outBuffer[x/down]=(m_noDataValues.size())? m_noDataValues[0] : 0;
          break;
        }
        case(filter2d::mode):{
          if(occurrence.size()){
            std::map<long int,int>::const_iterator maxit=occurrence.begin();
            for(std::map<long int,int>::const_iterator mit=occurrence.begin();mit!=occurrence.end();++mit){
              if(mit->second>maxit->second)
                maxit=mit;
            }
            if(occurrence[inBuffer[(dimY-1)/2][x]]<maxit->second)//
              outBuffer[x/down]=maxit->first;
            else//favorize original value in case of ties
              outBuffer[x/down]=inBuffer[(dimY-1)/2][x];
          }
          else
            outBuffer[x/down]=(m_noDataValues.size())? m_noDataValues[0] : 0;
          break;
        }
        case(filter2d::threshold):{
          assert(m_class.size()==m_threshold.size());
          if(windowBuffer.size()){
            outBuffer[x/down]=inBuffer[(dimY-1)/2][x];//initialize with original value (in case thresholds not met)
            for(int iclass=0;iclass<m_class.size();++iclass){
              if(100.0*(occurrence[m_class[iclass]])/windowBuffer.size()>m_threshold[iclass])
                outBuffer[x/down]=m_class[iclass];
            }
          }
          else
            outBuffer[x/down]=(m_noDataValues.size())? m_noDataValues[0] : 0;
          break;
        }
        case(filter2d::scramble):{//could be done more efficiently window by window with random shuffling entire buffer and assigning entire buffer at once to output image...
          if(windowBuffer.size()){
            int randomIndex=std::rand()%windowBuffer.size();
            if(randomIndex>=windowBuffer.size())
              outBuffer[x/down]=windowBuffer.back();
            else if(randomIndex<0)
              outBuffer[x/down]=windowBuffer[0];
            else
              outBuffer[x/down]=windowBuffer[randomIndex];
          }
          else
            outBuffer[x/down]=(m_noDataValues.size())? m_noDataValues[0] : 0;
          break;
        }
        case(filter2d::mixed):{
          enum Type { BF=11, CF=12, MF=13, NF=20, W=30 };
          double nBF=occurrence[BF];
          double nCF=occurrence[CF];
          double nMF=occurrence[MF];
          double nNF=occurrence[NF];
          double nW=occurrence[W];
          if(windowBuffer.size()){
            if((nBF+nCF+nMF)&&(nBF+nCF+nMF>=nNF+nW)){//forest
              if(nBF/(nBF+nCF)>=0.75)
                outBuffer[x/down]=BF;
              else if(nCF/(nBF+nCF)>=0.75)
                outBuffer[x/down]=CF;
              else
                outBuffer[x/down]=MF;
            }
            else{//non-forest
              if(nW&&(nW>=nNF))
                outBuffer[x/down]=W;
              else
                outBuffer[x/down]=NF;
            }
          }
          else
            outBuffer[x/down]=inBuffer[indexJ][indexI];
          break;
        }
        default:{
          std::ostringstream ess;
          ess << "Error: filter method " << method << " not supported" << std::endl;
          throw(ess.str());
          break;
        }
        }
      }
      // progress=(1.0+y/down);
      // progress+=(output.nrOfRow()*iband);
      // progress/=output.nrOfBand()*output.nrOfRow();
      // MyProgressFunc(progress,pszMessage,pProgressArg);
      //write outBuffer to file
      try{
        output.writeData(outBuffer,y/down,iband);
      }
      catch(std::string errorstring){
        std::cerr << errorstring << "in band " << iband << ", line " << y << std::endl;
      }
    }
  }
  // MyProgressFunc(1.0,pszMessage,pProgressArg);
}

void filter2d::Filter2d::mrf(Jim& input, Jim& output, int dimX, int dimY, double beta, bool eightConnectivity, short down, bool verbose){
  assert(m_class.size()>1);
  Vector2d<double> fullBeta(m_class.size(),m_class.size());
  for(int iclass1=0;iclass1<m_class.size();++iclass1)
    for(int iclass2=0;iclass2<m_class.size();++iclass2)
      fullBeta[iclass1][iclass2]=beta;
  mrf(input,output,dimX,dimY,fullBeta,eightConnectivity,down,verbose);
}

//beta[classTo][classFrom]
void filter2d::Filter2d::mrf(Jim& input, Jim& output, int dimX, int dimY, Vector2d<double> beta, bool eightConnectivity, short down, bool verbose)
{
  if(!output.isInit())
    output.open(input);
  output.setNoData(m_noDataValues);

  const char* pszMessage;
  // void* pProgressArg=NULL;
  // GDALProgressFunc pfnProgress=GDALTermProgress;
  // double progress=0;
  // MyProgressFunc(progress,pszMessage,pProgressArg);

  assert(dimX);
  assert(dimY);

  Vector2d<short> inBuffer(dimY,input.nrOfCol());
  Vector2d<double> outBuffer(m_class.size(),(input.nrOfCol()+down-1)/down);
  assert(input.nrOfBand()==1);
  assert(output.nrOfBand()==m_class.size());
  assert(m_class.size()>1);
  assert(beta.size()==m_class.size());
  int indexI=0;
  int indexJ=0;
  //initialize last half of inBuffer
  for(int j=-(dimY-1)/2;j<=dimY/2;++j){
    try{
      input.readData(inBuffer[indexJ],static_cast<unsigned int>(abs(j)));
    }
    catch(std::string errorstring){
      std::cerr << errorstring << "in line " << indexJ << std::endl;
    }
    ++indexJ;
  }
  for(unsigned int y=0;y<input.nrOfRow();++y){
    if(y){//inBuffer already initialized for y=0
      //erase first line from inBuffer
      if(dimY>1)
        inBuffer.erase(inBuffer.begin());
      //read extra line and push back to inBuffer if not out of bounds
      if(y+dimY/2<input.nrOfRow()){
        //allocate buffer
        if(dimY>1)
          inBuffer.push_back(inBuffer.back());
        try{
          input.readData(inBuffer[inBuffer.size()-1],y+dimY/2);
        }
        catch(std::string errorstring){
          std::cerr << errorstring << "in line " << y << std::endl;
        }
      }
      else{
        int over=y+dimY/2-input.nrOfRow();
        int index=(inBuffer.size()-1)-over;
        assert(index>=0);
        assert(index<inBuffer.size());
        inBuffer.push_back(inBuffer[index]);
      }
    }
    if((y+1+down/2)%down)
      continue;
    for(unsigned int x=0;x<input.nrOfCol();++x){
      if((x+1+down/2)%down)
        continue;
      std::vector<short> potential(m_class.size());
      for(int iclass=0;iclass<m_class.size();++iclass){
        potential[iclass]=0;
        outBuffer[iclass][x/down]=0;
      }
      std::vector<double> windowBuffer;
      // int centre=dimX*(dimY-1)/2+(dimX-1)/2;
      for(int j=-(dimY-1)/2;j<=dimY/2;++j){
        for(int i=-(dimX-1)/2;i<=dimX/2;++i){
          if(i!=0&&j!=0&&!eightConnectivity)
            continue;
          if(i==0&&j==0)
            continue;
          indexI=x+i;
          //check if out of bounds
          if(indexI<0)
            indexI=-indexI;
          else if(indexI>=input.nrOfCol())
            indexI=input.nrOfCol()-i;
          if(y+j<0)
            indexJ=-j;
          else if(y+j>=input.nrOfRow())
            indexJ=(dimY>2) ? (dimY-1)/2-j : 0;
          else
            indexJ=(dimY-1)/2+j;
          bool masked=false;
          for(int imask=0;imask<m_noDataValues.size();++imask){
            if(inBuffer[indexJ][indexI]==m_noDataValues[imask]){
              masked=true;
              break;
            }
          }
          if(!masked){
            for(int iclass=0;iclass<m_class.size();++iclass){
              if(inBuffer[indexJ][indexI]==m_class[iclass])
                potential[iclass]+=1;
            }
          }
        }
      }
      double norm=0;
      for(int iclass1=0;iclass1<m_class.size();++iclass1){
        assert(beta[iclass1].size()==m_class.size());
        double pot=0;
        for(int iclass2=0;iclass2<m_class.size();++iclass2)
          if(iclass2!=iclass1)
            pot+=potential[iclass2]*beta[iclass1][iclass2];
        double prior=exp(-pot);
        outBuffer[iclass1][x/down]=prior;
        norm+=prior;
      }
      if(norm){
        for(int iclass1=0;iclass1<m_class.size();++iclass1)
          outBuffer[iclass1][x/down]/=norm;
      }
    }
    // progress=(1.0+y/down)/output.nrOfRow();
    // MyProgressFunc(progress,pszMessage,pProgressArg);
    //write outBuffer to file
    assert(outBuffer.size()==m_class.size());
    assert(y<output.nrOfRow());
    for(int iclass=0;iclass<m_class.size();++iclass){
      assert(outBuffer[iclass].size()==output.nrOfCol());
      try{
        output.writeData(outBuffer[iclass],y/down,iclass);
      }
      catch(std::string errorstring){
        std::cerr << errorstring << "in class " << iclass << ", line " << y << std::endl;
      }
    }
  }
}

//todo: perform on Jim directly instead of copying into Vector2d
void filter2d::Filter2d::shift(Jim& input, Jim& output, double offsetX, double offsetY, double randomSigma, RESAMPLE resample, bool verbose)
{
  if(!output.isInit())
    output.open(input);
  output.setNoData(m_noDataValues);

  assert(input.nrOfCol()==output.nrOfCol());
  assert(input.nrOfRow()==output.nrOfRow());
  assert(input.nrOfBand()==output.nrOfBand());
  const char* pszMessage;
  // void* pProgressArg=NULL;
  // GDALProgressFunc pfnProgress=GDALTermProgress;
  // double progress=0;
  // MyProgressFunc(progress,pszMessage,pProgressArg);
  //process band per band in memory
  Vector2d<double> inBuffer(input.nrOfRow(),output.nrOfCol());
  Vector2d<double> outBuffer(input.nrOfRow(),output.nrOfCol());
  for(int iband=0;iband<input.nrOfBand();++iband){
    input.readDataBlock(inBuffer,0,inBuffer.nCols()-1,0,inBuffer.nRows()-1,iband);
    shift(inBuffer,outBuffer,offsetX,offsetY,randomSigma,resample,verbose);
    output.writeDataBlock(outBuffer,0,outBuffer.nCols()-1,0,outBuffer.nRows()-1,iband);
  }
}

//todo: re-implement without dependency of CImg and reg libraries
// void filter2d::Filter2d::dwt_texture(const std::string& inputFilename, const std::string& outputFilename,int dim, int scale, int down, int iband, bool verbose)
// {
//   Jim input;
//   Jim output;
//   if(verbose)
//     std::cout << "opening file " << inputFilename << std::endl;
//   input.open(inputFilename);
//   double magicX=1,magicY=1;
//   output.open(outputFilename,(input.nrOfCol()+down-1)/down,(input.nrOfRow()+down-1)/down,scale*3,GDT_Float32,input.getImageType());
//   if(input.isGeoRef()){
//     output.setProjection(input.getProjection());
//     output.copyGeoTransform(input);
//   }
//   if(verbose)
//     std::cout << "Dimension texture (row x col x band) = " << (input.nrOfCol()+down-1)/down << " x " << (input.nrOfRow()+down-1)/down << " x " << scale*3 << std::endl;
//   assert(dim%2);
//   int dimX=dim;
//   int dimY=dim;
//   Vector2d<float> inBuffer(dimY,input.nrOfCol());
//   Vector2d<float> outBuffer(scale*3,(input.nrOfCol()+down-1)/down);
//   //initialize last half of inBuffer
//   int indexI=0;
//   int indexJ=0;
//   for(int j=-dimY/2;j<(dimY+1)/2;++j){
//     try{
//       if(verbose)
//  cout << "reading input line " << abs(j) << std::endl;
//       input.readData(inBuffer[indexJ],GDT_Float32,abs(j),iband);
//       ++indexJ;
//     }
//     catch(std::string errorstring){
//       std::cerr << errorstring << "in band " << iband << ", line " << indexJ << std::endl;
//     }
//   }
//   const char* pszMessage;
//   void* pProgressArg=NULL;
//   GDALProgressFunc pfnProgress=GDALTermProgress;
//   double progress=0;
//   MyProgressFunc(progress,pszMessage,pProgressArg);
//   for(int y=0;y<input.nrOfRow();y+=down){
//     if(verbose)
//       std::cout << "calculating line " << y/down << std::endl;
//     if(y){//inBuffer already initialized for y=0
//       //erase first line from inBuffer
//       inBuffer.erase(inBuffer.begin());
//       //read extra line and push back to inBuffer if not out of bounds
//       if(y+dimY/2<input.nrOfRow()){
//  //allocate buffer
//  inBuffer.push_back(inBuffer.back());
//  try{
//    if(verbose)
//      std::cout << "reading input line " << y+dimY/2 << std::endl;
//           input.readData(inBuffer[inBuffer.size()-1],GDT_Float32,y+dimY/2,iband);
//  }
//  catch(std::string errorstring){
//    std::cerr << errorstring << "in band " << iband << ", line " << y << std::endl;
//  }
//       }
//     }
//     for(int x=0;x<input.nrOfCol();x+=down){
//       Vector2d<double> texture_feature(scale,3);
//       CImg<> texture_in(dimX,dimY);
//       int r=0;//index for row of texture_in
//       for(int j=-dimY/2;j<(dimY+1)/2;++j){
//  int c=0;
//  for(int i=-dimX/2;i<(dimX+1)/2;++i){
//    indexI=x+i;
//    //check if out of bounds
//    if(indexI<0)
//      indexI=-indexI;
//    else if(indexI>=input.nrOfCol())
//      indexI=input.nrOfCol()-i;
//    if(y+j<0)
//      indexJ=-j;
//    else if(y+j>=input.nrOfRow())
//      indexJ=dimY/2-j;//indexJ=inBuffer.size()-1-j;
//    else
//      indexJ=dimY/2+j;
//    assert(indexJ<inBuffer.size());
//    assert(indexI<inBuffer[indexJ].size());
//    texture_in(r,c)=inBuffer[indexJ][indexI];
//    c++;
//  }
//  ++r;
//       }
//       texture_in.dwt_texture(texture_feature,scale);
//       for(int v=0;v<scale*3;++v)
//  outBuffer[v][x/down]=texture_feature[v/3][v%3];
//     }
//     //write outBuffer to file
//     try{
//       if(verbose)
//         std::cout << "writing line " << y/down << std::endl;
//       for(int v=0;v<scale*3;++v)
//         output.writeData(outBuffer[v],GDT_Float32,y/down,v);
//     }
//     catch(std::string errorstring){
//       std::cerr << errorstring << "in band " << iband << ", line " << y << std::endl;
//     }
//     progress=(1.0+y)/output.nrOfRow();
//     MyProgressFunc(progress,pszMessage,pProgressArg);
//   }
//   input.close();
//   output.close();
// }

void filter2d::Filter2d::morphology(Jim& input, Jim& output, const std::string& method, int dimX, int dimY, const std::vector<double> &angle, bool disc)
{
  if(!output.isInit())
    output.open(input);
  output.setNoData(m_noDataValues);

  const char* pszMessage;
  // void* pProgressArg=NULL;
  // GDALProgressFunc pfnProgress=GDALTermProgress;
  // double progress=0;
  // MyProgressFunc(progress,pszMessage,pProgressArg);

  assert(dimX);
  assert(dimY);

  statfactory::StatFactory stat;
#if JIPLIB_PROCESS_IN_PARALLEL == 1
#pragma omp parallel for
#else
#endif
  for(unsigned int iband=0;iband<input.nrOfBand();++iband){
    Vector2d<double> inBuffer(dimY,input.nrOfCol());
    std::vector<double> outBuffer(input.nrOfCol());
    int indexI=0;
    int indexJ=0;
    //initialize last half of inBuffer
    for(int j=-(dimY-1)/2;j<=dimY/2;++j){
      try{
        input.readData(inBuffer[indexJ],static_cast<unsigned int>(abs(j)),iband);
        ++indexJ;
      }
      catch(std::string errorstring){
        std::cerr << errorstring << "in line " << indexJ << std::endl;
      }
    }
    for(unsigned int y=0;y<input.nrOfRow();++y){
      if(y){//inBuffer already initialized for y=0
        //erase first line from inBuffer
        if(dimY>1)
          inBuffer.erase(inBuffer.begin());
        //read extra line and push back to inBuffer if not out of bounds
        if(y+dimY/2<input.nrOfRow()){
          //allocate buffer
          if(dimY>1)
            inBuffer.push_back(inBuffer.back());
          try{
            input.readData(inBuffer[inBuffer.size()-1],y+dimY/2,iband);
          }
          catch(std::string errorstring){
            std::cerr << errorstring << "in band " << iband << ", line " << y << std::endl;
          }
        }
        else{
          int over=y+dimY/2-input.nrOfRow();
          int index=(inBuffer.size()-1)-over;
          assert(index>=0);
          assert(index<inBuffer.size());
          inBuffer.push_back(inBuffer[index]);
        }
      }
      for(unsigned int x=0;x<input.nrOfCol();++x){
        double currentValue=inBuffer[(dimY-1)/2][x];
        outBuffer[x]=currentValue;
        std::vector<double> statBuffer;
        bool currentMasked=false;
        // int centre=dimX*(dimY-1)/2+(dimX-1)/2;
        for(int imask=0;imask<m_noDataValues.size();++imask){
          if(currentValue==m_noDataValues[imask]){
            currentMasked=true;
            break;
          }
        }
        if(currentMasked){
          outBuffer[x]=currentValue;
        }
        else{
          for(int j=-(dimY-1)/2;j<=dimY/2;++j){
            for(int i=-(dimX-1)/2;i<=dimX/2;++i){
              double d2=i*i+j*j;//square distance
              if(disc&&(d2>(dimX/2)*(dimY/2)))
                continue;
              if(angle.size()){
                double theta;
                //use polar coordinates in radians
                if(i>0){
                  if(j<0)
                    theta=atan(static_cast<double>(-j)/(static_cast<double>(i)));
                  else
                    theta=-atan(static_cast<double>(j)/(static_cast<double>(i)));
                }
                else if(i<0){
                  if(j<0)
                    theta=PI-atan(static_cast<double>(-j)/(static_cast<double>(-i)));
                  else
                    theta=PI+atan(static_cast<double>(j)/(static_cast<double>(-i)));
                }
                else if(j<0)
                  theta=PI/2.0;
                else if(j>0)
                  theta=3.0*PI/2.0;
                //convert to North (0), East (90), South (180), West (270) in degrees
                theta=360-(theta/PI*180)+90;
                if(theta<0)
                  theta+=360;
                while(theta>360)
                  theta-=360;
                bool alligned=false;
                for(int iangle=0;iangle<angle.size();++iangle){
                  if(sqrt((theta-angle[iangle])*(theta-angle[iangle]))<10){
                    alligned=true;
                    break;
                  }
                }
                if(!alligned)
                  continue;
              }
              indexI=x+i;
              //check if out of bounds
              if(indexI<0)
                indexI=-indexI;
              else if(indexI>=input.nrOfCol())
                indexI=input.nrOfCol()-i;
              if(y+j<0)
                indexJ=-j;
              else if(y+j>=input.nrOfRow())
                indexJ=(dimY>2) ? (dimY-1)/2-j : 0;
              else
                indexJ=(dimY-1)/2+j;
              //todo: introduce novalue as this: ?
              // if(inBuffer[indexJ][indexI]==(m_noDataValues.size())? m_noDataValues[0] : 0)
              //   continue;
              bool masked=false;
              for(int imask=0;imask<m_noDataValues.size();++imask){
                if(inBuffer[indexJ][indexI]==m_noDataValues[imask]){
                  masked=true;
                  break;
                }
              }
              if(!masked){
                short binValue=0;
                for(int iclass=0;iclass<m_class.size();++iclass){
                  if(inBuffer[indexJ][indexI]==m_class[iclass]){
                    binValue=1;
                    break;
                  }
                }
                if(m_class.size())
                  statBuffer.push_back(binValue);
                else
                  statBuffer.push_back(inBuffer[indexJ][indexI]);
              }
            }
          }
          if(statBuffer.size()){
            switch(getFilterType(method)){
            case(filter2d::dilate):
              outBuffer[x]=stat.mymax(statBuffer);
              break;
            case(filter2d::erode):
              outBuffer[x]=stat.mymin(statBuffer);
              break;
            default:
              std::ostringstream ess;
              ess << "Error:  morphology method " << method << " not supported, choose " << filter2d::dilate << " (dilate) or " << filter2d::erode << " (erode)" << std::endl;
              throw(ess.str());
              break;
            }
          }
          if(outBuffer[x]&&m_class.size())
            outBuffer[x]=m_class[0];
        }
      }
      //write outBuffer to file
      try{
        output.writeData(outBuffer,y,iband);
      }
      catch(std::string errorstring){
        std::cerr << errorstring << "in band " << iband << ", line " << y << std::endl;
      }
      // progress=(1.0+y);
      // progress+=(output.nrOfRow()*iband);
      // progress/=output.nrOfBand()*output.nrOfRow();
      // MyProgressFunc(progress,pszMessage,pProgressArg);
    }
  }
}

//todo: perform on Jim directly instead of copying into Vector2d
void filter2d::Filter2d::dsm2shadow(Jim& input, Jim& output, double sza, double saa, double pixelSize, short shadowFlag){
  if(!output.isInit())
    output.open(input);
  output.setNoData(m_noDataValues);

  Vector2d<float> inputBuffer;
  Vector2d<float> outputBuffer;
  input.readDataBlock(inputBuffer,  0, input.nrOfCol()-1, 0, input.nrOfRow()-1, 0);
  dsm2shadow(inputBuffer, outputBuffer, sza, saa, pixelSize, shadowFlag);
  output.writeDataBlock(outputBuffer,0,output.nrOfCol()-1,0,output.nrOfRow()-1,0);
}

void filter2d::Filter2d::dwtForward(Jim& input, Jim& output, const std::string& wavelet_type, int family){
  if(!output.isInit())
    output.open(input);
  output.setNoData(m_noDataValues);

  Vector2d<double> theBuffer;
  for(int iband=0;iband<input.nrOfBand();++iband){
    std::cout << "filtering band " << iband << std::endl << std::flush;
    input.readDataBlock(theBuffer,  0, input.nrOfCol()-1, 0, input.nrOfRow()-1, iband);
    dwtForward(theBuffer, wavelet_type, family);
    output.writeDataBlock(theBuffer,0,output.nrOfCol()-1,0,output.nrOfRow()-1,iband);
  }
}

void filter2d::Filter2d::dwtInverse(Jim& input, Jim& output, const std::string& wavelet_type, int family){
  if(!output.isInit())
    output.open(input);
  output.setNoData(m_noDataValues);

  Vector2d<float> theBuffer;
  for(int iband=0;iband<input.nrOfBand();++iband){
    input.readDataBlock(theBuffer,  0, input.nrOfCol()-1, 0, input.nrOfRow()-1, iband);
    std::cout << "filtering band " << iband << std::endl << std::flush;
    dwtInverse(theBuffer, wavelet_type, family);
    output.writeDataBlock(theBuffer,0,output.nrOfCol()-1,0,output.nrOfRow()-1,iband);
  }
}

void filter2d::Filter2d::dwtCut(Jim& input, Jim& output, const std::string& wavelet_type, int family, double cut, bool verbose){
  if(!output.isInit())
    output.open(input);
  output.setNoData(m_noDataValues);

  Vector2d<float> theBuffer;
  for(int iband=0;iband<input.nrOfBand();++iband){
    input.readDataBlock(theBuffer,  0, input.nrOfCol()-1, 0, input.nrOfRow()-1, iband);
    std::cout << "filtering band " << iband << std::endl << std::flush;
    dwtCut(theBuffer, wavelet_type, family, cut);
    output.writeDataBlock(theBuffer,0,output.nrOfCol()-1,0,output.nrOfRow()-1,iband);
  }
}

void filter2d::Filter2d::linearFeature(Jim& input, Jim& output, float angle, float angleStep, float maxDistance, float eps, bool l1, bool a1, bool l2, bool a2, int band, bool verbose){
  if(!output.isInit())
    output.open(input);
  output.setNoData(m_noDataValues);

  Vector2d<float> inputBuffer;
  std::vector< Vector2d<float> > outputBuffer;
  input.readDataBlock(inputBuffer,  0, input.nrOfCol()-1, 0, input.nrOfRow()-1, band);
  if(maxDistance<=0)
    maxDistance=sqrt(static_cast<float>(input.nrOfCol()*input.nrOfRow()));
  linearFeature(inputBuffer,outputBuffer,angle,angleStep,maxDistance,eps, l1, a1, l2, a2,verbose);
  for(int iband=0;iband<outputBuffer.size();++iband)
    output.writeDataBlock(outputBuffer[iband],0,output.nrOfCol()-1,0,output.nrOfRow()-1,iband);
}

void filter2d::Filter2d::linearFeature(const Vector2d<float>& input, std::vector< Vector2d<float> >& output, float angle, float angleStep, float maxDistance, float eps, bool l1, bool a1, bool l2, bool a2, bool verbose)
{
  output.clear();
  int nband=0;//linear feature
  if(l1)
    ++nband;
  if(a1)
    ++nband;
  if(l2)
    ++nband;
  if(a2)
    ++nband;
  output.resize(nband);
  for(int iband=0;iband<output.size();++iband)
    output[iband].resize(input.nRows(),input.nCols());
  if(maxDistance<=0)
    maxDistance=sqrt(static_cast<float>(input.nRows()*input.nCols()));
  int indexI=0;
  int indexJ=0;
  const char* pszMessage;
  // void* pProgressArg=NULL;
  // GDALProgressFunc pfnProgress=GDALTermProgress;
  // double progress=0;
  // MyProgressFunc(progress,pszMessage,pProgressArg);
  for(unsigned int y=0;y<input.nRows();++y){
    for(unsigned int x=0;x<input.nCols();++x){
      float currentValue=input[y][x];
      //find values equal to current value with some error margin
      //todo: add distance for two opposite directions
      float lineDistance1=0;//longest line of object
      float lineDistance2=maxDistance;//shortest line of object
      float lineAngle1=0;//angle to longest line (North=0)
      float lineAngle2=0;//angle to shortest line (North=0)
      float northAngle=0;//rotating angle
      for(northAngle=0;northAngle<180;northAngle+=angleStep){
        if(angle<=360&&angle>=0&&angle!=northAngle)
          continue;
        float currentDistance=0;
        float theDir=0;
        for(short side=0;side<=1;side+=1){
          theDir=PI/2.0-DEG2RAD(northAngle)+side*PI;//in radians
          if(theDir<0)
            theDir+=2*PI;
          float nextValue=currentValue;
          for(float currentRay=1;currentRay<maxDistance;++currentRay){
            indexI=x+currentRay*cos(theDir);
            indexJ=y-currentRay*sin(theDir);
            if(indexJ<0||indexJ>=input.size())
              break;
            if(indexI<0||indexI>=input[indexJ].size())
              break;
            nextValue=input[indexJ][indexI];
            if(verbose){
              std::cout << "x: " << x << std::endl;
              std::cout << "y: " << y << std::endl;
              std::cout << "currentValue: " << currentValue << std::endl;
              std::cout << "theDir in degrees: " << RAD2DEG(theDir) << std::endl;
              std::cout << "cos(theDir): " << cos(theDir) << std::endl;
              std::cout << "sin(theDir): " << sin(theDir) << std::endl;
              std::cout << "currentRay: " << currentRay << std::endl;
              std::cout << "currentDistance: " << currentDistance << std::endl;
              std::cout << "indexI: " << indexI << std::endl;
              std::cout << "indexJ: " << indexJ << std::endl;
              std::cout << "nextValue: " << nextValue << std::endl;
            }
            if(fabs(currentValue-nextValue)<=eps){
              ++currentDistance;
            }
            else{
              break;
            }
          }
        }
        if(lineDistance1<currentDistance){
          lineDistance1=currentDistance;
          lineAngle1=northAngle;
        }
        if(lineDistance2>currentDistance){
          lineDistance2=currentDistance;
          lineAngle2=northAngle;
        }
        if(verbose){
          std::cout << "lineDistance1: " << lineDistance1 << std::endl;
          std::cout << "lineAngle1: " << lineAngle1 << std::endl;
          std::cout << "lineDistance2: " << lineDistance2 << std::endl;
          std::cout << "lineAngle2: " << lineAngle2 << std::endl;
        }
      }
      int iband=0;
      if(l1)
        output[iband++][y][x]=lineDistance1;
      if(a1)
        output[iband++][y][x]=lineAngle1;
      if(l2)
        output[iband++][y][x]=lineDistance2;
      if(a2)
        output[iband++][y][x]=lineAngle2;
      assert(iband==nband);
    }
    // progress=(1.0+y);
    // progress/=input.nRows();
    // MyProgressFunc(progress,pszMessage,pProgressArg);
  }
}
