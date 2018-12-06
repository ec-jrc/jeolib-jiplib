/**********************************************************************
StatFactory.h: class for statistical operations on vectors
Author(s): Pieter.Kempeneers@ec.europa.eu
Copyright (c) 2016-2018 European Union (Joint Research Centre)
License EUPLv1.2

This file is part of jiplib
***********************************************************************/
#ifndef _STATFACTORY_H_
#define _STATFACTORY_H_

#include <iostream>
#include <vector>
#include <map>
#include <math.h>
#include <assert.h>
#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <gsl/gsl_fit.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_sort.h>

namespace statfactory
{

  class StatFactory{

  public:
    enum INTERPOLATION_TYPE {undefined=0,linear=1,polynomial=2,cspline=3,cspline_periodic=4,akima=5,akima_periodic=6};
    //todo: expand with other distributions as in http://www.gnu.org/software/gsl/manual/gsl-ref_toc.html#TOC320
    enum DISTRIBUTION_TYPE {none=0,uniform=1,gaussian=2};

    StatFactory(void){};
    virtual ~StatFactory(void){};
    INTERPOLATION_TYPE getInterpolationType(const std::string interpolationType){
      std::map<std::string, INTERPOLATION_TYPE> m_interpMap;
      initMap(m_interpMap);
      return m_interpMap[interpolationType];
    };
    DISTRIBUTION_TYPE getDistributionType(const std::string distributionType){
      std::map<std::string, DISTRIBUTION_TYPE> m_distMap;
      initDist(m_distMap);
      return m_distMap[distributionType];
    };

    static void allocAcc(gsl_interp_accel *&acc){
      acc = gsl_interp_accel_alloc ();
    };

    static void getSpline(const std::string type, int size, gsl_spline *& spline){
      std::map<std::string, INTERPOLATION_TYPE> m_interpMap;
      initMap(m_interpMap);
      switch(m_interpMap[type]){
      case(polynomial):
        spline=gsl_spline_alloc(gsl_interp_polynomial,size);
        break;
      case(cspline):
        spline=gsl_spline_alloc(gsl_interp_cspline,size);
        break;
      case(cspline_periodic):
        spline=gsl_spline_alloc(gsl_interp_cspline_periodic,size);
        break;
      case(akima):
        spline=gsl_spline_alloc(gsl_interp_akima,size);
        break;
      case(akima_periodic):
        spline=gsl_spline_alloc(gsl_interp_akima_periodic,size);
        break;
      case(linear):
      default:
        spline=gsl_spline_alloc(gsl_interp_linear,size);
        break;
      }
      assert(spline);
    };
    static int initSpline(gsl_spline *spline, const double *x, const double *y, int size){
      return gsl_spline_init (spline, x, y, size);
    };
    static double evalSpline(gsl_spline *spline, double x, gsl_interp_accel *acc){
      return gsl_spline_eval (spline, x, acc);
    };

    static gsl_rng* getRandomGenerator(unsigned long int theSeed){
      gsl_rng * r;
      // select random number generator
      r = gsl_rng_alloc (gsl_rng_mt19937);
      gsl_rng_set(r,theSeed);
      return r;
    }
    static void freeRandomGenerator(gsl_rng* r){
      // release random number generator
      gsl_rng_free(r);
    }
    void getNodataValues(std::vector<double>& nodatav) const{nodatav=m_noDataValues;};
    bool isNoData(const double& value) const{
      if(m_noDataValues.empty())
        return false;
      else
        return find(m_noDataValues.begin(),m_noDataValues.end(),value)!=m_noDataValues.end();
    };
    unsigned int pushNodDataValue(double noDataValue){
      if(find(m_noDataValues.begin(),m_noDataValues.end(),noDataValue)==m_noDataValues.end())
        m_noDataValues.push_back(noDataValue);
      return m_noDataValues.size();
    };
    unsigned int clearNoDataValues(){
      m_noDataValues.clear();
      return m_noDataValues.size();
    };
    unsigned int setNoDataValues(std::vector<double> vnodata){
      m_noDataValues=vnodata;
      return m_noDataValues.size();
    };
    double getRandomValue(const gsl_rng* r, const std::string type, double a=0, double b=1) const{
      std::map<std::string, DISTRIBUTION_TYPE> m_distMap;
      initDist(m_distMap);
      double randValue=0;
      switch(m_distMap[type]){
      case(none):
        randValue = a;
        break;
      case(uniform):
        randValue = a+(b-a)*gsl_rng_uniform(r);
        break;
      case(gaussian):
        randValue = a+gsl_ran_gaussian(r, b);
        break;
      default:
        std::string errorString="Error: distribution not valid";
        throw(errorString);
        break;
      }
      return randValue;
    };

    template<class T> T mymin(const typename std::vector<T>& v) const;
    template<class T> T mymax(const typename std::vector<T>& v) const;
    template<class T> T mymin(const typename std::vector<T>& v, T minConstraint) const;
    template<class T> T mymax(const typename std::vector<T>& v, T maxConstraint) const;
    template<class T> T absmin(const typename std::vector<T>& v) const;
    template<class T> T absmax(const typename std::vector<T>& v) const;

    //   template<class T> typename std::vector<T>::const_iterator mymax(const std::vector<T>& v, typename std::vector<T>::const_iterator begin, typename std::vector<T>::const_iterator end) const;
    template<class T> typename std::vector<T>::const_iterator mymin(const typename std::vector<T>& v, typename std::vector<T>::const_iterator begin, typename std::vector<T>::const_iterator end) const;
    template<class T> typename std::vector<T>::iterator mymin(const typename std::vector<T>& v, typename std::vector<T>::iterator begin, typename std::vector<T>::iterator end) const;
    template<class T> typename std::vector<T>::const_iterator mymin(const typename std::vector<T>& v, typename std::vector<T>::const_iterator begin, typename std::vector<T>::const_iterator end, T minConstraint) const;
    template<class T> typename std::vector<T>::iterator mymin(const typename std::vector<T>& v, typename std::vector<T>::iterator begin, typename std::vector<T>::iterator end, T minConstraint) const;
    template<class T> typename std::vector<T>::const_iterator mymax(const std::vector<T>& v, typename std::vector<T>::const_iterator begin, typename std::vector<T>::const_iterator end) const;
    template<class T> typename std::vector<T>::iterator mymax(const std::vector<T>& v, typename std::vector<T>::iterator begin, typename std::vector<T>::iterator end) const;
    template<class T> typename std::vector<T>::const_iterator mymax(const std::vector<T>& v, typename std::vector<T>::const_iterator begin, typename std::vector<T>::const_iterator end, T maxConstraint) const;
    template<class T> typename std::vector<T>::iterator mymax(const std::vector<T>& v, typename std::vector<T>::iterator begin, typename std::vector<T>::iterator end, T maxConstraint) const;
    template<class T> typename std::vector<T>::const_iterator absmin(const std::vector<T>& v, typename std::vector<T>::const_iterator begin, typename std::vector<T>::const_iterator end) const;
    template<class T> typename std::vector<T>::const_iterator absmax(const std::vector<T>& v, typename std::vector<T>::const_iterator begin, typename std::vector<T>::const_iterator end) const;

    template<class T> void minmax(const std::vector<T>& v, typename std::vector<T>::const_iterator begin, typename std::vector<T>::const_iterator end, T& theMin, T& theMax) const;
    template<class T> T sum(const std::vector<T>& v) const;
    template<class T> double mean(const std::vector<T>& v) const;
    template<class T> void eraseNoData(std::vector<T>& v) const;
    void eraseBelow(std::vector<double>& v, double min) const;
    void eraseAbove(std::vector<double>& v, double max) const;
    void setThreshold(std::vector<double>& v, double min, double max) const;
    template<class T> unsigned int nvalid(const std::vector<T>& v) const;
    template<class T> T median(const std::vector<T>& v) const;
    template<class T> double var(const std::vector<T>& v) const;
    template<class T> double moment(const std::vector<T>& v, int n) const;
    template<class T> double cmoment(const std::vector<T>& v, int n) const;
    template<class T> double skewness(const std::vector<T>& v) const;
    template<class T> double kurtosis(const std::vector<T>& v) const;
    template<class T> void meanVar(const std::vector<T>& v, double& m1, double& v1) const;
    template<class T1, class T2> void  scale2byte(const std::vector<T1>& input, std::vector<T2>& output, unsigned char lbound=0, unsigned char ubound=255) const;
    template<class T> void distribution(const std::vector<T>& input, typename std::vector<T>::const_iterator begin, typename std::vector<T>::const_iterator end,  std::vector<double>& output, int nbin, T &minimum, T &maximum, double sigma=0, const std::string &filename="") const;
    template<class T> void distribution(const std::vector<T>& input,  std::vector<double>& output, int nbin, double sigma=0, const std::string &filename="") const{distribution(input,input.begin(),input.end(),output,nbin,0,0,sigma,filename);};
    template<class T> void  distribution2d(const std::vector<T>& inputX, const std::vector<T>& inputY, std::vector< std::vector<double> >& output, int nbin, T& minX, T& maxX, T& minY, T& maxY, double sigma=0, const std::string& filename="") const;
    template<class T> void cumulative (const std::vector<T>& input, typename std::vector<T>::const_iterator begin, typename std::vector<T>::const_iterator end, std::vector<int>& output, int nbin, T &minimum, T &maximum) const;
    template<class T> void  percentiles (const std::vector<T>& input, typename std::vector<T>::const_iterator begin, typename std::vector<T>::const_iterator end, std::vector<T>& output, int nbin, T &minimum, T &maximum, const std::string &filename="") const;
    template<class T> T percentile(const std::vector<T>& input, typename std::vector<T>::const_iterator begin, typename std::vector<T>::const_iterator end, double percent, T minimum=0, T maximum=0) const;
    template<class T> void signature(const std::vector<T>& input, double& k, double& alpha, double& beta, double e) const;
    void signature(double m1, double m2, double& k, double& alpha, double& beta, double e) const;
    template<class T> void normalize(const std::vector<T>& input, std::vector<double>& output) const;
    template<class T> void normalize_pct(std::vector<T>& input) const;
    template<class T> double rmse(const std::vector<T>& x, const std::vector<T>& y) const;
    template<class T> double nrmse(const std::vector<T>& x, const std::vector<T>& y) const;
    template<class T> double cvrmse(const std::vector<T>& x, const std::vector<T>& y) const;
    template<class T> double correlation(const std::vector<T>& x, const std::vector<T>& y, int delay=0) const;
    //  template<class T> double gsl_correlation(const std::vector<T>& x, const std::vector<T>& y) const;
    template<class T> double gsl_covariance(const std::vector<T>& x, const std::vector<T>& y) const;
    template<class T> double cross_correlation(const std::vector<T>& x, const std::vector<T>& y, int maxdelay, std::vector<T>& z) const;
    template<class T> double linear_regression(const std::vector<T>& x, const std::vector<T>& y, double &c0, double &c1) const;
    template<class T> double linear_regression_err(const std::vector<T>& x, const std::vector<T>& y, double &c0, double &c1) const;
    template<class T> void interpolateNoData(const std::vector<double>& wavelengthIn, const std::vector<T>& input, const std::string& type, std::vector<T>& output, bool verbose=false) const;
    template<class T> void interpolateUp(const std::vector<double>& wavelengthIn, const std::vector<T>& input, const std::vector<double>& wavelengthOut, const std::string& type, std::vector<T>& output, bool verbose=false) const;
    template<class T> void interpolateUp(const std::vector<double>& wavelengthIn, const std::vector< std::vector<T> >& input, const std::vector<double>& wavelengthOut, const std::string& type, std::vector< std::vector<T> >& output, bool verbose=false) const;
    // template<class T> void interpolateUp(const std::vector< std::vector<T> >& input, std::vector< std::vector<T> >& output, double start, double end, double step, const gsl_interp_type* type);
    // template<class T> void interpolateUp(const std::vector< std::vector<T> >& input, const std::vector<double>& wavelengthIn, std::vector< std::vector<T> >& output, std::vector<double>& wavelengthOut, double start, double end, double step, const gsl_interp_type* type);
    template<class T> void interpolateUp(const std::vector<T>& input, std::vector<T>& output, int nbin) const;
    template<class T> void nearUp(const std::vector<T>& input, std::vector<T>& output) const;
    template<class T> void interpolateUp(double* input, int dim, std::vector<T>& output, int nbin);
    template<class T> void interpolateDown(const std::vector<T>& input, std::vector<T>& output, int nbin) const;
    template<class T> void interpolateDown(double* input, int dim, std::vector<T>& output, int nbin);

  private:
    static void initMap(std::map<std::string, INTERPOLATION_TYPE>& m_interpMap){
      //initialize selMap
      m_interpMap["linear"]=linear;
      m_interpMap["polynomial"]=polynomial;
      m_interpMap["cspline"]=cspline;
      m_interpMap["cspline_periodic"]=cspline_periodic;
      m_interpMap["akima"]=akima;
      m_interpMap["akima_periodic"]=akima_periodic;
    }
    static void initDist(std::map<std::string, DISTRIBUTION_TYPE>& m_distMap){
      //initialize distMap
      m_distMap["gaussian"]=gaussian;
      m_distMap["uniform"]=uniform;
      m_distMap["none"]=none;
    }
    std::vector<double> m_noDataValues;
  };


  template<class T> inline typename std::vector<T>::const_iterator StatFactory::mymin(const std::vector<T>& v, typename std::vector<T>::const_iterator begin, typename std::vector<T>::const_iterator end) const
    {
      bool isValid=false;
      typename std::vector<T>::const_iterator tmpIt;
      for(typename std::vector<T>::const_iterator it = begin; it!=end; ++it){
        if(!isNoData(*it)){
          if(isValid){
            if(*tmpIt>*it)
              tmpIt=it;
          }
          else{
            tmpIt=it;
            isValid=true;
          }
        }
      }
      if(isValid)
        return tmpIt;
      else if(m_noDataValues.size())
        return m_noDataValues[0];
      else{
        std::string errorString="Error: no valid data found";
        throw(errorString);
      }
    }

  template<class T> inline typename std::vector<T>::iterator StatFactory::mymin(const std::vector<T>& v, typename std::vector<T>::iterator begin, typename std::vector<T>::iterator end) const
    {
      bool isValid=false;
      typename std::vector<T>::iterator tmpIt;
      for (typename std::vector<T>::const_iterator it = begin; it!=end; ++it){
        if(!isNoData(*it)){
          if(isValid){
            if(*tmpIt>*it)
              tmpIt=it;
          }
          else{
            tmpIt=it;
            isValid=true;
          }
        }
      }
      if(isValid)
        return tmpIt;
      else if(m_noDataValues.size())
        return m_noDataValues[0];
      else{
        std::string errorString="Error: no valid data found";
        throw(errorString);
      }
    }

  template<class T> inline  typename std::vector<T>::const_iterator StatFactory::mymin(const std::vector<T>& v, typename std::vector<T>::const_iterator begin, typename std::vector<T>::const_iterator end, T minConstraint) const
    {
      bool isValid=false;
      typename std::vector<T>::const_iterator tmpIt;
      T minValue=minConstraint;
      for (typename std::vector<T>::const_iterator it = begin; it!=end; ++it){
        if(isNoData(*it))
          continue;
        if(isValid){
          if((minConstraint<=*it)&&(*it<minValue)){
            tmpIt=it;
            minValue=*it;
          }
        }
        else{
          if(*it<minValue)
            continue;
          tmpIt=it;
          minValue=*it;
          isValid=true;
        }
      }
      if(isValid)
        return tmpIt;
      else if(m_noDataValues.size())
        return m_noDataValues[0];
      else{
        std::string errorString="Error: no valid data found";
        throw(errorString);
      }
    }

  template<class T> inline typename std::vector<T>::iterator StatFactory::mymin(const std::vector<T>& v, typename std::vector<T>::iterator begin, typename std::vector<T>::iterator end, T minConstraint) const
    {
      bool isValid=false;
      typename std::vector<T>::iterator tmpIt;
      T minValue=minConstraint;
      for (typename std::vector<T>::const_iterator it = begin; it!=end; ++it){
        if(isNoData(*it))
          continue;
        if(isValid){
          if((minConstraint<=*it)&&(*it<minValue)){
            tmpIt=it;
            minValue=*it;
          }
        }
        else{
          if(*it<minConstraint)
            continue;
          tmpIt=it;
          minValue=*it;
          isValid=true;
        }
      }
      if(isValid)
        return tmpIt;
      else if(m_noDataValues.size())
        return m_noDataValues[0];
      else{
        std::string errorString="Error: no valid data found";
        throw(errorString);
      }
    }

  template<class T> inline typename std::vector<T>::const_iterator StatFactory::mymax(const std::vector<T>& v, typename std::vector<T>::const_iterator begin, typename std::vector<T>::const_iterator end) const
    {
      bool isValid=false;
      typename std::vector<T>::const_iterator tmpIt;
      for (typename std::vector<T>::iterator it = begin; it!=end; ++it){
        if(isNoData(*it))
          continue;
        if(isValid){
          if(*tmpIt<*it)
            tmpIt=it;
        }
        else{
          tmpIt=it;
          isValid=true;
        }
      }
      if(isValid)
        return tmpIt;
      else if(m_noDataValues.size())
        return m_noDataValues[0];
      else{
        std::string errorString="Error: no valid data found";
        throw(errorString);
      }
    }

  template<class T> inline typename std::vector<T>::iterator StatFactory::mymax(const std::vector<T>& v, typename std::vector<T>::iterator begin, typename std::vector<T>::iterator end) const
    {
      bool isValid=false;
      typename std::vector<T>::iterator tmpIt;
      for (typename std::vector<T>::iterator it = begin; it!=end; ++it){
        if(isNoData(*it))
          continue;
        if(isValid){
          if(*tmpIt<*it)
            tmpIt=it;
        }
        else{
          tmpIt=it;
          isValid=true;
        }
      }
      if(isValid)
        return tmpIt;
      else
        return end;
    }

  template<class T> inline typename std::vector<T>::const_iterator StatFactory::mymax(const std::vector<T>& v, typename std::vector<T>::const_iterator begin, typename std::vector<T>::const_iterator end, T maxConstraint) const
    {
      bool isValid=false;
      typename std::vector<T>::const_iterator tmpIt;
      T maxValue=maxConstraint;
      for (typename std::vector<T>::const_iterator it = begin; it!=end; ++it){
        if(isNoData(*it))
          continue;
        if(isValid){
          if((maxConstraint>=*it)&&(*it>maxValue)){
            tmpIt=it;
            maxValue=*it;
          }
        }
        else{
          if(*it>maxConstraint)
            continue;
          tmpIt=it;
          maxValue=*it;
          isValid=true;
        }
      }
      if(isValid)
        return tmpIt;
      else
        return end;
    }

  template<class T> inline typename std::vector<T>::iterator StatFactory::mymax(const std::vector<T>& v, typename std::vector<T>::iterator begin, typename std::vector<T>::iterator end, T maxConstraint) const
    {
      bool isValid=false;
      typename std::vector<T>::iterator tmpIt=v.end();
      T maxValue=maxConstraint;
      for (typename std::vector<T>::const_iterator it = begin; it!=end; ++it){
        if(isNoData(*it))
          continue;
        if(isValid){
          if((maxConstraint>=*it)&&(*it>maxValue)){
            tmpIt=it;
            maxValue=*it;
          }
        }
        else{
          if(*it>maxValue)
            continue;
          tmpIt=it;
          maxValue=*it;
          isValid=true;
        }
      }
      if(isValid)
        return tmpIt;
      else
        return end;
    }

  template<class T> inline T StatFactory::mymin(const std::vector<T>& v) const
    {
      bool isValid=false;
      if(v.empty()){
        std::string errorString="Error: vector is empty";
        throw(errorString);
      }
      T minValue;
      for (typename std::vector<T>::const_iterator it = v.begin(); it!=v.end(); ++it){
        if(isNoData(*it))
          continue;
        if(isValid){
          if(minValue>*it)
            minValue=*it;
        }
        else{
          minValue=*it;
          isValid=true;
        }
      }
      if(isValid)
        return minValue;
      else if(m_noDataValues.size())
        return m_noDataValues[0];
      else{
        std::string errorString="Error: no valid data found";
        throw(errorString);
      }
    }

  template<class T> inline T StatFactory::mymin(const std::vector<T>& v, T minConstraint) const
    {
      bool isValid=false;
      T minValue=minConstraint;
      for (typename std::vector<T>::const_iterator it = v.begin(); it!=v.end(); ++it){
        if(isNoData(*it))
          continue;
        if(isValid){
          if((minConstraint<=*it)&&(*it<minValue))
            minValue=*it;
        }
        else{
          if(*it<minValue)
            continue;
          minValue=*it;
          isValid=true;
        }
      }
      if(isValid)
        return minValue;
      else if(m_noDataValues.size())
        return m_noDataValues[0];
      else{
        std::string errorString="Error: no valid data found";
        throw(errorString);
      }
    }

  template<class T> inline T StatFactory::mymax(const std::vector<T>& v) const
    {
      bool isValid=false;
      if(v.empty()){
        std::string errorString="Error: vector is empty";
        throw(errorString);
      }
      T maxValue;
      for (typename std::vector<T>::const_iterator it = v.begin(); it!=v.end(); ++it){
        if(isNoData(*it))
          continue;
        if(isValid){
          if(maxValue<*it)
            maxValue=*it;
        }
        else{
          maxValue=*it;
          isValid=true;
        }
      }
      if(isValid)
        return maxValue;
      else if(m_noDataValues.size())
        return m_noDataValues[0];
      else{
        std::string errorString="Error: no valid data found";
        throw(errorString);
      }
    }

  template<class T> inline T StatFactory::mymax(const std::vector<T>& v, T maxConstraint) const
    {
      bool isValid=false;
      T maxValue=maxConstraint;
      for (typename std::vector<T>::const_iterator it = v.begin(); it!=v.end(); ++it){
        if(isNoData(*it))
          continue;
        if(isValid){
          if((*it<=maxConstraint)&&(*it>maxValue))
            maxValue=*it;
        }
        else{
          if(*it>maxValue)
            continue;
          maxValue=*it;
          isValid=true;
        }
      }
      if(isValid)
        return maxValue;
      else if(m_noDataValues.size())
        return m_noDataValues[0];
      else{
        std::string errorString="Error: no valid data found";
        throw(errorString);
      }
    }

  template<class T> inline typename std::vector<T>::const_iterator StatFactory::absmin(const std::vector<T>& v, typename std::vector<T>::const_iterator begin, typename std::vector<T>::const_iterator end) const
    {
      bool isValid=false;
      if(v.empty()){
        std::string errorString="Error: vector is empty";
        throw(errorString);
      }
      typename std::vector<T>::const_iterator tmpIt;
      for (typename std::vector<T>::const_iterator it = begin; it!=end; ++it){
        if(isNoData(*it))
          continue;
        if(isValid){
          if(abs(*tmpIt)>abs(*it))
            tmpIt=it;
        }
        else{
          tmpIt=it;
          isValid=true;
        }
      }
      if(isValid)
        return tmpIt;
      else
        return end;
    }

  template<class T> inline T StatFactory::absmin(const std::vector<T>& v) const
    {
      typename std::vector<T>::const_iterator tmpIt;
      tmpIt=absmin(v, v.begin(), v.end());
      if(tmpIt!=v.end())
        return *tmpIt;
      else if(m_noDataValues.size())
        return m_noDataValues[0];
      else{
        std::string errorString="Error: no valid data found";
        throw(errorString);
      }
    }

  template<class T> inline typename std::vector<T>::const_iterator StatFactory::absmax(const std::vector<T>& v, typename std::vector<T>::const_iterator begin, typename std::vector<T>::const_iterator end) const
    {
      bool isValid=false;
      typename std::vector<T>::const_iterator tmpIt;
      for (typename std::vector<T>::const_iterator it = begin; it!=end; ++it){
        if(isNoData(*it))
          continue;
        if(isValid){
          if(abs(*tmpIt)<abs(*it))
            tmpIt=it;
        }
        else{
          tmpIt=it;
          isValid=true;
        }
      }
      if(isValid)
        return tmpIt;
      else
        return end;
    }

  template<class T> inline T StatFactory::absmax(const std::vector<T>& v) const
    {
      typename std::vector<T>::const_iterator tmpIt;
      tmpIt=absmax(v, v.begin(), v.end());
      if(tmpIt!=v.end())
        return *tmpIt;
      else if(m_noDataValues.size())
        return m_noDataValues[0];
      else{
        std::string errorString="Error: no valid data found";
        throw(errorString);
      }
    }

  template<class T> inline void StatFactory::minmax(const std::vector<T>& v, typename std::vector<T>::const_iterator begin, typename std::vector<T>::const_iterator end, T& theMin, T& theMax) const
    {
      bool isConstraint=(theMax>theMin);
      double minConstraint=theMin;
      double maxConstraint=theMax;
      bool isValid=false;
      for (typename std::vector<T>::const_iterator it = begin; it!=end; ++it){
        if(isNoData(*it))
          continue;
        if(isValid){
          if(isConstraint){
            if(*it<minConstraint)
              continue;
            if(*it>maxConstraint)
              continue;
          }
          if(*it<theMin)
            theMin=*it;
          if(*it>theMax)
            theMax=*it;
        }
        else{
          if(isConstraint){
            if(*it<minConstraint)
              continue;
            if(*it>maxConstraint)
              continue;
          }
          theMin=*it;
          theMax=*it;
          isValid=true;
        }
      }
      if(!isValid){
        if(m_noDataValues.size()){
          theMin=m_noDataValues[0];
          theMax=m_noDataValues[0];
        }
        else{
          std::string errorString="Error: no valid data found";
          throw(errorString);
        }
      }
    }

  template<class T> inline T StatFactory::sum(const std::vector<T>& v) const
    {
      bool isValid=false;
      typename std::vector<T>::const_iterator it;
      T tmpSum=0;
      for (it = v.begin(); it!= v.end(); ++it){
        if(isNoData(*it))
          continue;
        isValid=true;
        tmpSum+=*it;
      }
      if(isValid)
        return tmpSum;
      else if(m_noDataValues.size())
        return m_noDataValues[0];
      else{
        std::string errorString="Error: no valid data found";
        throw(errorString);
      }
    }

  template<class T> inline double StatFactory::mean(const std::vector<T>& v) const
    {
      typename std::vector<T>::const_iterator it;
      T tmpSum=0;
      unsigned int validSize=0;
      for (it = v.begin(); it!= v.end(); ++it){
        if(isNoData(*it))
          continue;
        ++validSize;
        tmpSum+=*it;
      }
      if(validSize)
        return static_cast<double>(tmpSum)/validSize;
      else if(m_noDataValues.size())
        return m_noDataValues[0];
      else{
        std::string errorString="Error: no valid data found";
        throw(errorString);
      }
    }

  struct isAbove
  {
  isAbove(double maxValue) : m_max(maxValue) {}
    double m_max;
    bool operator()(double value)
    {
      return value>m_max;
    }
  };

  struct isBelow
  {
  isBelow(double minValue) : m_min(minValue) {}
    double m_min;
    bool operator()(double value)
    {
      return value<m_min;
    }
  };

  struct notWithin
  {
  notWithin(double minValue, double maxValue) : m_min(minValue), m_max(maxValue) {}
    double m_min;
    double m_max;
    bool operator()(double value)
    {
      return value<m_min||value>m_max;
    }
  };

 //template<class T> inline void StatFactory::setThreshold(std::vector<T>& v, double min, T double) const
 inline void StatFactory::setThreshold(std::vector<double>& v, double min, double max) const
 {
   //apply erase-remove idiom (see https://en.wikipedia.org/wiki/Erase%E2%80%93remove_idiom)
   v.erase( std::remove_if(v.begin(), v.end(), notWithin(min,max)), v.end() );
 }

 //template<class T> inline void StatFactory::setThreshold(std::vector<T>& v, double min, T double) const
 inline void StatFactory::eraseAbove(std::vector<double>& v, double max) const
 {
   //apply erase-remove idiom (see https://en.wikipedia.org/wiki/Erase%E2%80%93remove_idiom)
   v.erase( std::remove_if(v.begin(), v.end(), isAbove(max)), v.end() );
 }

//template<class T> inline void StatFactory::setThreshold(std::vector<T>& v, double min, T double) const
 inline void StatFactory::eraseBelow(std::vector<double>& v, double min) const
   {
     //apply erase-remove idiom (see https://en.wikipedia.org/wiki/Erase%E2%80%93remove_idiom)
     v.erase( std::remove_if(v.begin(), v.end(), isBelow(min)), v.end() );
   }

 template<class T> inline void StatFactory::eraseNoData(std::vector<T>& v) const
   {
     //apply erase-remove idiom (see https://en.wikipedia.org/wiki/Erase%E2%80%93remove_idiom)
     std::vector<double>::const_iterator vit=m_noDataValues.begin();
     for(vit=m_noDataValues.begin();vit!=m_noDataValues.end();++vit)
       v.erase( std::remove(v.begin(), v.end(), *vit), v.end() );
   }

  /* template<class T> inline void StatFactory::eraseNoData(std::vector<T>& v) const */
  /*   { */
  /*     if(m_noDataValues.size()){ */
  /*       typename std::vector<T>::iterator it=v.begin(); */
  /*       while(it!=v.end()){ */
  /*         if(isNoData(*it)) */
  /*           v.erase(it); */
  /*         else */
  /*           ++it; */
  /*       } */
  /*     } */
  /*   } */

  template<class T> unsigned int StatFactory::nvalid(const std::vector<T>& v) const{
    std::vector<T> tmpV=v;
    eraseNoData(tmpV);
    return(tmpV.size());
  }

  template<class T> T StatFactory::median(const std::vector<T>& v) const
    {
      if(v.empty()){
        std::ostringstream s;
        s<<"Error: input is empty";
        throw(s.str());
      }
      std::vector<double> inputSort;
      inputSort.assign(v.begin(),v.end());
      size_t stride=1;
      gsl_sort(&(inputSort[0]),stride,inputSort.size());
      return gsl_stats_median_from_sorted_data(&(inputSort[0]),stride,inputSort.size());
      /* std::vector<T> tmpV=v; */
      /* eraseNoData(tmpV); */
      /* if(tmpV.size()){ */
      /*   sort(tmpV.begin(),tmpV.end()); */
      /*   if(tmpV.size()%2) */
      /*     return tmpV[tmpV.size()/2]; */
      /*   else */
      /*     return 0.5*(tmpV[tmpV.size()/2-1]+tmpV[tmpV.size()/2]); */
      /* } */
      /* else if(m_noDataValues.size()) */
      /*   return m_noDataValues[0]; */
      /* else{ */
      /*   std::string errorString="Error: no valid data found"; */
      /*   throw(errorString); */
      /* } */
    }

  template<class T> double StatFactory::var(const std::vector<T>& v) const
    {
      if(v.empty()){
        std::ostringstream s;
        s<<"Error: input is empty";
        throw(s.str());
      }
      size_t stride=1;
      return gsl_stats_variance(&(v[0]),stride,v.size());
      /* typename std::vector<T>::const_iterator it; */
      /* unsigned int validSize=0; */
      /* double m1=0; */
      /* double m2=0; */
      /* for (it = v.begin(); it!= v.end(); ++it){ */
      /*   if(isNoData(*it)) */
      /*     continue; */
      /*   m1+=*it; */
      /*   m2+=(*it)*(*it); */
      /*   ++validSize; */
      /* } */
      /* if(validSize){ */
      /*   m2/=validSize; */
      /*   m1/=validSize; */
      /*   return m2-m1*m1; */
      /* } */
      /* else if(m_noDataValues.size()) */
      /*   return m_noDataValues[0]; */
      /* else{ */
      /*   std::string errorString="Error: no valid data found"; */
      /*   throw(errorString); */
      /* } */
    }

  template<class T> double StatFactory::moment(const std::vector<T>& v, int n) const
    {
      unsigned int validSize=0;
      typename std::vector<T>::const_iterator it;
      double m=0;
      //   double m1=mean(v);
      for(it = v.begin(); it!= v.end(); ++it){
        if(isNoData(*it))
          continue;
        m+=pow((*it),n);
        ++validSize;
      }
      if(validSize)
        return m/validSize;
      else if(m_noDataValues.size())
        return m_noDataValues[0];
      else{
        std::string errorString="Error: no valid data found";
        throw(errorString);
      }
    }

  //central moment
  template<class T> double StatFactory::cmoment(const std::vector<T>& v, int n) const
    {
      unsigned int validSize=0;
      typename std::vector<T>::const_iterator it;
      double m=0;
      double m1=mean(v);
      for(it = v.begin(); it!= v.end(); ++it){
        if(isNoData(*it))
          continue;
        m+=pow((*it-m1),n);
        ++validSize;
      }
      if(validSize)
        return m/validSize;
      else if(m_noDataValues.size())
        return m_noDataValues[0];
      else{
        std::string errorString="Error: no valid data found";
        throw(errorString);
      }
    }

  template<class T> double StatFactory::skewness(const std::vector<T>& v) const
    {
      //todo: what if nodata value?
      return cmoment(v,3)/pow(var(v),1.5);
    }

  template<class T> double StatFactory::kurtosis(const std::vector<T>& v) const
    {
      //todo: what if nodata value?
      double m2=cmoment(v,2);
      double m4=cmoment(v,4);
      return m4/m2/m2-3.0;
    }

  template<class T> void StatFactory::meanVar(const std::vector<T>& v, double& m1, double& v1) const
    {
      if(v.empty()){
        std::ostringstream s;
        s<<"Error: input is empty";
        throw(s.str());
      }
      size_t stride=1; 
      m1=gsl_stats_mean(&(v[0]),stride,v.size());
      v1=gsl_stats_variance(&(v[0]),stride,v.size());
      /* typename std::vector<T>::const_iterator it; */
      /* unsigned int validSize=0; */
      /* m1=0; */
      /* v1=0; */
      /* double m2=0; */
      /* for (it = v.begin(); it!= v.end(); ++it){ */
      /*   if(isNoData(*it)) */
      /*     continue; */
      /*   m1+=*it; */
      /*   m2+=(*it)*(*it); */
      /*   ++validSize; */
      /* } */
      /* if(validSize){ */
      /*   m2/=validSize; */
      /*   m1/=validSize; */
      /*   v1=m2-m1*m1; */
      /* } */
      /* else if(m_noDataValues.size()){ */
      /*   m1=m_noDataValues[0]; */
      /*   v1=m_noDataValues[0]; */
      /* } */
      /* else{ */
      /*   std::string errorString="Error: no valid data found"; */
      /*   throw(errorString); */
      /* } */
    }

  template<class T1, class T2> void StatFactory::scale2byte(const std::vector<T1>& input, std::vector<T2>& output, unsigned char lbound,  unsigned char ubound) const
  {
    output.resize(input.size());
    T1 minimum=mymin(input);
    T1 maximum=mymax(input);
    if(minimum>=maximum){
      std::string errorString="Error: no valid data found";
      throw(errorString);
    }
    double scale=(ubound-lbound)/(maximum-minimum);
    //todo: what if nodata value?
    for (int i=0;i<input.size();++i){
      double value=scale*(input[i]-(minimum))+lbound;
      output[i]=static_cast<T2>(value);
    }
  }

  template<class T> void  StatFactory::distribution(const std::vector<T>& input, typename std::vector<T>::const_iterator begin, typename std::vector<T>::const_iterator end, std::vector<double>& output, int nbin, T &minimum, T &maximum, double sigma, const std::string &filename) const
    {
      double minValue=0;
      double maxValue=0;
      minmax(input,begin,end,minimum,maximum);
      /* if(minimum<maximum&&minimum>minValue) */
      /* minValue=minimum; */
      /* if(minimum<maximum&&maximum<maxValue) */
      /* maxValue=maximum; */

      if(maximum<=minimum){
        std::ostringstream s;
        s<<"Error: could not calculate distribution (min>=max)";
        throw(s.str());
      }
      if(!nbin){
        std::string errorString="Error: nbin not defined";
        throw(errorString);
      }
      if(!input.size()){
        std::string errorString="Error: no valid data found";
        throw(errorString);
      }
      if(output.size()!=nbin){
        output.resize(nbin);
        for(int i=0;i<nbin;output[i++]=0);
      }
      bool isValid=false;
      typename std::vector<T>::const_iterator it;
      for(it=begin;it!=end;++it){
        if(*it<minimum)
          continue;
        if(*it>maximum)
          continue;
        if(isNoData(*it))
          continue;
        isValid=true;
        if(sigma>0){
          // minimum-=2*sigma;
          // maximum+=2*sigma;
          //create kde for Gaussian basis function
          //todo: speed up by calculating first and last bin with non-zero contriubtion...
          //todo: calculate real surface below pdf by using gsl_cdf_gaussian_P(x-mean+binsize,sigma)-gsl_cdf_gaussian_P(x-mean,sigma)
          for(int ibin=0;ibin<nbin;++ibin){
            double icenter=minimum+static_cast<double>(maximum-minimum)*(ibin+0.5)/nbin;
            double thePdf=gsl_ran_gaussian_pdf(*it-icenter, sigma);
            output[ibin]+=thePdf;
          }
        }
        else{
          int theBin=0;
          if(*it==maximum)
            theBin=nbin-1;
          else if(*it>minimum && *it<maximum)
            theBin=static_cast<int>(static_cast<double>((nbin-1)*(*it)-minimum)/(maximum-minimum));
          ++output[theBin];
          // if(*it==maximum)
          //   ++output[nbin-1];
          // else if(*it>=minimum && *it<maximum)
          //   ++output[static_cast<int>(static_cast<double>((*it)-minimum)/(maximum-minimum)*nbin)];
        }
      }
      if(!isValid){
        std::string errorString="Error: no valid data found";
        throw(errorString);
      }
      else if(!filename.empty()){
        std::ofstream outputfile;
        outputfile.open(filename.c_str());
        if(!outputfile){
          std::ostringstream s;
          s<<"Error opening distribution file , " << filename;
          throw(s.str());
        }
        for(int ibin=0;ibin<nbin;++ibin)
          outputfile << minimum+static_cast<double>(maximum-minimum)*(ibin+0.5)/nbin << " " << static_cast<double>(output[ibin])/input.size() << std::endl;
        outputfile.close();
      }
    }

  template<class T> void  StatFactory::distribution2d(const std::vector<T>& inputX, const std::vector<T>& inputY, std::vector< std::vector<double> >& output, int nbin, T& minX, T& maxX, T& minY, T& maxY, double sigma, const std::string& filename) const
    {
      if(inputX.empty()){
        std::ostringstream s;
        s<<"Error: inputX is empty";
        throw(s.str());
      }
      if(inputX.size()!=inputY.size()){
        std::ostringstream s;
        s<<"Error: inputX is empty";
        throw(s.str());
      }
      unsigned long int npoint=inputX.size();
      if(maxX<=minX)
        minmax(inputX,inputX.begin(),inputX.end(),minX,maxX);
      if(maxX<=minX){
        std::ostringstream s;
        s<<"Error: could not calculate distribution (minX>=maxX)";
        throw(s.str());
      }
      if(maxY<=minY)
        minmax(inputY,inputY.begin(),inputY.end(),minY,maxY);
      if(maxY<=minY){
        std::ostringstream s;
        s<<"Error: could not calculate distribution (minY>=maxY)";
        throw(s.str());
      }
      if(nbin<=1){
        std::ostringstream s;
        s<<"Error: nbin must be larger than 1";
        throw(s.str());
      }
      output.resize(nbin);
      for(int i=0;i<nbin;++i){
        output[i].resize(nbin);
        for(int j=0;j<nbin;++j)
          output[i][j]=0;
      }
      int binX=0;
      int binY=0;
      for(unsigned long int ipoint=0;ipoint<npoint;++ipoint){
        if(inputX[ipoint]==maxX)
          binX=nbin-1;
        else
          binX=static_cast<int>(static_cast<double>(inputX[ipoint]-minX)/(maxX-minX)*nbin);
        if(inputY[ipoint]==maxY)
          binY=nbin-1;
        else
          binY=static_cast<int>(static_cast<double>(inputY[ipoint]-minY)/(maxY-minY)*nbin);
        if(binX<0){
          std::ostringstream s;
          s<<"Error: binX is smaller than 0";
          throw(s.str());
        }
        if(output.size()<=binX){
          std::ostringstream s;
          s<<"Error: output size must be larger than binX";
          throw(s.str());
        }
        if(binY<0){
          std::ostringstream s;
          s<<"Error: binY is smaller than 0";
          throw(s.str());
        }
        if(output.size()<=binY){
          std::ostringstream s;
          s<<"Error: output size must be larger than binY";
          throw(s.str());
        }
        if(sigma>0){
          // minX-=2*sigma;
          // maxX+=2*sigma;
          // minY-=2*sigma;
          // maxY+=2*sigma;
          //create kde for Gaussian basis function
          //todo: speed up by calculating first and last bin with non-zero contriubtion...
          for(int ibinX=0;ibinX<nbin;++ibinX){
            double centerX=minX+static_cast<double>(maxX-minX)*ibinX/nbin;
            double pdfX=gsl_ran_gaussian_pdf(inputX[ipoint]-centerX, sigma);
            for(int ibinY=0;ibinY<nbin;++ibinY){
              //calculate  \integral_ibinX^(ibinX+1)
              double centerY=minY+static_cast<double>(maxY-minY)*ibinY/nbin;
              double pdfY=gsl_ran_gaussian_pdf(inputY[ipoint]-centerY, sigma);
              output[ibinX][binY]+=pdfX*pdfY;
            }
          }
        }
        else
          ++output[binX][binY];
      }
      if(!filename.empty()){
        std::ofstream outputfile;
        outputfile.open(filename.c_str());
        if(!outputfile){
          std::ostringstream s;
          s<<"Error opening distribution file , " << filename;
          throw(s.str());
        }
        for(int binX=0;binX<nbin;++binX){
          outputfile << std::endl;
          for(int binY=0;binY<nbin;++binY){
            double binValueX=0;
            if(nbin==maxX-minX+1)
              binValueX=minX+binX;
            else
              binValueX=minX+static_cast<double>(maxX-minX)*(binX+0.5)/nbin;
            double binValueY=0;
            if(nbin==maxY-minY+1)
              binValueY=minY+binY;
            else
              binValueY=minY+static_cast<double>(maxY-minY)*(binY+0.5)/nbin;
            double value=0;
            value=static_cast<double>(output[binX][binY])/npoint;
            outputfile << binValueX << " " << binValueY << " " << value << std::endl;
            /* double value=static_cast<double>(output[binX][binY])/npoint; */
            /* outputfile << (maxX-minX)*bin/(nbin-1)+minX << " " << (maxY-minY)*bin/(nbin-1)+minY << " " << value << std::endl; */
          }
        }
        outputfile.close();
      }
    }

  //todo: what with nodata values?
  template<class T> void  StatFactory::percentiles (const std::vector<T>& input, typename std::vector<T>::const_iterator begin, typename std::vector<T>::const_iterator end, std::vector<T>& output, int nbin, T &minimum, T &maximum, const std::string &filename) const
    {
      if(maximum<=minimum)
        minmax(input,begin,end,minimum,maximum);
      if(maximum<=minimum){
        std::ostringstream s;
        s<<"Error: maximum must be at least minimum";
        throw(s.str());
      }
      if(nbin<=1){
        std::ostringstream s;
        s<<"Error: nbin must be larger than 1";
        throw(s.str());
      }
      if(input.empty()){
        std::ostringstream s;
        s<<"Error: input is empty";
        throw(s.str());
      }
      output.resize(nbin);
      std::vector<T> inputSort;
      inputSort.assign(begin,end);
      typename std::vector<T>::iterator vit=inputSort.begin();
      while(vit!=inputSort.end()){
        if(*vit<minimum||*vit>maximum)
          inputSort.erase(vit);
        else
          ++vit;
      }
      eraseNoData(inputSort);
      std::sort(inputSort.begin(),inputSort.end());
      vit=inputSort.begin();
      std::vector<T> inputBin;
      for(int ibin=0;ibin<nbin;++ibin){
        inputBin.clear();
        while(inputBin.size()<inputSort.size()/nbin&&vit!=inputSort.end()){
          inputBin.push_back(*vit);
          ++vit;
        }
        if(inputBin.size()){
          output[ibin]=mymax(inputBin);
        }
      }
      if(!filename.empty()){
        std::ofstream outputfile;
        outputfile.open(filename.c_str());
        if(!outputfile){
          std::ostringstream s;
          s<<"error opening distribution file , " << filename;
          throw(s.str());
        }
        for(int ibin=0;ibin<nbin;++ibin)
          outputfile << ibin*100.0/nbin << " " << static_cast<double>(output[ibin])/input.size() << std::endl;
        outputfile.close();
      }
    }

  //todo: what with nodata values?
  template<class T> T  StatFactory::percentile(const std::vector<T>& input, typename std::vector<T>::const_iterator begin, typename std::vector<T>::const_iterator end, double percent, T minimum, T maximum) const
    {
      if(input.empty()){
        std::ostringstream s;
        s<<"Error: input is empty";
        throw(s.str());
      }
      std::vector<double> inputSort;
      inputSort.assign(begin,end);
      typename std::vector<double>::iterator vit=inputSort.begin();
      while(vit!=inputSort.end()){
        if(maximum>minimum){
          if(*vit<minimum||*vit>maximum)
            inputSort.erase(vit);
        }
        else
          ++vit;
      }
      eraseNoData(inputSort);
      std::sort(inputSort.begin(),inputSort.end());
      return gsl_stats_quantile_from_sorted_data(&(inputSort[0]),1,inputSort.size(),percent/100.0);
    }

  template<class T> void StatFactory::signature(const std::vector<T>& input, double&k, double& alpha, double& beta, double e) const
    {
      double m1=moment(input,1);
      double m2=moment(input,2);
      signature(m1,m2,k,alpha,beta,e);
    }

  //todo: what with nodata values?
  template<class T> void StatFactory::normalize(const std::vector<T>& input, std::vector<double>& output) const{
    double total=sum(input);
    if(total){
      output.resize(input.size());
      for(int index=0;index<input.size();++index)
        output[index]=input[index]/total;
    }
    else
      output=input;
  }

  //todo: what with nodata values?
  template<class T> void StatFactory::normalize_pct(std::vector<T>& input) const{
    double total=sum(input);
    if(total){
      typename std::vector<T>::iterator it;
      for(it=input.begin();it!=input.end();++it)
        *it=100.0*(*it)/total;
    }
  }

  template<class T> double StatFactory::rmse(const std::vector<T>& x, const std::vector<T>& y) const{
    if(x.size()!=y.size()){
      std::ostringstream s;
      s<<"Error: x and y not equal in size";
      throw(s.str());
    }
    if(x.empty()){
      std::ostringstream s;
      s<<"Error: x is empty";
      throw(s.str());
    }
    double mse=0;
    for(int isample=0;isample<x.size();++isample){
      if(isNoData(x[isample])||isNoData(y[isample]))
        continue;
      double e=x[isample]-y[isample];
      mse+=e*e/x.size();
    }
    return sqrt(mse);
  }

  //normalized root mean square error
  template<class T> double StatFactory::nrmse(const std::vector<T>& x, const std::vector<T>& y) const{
    if(x.size()!=y.size()){
      std::ostringstream s;
      s<<"Error: x and y not equal in size";
      throw(s.str());
    }
    if(x.empty()){
      std::ostringstream s;
      s<<"Error: x is empty";
      throw(s.str());
    }
    std::vector<T> tmpX=x;
    eraseNoData(tmpX);
    std::vector<T> tmpY=y;
    eraseNoData(tmpY);
    double maxY=mymax(y);
    double minY=mymin(y);
    double rangeY=maxY-minY;
    double mse=0;
    for(int isample=0;isample<x.size();++isample){
      double e=x[isample]-y[isample];
      mse+=e*e/x.size();
    }
    return sqrt(mse)/rangeY;
  }

  // coefficient of variation root mean square error
  template<class T> double StatFactory::cvrmse(const std::vector<T>& x, const std::vector<T>& y) const{
    if(x.size()!=y.size()){
      std::ostringstream s;
      s<<"Error: x and y not equal in size";
      throw(s.str());
    }
    if(x.empty()){
      std::ostringstream s;
      s<<"Error: x is empty";
      throw(s.str());
    }
    std::vector<T> tmpX=x;
    eraseNoData(tmpX);
    std::vector<T> tmpY=y;
    eraseNoData(tmpY);
    double maxY=mymax(tmpY);
    double minY=mymin(tmpY);
    double rangeY=maxY-minY;
    double mse=0;
    for(int isample=0;isample<x.size();++isample){
      double e=x[isample]-y[isample];
      mse+=e*e/x.size();
    }
    return sqrt(mse)/mean(tmpY);
  }

  // template<class T> double StatFactory::gsl_correlation(const std::vector<T>& x, const std::vector<T>& y) const{
  //  return(gsl_stats_correlation(&(x[0]),1,&(y[0]),1,x.size()));
  // }

  template<class T> double StatFactory::gsl_covariance(const std::vector<T>& x, const std::vector<T>& y) const{
    return(gsl_stats_covariance(&(x[0]),1,&(y[0]),1,x.size()));
  }


  template<class T> double StatFactory::correlation(const std::vector<T>& x, const std::vector<T>& y, int delay) const{
    double meanX=0;
    double meanY=0;
    double varX=0;
    double varY=0;
    double sXY=0;
    meanVar(x,meanX,varX);
    meanVar(y,meanY,varY);
    double denom = sqrt(varX*varY);
    bool isValid=false;
    if(denom){
      //Calculate the correlation series
      sXY = 0;
      for (int i=0;i<x.size();++i) {
        int j = i + delay;
        if (j < 0 || j >= y.size())
          continue;
        else if(isNoData(x[i])||isNoData(y[j]))
          continue;
        else{
          isValid=true;
          if(i<0){
            std::ostringstream s;
            s<<"Error: i must be positive";
            throw(s.str());
          }
          if(i>=x.size()){
            std::ostringstream s;
            s<<"Error: i must be smaller than x.size()";
            throw(s.str());
          }
          if(j<0){
            std::ostringstream s;
            s<<"Error: j must be positive";
            throw(s.str());
          }
          if(j>=y.size()){
            std::ostringstream s;
            s<<"Error: j must be smaller than y.size()";
            throw(s.str());
          }
          sXY += (x[i] - meanX) * (y[j] - meanY);
        }
      }
      if(isValid){
        double minSize=(x.size()<y.size())?x.size():y.size();
        return(sXY / denom / (minSize-1));
      }
      else if(m_noDataValues.size())
        return m_noDataValues[0];
      else{
        std::string errorString="Error: no valid data found";
        throw(errorString);
      }
    }
    else
      return 0;
  }

  //todo: what if no valid data?
  template<class T> double StatFactory::cross_correlation(const std::vector<T>& x, const std::vector<T>& y, int maxdelay, std::vector<T>& z) const{
    z.clear();
    double sumCorrelation=0;
    for (int delay=-maxdelay;delay<maxdelay;delay++) {
      z.push_back(correlation(x,y,delay));
      sumCorrelation+=z.back();
    }
    return sumCorrelation;
  }

  //todo: nodata?
  template<class T> double StatFactory::linear_regression(const std::vector<T>& x, const std::vector<T>& y, double &c0, double &c1) const{
    if(x.size()!=y.size()){
      std::ostringstream s;
      s<<"Error: x and y not equal in size";
      throw(s.str());
    }
    if(x.empty()){
      std::ostringstream s;
      s<<"Error: x is empty";
      throw(s.str());
    }
    double cov00;
    double cov01;
    double  cov11;
    double sumsq;
    gsl_fit_linear(&(x[0]),1,&(y[0]),1,x.size(),&c0,&c1,&cov00,&cov01,&cov11,&sumsq);
    return (1-sumsq/var(y)/(y.size()-1));
  }

  //todo: nodata?
  template<class T> double StatFactory::linear_regression_err(const std::vector<T>& x, const std::vector<T>& y, double &c0, double &c1) const{
    if(x.size()!=y.size()){
      std::ostringstream s;
      s<<"Error: x and y not equal in size";
      throw(s.str());
    }
    if(x.empty()){
      std::ostringstream s;
      s<<"Error: x is empty";
      throw(s.str());
    }
    double cov00;
    double cov01;
    double  cov11;
    double sumsq;
    gsl_fit_linear(&(x[0]),1,&(y[0]),1,x.size(),&c0,&c1,&cov00,&cov01,&cov11,&sumsq);
    return sqrt((sumsq)/(y.size()));
  }

  //alternatively: use GNU scientific library:
  // gsl_stats_correlation (const double data1[], const size_t stride1, const double data2[], const size_t stride2, const size_t n)

  template<class T> void StatFactory::interpolateNoData(const std::vector<double>& wavelengthIn, const std::vector<T>& input, const std::string& type, std::vector<T>& output, bool verbose) const{
    if(wavelengthIn.empty()){
      std::ostringstream s;
      s<<"Error: wavelengthIn is empty";
      throw(s.str());
    }
    std::vector<double> wavelengthOut=wavelengthIn;
    std::vector<T> validIn=input;
    if(input.size()!=wavelengthIn.size()){
      std::ostringstream s;
      s<<"Error: x and y not equal in size";
      throw(s.str());
    }
    output.clear();
    //remove nodata from input and corresponding wavelengthIn
    if(m_noDataValues.size()){
      typename std::vector<T>::iterator itValue=validIn.begin();
      typename std::vector<T>::iterator itWavelength=wavelengthOut.begin();
      while(itValue!=validIn.end()&&itWavelength!=wavelengthOut.end()){
        if(isNoData(*itValue)){
          validIn.erase(itValue);
          wavelengthOut.erase(itWavelength);
        }
        else{
          ++itValue;
          ++itWavelength;
        }
      }
      if(validIn.size()>1){
        try{
          interpolateUp(wavelengthOut, validIn, wavelengthIn, type, output, verbose);
        }
        catch(...){
          output=input;
        }
      }
      else//we can not interpolate if no valid data
        output=input;
    }
    else//no nodata values to interpolate
      output=input;
  }

  template<class T> void StatFactory::interpolateUp(const std::vector<double>& wavelengthIn, const std::vector<T>& input, const std::vector<double>& wavelengthOut, const std::string& type, std::vector<T>& output, bool verbose) const{
    if(wavelengthIn.empty()){
      std::ostringstream s;
      s<<"Error: wavelengthIn is empty";
      throw(s.str());
    }
    if(input.size()!=wavelengthIn.size()){
      std::ostringstream s;
      s<<"Error: input and wavelengthIn not equal in size";
      throw(s.str());
    }
    if(wavelengthOut.empty()){
      std::ostringstream s;
      s<<"Error: wavelengthOut is empty";
      throw(s.str());
    }
    int nband=wavelengthIn.size();
    output.clear();
    gsl_interp_accel *acc;
    allocAcc(acc);
    gsl_spline *spline;
    getSpline(type,nband,spline);
    assert(spline);
    assert(&(wavelengthIn[0]));
    assert(&(input[0]));
    int status=initSpline(spline,&(wavelengthIn[0]),&(input[0]),nband);
    if(status){
      std::string errorString="Could not initialize spline";
      throw(errorString);
    }
    for(int index=0;index<wavelengthOut.size();++index){
      if(wavelengthOut[index]<*wavelengthIn.begin()){
        output.push_back(*(input.begin()));
        continue;
      }
      else if(wavelengthOut[index]>wavelengthIn.back()){
        output.push_back(input.back());
        continue;
      }
      double dout=evalSpline(spline,wavelengthOut[index],acc);
      output.push_back(dout);
    }
    gsl_spline_free(spline);
    gsl_interp_accel_free(acc);
  }

  // template<class T> void StatFactory::interpolateUp(const std::vector<double>& wavelengthIn, const std::vector< std::vector<T> >& input, const std::vector<double>& wavelengthOut, const std::string& type, std::vector< std::vector<T> >& output, bool verbose){
  //   assert(wavelengthIn.size());
  //   assert(wavelengthOut.size());
  //   int nsample=input.size();
  //   int nband=wavelengthIn.size();
  //   output.clear();
  //   output.resize(nsample);
  //   gsl_interp_accel *acc;
  //   allocAcc(acc);
  //   gsl_spline *spline;
  //   getSpline(type,nband,spline);
  //   for(int isample=0;isample<nsample;++isample){
  //     assert(input[isample].size()==wavelengthIn.size());
  //     initSpline(spline,&(wavelengthIn[0]),&(input[isample][0]),nband);
  //     for(int index=0;index<wavelengthOut.size();++index){
  //       if(type=="linear"){
  //         if(wavelengthOut[index]<wavelengthIn.back())
  //           output[isample].push_back(*(input.begin()));
  //         else if(wavelengthOut[index]>wavelengthIn.back())
  //           output[isample].push_back(input.back());
  //       }
  //       else{
  //         double dout=evalSpline(spline,wavelengthOut[index],acc);
  //         output[isample].push_back(dout);
  //       }
  //     }
  //   }
  //   gsl_spline_free(spline);
  //   gsl_interp_accel_free(acc);
  // }

  //todo: nodata?
  template<class T> void StatFactory::interpolateUp(const std::vector<T>& input, std::vector<T>& output, int nbin) const
    {
      if(input.empty()){
        std::ostringstream s;
        s<<"Error: input is empty";
        throw(s.str());
      }
      if(!nbin){
        std::ostringstream s;
        s<<"Error: nbin must be larger than 0";
        throw(s.str());
      }
      output.clear();
      int dim=input.size();
      for(int i=0;i<dim;++i){
        double deltaX=0;
        double left=input[i];
        if(i<dim-1){
          double right=(i<dim-1)? input[i+1]:input[i];
          deltaX=(right-left)/static_cast<double>(nbin);
          for(int x=0;x<nbin;++x){
            output.push_back(left+x*deltaX);
          }
        }
        else
          output.push_back(input.back());
      }
    }

  //todo: nodata?
  template<class T> void StatFactory::nearUp(const std::vector<T>& input, std::vector<T>& output) const
    {
      if(input.empty()){
        std::ostringstream s;
        s<<"Error: input is empty";
        throw(s.str());
      }
      if(output.size()<input.size()){
        std::ostringstream s;
        s<<"Error: output size is smaller than input size";
        throw(s.str());
      }
      int dimInput=input.size();
      int dimOutput=output.size();

      for(int iin=0;iin<dimInput;++iin){
        for(int iout=0;iout<dimOutput/dimInput;++iout){
          int indexOutput=iin*dimOutput/dimInput+iout;
          if(indexOutput>=output.size()){
            std::ostringstream s;
            s<<"Error: indexOutput must be smaller than output.size()";
            throw(s.str());
          }
          output[indexOutput]=input[iin];
        }
      }
    }

  //todo: nodata?
  template<class T> void StatFactory::interpolateUp(double* input, int dim, std::vector<T>& output, int nbin)
    {
      if(!nbin){
        std::ostringstream s;
        s<<"Error: nbin must be larger than 0";
        throw(s.str());
      }
      output.clear();
      for(int i=0;i<dim;++i){
        double deltaX=0;
        double left=input[i];
        if(i<dim-1){
          double right=(i<dim-1)? input[i+1]:input[i];
          deltaX=(right-left)/static_cast<double>(nbin);
          for(int x=0;x<nbin;++x){
            output.push_back(left+x*deltaX);
          }
        }
        else
          output.push_back(input[dim-1]);
      }
    }

  //todo: nodata?
  template<class T> void StatFactory::interpolateDown(const std::vector<T>& input, std::vector<T>& output, int nbin) const
    {
      if(input.empty()){
        std::ostringstream s;
        s<<"Error: input is empty";
        throw(s.str());
      }
      if(!nbin){
        std::ostringstream s;
        s<<"Error: nbin must be larger than 0";
        throw(s.str());
      }
      output.clear();
      int dim=input.size();
      int x=0;
      output.push_back(input[0]);
      for(int i=1;i<dim;++i){
        if(i%nbin)
          continue;
        else{
          x=(i-1)/nbin+1;
          output.push_back(input[i]);
        }
      }
    }

  //todo: nodata?
  template<class T> void StatFactory::interpolateDown(double* input, int dim, std::vector<T>& output, int nbin)
    {
      if(!nbin){
        std::ostringstream s;
        s<<"Error: nbin must be larger than 0";
        throw(s.str());
      }
      output.clear();
      int x=0;
      output.push_back(input[0]);
      for(int i=1;i<dim;++i){
        if(i%nbin)
          continue;
        else{
          x=(i-1)/nbin+1;
          output.push_back(input[i]);
        }
      }
    }
}

#endif /* _STATFACTORY_H_ */

// void Histogram::signature(double m1, double m2, double& k, double& alpha, double& beta, double e)
// {
//   double y=m1*m1/m2;
//   beta=F_1(y,0.1,10.0,e);
//   double fb=F(beta);
//   double g=exp(lgamma(1.0/beta));
//   alpha=m1*g/exp(lgamma(2.0/beta));
//   k=beta/(2*alpha*g);
// //   std::cout << "y, alpha, beta: " << y << ", " << alpha << ", " << beta << std::endl;
// }

// double Histogram::F(double x)
// {
//   double g2=exp(lgamma(2.0/x));
//   return(g2*g2/exp(lgamma(3.0/x))/exp(lgamma(1.0/x)));
// }

// //x1 is under estimate, x2 is over estimate, e is error
// double Histogram::F_1(double y, double x1, double x2, double e)
// {
//   double f1=F(x1);
//   double f2=F(x2);
//   assert(f1!=f2);
//   double x=x1+(x2-x1)*(y-f1)/(f2-f1);
//   double f=F(x);
//   while(f-y>=e||y-f>=e){
//     if(f<y)
//       x1=x;
//     else
//       x2=x;
//     if(x1==x2)
//       return x1;
//     assert(f1!=f2);
//     x=x1+(x2-x1)*(y-f1)/(f2-f1);
//     f=F(x);
//   }
//   return x;
// }
