/**********************************************************************
ImgRegression.h: class to calculate regression between two raster datasets
Copyright (c) 2016-2018 European Union (Joint Research Centre)
License EUPLv1.2

This file is part of jiplib
***********************************************************************/
#ifndef _IMGREGRESSION_H_
#define _IMGREGRESSION_H_

#include <vector>
#include "imageclasses/Jim.h"
#include "StatFactory.h"

namespace imgregression
{
  class ImgRegression{
  public:
    ImgRegression(void);
    ~ImgRegression(void);
    double getRMSE(Jim& imgReader1, Jim& imgReader2, double &c0, double &c1, unsigned int b1=0, unsigned int b2=0, short verbose=0) const;
    double getRMSE(Jim& imgReader, unsigned int b1, unsigned int b2, double& c0, double& c1, short verbose=0) const;
    double getR2(Jim& imgReader1, Jim& imgReader2, double &c0, double &c1, unsigned int b1=0, unsigned int b2=0, short verbose=0) const;
    double pgetR2(Jim& imgReader1, Jim& imgReader2, double& c0, double& c1, unsigned int band1, unsigned int band2, short verbose=0) const;
    double getR2(Jim& imgReader, unsigned int b1, unsigned int b2, double& c0, double& c1, short verbose=0) const;
    double pgetR2(Jim& imgReader, unsigned int band1, unsigned int band2, double& c0, double& c1, short verbose=0) const;

    void setThreshold(double theThreshold){m_threshold=theThreshold;};
    void setDown(int theDown){m_down=theDown;};
  private:
    int m_down;
    double m_threshold;
  };
}
#endif //_IMGREGRESSION_H_
