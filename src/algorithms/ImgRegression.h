/**********************************************************************
ImgRegression.h: class to calculate regression between two raster datasets
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
