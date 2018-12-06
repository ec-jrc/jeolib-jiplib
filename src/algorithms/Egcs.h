/**********************************************************************
Egcs.h: Conversions from and to european grid coding system
Author(s): Pieter.Kempeneers@ec.europa.eu
Copyright (c) 2016-2018 European Union (Joint Research Centre)
License EUPLv1.2

This file is part of jiplib
***********************************************************************/
#include <math.h>
#include <string>

#ifndef _EGCS_H_
#define _EGCS_H_

class Egcs
{
public:
  Egcs();
  Egcs(unsigned short level);
  /* Egcs(unsigned short level); */
  ~Egcs();
  unsigned short cell2level(const std::string& cellCode) const;
  std::string geo2cell(double x, double y) const;
  double getSize() const {return getBaseSize()*pow(2.0,(m_level-19)%3);};
  void setLevel(unsigned short level){m_level=level;};
  unsigned short getLevel() const{return m_level;};
  unsigned short res2level(double resolution) const;
  double getResolution() const;
  void force2grid(double& ulx, double& uly, double& lrx, double &lry) const;
  void cell2bb(const std::string& cellCode, int &ulx, int &uly, int &lrx, int &lry) const;
  void cell2mid(const std::string& cellCode, double& midX, double& midY) const;
private:
  int getBaseSize() const {return pow(10.0,(m_level+1)/3);};
  unsigned short m_level;
// level square scheme         example
// 19    1000km xy             32
// 18     500km xy-q           32-A
// 17     250km xy-qq          32-AB
// 16     100km xxyy           3320
// 5       25m  xxxxxyyyyy-qq  3346720658-DC
// 1        1m  xxxxxxxyyyyyyy 33467652065889
};
#endif // _EGCS_H_

