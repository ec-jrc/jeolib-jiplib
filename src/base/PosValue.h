/**********************************************************************
PosValue.h: class to work with structs containing a position and a value
Author(s): Pieter.Kempeneers@ec.europa.eu
Copyright (c) 2016-2019 European Union (Joint Research Centre)
License EUPLv1.2

This file is part of jiplib
***********************************************************************/
#ifndef _POSVALUE_H_
#define _POSVALUE_H_

struct PosValue{
  double posx;
  double posy;
  double value;
};
class Compare_PosValue{
public:
  int operator() (const PosValue& pv1, const PosValue& pv2) const{
    return pv1.value>pv2.value;//for decreasing order
  }
};
class Decrease_PosValue{
public:
  int operator() (const PosValue& pv1, const PosValue& pv2) const{
    return pv1.value>pv2.value;//for decreasing order
  }
};
class Increase_PosValue{
public:
  int operator() (const PosValue& pv1, const PosValue& pv2) const{
    return pv1.value<pv2.value;//for increasing order
  }
};
#endif /* _POSVALUE_H_ */
