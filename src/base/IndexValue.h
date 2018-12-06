/**********************************************************************
IndexValue.h: class to work with structs containing an index and a value
Author(s): Pieter.Kempeneers@ec.europa.eu
Copyright (c) 2016-2018 European Union (Joint Research Centre)
License EUPLv1.2

This file is part of jiplib
***********************************************************************/
#ifndef _INDEXVALUE_H_
#define _INDEXVALUE_H_

struct IndexValue{
  int position;
  double value;
};
class Compare_IndexValue{
public:
  int operator() (const IndexValue& pv1, const IndexValue& pv2) const{
    return pv1.value>pv2.value;//for decreasing order
  }
};
class Decrease_IndexValue{
public:
  int operator() (const IndexValue& pv1, const IndexValue& pv2) const{
    return pv1.value>pv2.value;//for decreasing order
  }
};
class Increase_IndexValue{
public:
  int operator() (const IndexValue& pv1, const IndexValue& pv2) const{
    return pv1.value<pv2.value;//for increasing order
  }
};
#endif /* _INDEXVALUE_H_ */
