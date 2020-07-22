/**********************************************************************
PosValue.h: class to work with structs containing a position and a value
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
