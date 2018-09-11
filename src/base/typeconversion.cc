/**********************************************************************
typeconversion.cc: class to handle type conversions
Copyright (C) 2008-2016 Pieter Kempeneers

This file is part of pktools

pktools is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

pktools is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with pktools.  If not, see <http://www.gnu.org/licenses/>.
***********************************************************************/
/////////////////// Specializations /////////////////
#include "typeconversion.h"
#include "ogr_feature.h"

///specialization for string
template<> std::string string2type(std::string const& s){
  return s;
}

template<> double string2type(std::string const& s){
  std::istringstream i;
  i.precision(12);
  i.str(s);
  double x;
  if (!(i >> std::setprecision(12) >> x) )
     throw BadConversion(s);
  return x;
}

template<> float string2type(std::string const& s){
  std::istringstream i;
  i.precision(12);
  i.str(s);
  float x;
  if (!(i >> std::setprecision(12) >> x) )
     throw BadConversion(s);
  return x;
}

///specialization for OGRFieldType
template<> OGRFieldType string2type(std::string const& s){
  OGRFieldType ftype;
  int ogr_typecount=11;//hard coded for now!
  for(int iType = 0; iType < ogr_typecount; ++iType){
    if( OGRFieldDefn::GetFieldTypeName((OGRFieldType)iType) != NULL
        && EQUAL(OGRFieldDefn::GetFieldTypeName((OGRFieldType)iType),s.c_str()))
      ftype=(OGRFieldType) iType;
  }
  return ftype;
}

///specialization for bool
template<> std::string type2string(bool const& value){
  if(value)
    return("true");
  else
    return("false");
}

///specialization for string
template<> std::string type2string(std::string const& value){
  // if(value.empty())
  //   return("<empty string>");
  // else
    return(value);
}

///specialization for float
template<> std::string type2string(float const& value){
  std::ostringstream oss;
  // oss.precision(1);
  // oss.setf(ios::fixed);
  oss << value;
  return oss.str();
}

///specialization for double
template<> std::string type2string(double const& value){
  std::ostringstream oss;
  // oss.precision(1);
  //  oss.setf(ios::fixed);
  oss << value;
  return oss.str();
}
