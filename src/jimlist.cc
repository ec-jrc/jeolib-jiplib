/**********************************************************************
jimlist.cc: class to read raster files
History
2016/12/05 - Created by Pieter Kempeneers
Change log
***********************************************************************/
#include "base/Optionpk.h"
#include "json/json.h"
#include "jimlist.h"

using namespace jiplib;
///constructor using vector of images
JimList::JimList(const std::list<std::shared_ptr<jiplib::Jim> > &jimlist) : ImgList(){
  std::list<std::shared_ptr<jiplib::Jim> >::const_iterator lit=jimlist.begin();
  for(lit=jimlist.begin();lit!=jimlist.end();++lit)
    pushImage(*lit);
  // for(int ijim=0;ijim<jimVector.size();++ijim){
    // pushImage(jimVector[ijim]);
  // }
}

///constructor using a json string coming from a custom colllection
JimList& JimList::open(const std::string& strjson){
  Json::Value custom;
  Json::Reader reader;
  bool parsedSuccess=reader.parse(strjson,custom,false);
  if(parsedSuccess){
    for(int iimg=0;iimg<custom["size"].asInt();++iimg){
      std::ostringstream os;
      os << iimg;
      Json::Value image=custom[os.str()];
      std::string filename=image["path"].asString();
      //todo: open without reading?
      app::AppFactory theApp;
      theApp.setLongOption("filename",filename);
      std::shared_ptr<jiplib::Jim> theImage=jiplib::Jim::createImg(theApp);
      pushImage(theImage);
    }
  }
  return(*this);
}

JimList& JimList::open(app::AppFactory& theApp){
  Optionpk<std::string> json_opt("json", "json", "The json object");
  bool doProcess;//stop process when program was invoked with help option (-h --help)
  try{
    doProcess=json_opt.retrieveOption(theApp);
  }
  catch(std::string predefinedString){
    std::cout << predefinedString << std::endl;
  }
  if(!doProcess){
    std::cout << std::endl;
    std::ostringstream helpStream;
    helpStream << "exception thrown due to help info";
    throw(helpStream.str());//help was invoked, stop processing
  }

  std::vector<std::string> badKeys;
  theApp.badKeys(badKeys);
  if(badKeys.size()){
    std::ostringstream errorStream;
    if(badKeys.size()>1)
      errorStream << "Error: unknown keys: ";
    else
      errorStream << "Error: unknown key: ";
    for(int ikey=0;ikey<badKeys.size();++ikey){
      errorStream << badKeys[ikey] << " ";
    }
    errorStream << std::endl;
    throw(errorStream.str());
  }
  if(json_opt.empty()){
    std::string errorString="Error: json string is empty";
    throw(errorString);
  }
  return(open(json_opt[0]));
  // JimList(std::string(""));
}

///get image from collection
std::shared_ptr<jiplib::Jim> JimList::getImage(int index) const{
  return(std::dynamic_pointer_cast<jiplib::Jim>(ImgList::getImage(index)));
}

///convert a JimList to a json string
std::string JimList::jl2json(){
  Json::Value custom;
  custom["size"]=static_cast<int>(size());
  int iimg=0;
  for(std::list<std::shared_ptr<ImgRaster> >::iterator lit=begin();lit!=end();++lit){
    Json::Value image;
    image["path"]=(*lit)->getFileName();
    std::string wktString=(*lit)->getProjectionRef();
    std::string key("EPSG");
    std::size_t foundEPSG=wktString.rfind(key);
    std::string fromEPSG=wktString.substr(foundEPSG);//EPSG","32633"]]'
    std::size_t foundFirstDigit=fromEPSG.find_first_of("0123456789");
    std::size_t foundLastDigit=fromEPSG.find_last_of("0123456789");
    std::string epsgString=fromEPSG.substr(foundFirstDigit,foundLastDigit-foundFirstDigit+1);
    image["epsg"]=atoi(epsgString.c_str());
    std::ostringstream os;
    os << iimg++;
    custom[os.str()]=image;
  }
  Json::FastWriter fastWriter;
  return(fastWriter.write(custom));
}

///push image to collection
JimList& JimList::pushImage(const std::shared_ptr<jiplib::Jim> imgRaster){
  this->emplace_back(imgRaster);
  return(*this);
}

///composite image only for in memory
std::shared_ptr<jiplib::Jim> JimList::composite(app::AppFactory& app){
  std::shared_ptr<jiplib::Jim> imgWriter=std::make_shared<jiplib::Jim>();
  ImgList::composite(*imgWriter, app);
  return(imgWriter);
}

std::shared_ptr<jiplib::Jim> JimList::crop(app::AppFactory& app){
  /* std::shared_ptr<jiplib::Jim> imgWriter=Jim::createImg(); */
  std::shared_ptr<jiplib::Jim> imgWriter=std::make_shared<jiplib::Jim>();
  ImgList::crop(*imgWriter, app);
  return(imgWriter);
}

///stack all images in collection to multiband image (alias for crop)
std::shared_ptr<jiplib::Jim> JimList::stack(app::AppFactory& app){return(crop(app));};

////stack all images in collection to multiband image (alias for crop)
std::shared_ptr<jiplib::Jim> JimList::stack(){app::AppFactory app;return(stack(app));};

///create statistical profile from a collection
std::shared_ptr<jiplib::Jim> JimList::statProfile(app::AppFactory& app){
  std::shared_ptr<Jim> imgWriter=Jim::createImg();
  ImgList::statProfile(*imgWriter, app);
  return(imgWriter);
}

///get statistics on image list
std::multimap<std::string,std::string> JimList::getStats(app::AppFactory& app){
  return(ImgList::getStats(app));
}

JimList& JimList::validate(app::AppFactory& app){
  ImgList::validate(app);
  return(*this);
}

// std::shared_ptr<VectorOgr> JimList::extractOgr(VectorOgr& sampleReader, app::AppFactory& app){
//   return(ImgList::extractOgr(sampleReader,app));
// }

//automatically ported for now, but should probably better via JimList as implemented here:
///functions from mialib
// std::shared_ptr<jiplib::Jim> JimList::labelConstrainedCCsMultiband(Jim &imgRaster, int ox, int oy, int oz, int r1, int r2){
//   try{
//     IMAGE * imout = 0;
//     IMAGE * imse=imgRaster.getMIA();
//     IMAGE ** imap;
//     imap = (IMAGE **) malloc(this->size()*sizeof(IMAGE **));
//     for(int iimg=0;iimg=this->size();++iimg)
//       imap[iimg]=getImage(iimg)->getMIA();
//     imout =::labelccms(imap,this->size(),imse,ox,oy,oz,r1,r2);
//     if (imout){
//       std::shared_ptr<Jim> imgWriter=std::make_shared<Jim>(imout);
//       imgWriter->copyGeoTransform(*front());
//       imgWriter->setProjection(front()->getProjectionRef());
//       return(imgWriter);
//     }
//     else{
//       std::string errorString="Error: labelConstrainedCCsMultiband() function in MIA failed, returning NULL pointer";
//       throw(errorString);
//     }
//   }
//   catch(std::string errorString){
//     std::cerr << errorString << std::endl;
//     return(0);
//   }
//   catch(...){
//     return(0);
//   }
// }

// JimList JimList::convertRgbToHsx(int x){
//   int ninput=3;
//   int noutput=3;
//   JimList listout;
//   try{
//     if(size()!=ninput){
//       std::ostringstream ess;
//       ess << "Error: input image list should have " << ninput << " images (got " << size() << ")";
//       throw(ess.str());
//     }
//     IMAGE ** imout;
//     imout=::imrgb2hsx(this->getImage(0)->getMIA(),this->getImage(1)->getMIA(),this->getImage(2)->getMIA(),x);
//     if(imout){
//       for(int iim=0;iim<noutput;++iim){
//         std::shared_ptr<Jim> imgWriter=std::make_shared<Jim>(imout[iim]);
//         imgWriter->copyGeoTransform(*front());
//         imgWriter->setProjection(front()->getProjectionRef());
//         listout.pushImage(imgWriter);
//       }
//       return(listout);
//     }
//     else{
//       std::string errorString="Error: imrgb2hsx() function in MIA failed, returning empty list";
//       throw(errorString);
//     }
//   }
//   catch(std::string errorString){
//     std::cerr << errorString << std::endl;
//     return(listout);
//   }
//   catch(...){
//     return(listout);
//   }
// }

// JimList JimList::alphaTreeDissimGet(int alphaMax){
//   int ninput=2;
//   int noutput=5;
//   JimList listout;
//   if(size()!=ninput){
//     std::ostringstream ess;
//     ess << "Error: input image list should have " << ninput << " images (got " << size() << ")";
//     throw(ess.str());
//   }
//   try{
//     IMAGE ** imout;
//     imout=::alphatree(this->getImage(0)->getMIA(),this->getImage(1)->getMIA(),alphaMax);
//     if(imout){
//       for(int iim=0;iim<noutput;++iim){
//         std::shared_ptr<Jim> imgWriter=std::make_shared<Jim>(imout[iim]);
//         imgWriter->copyGeoTransform(*front());
//         imgWriter->setProjection(front()->getProjectionRef());
//         listout.pushImage(imgWriter);
//       }
//       return(listout);
//     }
//     else{
//       std::string errorString="Error: alphatree() function in MIA failed, returning empty list";
//       throw(errorString);
//     }
//   }
//   catch(std::string errorString){
//     std::cerr << errorString << std::endl;
//     return(listout);
//   }
//   catch(...){
//     return(listout);
//   }
// }

// JimList JimList::histoMatchRgb(){
//   int ninput=4;
//   int noutput=3;
//   JimList listout;
//   if(size()!=ninput){
//     std::ostringstream ess;
//     ess << "Error: input image list should have " << ninput << " images (got " << size() << ")";
//     throw(ess.str());
//   }
//   try{
//     IMAGE ** imout;
//     imout=::histrgbmatch(this->getImage(0)->getMIA(),this->getImage(1)->getMIA(),this->getImage(2)->getMIA(),this->getImage(3)->getMIA());
//     if(imout){
//       for(int iim=0;iim<noutput;++iim){
//         std::shared_ptr<Jim> imgWriter=std::make_shared<Jim>(imout[iim]);
//         imgWriter->copyGeoTransform(*front());
//         imgWriter->setProjection(front()->getProjectionRef());
//         listout.pushImage(imgWriter);
//       }
//       return(listout);
//     }
//     else{
//       std::string errorString="Error: histrgbmatch() function in MIA failed, returning empty list";
//       throw(errorString);
//     }
//   }
//   catch(std::string errorString){
//     std::cerr << errorString << std::endl;
//     return(listout);
//   }
//   catch(...){
//     return(listout);
//   }
// }

// JimList JimList::histoMatch3dRgb(){
//   int ninput=4;
//   int noutput=3;
//   JimList listout;
//   if(size()!=ninput){
//     std::ostringstream ess;
//     ess << "Error: input image list should have " << ninput << " images (got " << size() << ")";
//     throw(ess.str());
//   }
//   try{
//     IMAGE ** imout;
//     imout=::histrgb3dmatch(this->getImage(0)->getMIA(),this->getImage(1)->getMIA(),this->getImage(2)->getMIA(),this->getImage(3)->getMIA());
//     if(imout){
//       for(int iim=0;iim<noutput;++iim){
//         std::shared_ptr<Jim> imgWriter=std::make_shared<Jim>(imout[iim]);
//         imgWriter->copyGeoTransform(*front());
//         imgWriter->setProjection(front()->getProjectionRef());
//         listout.pushImage(imgWriter);
//       }
//       return(listout);
//     }
//     else{
//       std::string errorString="Error: histrgb3dmatch() function in MIA failed, returning empty list";
//       throw(errorString);
//     }
//   }
//   catch(std::string errorString){
//     std::cerr << errorString << std::endl;
//     return(listout);
//   }
//   catch(...){
//     return(listout);
//   }
// }
