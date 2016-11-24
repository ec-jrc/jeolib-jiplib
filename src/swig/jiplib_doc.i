
// File: index.xml

// File: classjiplib_1_1Jim.xml
%feature("docstring") jiplib::Jim "

class for raster dataset (read and write).

Jim is a class that enables the integration of functionalities from
both pktools and mia image processing libraries Pierre Soille, Pieter
Kempeneers

C++ includes: jim.h ";

%feature("docstring")  jiplib::Jim::Jim "jiplib::Jim::Jim()

default constructor ";

%feature("docstring")  jiplib::Jim::Jim "jiplib::Jim::Jim(void
*dataPointer, int ncol, int nrow, const GDALDataType &dataType)

constructor opening an image in memory using an external data pointer
(not tested yet) ";

%feature("docstring")  jiplib::Jim::Jim "jiplib::Jim::Jim(IMAGE *mia)

constructor input image ";

%feature("docstring")  jiplib::Jim::Jim "jiplib::Jim::Jim(const
std::string &filename, unsigned int memory=0)

constructor input image ";

%feature("docstring")  jiplib::Jim::Jim "jiplib::Jim::Jim(const
std::string &filename, const Jim &imgSrc, unsigned int memory=0, const
std::vector< std::string > &options=std::vector< std::string >())

constructor input image ";

%feature("docstring")  jiplib::Jim::Jim "jiplib::Jim::Jim(Jim
&imgSrc, bool copyData=true)

constructor input image

constructor input image ";

%feature("docstring")  jiplib::Jim::Jim "jiplib::Jim::Jim(const
std::string &filename, int ncol, int nrow, int nband, const
GDALDataType &dataType, const std::string &imageType, unsigned int
memory=0, const std::vector< std::string > &options=std::vector<
std::string >())

constructor output image ";

%feature("docstring")  jiplib::Jim::Jim "jiplib::Jim::Jim(int ncol,
int nrow, int nband, const GDALDataType &dataType)

constructor output image ";

%feature("docstring")  jiplib::Jim::Jim "jiplib::Jim::Jim(app::AppFactory &theApp)

constructor from app ";

%feature("docstring")  jiplib::Jim::~Jim "jiplib::Jim::~Jim(void)

destructor ";

%feature("docstring")  jiplib::Jim::clone "virtual
std::shared_ptr<ImgRaster> jiplib::Jim::clone()

Open an image for writing in memory, defining image attributes.

Clone as new shared pointer to ImgRaster object

shared pointer to new ImgRaster object alllowing polymorphism ";

%feature("docstring")  jiplib::Jim::reset "void
jiplib::Jim::reset(void)

reset all member variables ";

%feature("docstring")  jiplib::Jim::nrOfPlane "int
jiplib::Jim::nrOfPlane(void) const

Get the number of planes of this dataset. ";

%feature("docstring")  jiplib::Jim::band2plane "CPLErr
jiplib::Jim::band2plane()

convert single plane multiband image to single band image with
multiple planes ";

%feature("docstring")  jiplib::Jim::plane2band "CPLErr
jiplib::Jim::plane2band()

convert single band multiple plane image to single plane multiband
image ";

%feature("docstring")  jiplib::Jim::getMIA "IMAGE * Jim::getMIA(int
band=0)

get MIA representation for a particular band

Parameters:
-----------

band:  the band to get the MIA image representation for

pointer to MIA image representation ";

%feature("docstring")  jiplib::Jim::setMIA "CPLErr Jim::setMIA(int
band=0)

set memory from internal MIA representation for particular band

Parameters:
-----------

band:  the band for which the MIA image pointer needs to be set

C_None if successful ";

%feature("docstring")  jiplib::Jim::setMIA "CPLErr Jim::setMIA(IMAGE
*mia, int band=0)

Parameters:
-----------

mia:  the MIA image pointer to be set

band:  the band for which the MIA image pointer needs to be set

C_None if successful ";

%feature("docstring")  jiplib::Jim::GDAL2MIADataType "int
jiplib::Jim::GDAL2MIADataType(GDALDataType aGDALDataType)

convert a GDAL data type to MIA data type

Parameters:
-----------

aGDALDataType:

MIA data type ";

%feature("docstring")  jiplib::Jim::MIA2GDALDataType "GDALDataType
jiplib::Jim::MIA2GDALDataType(int aMIADataType)

convert a MIA data type to GDAL data type

Parameters:
-----------

aMIADataType:  the MIA data type to be converted

GDAL data type (GDT_Byte, GDT_UInt16, GDT_Int16, GDT_UInt32,
GDT_Int32, GDT_Float32, GDT_Float64) ";

%feature("docstring")  jiplib::Jim::crop "std::shared_ptr<Jim>
jiplib::Jim::crop(app::AppFactory &app)

crop Jim image in memory returning Jim image

Parameters:
-----------

app:  application specific option arguments

output image ";

%feature("docstring")  jiplib::Jim::getShared "std::shared_ptr<Jim>
jiplib::Jim::getShared() ";

%feature("docstring")  jiplib::Jim::azimuth "CPLErr Jim::azimuth(Jim
&imRaster_iy, int iband=0) ";

%feature("docstring")  jiplib::Jim::mapori "CPLErr Jim::mapori(int
ox, int oy, int iband=0) ";

%feature("docstring")  jiplib::Jim::dir "CPLErr Jim::dir(int graph,
int iband=0) ";

%feature("docstring")  jiplib::Jim::cboutlet "CPLErr
Jim::cboutlet(Jim &imRaster_d8, int iband=0) ";

%feature("docstring")  jiplib::Jim::cbconfluence "CPLErr
Jim::cbconfluence(Jim &imRaster_d8, int iband=0) ";

%feature("docstring")  jiplib::Jim::strahler "CPLErr
Jim::strahler(int iband=0) ";

%feature("docstring")  jiplib::Jim::FlatIGeodAFAB "CPLErr
Jim::FlatIGeodAFAB(Jim &imRaster_im, int graph, int iband=0) ";

%feature("docstring")  jiplib::Jim::dst2d4 "CPLErr Jim::dst2d4(int
iband=0) ";

%feature("docstring")  jiplib::Jim::dst2dchamfer "CPLErr
Jim::dst2dchamfer(int iband=0) ";

%feature("docstring")  jiplib::Jim::chamfer2d "CPLErr
Jim::chamfer2d(int type, int iband=0) ";

%feature("docstring")  jiplib::Jim::oiiz "CPLErr Jim::oiiz(int
iband=0) ";

%feature("docstring")  jiplib::Jim::geodist "CPLErr Jim::geodist(Jim
&imRaster_im_r, int graph, int iband=0) ";

%feature("docstring")  jiplib::Jim::linero "CPLErr Jim::linero(int
dx, int dy, int n, int line_type, int iband=0) ";

%feature("docstring")  jiplib::Jim::lindil "CPLErr Jim::lindil(int
dx, int dy, int n, int line_type, int iband=0) ";

%feature("docstring")  jiplib::Jim::herkpldil "CPLErr
Jim::herkpldil(int dx, int dy, int k, int o, int t, int iband=0) ";

%feature("docstring")  jiplib::Jim::herkplero "CPLErr
Jim::herkplero(int dx, int dy, int k, int o, int t, int iband=0) ";

%feature("docstring")  jiplib::Jim::linerank "CPLErr
Jim::linerank(int dx, int dy, int k, int rank, int o, int iband=0) ";

%feature("docstring")  jiplib::Jim::to_uchar "CPLErr
Jim::to_uchar(int iband=0) ";

%feature("docstring")  jiplib::Jim::dbltofloat "CPLErr
Jim::dbltofloat(int iband=0) ";

%feature("docstring")  jiplib::Jim::uint32_to_float "CPLErr
Jim::uint32_to_float(int iband=0) ";

%feature("docstring")  jiplib::Jim::swap "CPLErr Jim::swap(int
iband=0) ";

%feature("docstring")  jiplib::Jim::rdil "CPLErr Jim::rdil(Jim
&imRaster_mask, int graph, int flag, int iband=0) ";

%feature("docstring")  jiplib::Jim::rero "CPLErr Jim::rero(Jim
&imRaster_mask, int graph, int flag, int iband=0) ";

%feature("docstring")  jiplib::Jim::rerodilp "CPLErr
Jim::rerodilp(Jim &imRaster_mask, int graph, int flag, int version,
int iband=0) ";

%feature("docstring")  jiplib::Jim::complete "CPLErr
Jim::complete(Jim &imRaster_im_rmin, int graph, int iband=0) ";

%feature("docstring")  jiplib::Jim::sqtgpla "CPLErr Jim::sqtgpla(Jim
&imRaster_im_r, int graph, int iband=0) ";

%feature("docstring")  jiplib::Jim::sqtg "CPLErr Jim::sqtg(Jim
&imRaster_im_r, int graph, int iband=0) ";

%feature("docstring")  jiplib::Jim::framebox "CPLErr
Jim::framebox(int *box, double d_gval, int iband=0) ";

%feature("docstring")  jiplib::Jim::addframebox "CPLErr
Jim::addframebox(int *box, double d_gval, int iband=0) ";

%feature("docstring")  jiplib::Jim::subframebox "CPLErr
Jim::subframebox(int *box, int iband=0) ";

%feature("docstring")  jiplib::Jim::dumpxyz "CPLErr Jim::dumpxyz(int
x, int y, int z, int dx, int dy, int iband=0) ";

%feature("docstring")  jiplib::Jim::imputop "CPLErr Jim::imputop(Jim
&imRaster_im2, int x, int y, int z, int op, int iband=0) ";

%feature("docstring")  jiplib::Jim::imputcompose "CPLErr
Jim::imputcompose(Jim &imRaster_imlbl, Jim &imRaster_im2, int x, int
y, int z, int val, int iband=0) ";

%feature("docstring")  jiplib::Jim::szcompat "CPLErr
Jim::szcompat(Jim &imRaster_im2, int iband=0) ";

%feature("docstring")  jiplib::Jim::szgeocompat "CPLErr
Jim::szgeocompat(Jim &imRaster_im2, int iband=0) ";

%feature("docstring")  jiplib::Jim::plotline "CPLErr
Jim::plotline(int x1, int y1, int x2, int y2, int val, int iband=0) ";

%feature("docstring")  jiplib::Jim::ovlmatrix "CPLErr
Jim::ovlmatrix(Jim &imRaster_maxg_array, char *odir, int iband=0) ";

%feature("docstring")  jiplib::Jim::skeleton "CPLErr
Jim::skeleton(int iband=0) ";

%feature("docstring")  jiplib::Jim::bprune "CPLErr Jim::bprune(int
occa, int graph, int iband=0) ";

%feature("docstring")  jiplib::Jim::oiskeleton "CPLErr
Jim::oiskeleton(Jim &imRaster_imanchor, int iband=0) ";

%feature("docstring")  jiplib::Jim::oiask "CPLErr Jim::oiask(Jim
&imRaster_imanchor, int iband=0) ";

%feature("docstring")  jiplib::Jim::binODthin_noqueue "CPLErr
Jim::binODthin_noqueue(int stype, int atype, Jim &imRaster_imanchor,
int iband=0) ";

%feature("docstring")  jiplib::Jim::binODthin_FIFO "CPLErr
Jim::binODthin_FIFO(int stype, int atype, Jim &imRaster_imanchor, int
iband=0) ";

%feature("docstring")  jiplib::Jim::binOIthin_noqueue "CPLErr
Jim::binOIthin_noqueue(int stype, int atype, Jim &imRaster_imanchor,
int iband=0) ";

%feature("docstring")  jiplib::Jim::binOIthin_FIFO "CPLErr
Jim::binOIthin_FIFO(int stype, int atype, Jim &imRaster_imanchor, int
iband=0) ";

%feature("docstring")  jiplib::Jim::iminfo "CPLErr Jim::iminfo(int
iband=0) ";

%feature("docstring")  jiplib::Jim::copy_lut "CPLErr
Jim::copy_lut(Jim &imRaster_im2, int iband=0) ";

%feature("docstring")  jiplib::Jim::create_lut "CPLErr
Jim::create_lut(int iband=0) ";

%feature("docstring")  jiplib::Jim::setpixval "CPLErr
Jim::setpixval(unsigned long offset, double d_g, int iband=0) ";

%feature("docstring")  jiplib::Jim::write_ColorMap_tiff "CPLErr
Jim::write_ColorMap_tiff(char *fn, int iband=0) ";

%feature("docstring")  jiplib::Jim::write_tiff "CPLErr
Jim::write_tiff(char *fn, int iband=0) ";

%feature("docstring")  jiplib::Jim::writeTiffOneStripPerLine "CPLErr
Jim::writeTiffOneStripPerLine(char *fn, char *desc, int iband=0) ";

%feature("docstring")  jiplib::Jim::writeGeoTiffOneStripPerLine "CPLErr Jim::writeGeoTiffOneStripPerLine(char *fn, int PCSCode, double
xoff, double yoff, double scale, unsigned short RasterType, int
nodata_flag, int nodata_val, int metadata_flag, char *metadata_str,
int iband=0) ";

%feature("docstring")  jiplib::Jim::label "CPLErr Jim::label(Jim
&imRaster_im2, int ox, int oy, int oz, int iband=0) ";

%feature("docstring")  jiplib::Jim::labelpixngb "CPLErr
Jim::labelpixngb(Jim &imRaster_im2, int ox, int oy, int oz, int
iband=0) ";

%feature("docstring")  jiplib::Jim::labelplat "CPLErr
Jim::labelplat(Jim &imRaster_im2, int ox, int oy, int oz, int iband=0)
";

%feature("docstring")  jiplib::Jim::seededlabelplat "CPLErr
Jim::seededlabelplat(Jim &imRaster_im2, Jim &imRaster_im3, int ox, int
oy, int oz, int iband=0) ";

%feature("docstring")  jiplib::Jim::seededplat "CPLErr
Jim::seededplat(Jim &imRaster_im2, Jim &imRaster_im3, int ox, int oy,
int oz, int iband=0) ";

%feature("docstring")  jiplib::Jim::labelpix "CPLErr
Jim::labelpix(int iband=0) ";

%feature("docstring")  jiplib::Jim::resolveLabels "CPLErr
Jim::resolveLabels(Jim &imRaster_imlut, Jim &imRaster_imlutback, int
graph, int iband=0) ";

%feature("docstring")  jiplib::Jim::gorder "CPLErr Jim::gorder(Jim
&imRaster_g, int n, int iband=0) ";

%feature("docstring")  jiplib::Jim::propagate "CPLErr
Jim::propagate(Jim &imRaster_dst, IMAGE **imap, int nc, int graph, int
iband=0) ";

%feature("docstring")  jiplib::Jim::set_regions "CPLErr
Jim::set_regions(Jim &imRaster_ival, int indic, int iband=0) ";

%feature("docstring")  jiplib::Jim::setregionsgraph "CPLErr
Jim::setregionsgraph(Jim &imRaster_ival, int indic, int graph, int
iband=0) ";

%feature("docstring")  jiplib::Jim::tessel_surface "CPLErr
Jim::tessel_surface(int iband=0) ";

%feature("docstring")  jiplib::Jim::relabel "CPLErr Jim::relabel(Jim
&imRaster_ilbl2, Jim &imRaster_iarea2, int iband=0) ";

%feature("docstring")  jiplib::Jim::bitwise_op "CPLErr
Jim::bitwise_op(Jim &imRaster_im2, int op, int iband=0) ";

%feature("docstring")  jiplib::Jim::negation "CPLErr
Jim::negation(int iband=0) ";

%feature("docstring")  jiplib::Jim::arith "CPLErr Jim::arith(Jim
&imRaster_im2, int op, int iband=0) ";

%feature("docstring")  jiplib::Jim::arithcst "CPLErr
Jim::arithcst(double d_gt, int op, int iband=0) ";

%feature("docstring")  jiplib::Jim::imabs "CPLErr Jim::imabs(int
iband=0) ";

%feature("docstring")  jiplib::Jim::imsqrt "CPLErr Jim::imsqrt(int
iband=0) ";

%feature("docstring")  jiplib::Jim::imlog "CPLErr Jim::imlog(int
iband=0) ";

%feature("docstring")  jiplib::Jim::imatan "CPLErr Jim::imatan(int
iband=0) ";

%feature("docstring")  jiplib::Jim::imcos "CPLErr Jim::imcos(int
iband=0) ";

%feature("docstring")  jiplib::Jim::imacos "CPLErr Jim::imacos(int
iband=0) ";

%feature("docstring")  jiplib::Jim::imsin "CPLErr Jim::imsin(int
iband=0) ";

%feature("docstring")  jiplib::Jim::imasin "CPLErr Jim::imasin(int
iband=0) ";

%feature("docstring")  jiplib::Jim::thresh "CPLErr Jim::thresh(double
d_gt1, double d_gt2, double d_gbg, double d_gfg, int iband=0) ";

%feature("docstring")  jiplib::Jim::setlevel "CPLErr
Jim::setlevel(double d_gt1, double d_gt2, double d_gval, int iband=0)
";

%feature("docstring")  jiplib::Jim::modulo "CPLErr Jim::modulo(int
val, int iband=0) ";

%feature("docstring")  jiplib::Jim::complement "CPLErr
Jim::complement(int iband=0) ";

%feature("docstring")  jiplib::Jim::power2p "CPLErr Jim::power2p(int
iband=0) ";

%feature("docstring")  jiplib::Jim::blank "CPLErr Jim::blank(double
d_gval, int iband=0) ";

%feature("docstring")  jiplib::Jim::shift "CPLErr Jim::shift(int val,
int iband=0) ";

%feature("docstring")  jiplib::Jim::setrange "CPLErr
Jim::setrange(double d_gt1, double d_gt2, int iband=0) ";

%feature("docstring")  jiplib::Jim::FindPixWithVal "CPLErr
Jim::FindPixWithVal(double d_gval, unsigned long int *ofs, int
iband=0) ";

%feature("docstring")  jiplib::Jim::wsfah "CPLErr Jim::wsfah(Jim
&imRaster_imr, int graph, int maxfl, int iband=0) ";

%feature("docstring")  jiplib::Jim::skelfah "CPLErr Jim::skelfah(Jim
&imRaster_imr, Jim &imRaster_imdir, int graph, int maxfl, int iband=0)
";

%feature("docstring")  jiplib::Jim::skelfah2 "CPLErr
Jim::skelfah2(Jim &imRaster_impskp, int n, int graph, int iband=0) ";

%feature("docstring")  jiplib::Jim::compose "CPLErr Jim::compose(Jim
&imRaster_mask, Jim &imRaster_g, Jim &imRaster_lbl, int graph, int
iband=0) ";

%feature("docstring")  jiplib::Jim::oiws "CPLErr Jim::oiws(int
iband=0) ";

%feature("docstring")  jiplib::Jim::srg "CPLErr Jim::srg(Jim
&imRaster_im2, Jim &imRaster_im3, int ox, int oy, int oz, int iband=0)
";

%feature("docstring")  jiplib::Jim::IsPartitionEqual "CPLErr
Jim::IsPartitionEqual(Jim &imRaster_im2, int *result, int iband=0) ";

%feature("docstring")  jiplib::Jim::IsPartitionFiner "CPLErr
Jim::IsPartitionFiner(Jim &imRaster_im2, int graph, unsigned long int
*res, int iband=0) ";

%feature("docstring")  jiplib::Jim::getfirstmaxpos "CPLErr
Jim::getfirstmaxpos(unsigned long int *pos, int iband=0) ";

%feature("docstring")  jiplib::Jim::histcompress "CPLErr
Jim::histcompress(int iband=0) ";

%feature("docstring")  jiplib::Jim::lookup "CPLErr Jim::lookup(Jim
&imRaster_imlut, int iband=0) ";

%feature("docstring")  jiplib::Jim::lookuptypematch "CPLErr
Jim::lookuptypematch(Jim &imRaster_imlut, int iband=0) ";

%feature("docstring")  jiplib::Jim::volume "CPLErr Jim::volume(int
iband=0) ";

%feature("docstring")  jiplib::Jim::dirmax "CPLErr Jim::dirmax(int
dir, int iband=0) ";

%feature("docstring")  jiplib::Jim::imequalp "CPLErr
Jim::imequalp(Jim &imRaster_im2, int iband=0) ";

%feature("docstring")  jiplib::Jim::getmax "CPLErr Jim::getmax(double
*maxval, int iband=0) ";

%feature("docstring")  jiplib::Jim::getminmax "CPLErr
Jim::getminmax(double *minval, double *maxval, int iband=0) ";

%feature("docstring")  jiplib::Jim::classstatsinfo "CPLErr
Jim::classstatsinfo(Jim &imRaster_imin, int iband=0) ";

%feature("docstring")  jiplib::Jim::clmindist "CPLErr
Jim::clmindist(Jim &imRaster_imin, int bklabel, int mode, double thr,
int iband=0) ";

%feature("docstring")  jiplib::Jim::clparpip "CPLErr
Jim::clparpip(Jim &imRaster_imin, int bklabel, int mode, double mult,
int iband=0) ";

%feature("docstring")  jiplib::Jim::clmaha "CPLErr Jim::clmaha(Jim
&imRaster_imin, int bklabel, int mode, double thr, int iband=0) ";

%feature("docstring")  jiplib::Jim::clmaxlike "CPLErr
Jim::clmaxlike(Jim &imRaster_imin, int bklabel, int type, double thr,
int iband=0) ";

%feature("docstring")  jiplib::Jim::getVolume "double
jiplib::Jim::getVolume(int iband=0)

get volume (from mialib) ";

%feature("docstring")  jiplib::Jim::filter "std::shared_ptr<Jim>
jiplib::Jim::filter(app::AppFactory &theApp)

filter Jim image and return filtered image as shared pointer

Parameters:
-----------

input:  (type: std::string)Input image file(s). If input contains
multiple images, a multi-band output is created

output:  (type: std::string)Output image file

oformat:  (type: std::string) (default: GTiff) Output image format
(see also gdal_translate).

co:  (type: std::string)Creation option for output file. Multiple
options can be specified.

a_srs:  (type: std::string)Override the spatial reference for the
output file (leave blank to copy from input file, use epsg:3035 to use
European projection and force to European grid

mem:  (type: unsigned long) (default: 0) Buffer size (in MB) to read
image data blocks in memory

a_srs:  (type: std::string)Override the projection for the output file
(leave blank to copy from input file, use epsg:3035 to use European
projection and force to European grid

ulx:  (type: double) (default: 0) Upper left x value bounding box

uly:  (type: double) (default: 0) Upper left y value bounding box

lrx:  (type: double) (default: 0) Lower right x value bounding box

lry:  (type: double) (default: 0) Lower right y value bounding box

band:  (type: unsigned int)band index to crop (leave empty to retain
all bands)

startband:  (type: unsigned int)Start band sequence number

endband:  (type: unsigned int)End band sequence number

autoscale:  (type: double)scale output to min and max, e.g.,autoscale
0autoscale 255

otype:  (type: std::string)Data type for output image
({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}).
Empty string: inherit type from input image

oformat:  (type: std::string) (default: GTiff) Output image format
(see also gdal_translate).

ct:  (type: std::string)color table (file with 5 columns: id R G B
ALFA (0: transparent, 255: solid)

dx:  (type: double)Output resolution in x (in meter) (empty: keep
original resolution)

dy:  (type: double)Output resolution in y (in meter) (empty: keep
original resolution)

resampling-method:  (type: std::string) (default: near) Resampling
method (near: nearest neighbor, bilinear: bi-linear interpolation).

extent:  (type: std::string)get boundary from extent from polygons in
vector file

crop_to_cutline:  (type: bool) (default: 0) Crop the extent of the
target dataset to the extent of the cutline.

eo:  (type: std::string)special extent options controlling
rasterization:
ATTRIBUTE|CHUNKYSIZE|ALL_TOUCHED|BURN_VALUE_FROM|MERGE_ALG, e.g., -eo
ATTRIBUTE=fieldname

mask:  (type: std::string)Use the the specified file as a validity
mask (0 is nodata).

msknodata:  (type: double) (default: 0) Mask value not to consider for
crop.

mskband:  (type: unsigned int) (default: 0) Mask band to read (0
indexed)

x:  (type: double)x-coordinate of image center to crop (in meter)

y:  (type: double)y-coordinate of image center to crop (in meter)

nx:  (type: double)image size in x to crop (in meter)

ny:  (type: double)image size in y to crop (in meter)

ns:  (type: unsigned int)number of samples to crop (in pixels)

nl:  (type: unsigned int)number of lines to crop (in pixels)

scale:  (type: double)output=scale*input+offset

offset:  (type: double)output=scale*input+offset

nodata:  (type: double)Nodata value to put in image if out of bounds.

description:  (type: std::string)Set image description

align:  (type: bool) (default: 0) Align output bounding box to input
image ";

%feature("docstring")  jiplib::Jim::diff "std::shared_ptr<Jim>
jiplib::Jim::diff(app::AppFactory &app)

create statistical profile from a collection

Parameters:
-----------

input:  (type: std::string) input image

reference:  (type: std::string) Reference (raster or vector) dataset

mem:  (type: unsigned long) (default: 0) Buffer size (in MB) to read
image data blocks in memory

band:  (type: unsigned int) (default: 0) Input (reference) raster
band. Optionally, you can define different bands for input and
reference bands respectively: -b 1 -b 0.

rmse:  (type: bool) (default: 0) Report root mean squared error

reg:  (type: bool) (default: 0) Report linear regression (Input =
c0+c1*Reference)

confusion:  (type: bool) (default: 0) Create confusion matrix (to std
out)

class:  (type: std::string) List of class names.

reclass:  (type: short) List of class values (use same order as in
classname option).

nodata:  (type: double) No data value(s) in input or reference dataset
are ignored

mask:  (type: std::string) Use the first band of the specified file as
a validity mask. Nodata values can be set with the option msknodata.

msknodata:  (type: double) (default: 0) Mask value(s) where image is
invalid. Use negative value for valid data (example: use -t -1: if
only -1 is valid value)

output:  (type: std::string) Output dataset (optional)

cmf:  (type: std::string) (default: ascii) Format for confusion matrix
(ascii or latex)

cmo:  (type: std::string) Output file for confusion matrix

se95:  (type: bool) (default: 0) Report standard error for 95
confidence interval

ct:  (type: std::string) Color table in ASCII format having 5 columns:
id R G B ALFA (0: transparent, 255: solid).

commission:  (type: short) (default: 2) Value for commission errors:
input label < reference label

output image ";

%feature("docstring")  jiplib::Jim::svm "std::shared_ptr<Jim>
jiplib::Jim::svm(app::AppFactory &app)

supervised classification using support vector machine (train with
extractImg/extractOgr)

Parameters:
-----------

input:  (type: std::string) input image

output:  (type: std::string) Output classification image

oformat:  (type: std::string) (default: GTiff) Output image format
(see also gdal_translate).

co:  (type: std::string) Creation option for output file. Multiple
options can be specified.

mem:  (type: unsigned long) (default: 0) Buffer size (in MB) to read
image data blocks in memory

training:  (type: std::string) Training vector file. A single vector
file contains all training features (must be set as: b0, b1, b2,...)
for all classes (class numbers identified by label option). Use
multiple training files for bootstrap aggregation (alternative to the
bag and bsize options, where a random subset is taken from a single
training file)

cv:  (type: unsigned short) (default: 0) N-fold cross validation mode

cmf:  (type: std::string) (default: ascii) Format for confusion matrix
(ascii or latex)

tln:  (type: std::string) Training layer name(s)

class:  (type: std::string) List of class names.

reclass:  (type: short) List of class values (use same order as in
class opt).

f:  (type: std::string) (default: SQLite) Output ogr format for active
training sample

ct:  (type: std::string) Color table in ASCII format having 5 columns:
id R G B ALFA (0: transparent, 255: solid)

label:  (type: std::string) (default: label) Attribute name for class
label in training vector file.

prior:  (type: double) (default: 0) Prior probabilities for each class
(e.g., -p 0.3 -p 0.3 -p 0.2 ). Used for input only (ignored for cross
validation)

gamma:  (type: float) (default: 1) Gamma in kernel function

ccost:  (type: float) (default: 1000) The parameter C of C_SVC,
epsilon_SVR, and nu_SVR

extent:  (type: std::string) Only classify within extent from polygons
in vector file

eo:  (type: std::string) special extent options controlling
rasterization:
ATTRIBUTE|CHUNKYSIZE|ALL_TOUCHED|BURN_VALUE_FROM|MERGE_ALG, e.g., -eo
ATTRIBUTE=fieldname

mask:  (type: std::string) Only classify within specified mask. For
raster mask, set nodata values with the option msknodata.

msknodata:  (type: short) (default: 0) Mask value(s) not to consider
for classification. Values will be taken over in classification image.

nodata:  (type: unsigned short) (default: 0) Nodata value to put where
image is masked as nodata

band:  (type: unsigned int) Band index (starting from 0, either use
band option or use start to end)

startband:  (type: unsigned int) Start band sequence number

endband:  (type: unsigned int) End band sequence number

balance:  (type: unsigned int) (default: 0) Balance the input data to
this number of samples for each class

min:  (type: unsigned int) (default: 0) If number of training pixels
is less then min, do not take this class into account (0: consider all
classes)

bag:  (type: unsigned short) (default: 1) Number of bootstrap
aggregations

bagsize:  (type: int) (default: 100) Percentage of features used from
available training features for each bootstrap aggregation (one size
for all classes, or a different size for each class respectively

comb:  (type: unsigned short) (default: 0) How to combine bootstrap
aggregation classifiers (0: sum rule, 1: product rule, 2: max rule).
Also used to aggregate classes with rc option.

classbag:  (type: std::string) Output for each individual bootstrap
aggregation

prob:  (type: std::string) Probability image.

priorimg:  (type: std::string) (default: ) Prior probability image
(multi-band img with band for each class

offset:  (type: double) (default: 0) Offset value for each spectral
band input features: refl[band]=(DN[band]-offset[band])/scale[band]

scale:  (type: double) (default: 0) Scale value for each spectral band
input features: refl=(DN[band]-offset[band])/scaleband

svmtype:  (type: std::string) (default: C_SVC) Type of SVM (C_SVC,
nu_SVC,one_class, epsilon_SVR, nu_SVR)

kerneltype:  (type: std::string) (default: radial) Type of kernel
function (linear,polynomial,radial,sigmoid)

kd:  (type: unsigned short) (default: 3) Degree in kernel function

coef0:  (type: float) (default: 0) Coef0 in kernel function

nu:  (type: float) (default: 0.5) The parameter nu of nu_SVC,
one_class SVM, and nu_SVR

eloss:  (type: float) (default: 0.1) The epsilon in loss function of
epsilon_SVR

cache:  (type: int) (default: 100) Cache memory size in MB

etol:  (type: float) (default: 0.001) The tolerance of termination
criterion

shrink:  (type: bool) (default: 0) Whether to use the shrinking
heuristics

probest:  (type: bool) (default: 1) Whether to train a SVC or SVR
model for probability estimates

entropy:  (type: std::string) (default: ) Entropy image (measure for
uncertainty of classifier output

active:  (type: std::string) (default: ) Ogr output for active
training sample.

nactive:  (type: unsigned int) (default: 1) Number of active training
points

random:  (type: bool) (default: 1) Randomize training data for
balancing and bagging

output image ";

%feature("docstring")  jiplib::Jim::stretch "std::shared_ptr<Jim>
jiplib::Jim::stretch(app::AppFactory &app)

stretch Jim image and return stretched image as shared pointer ";


// File: classjiplib_1_1JimList.xml
%feature("docstring") jiplib::JimList "C++ includes: jim.h ";

%feature("docstring")  jiplib::JimList::JimList "jiplib::JimList::JimList() ";

%feature("docstring")  jiplib::JimList::JimList "jiplib::JimList::JimList(const std::vector< std::shared_ptr< Jim > >
&jimVector)

constructor using vector of images ";

%feature("docstring")  jiplib::JimList::pushImage "void
jiplib::JimList::pushImage(const std::shared_ptr< Jim > imgRaster)

push image to collection ";

%feature("docstring")  jiplib::JimList::getImage "const
std::shared_ptr<Jim> jiplib::JimList::getImage(int index)

get image from collection ";

%feature("docstring")  jiplib::JimList::composite "std::shared_ptr<Jim> jiplib::JimList::composite(app::AppFactory &app)

composite image only for in memory

Parameters:
-----------

input:  (type: std::string) Input image file(s). If input contains
multiple images, a multi-band output is created

output:  (type: std::string) Output image file

oformat:  (type: std::string) (default: GTiff) Output image format
(see also gdal_translate).

co:  (type: std::string) Creation option for output file. Multiple
options can be specified.

scale:  (type: double) output=scale*input+offset

offset:  (type: double) output=scale*input+offset

mem:  (type: unsigned long) (default: 0) Buffer size (in MB) to read
image data blocks in memory

band:  (type: unsigned int) band index(es) to crop (leave empty if all
bands must be retained)

dx:  (type: double) Output resolution in x (in meter) (empty: keep
original resolution)

dy:  (type: double) Output resolution in y (in meter) (empty: keep
original resolution)

extent:  (type: std::string) get boundary from extent from polygons in
vector file

crop_to_cutline:  (type: bool) (default: 0) Crop the extent of the
target dataset to the extent of the cutline.

eo:  (type: std::string) special extent options controlling
rasterization:
ATTRIBUTE|CHUNKYSIZE|ALL_TOUCHED|BURN_VALUE_FROM|MERGE_ALG, e.g., -eo
ATTRIBUTE=fieldname

mask:  (type: std::string) Use the specified file as a validity mask.

msknodata:  (type: float) (default: 0) Mask value not to consider for
composite.

mskband:  (type: unsigned int) (default: 0) Mask band to read (0
indexed)

ulx:  (type: double) (default: 0) Upper left x value bounding box

uly:  (type: double) (default: 0) Upper left y value bounding box

lrx:  (type: double) (default: 0) Lower right x value bounding box

lry:  (type: double) (default: 0) Lower right y value bounding box

crule:  (type: std::string) (default: overwrite) Composite rule
(overwrite, maxndvi, maxband, minband, mean, mode (only for byte
images), median, sum, maxallbands, minallbands, stdev

cband:  (type: unsigned int) (default: 0) band index used for the
composite rule (e.g., for ndvi, usecband=0cband=1 with 0 and 1 indices
for red and nir band respectively

srcnodata:  (type: double) invalid value(s) for input raster dataset

bndnodata:  (type: unsigned int) (default: 0) Band(s) in input image
to check if pixel is valid (used for srcnodata, min and max options)

min:  (type: double) flag values smaller or equal to this value as
invalid.

max:  (type: double) flag values larger or equal to this value as
invalid.

dstnodata:  (type: double) (default: 0) nodata value to put in output
raster dataset if not valid or out of bounds.

resampling-method:  (type: std::string) (default: near) Resampling
method (near: nearest neighbor, bilinear: bi-linear interpolation).

otype:  (type: std::string) Data type for output image
({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}).
Empty string: inherit type from input image

a_srs:  (type: std::string) Override the spatial reference for the
output file (leave blank to copy from input file, use epsg:3035 to use
European projection and force to European grid

file:  (type: short) (default: 0) write number of observations (1) or
sequence nr of selected file (2) for each pixels as additional layer
in composite

weight:  (type: short) (default: 1) Weights (type: short) for the
composite, use one weight for each input file in same order as input
files are provided). Use value 1 for equal weights.

class:  (type: short) (default: 0) classes for multi-band output
image: each band represents the number of observations for one
specific class. Use value 0 for no multi-band output image.

ct:  (type: std::string) color table file with 5 columns: id R G B
ALFA (0: transparent, 255: solid)

description:  (type: std::string) Set image description

align:  (type: bool) (default: 0) Align output bounding box to input
image

output image ";

%feature("docstring")  jiplib::JimList::crop "std::shared_ptr<Jim>
jiplib::JimList::crop(app::AppFactory &app)

Parameters:
-----------

input:  (type: std::string) Input image file(s). If input contains
multiple images, a multi-band output is created

output:  (type: std::string) Output image file

oformat:  (type: std::string) (default: GTiff) Output image format
(see also gdal_translate).

co:  (type: std::string) Creation option for output file. Multiple
options can be specified.

a_srs:  (type: std::string) Override the spatial reference for the
output file (leave blank to copy from input file, use epsg:3035 to use
European projection and force to European grid

mem:  (type: unsigned long) (default: 0) Buffer size (in MB) to read
image data blocks in memory

a_srs:  (type: std::string) Override the projection for the output
file (leave blank to copy from input file, use epsg:3035 to use
European projection and force to European grid

ulx:  (type: double) (default: 0) Upper left x value bounding box

uly:  (type: double) (default: 0) Upper left y value bounding box

lrx:  (type: double) (default: 0) Lower right x value bounding box

lry:  (type: double) (default: 0) Lower right y value bounding box

band:  (type: unsigned int) band index to crop (leave empty to retain
all bands)

startband:  (type: unsigned int) Start band sequence number

endband:  (type: unsigned int) End band sequence number

autoscale:  (type: double) scale output to min and max, e.g.,autoscale
0autoscale 255

otype:  (type: std::string) Data type for output image
({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}).
Empty string: inherit type from input image

oformat:  (type: std::string) (default: GTiff) Output image format
(see also gdal_translate).

ct:  (type: std::string) color table (file with 5 columns: id R G B
ALFA (0: transparent, 255: solid)

dx:  (type: double) Output resolution in x (in meter) (empty: keep
original resolution)

dy:  (type: double) Output resolution in y (in meter) (empty: keep
original resolution)

resampling-method:  (type: std::string) (default: near) Resampling
method (near: nearest neighbor, bilinear: bi-linear interpolation).

extent:  (type: std::string) get boundary from extent from polygons in
vector file

crop_to_cutline:  (type: bool) (default: 0) Crop the extent of the
target dataset to the extent of the cutline.

eo:  (type: std::string) special extent options controlling
rasterization:
ATTRIBUTE|CHUNKYSIZE|ALL_TOUCHED|BURN_VALUE_FROM|MERGE_ALG, e.g., -eo
ATTRIBUTE=fieldname

mask:  (type: std::string) Use the the specified file as a validity
mask (0 is nodata).

msknodata:  (type: double) (default: 0) Mask value not to consider for
crop.

mskband:  (type: unsigned int) (default: 0) Mask band to read (0
indexed)

x:  (type: double) x-coordinate of image center to crop (in meter)

y:  (type: double) y-coordinate of image center to crop (in meter)

nx:  (type: double) image size in x to crop (in meter)

ny:  (type: double) image size in y to crop (in meter)

ns:  (type: unsigned int) number of samples to crop (in pixels)

nl:  (type: unsigned int) number of lines to crop (in pixels)

scale:  (type: double) output=scale*input+offset

offset:  (type: double) output=scale*input+offset

nodata:  (type: double) Nodata value to put in image if out of bounds.

description:  (type: std::string) Set image description

align:  (type: bool) (default: 0) Align output bounding box to input
image

output image ";

%feature("docstring")  jiplib::JimList::stack "std::shared_ptr<Jim>
jiplib::JimList::stack(app::AppFactory &app)

stack all images in collection to multiband image (alias for crop)

Parameters:
-----------

input:  (type: std::string) Input image file(s). If input contains
multiple images, a multi-band output is created

output:  (type: std::string) Output image file

oformat:  (type: std::string) (default: GTiff) Output image format
(see also gdal_translate).

co:  (type: std::string) Creation option for output file. Multiple
options can be specified.

a_srs:  (type: std::string) Override the spatial reference for the
output file (leave blank to copy from input file, use epsg:3035 to use
European projection and force to European grid

mem:  (type: unsigned long) (default: 0) Buffer size (in MB) to read
image data blocks in memory

a_srs:  (type: std::string) Override the projection for the output
file (leave blank to copy from input file, use epsg:3035 to use
European projection and force to European grid

ulx:  (type: double) (default: 0) Upper left x value bounding box

uly:  (type: double) (default: 0) Upper left y value bounding box

lrx:  (type: double) (default: 0) Lower right x value bounding box

lry:  (type: double) (default: 0) Lower right y value bounding box

band:  (type: unsigned int) band index to stack (leave empty to retain
all bands)

startband:  (type: unsigned int) Start band sequence number

endband:  (type: unsigned int) End band sequence number

otype:  (type: std::string) Data type for output image
({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}).
Empty string: inherit type from input image

oformat:  (type: std::string) (default: GTiff) Output image format
(see also gdal_translate).

ct:  (type: std::string) color table (file with 5 columns: id R G B
ALFA (0: transparent, 255: solid)

dx:  (type: double) Output resolution in x (in meter) (empty: keep
original resolution)

dy:  (type: double) Output resolution in y (in meter) (empty: keep
original resolution)

resampling-method:  (type: std::string) (default: near) Resampling
method (near: nearest neighbor, bilinear: bi-linear interpolation).

extent:  (type: std::string) get boundary from extent from polygons in
vector file

crop_to_cutline:  (type: bool) (default: 0) Crop the extent of the
target dataset to the extent of the cutline.

eo:  (type: std::string) special extent options controlling
rasterization:
ATTRIBUTE|CHUNKYSIZE|ALL_TOUCHED|BURN_VALUE_FROM|MERGE_ALG, e.g., -eo
ATTRIBUTE=fieldname

mask:  (type: std::string) Use the the specified file as a validity
mask (0 is nodata).

msknodata:  (type: double) (default: 0) Mask value not to consider for
crop.

mskband:  (type: unsigned int) (default: 0) Mask band to read (0
indexed)

x:  (type: double) x-coordinate of image center to crop (in meter)

y:  (type: double) y-coordinate of image center to crop (in meter)

nx:  (type: double) image size in x to crop (in meter)

ny:  (type: double) image size in y to crop (in meter)

ns:  (type: unsigned int) number of samples to crop (in pixels)

nl:  (type: unsigned int) number of lines to crop (in pixels)

scale:  (type: double) output=scale*input+offset

offset:  (type: double) output=scale*input+offset

nodata:  (type: double) Nodata value to put in image if out of bounds.

description:  (type: std::string) Set image description

align:  (type: bool) (default: 0) Align output bounding box to input
image

output image ";

%feature("docstring")  jiplib::JimList::stack "std::shared_ptr<Jim>
jiplib::JimList::stack()

stack all images in collection to multiband image (alias for crop)

Parameters:
-----------

input:  (type: std::string) Input image file(s). If input contains
multiple images, a multi-band output is created

output:  (type: std::string) Output image file

oformat:  (type: std::string) (default: GTiff) Output image format
(see also gdal_translate).

co:  (type: std::string) Creation option for output file. Multiple
options can be specified.

a_srs:  (type: std::string) Override the spatial reference for the
output file (leave blank to copy from input file, use epsg:3035 to use
European projection and force to European grid

mem:  (type: unsigned long) (default: 0) Buffer size (in MB) to read
image data blocks in memory

a_srs:  (type: std::string) Override the projection for the output
file (leave blank to copy from input file, use epsg:3035 to use
European projection and force to European grid

ulx:  (type: double) (default: 0) Upper left x value bounding box

uly:  (type: double) (default: 0) Upper left y value bounding box

lrx:  (type: double) (default: 0) Lower right x value bounding box

lry:  (type: double) (default: 0) Lower right y value bounding box

band:  (type: unsigned int) band index to stack (leave empty to retain
all bands)

startband:  (type: unsigned int) Start band sequence number

endband:  (type: unsigned int) End band sequence number

otype:  (type: std::string) Data type for output image
({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}).
Empty string: inherit type from input image

oformat:  (type: std::string) (default: GTiff) Output image format
(see also gdal_translate).

ct:  (type: std::string) color table (file with 5 columns: id R G B
ALFA (0: transparent, 255: solid)

dx:  (type: double) Output resolution in x (in meter) (empty: keep
original resolution)

dy:  (type: double) Output resolution in y (in meter) (empty: keep
original resolution)

resampling-method:  (type: std::string) (default: near) Resampling
method (near: nearest neighbor, bilinear: bi-linear interpolation).

extent:  (type: std::string) get boundary from extent from polygons in
vector file

crop_to_cutline:  (type: bool) (default: 0) Crop the extent of the
target dataset to the extent of the cutline.

eo:  (type: std::string) special extent options controlling
rasterization:
ATTRIBUTE|CHUNKYSIZE|ALL_TOUCHED|BURN_VALUE_FROM|MERGE_ALG, e.g., -eo
ATTRIBUTE=fieldname

mask:  (type: std::string) Use the the specified file as a validity
mask (0 is nodata).

msknodata:  (type: double) (default: 0) Mask value not to consider for
crop.

mskband:  (type: unsigned int) (default: 0) Mask band to read (0
indexed)

x:  (type: double) x-coordinate of image center to crop (in meter)

y:  (type: double) y-coordinate of image center to crop (in meter)

nx:  (type: double) image size in x to crop (in meter)

ny:  (type: double) image size in y to crop (in meter)

ns:  (type: unsigned int) number of samples to crop (in pixels)

nl:  (type: unsigned int) number of lines to crop (in pixels)

scale:  (type: double) output=scale*input+offset

offset:  (type: double) output=scale*input+offset

nodata:  (type: double) Nodata value to put in image if out of bounds.

description:  (type: std::string) Set image description

align:  (type: bool) (default: 0) Align output bounding box to input
image

output image ";

%feature("docstring")  jiplib::JimList::statProfile "std::shared_ptr<Jim> jiplib::JimList::statProfile(app::AppFactory
&app)

create statistical profile from a collection

Parameters:
-----------

input:  (type: std::string) input image file

output:  (type: std::string) Output image file

oformat:  (type: std::string) (default: GTiff) Output image format
(see also gdal_translate)

mem:  (type: unsigned long) (default: 0) Buffer size (in MB) to read
image data blocks in memory

function:  (type: std::string) Statistics function (mean, median, var,
stdev, min, max, sum, mode (provide classes), ismin, ismax, proportion
(provide classes), percentile, nvalid

perc:  (type: double) Percentile value(s) used for rule percentile

nodata:  (type: double) nodata value(s)

otype:  (type: std::string) (default: GDT_Unknown) Data type for
output image
({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}).
Empty string: inherit type from input image

output image ";


// File: namespacefun2method.xml
%feature("docstring")  fun2method::fun2method "def
fun2method.fun2method converts MIALib C function declarations into
JIPLib C++ methods (outputfile_basename.cc file) and C++ method
declarations (outputfile_basename.h file).  Currently only convert
desctuctive functions, i.e., ERROR_TYPE functions, with IMAGE * as
first argument (IMAGE ** not yet taken into account).  :param
inputfile: string for input file containing extern declarations :param
outputfile_basename: string for output file basename (i.e. without
extension) :returns: True on success, False otherwise :rtype: ";

%feature("docstring")  fun2method::main "def fun2method.main";


// File: namespacejiplib.xml


// File: fun2method_8cc.xml


// File: fun2method_8h.xml
%feature("docstring")  azimuth "CPLErr azimuth(Jim &imRaster_iy, int
iband=0) ";

%feature("docstring")  mapori "CPLErr mapori(int ox, int oy, int
iband=0) ";

%feature("docstring")  dir "CPLErr dir(int graph, int iband=0) ";

%feature("docstring")  cboutlet "CPLErr cboutlet(Jim &imRaster_d8,
int iband=0) ";

%feature("docstring")  cbconfluence "CPLErr cbconfluence(Jim
&imRaster_d8, int iband=0) ";

%feature("docstring")  strahler "CPLErr strahler(int iband=0) ";

%feature("docstring")  FlatIGeodAFAB "CPLErr FlatIGeodAFAB(Jim
&imRaster_im, int graph, int iband=0) ";

%feature("docstring")  dst2d4 "CPLErr dst2d4(int iband=0) ";

%feature("docstring")  dst2dchamfer "CPLErr dst2dchamfer(int iband=0)
";

%feature("docstring")  chamfer2d "CPLErr chamfer2d(int type, int
iband=0) ";

%feature("docstring")  oiiz "CPLErr oiiz(int iband=0) ";

%feature("docstring")  geodist "CPLErr geodist(Jim &imRaster_im_r,
int graph, int iband=0) ";

%feature("docstring")  linero "CPLErr linero(int dx, int dy, int n,
int line_type, int iband=0) ";

%feature("docstring")  lindil "CPLErr lindil(int dx, int dy, int n,
int line_type, int iband=0) ";

%feature("docstring")  herkpldil "CPLErr herkpldil(int dx, int dy,
int k, int o, int t, int iband=0) ";

%feature("docstring")  herkplero "CPLErr herkplero(int dx, int dy,
int k, int o, int t, int iband=0) ";

%feature("docstring")  linerank "CPLErr linerank(int dx, int dy, int
k, int rank, int o, int iband=0) ";

%feature("docstring")  to_uchar "CPLErr to_uchar(int iband=0) ";

%feature("docstring")  dbltofloat "CPLErr dbltofloat(int iband=0) ";

%feature("docstring")  uint32_to_float "CPLErr uint32_to_float(int
iband=0) ";

%feature("docstring")  swap "CPLErr swap(int iband=0) ";

%feature("docstring")  rdil "CPLErr rdil(Jim &imRaster_mask, int
graph, int flag, int iband=0) ";

%feature("docstring")  rero "CPLErr rero(Jim &imRaster_mask, int
graph, int flag, int iband=0) ";

%feature("docstring")  rerodilp "CPLErr rerodilp(Jim &imRaster_mask,
int graph, int flag, int version, int iband=0) ";

%feature("docstring")  complete "CPLErr complete(Jim
&imRaster_im_rmin, int graph, int iband=0) ";

%feature("docstring")  sqtgpla "CPLErr sqtgpla(Jim &imRaster_im_r,
int graph, int iband=0) ";

%feature("docstring")  sqtg "CPLErr sqtg(Jim &imRaster_im_r, int
graph, int iband=0) ";

%feature("docstring")  framebox "CPLErr framebox(int *box, double
d_gval, int iband=0) ";

%feature("docstring")  addframebox "CPLErr addframebox(int *box,
double d_gval, int iband=0) ";

%feature("docstring")  subframebox "CPLErr subframebox(int *box, int
iband=0) ";

%feature("docstring")  dumpxyz "CPLErr dumpxyz(int x, int y, int z,
int dx, int dy, int iband=0) ";

%feature("docstring")  imputop "CPLErr imputop(Jim &imRaster_im2, int
x, int y, int z, int op, int iband=0) ";

%feature("docstring")  imputcompose "CPLErr imputcompose(Jim
&imRaster_imlbl, Jim &imRaster_im2, int x, int y, int z, int val, int
iband=0) ";

%feature("docstring")  szcompat "CPLErr szcompat(Jim &imRaster_im2,
int iband=0) ";

%feature("docstring")  szgeocompat "CPLErr szgeocompat(Jim
&imRaster_im2, int iband=0) ";

%feature("docstring")  plotline "CPLErr plotline(int x1, int y1, int
x2, int y2, int val, int iband=0) ";

%feature("docstring")  ovlmatrix "CPLErr ovlmatrix(Jim
&imRaster_maxg_array, char *odir, int iband=0) ";

%feature("docstring")  skeleton "CPLErr skeleton(int iband=0) ";

%feature("docstring")  bprune "CPLErr bprune(int occa, int graph, int
iband=0) ";

%feature("docstring")  oiskeleton "CPLErr oiskeleton(Jim
&imRaster_imanchor, int iband=0) ";

%feature("docstring")  oiask "CPLErr oiask(Jim &imRaster_imanchor,
int iband=0) ";

%feature("docstring")  binODthin_noqueue "CPLErr
binODthin_noqueue(int stype, int atype, Jim &imRaster_imanchor, int
iband=0) ";

%feature("docstring")  binODthin_FIFO "CPLErr binODthin_FIFO(int
stype, int atype, Jim &imRaster_imanchor, int iband=0) ";

%feature("docstring")  binOIthin_noqueue "CPLErr
binOIthin_noqueue(int stype, int atype, Jim &imRaster_imanchor, int
iband=0) ";

%feature("docstring")  binOIthin_FIFO "CPLErr binOIthin_FIFO(int
stype, int atype, Jim &imRaster_imanchor, int iband=0) ";

%feature("docstring")  iminfo "CPLErr iminfo(int iband=0) ";

%feature("docstring")  copy_lut "CPLErr copy_lut(Jim &imRaster_im2,
int iband=0) ";

%feature("docstring")  create_lut "CPLErr create_lut(int iband=0) ";

%feature("docstring")  setpixval "CPLErr setpixval(unsigned long
offset, double d_g, int iband=0) ";

%feature("docstring")  write_ColorMap_tiff "CPLErr
write_ColorMap_tiff(char *fn, int iband=0) ";

%feature("docstring")  write_tiff "CPLErr write_tiff(char *fn, int
iband=0) ";

%feature("docstring")  writeTiffOneStripPerLine "CPLErr
writeTiffOneStripPerLine(char *fn, char *desc, int iband=0) ";

%feature("docstring")  writeGeoTiffOneStripPerLine "CPLErr
writeGeoTiffOneStripPerLine(char *fn, int PCSCode, double xoff, double
yoff, double scale, unsigned short RasterType, int nodata_flag, int
nodata_val, int metadata_flag, char *metadata_str, int iband=0) ";

%feature("docstring")  label "CPLErr label(Jim &imRaster_im2, int ox,
int oy, int oz, int iband=0) ";

%feature("docstring")  labelpixngb "CPLErr labelpixngb(Jim
&imRaster_im2, int ox, int oy, int oz, int iband=0) ";

%feature("docstring")  labelplat "CPLErr labelplat(Jim &imRaster_im2,
int ox, int oy, int oz, int iband=0) ";

%feature("docstring")  seededlabelplat "CPLErr seededlabelplat(Jim
&imRaster_im2, Jim &imRaster_im3, int ox, int oy, int oz, int iband=0)
";

%feature("docstring")  seededplat "CPLErr seededplat(Jim
&imRaster_im2, Jim &imRaster_im3, int ox, int oy, int oz, int iband=0)
";

%feature("docstring")  labelpix "CPLErr labelpix(int iband=0) ";

%feature("docstring")  resolveLabels "CPLErr resolveLabels(Jim
&imRaster_imlut, Jim &imRaster_imlutback, int graph, int iband=0) ";

%feature("docstring")  gorder "CPLErr gorder(Jim &imRaster_g, int n,
int iband=0) ";

%feature("docstring")  propagate "CPLErr propagate(Jim &imRaster_dst,
IMAGE **imap, int nc, int graph, int iband=0) ";

%feature("docstring")  set_regions "CPLErr set_regions(Jim
&imRaster_ival, int indic, int iband=0) ";

%feature("docstring")  setregionsgraph "CPLErr setregionsgraph(Jim
&imRaster_ival, int indic, int graph, int iband=0) ";

%feature("docstring")  tessel_surface "CPLErr tessel_surface(int
iband=0) ";

%feature("docstring")  relabel "CPLErr relabel(Jim &imRaster_ilbl2,
Jim &imRaster_iarea2, int iband=0) ";

%feature("docstring")  bitwise_op "CPLErr bitwise_op(Jim
&imRaster_im2, int op, int iband=0) ";

%feature("docstring")  negation "CPLErr negation(int iband=0) ";

%feature("docstring")  arith "CPLErr arith(Jim &imRaster_im2, int op,
int iband=0) ";

%feature("docstring")  arithcst "CPLErr arithcst(double d_gt, int op,
int iband=0) ";

%feature("docstring")  imabs "CPLErr imabs(int iband=0) ";

%feature("docstring")  imsqrt "CPLErr imsqrt(int iband=0) ";

%feature("docstring")  imlog "CPLErr imlog(int iband=0) ";

%feature("docstring")  imatan "CPLErr imatan(int iband=0) ";

%feature("docstring")  imcos "CPLErr imcos(int iband=0) ";

%feature("docstring")  imacos "CPLErr imacos(int iband=0) ";

%feature("docstring")  imsin "CPLErr imsin(int iband=0) ";

%feature("docstring")  imasin "CPLErr imasin(int iband=0) ";

%feature("docstring")  thresh "CPLErr thresh(double d_gt1, double
d_gt2, double d_gbg, double d_gfg, int iband=0) ";

%feature("docstring")  setlevel "CPLErr setlevel(double d_gt1, double
d_gt2, double d_gval, int iband=0) ";

%feature("docstring")  modulo "CPLErr modulo(int val, int iband=0) ";

%feature("docstring")  complement "CPLErr complement(int iband=0) ";

%feature("docstring")  power2p "CPLErr power2p(int iband=0) ";

%feature("docstring")  blank "CPLErr blank(double d_gval, int
iband=0) ";

%feature("docstring")  shift "CPLErr shift(int val, int iband=0) ";

%feature("docstring")  setrange "CPLErr setrange(double d_gt1, double
d_gt2, int iband=0) ";

%feature("docstring")  FindPixWithVal "CPLErr FindPixWithVal(double
d_gval, unsigned long int *ofs, int iband=0) ";

%feature("docstring")  wsfah "CPLErr wsfah(Jim &imRaster_imr, int
graph, int maxfl, int iband=0) ";

%feature("docstring")  skelfah "CPLErr skelfah(Jim &imRaster_imr, Jim
&imRaster_imdir, int graph, int maxfl, int iband=0) ";

%feature("docstring")  skelfah2 "CPLErr skelfah2(Jim
&imRaster_impskp, int n, int graph, int iband=0) ";

%feature("docstring")  compose "CPLErr compose(Jim &imRaster_mask,
Jim &imRaster_g, Jim &imRaster_lbl, int graph, int iband=0) ";

%feature("docstring")  oiws "CPLErr oiws(int iband=0) ";

%feature("docstring")  srg "CPLErr srg(Jim &imRaster_im2, Jim
&imRaster_im3, int ox, int oy, int oz, int iband=0) ";

%feature("docstring")  IsPartitionEqual "CPLErr IsPartitionEqual(Jim
&imRaster_im2, int *result, int iband=0) ";

%feature("docstring")  IsPartitionFiner "CPLErr IsPartitionFiner(Jim
&imRaster_im2, int graph, unsigned long int *res, int iband=0) ";

%feature("docstring")  getfirstmaxpos "CPLErr getfirstmaxpos(unsigned
long int *pos, int iband=0) ";

%feature("docstring")  histcompress "CPLErr histcompress(int iband=0)
";

%feature("docstring")  lookup "CPLErr lookup(Jim &imRaster_imlut, int
iband=0) ";

%feature("docstring")  lookuptypematch "CPLErr lookuptypematch(Jim
&imRaster_imlut, int iband=0) ";

%feature("docstring")  volume "CPLErr volume(int iband=0) ";

%feature("docstring")  dirmax "CPLErr dirmax(int dir, int iband=0) ";

%feature("docstring")  imequalp "CPLErr imequalp(Jim &imRaster_im2,
int iband=0) ";

%feature("docstring")  getmax "CPLErr getmax(double *maxval, int
iband=0) ";

%feature("docstring")  getminmax "CPLErr getminmax(double *minval,
double *maxval, int iband=0) ";

%feature("docstring")  classstatsinfo "CPLErr classstatsinfo(Jim
&imRaster_imin, int iband=0) ";

%feature("docstring")  clmindist "CPLErr clmindist(Jim
&imRaster_imin, int bklabel, int mode, double thr, int iband=0) ";

%feature("docstring")  clparpip "CPLErr clparpip(Jim &imRaster_imin,
int bklabel, int mode, double mult, int iband=0) ";

%feature("docstring")  clmaha "CPLErr clmaha(Jim &imRaster_imin, int
bklabel, int mode, double thr, int iband=0) ";

%feature("docstring")  clmaxlike "CPLErr clmaxlike(Jim
&imRaster_imin, int bklabel, int type, double thr, int iband=0) ";


// File: fun2method_8py.xml


// File: jim_8cc.xml


// File: jim_8h.xml


// File: dir_68267d1309a1af8e8297ef4c3efbcdba.xml

