
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

%feature("docstring")  jiplib::Jim::Jim "jiplib::Jim::Jim(const
app::AppFactory &theApp)

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
jiplib::Jim::crop(const app::AppFactory &app=app::AppFactory())

crop Jim image in memory returning Jim image

Parameters:
-----------

app:  application specific option arguments

output image The utility pkcrop can subset and stack raster images. In
the spatial domain it can crop a bounding box from a larger image. The
output bounding box is selected by setting the new corner coordinates
using the options -ulx -uly -lrx -lry. Alternatively you can set the
new image center (-x -y) and size. This can be done either in
projected coordinates (using the options -nx -ny) or in image
coordinates (using the options -ns -nl). You can also use a vector
file to set the new bounding box (option -e). In the spectral domain,
pkcrop allows you to select individual bands from one or more input
image(s). Bands are stored in the same order as provided on the
command line, using the option -b. Band numbers start with index 0
(indicating the first band). The default is to select all input bands.
If more input images are provided, the bands are stacked into a multi-
band image. If the bounding boxes or spatial resolution are not
identical for all input images, you should explicitly set them via the
options. The pkcrop utility is not suitable to mosaic or composite
images. Consider the utility pkcomposite instead.

Options

use either -short or --long options (both --long=value and --long
value are supported)

short option -h shows basic options only, long option --help shows all
options short

long

type

default

description

i

input

std::string

Input image file(s). If input contains multiple images, a multi-band
output is created

o

output

std::string

Output image file

a_srs

a_srs

std::string

Override the projection for the output file (leave blank to copy from
input file, use epsg:3035 to use European projection and force to
European grid

ulx

ulx

double

0

Upper left x value bounding box

uly

uly

double

0

Upper left y value bounding box

lrx

lrx

double

0

Lower right x value bounding box

lry

lry

double

0

Lower right y value bounding box

b

band

unsigned int

band index to crop (leave empty to retain all bands)

sband

startband

unsigned int

Start band sequence number

eband

endband

unsigned int

End band sequence number

as

autoscale

double

scale output to min and max, e.g.,autoscale 0autoscale 255

ot

otype

std::string

Data type for output image
({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}).
Empty string: inherit type from input image

of

oformat

std::string

GTiff

Output image format (see also gdal_translate)

ct

ct

std::string

color table (file with 5 columns: id R G B ALFA (0: transparent, 255:
solid)

dx

dx

double

Output resolution in x (in meter) (empty: keep original resolution)

dy

dy

double

Output resolution in y (in meter) (empty: keep original resolution)

r

resampling-method

std::string

near

Resampling method (near: nearest neighbor, bilinear: bi-linear
interpolation).

e

extent

std::string

get boundary from extent from polygons in vector file

cut

crop_to_cutline

bool

false

Crop the extent of the target dataset to the extent of the cutline  |
eo | eo | std::string | |special extent options controlling
rasterization:
ATTRIBUTE|CHUNKYSIZE|ALL_TOUCHED|BURN_VALUE_FROM|MERGE_ALG, e.g., -eo
ATTRIBUTE=fieldname | | m | mask | std::string | |Use the specified
file as a validity mask (0 is nodata) | | msknodata | msknodata |
float | 0 |Mask value not to consider for crop | mskband | mskband |
short | 0 |Mask band to read (0 indexed) | | co | co | std::string |
|Creation option for output file. Multiple options can be specified. |
| x | x | double | |x-coordinate of image center to crop (in meter) |
| y | y | double | |y-coordinate of image center to crop (in meter) |
| nx | nx | double | |image size in x to crop (in meter) | | ny | ny |
double | |image size in y to crop (in meter) | | ns | ns | int |
|number of samples to crop (in pixels) | | nl | nl | int | |number of
lines to crop (in pixels) | | scale | scale | double |
|output=scale*input+offset | | off | offset | double |
|output=scale*input+offset | | nodata | nodata | float | |Nodata value
to put in image if out of bounds. | | align | align | bool | |Align
output bounding box to input image | | mem | mem | unsigned long int |
0 |Buffer size (in MB) to read image data blocks in memory | | d |
description | std::string | |Set image description |

Examples

Some examples how to use pkcrop can be found here ";

%feature("docstring")  jiplib::Jim::arith "CPLErr Jim::arith(Jim
&imgRaster, int theOperation, int band=0)

Parameters:
-----------

imgRaster:  is operand

theOperation:  the operation to be performed

iband:  is the band for which the function needs to be performed
(default 0 is first band)

CE_None if successful

Parameters:
-----------

imgRaster:  is operand

theOperation:  the operation to be performed

iband:  is the band for which the function needs to be performed
(default 0 is first band)

CE_None if successful ";

%feature("docstring")  jiplib::Jim::arithcst "CPLErr
Jim::arithcst(double dcst, int theOperation, int band=0)

Parameters:
-----------

imgRaster:  is operand

theOperation:  the operation to be performed

iband:  is the band for which the function needs to be performed
(default 0 is first band)

CE_None if successful ";

%feature("docstring")  jiplib::Jim::getVolume "double
jiplib::Jim::getVolume(int iband=0)

get volume (from mialib) ";

%feature("docstring")  jiplib::Jim::filter "std::shared_ptr<Jim>
jiplib::Jim::filter(const app::AppFactory &theApp)

filter Jim image and return filtered image as shared pointer ";

%feature("docstring")  jiplib::Jim::stretch "std::shared_ptr<Jim>
jiplib::Jim::stretch(const app::AppFactory &app)

stretch Jim image and return stretched image as shared pointer ";

%feature("docstring")  jiplib::Jim::diff "std::shared_ptr<Jim>
jiplib::Jim::diff(const app::AppFactory &app)

create statistical profile from a collection ";

%feature("docstring")  jiplib::Jim::svm "std::shared_ptr<Jim>
jiplib::Jim::svm(const app::AppFactory &app)

supervised classification using support vector machine (train with
extractImg/extractOgr) ";


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

%feature("docstring")  jiplib::JimList::composite "std::shared_ptr<Jim> jiplib::JimList::composite(const app::AppFactory
&app=app::AppFactory())

composite image only for in memory ";

%feature("docstring")  jiplib::JimList::crop "std::shared_ptr<Jim>
jiplib::JimList::crop(const app::AppFactory &app=app::AppFactory())

Parameters:
-----------

app:  application specific option arguments

output image The utility pkcrop can subset and stack raster images. In
the spatial domain it can crop a bounding box from a larger image. The
output bounding box is selected by setting the new corner coordinates
using the options -ulx -uly -lrx -lry. Alternatively you can set the
new image center (-x -y) and size. This can be done either in
projected coordinates (using the options -nx -ny) or in image
coordinates (using the options -ns -nl). You can also use a vector
file to set the new bounding box (option -e). In the spectral domain,
pkcrop allows you to select individual bands from one or more input
image(s). Bands are stored in the same order as provided on the
command line, using the option -b. Band numbers start with index 0
(indicating the first band). The default is to select all input bands.
If more input images are provided, the bands are stacked into a multi-
band image. If the bounding boxes or spatial resolution are not
identical for all input images, you should explicitly set them via the
options. The pkcrop utility is not suitable to mosaic or composite
images. Consider the utility pkcomposite instead.

Options

use either -short or --long options (both --long=value and --long
value are supported)

short option -h shows basic options only, long option --help shows all
options short

long

type

default

description

i

input

std::string

Input image file(s). If input contains multiple images, a multi-band
output is created

o

output

std::string

Output image file

a_srs

a_srs

std::string

Override the projection for the output file (leave blank to copy from
input file, use epsg:3035 to use European projection and force to
European grid

ulx

ulx

double

0

Upper left x value bounding box

uly

uly

double

0

Upper left y value bounding box

lrx

lrx

double

0

Lower right x value bounding box

lry

lry

double

0

Lower right y value bounding box

b

band

unsigned int

band index to crop (leave empty to retain all bands)

sband

startband

unsigned int

Start band sequence number

eband

endband

unsigned int

End band sequence number

as

autoscale

double

scale output to min and max, e.g.,autoscale 0autoscale 255

ot

otype

std::string

Data type for output image
({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64}).
Empty string: inherit type from input image

of

oformat

std::string

GTiff

Output image format (see also gdal_translate)

ct

ct

std::string

color table (file with 5 columns: id R G B ALFA (0: transparent, 255:
solid)

dx

dx

double

Output resolution in x (in meter) (empty: keep original resolution)

dy

dy

double

Output resolution in y (in meter) (empty: keep original resolution)

r

resampling-method

std::string

near

Resampling method (near: nearest neighbor, bilinear: bi-linear
interpolation).

e

extent

std::string

get boundary from extent from polygons in vector file

cut

crop_to_cutline

bool

false

Crop the extent of the target dataset to the extent of the cutline  |
eo | eo | std::string | |special extent options controlling
rasterization:
ATTRIBUTE|CHUNKYSIZE|ALL_TOUCHED|BURN_VALUE_FROM|MERGE_ALG, e.g., -eo
ATTRIBUTE=fieldname | | m | mask | std::string | |Use the specified
file as a validity mask (0 is nodata) | | msknodata | msknodata |
float | 0 |Mask value not to consider for crop | mskband | mskband |
short | 0 |Mask band to read (0 indexed) | | co | co | std::string |
|Creation option for output file. Multiple options can be specified. |
| x | x | double | |x-coordinate of image center to crop (in meter) |
| y | y | double | |y-coordinate of image center to crop (in meter) |
| nx | nx | double | |image size in x to crop (in meter) | | ny | ny |
double | |image size in y to crop (in meter) | | ns | ns | int |
|number of samples to crop (in pixels) | | nl | nl | int | |number of
lines to crop (in pixels) | | scale | scale | double |
|output=scale*input+offset | | off | offset | double |
|output=scale*input+offset | | nodata | nodata | float | |Nodata value
to put in image if out of bounds. | | align | align | bool | |Align
output bounding box to input image | | mem | mem | unsigned long int |
0 |Buffer size (in MB) to read image data blocks in memory | | d |
description | std::string | |Set image description |

Examples

Some examples how to use pkcrop can be found here ";

%feature("docstring")  jiplib::JimList::stack "std::shared_ptr<Jim>
jiplib::JimList::stack(const app::AppFactory &app=app::AppFactory())

stack all images in collection to multiband image (alias for crop) ";

%feature("docstring")  jiplib::JimList::statProfile "std::shared_ptr<Jim> jiplib::JimList::statProfile(const
app::AppFactory &app)

create statistical profile from a collection ";


// File: namespacejiplib.xml


// File: jim_8cc.xml


// File: jim_8h.xml


// File: dir_68267d1309a1af8e8297ef4c3efbcdba.xml

