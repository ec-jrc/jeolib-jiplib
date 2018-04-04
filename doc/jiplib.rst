MODULE
Joint image processing library, developed for the JEODPP infrastructure, European Commission JRC - Ispra
Developed by:
Pieter Kempeneers: pieter.kempeneers@ec.europa.eu
Davide De Marchi: davide.de-marchi@ec.europa.eu
Pierre Soille: pierre.soille@ec.europa.eu
END



###########################################################################################################################################################################
# JIPLIB GLOBAL FUNCTIONS
###########################################################################################################################################################################

FUNC createJim()
Creates an empty Jim object as an instance of the basis image class of the Joint image processing library.

Returns:
   This instance of Jim object (self)

END

FUNC createJim(dict)
Creates a Jim object as an instance of the basis image class of the Joint image processing library, using a Python Dictionary argument

Args:
    * ``dict`` (Python Dictionary) with key value pairs. Each key (a 'quoted' string) is separated from its value by a colon (:). The items are separated by commas and the dictionary is enclosed in curly braces. An empty dictionary without any items is written with just two curly braces, like this: {}. A value can be a list that is also separated by commas and enclosed in square brackets [].

Supported keys in the dict:

======== ===================================================
filename input filename to read from (GDAL supported format)
nodata   Nodata value to put in image
band     Bands to open, index starts from 0
ulx      Upper left x value bounding box
uly      Upper left y value bounding box
lrx      Lower right x value bounding box
lry      Lower right y value bounding box
dx       Resolution in x
dy       Resolution in y
resample Resample algorithm used for reading pixel data in case of interpolation (default: GRIORA_NearestNeighbour). Check http://www.gdal.org/gdal_8h.html#a640ada511cbddeefac67c548e009d5a for available options.
extent   get boundary from extent from polygons in vector dataset
noread   Set this flag to True to not read data when opening
======== ===================================================

###
# resample: (default: GRIORA_NearestNeighbour) Resample algorithm used for reading pixel data in case of interpolation GRIORA_NearestNeighbour | GRIORA_Bilinear | GRIORA_Cubic | GRIORA_CubicSpline | GRIORA_Lanczos | GRIORA_Average | GRIORA_Average | GRIORA_Gauss (check http://www.gdal.org/gdal_8h.html#a640ada511cbddeefac67c548e009d5a)
###

Supported keys when creating new Jim image object not read from file:
===== =================
ncol  Number of columns
nrow  Number of rows
nband (default: 1) Number of bands
otype (default: Byte) Data type ({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64})
a_srs Assign the spatial reference for the output file, e.g., psg:3035 to use European projection and force to European grid
===== =================

Supported keys used to initialize random pixel values in new Jim image object:
======= ============================================
seed    (default: 0) seed value for random generator
mean    (default: 0) Mean value for random generator
stdev   (default: 0) Standard deviation for Gaussian random generator
uniform (default: 0) Start and end values for random value with uniform distribution
======= ============================================

Returns:
   This instance of Jim object (self)

Example:
Create Jim image object by opening an existing file (file content will automatically be read in memory)::

    ifn='/eos/jeodpp/data/SRS/Copernicus/S2/scenes/source/L1C/2017/08/05/065/S2A_MSIL1C_20170805T102031_N0205_R065_T32TNR_20170805T102535.SAFE/GRANULE/L1C_T32TNR_A011073_20170805T102535/IMG_DATA/T32TNR_20170805T102031_B08.jp2'
    jim=jl.createJim({'filename':ifn})
    #do stuff with jim ...
    jim.close()

Create Jim image object by opening an existing file for specific region of interest and spatial resolution using cubic convolution resampling::

    ifn='/eos/jeodpp/data/SRS/Copernicus/S2/scenes/source/L1C/2017/08/05/065/S2A_MSIL1C_20170805T102031_N0205_R065_T32TNR_20170805T102535.SAFE/GRANULE/L1C_T32TNR_A011073_20170805T102535/IMG_DATA/T32TNR_20170805T102031_B08.jp2'
    jim0=jl.createJim({'filename':ifn,'noread':True})
    ULX=jim0.getUlx()
    ULY=jim0.getUly()
    LRX=jim0.getUlx()+100*jim0.getDeltaX()
    LRY=jim0.getUly()-100*jim0.getDeltaY()
    jim=jl.Jim.createImg({'filename':ifn,'ulx':ULX,'uly':ULY,'lrx':LRX,'lry':LRY,'dx':5,'dy':5,'resample':'GRIORA_Cubic'})
    #do stuff with jim ...
    jim.close()

Create a new georeferenced Jim image object by defining the projection epsg code, bounding box, and pixel size::

    projection='epsg:32612'
    ULX=600000.0
    ULY=4000020.0
    LRX=709800.0
    LRY=3890220.0
    dict={'ulx':ULX,'uly':ULY,'lrx':LRX,'lry':LRY,'a_srs':projection}
    dict.update({'otype':'GDT_UInt16'})
    dict.update({'dy':100,'dx':100})
    jim=jl.Jim.createImg(dict)
    #do stuff with jim ...
    jim.close()

Create a new georeferenced Jim image object for writing by defining the projection epsg code, bounding box and number of rows and columns::

    projection='epsg:32612'
    ULX=600000.0
    ULY=4000020.0
    LRX=709800.0
    LRY=3890220.0
    dict={'ulx':ULX,'uly':ULY,'lrx':LRX,'lry':LRY,'a_srs':projection}
    dict.update({'otype':'GDT_UInt16'})
    nrow=1098
    ncol=1098
    dict.update({'nrow':nrow,'ncol':ncol})
    jim=jl.Jim.createImg(dict)
    #do stuff with jim ...
    jim.close()

END

FUNC createJimList()
Creates an empty JimList object.

Returns:
   This instance of Jim object (self)

END

FUNC createVector()
Creates an empty VectorOgr object as an instance of the basis vector class of the Joint image processing library.
END

Returns:
   This instance of VectorOgr object (self)


*********
Jim class
*********

CLASS Jim
Jim class is the basis image class of the Joint image processing library.

Notes:
The calls to Jim methods can be chained together using the dot (.) syntax returning a new Jim instance::

    ifn='/eos/jeodpp/data/SRS/Copernicus/S2/scenes/source/L1C/2017/08/05/065/S2A_MSIL1C_20170805T102031_N0205_R065_T32TNR_20170805T102535.SAFE/GRANULE/L1C_T32TNR_A011073_20170805T102535/IMG_DATA/T32TNR_20170805T102031_B08.jp2'
    jim0=createJim()
    ULX=600000.0
    ULY=4000020.0
    LRX=709800.0
    LRY=3890220.0
    jim = jim0.open({'filename':ifn}).crop({'ulx':ULX,'uly':ULY,'lrx':LRX,'lry':LRY})
    jim0.close()
    #do stuff with jim ...
    jim.close()

END

---------------------
Access Jim attributes
---------------------

METHOD nrOfCol()
Get number of columns in this raster dataset

Returns:
   The number of columns in this raster dataset
END

METHOD nrOfRow()
Get number of rows in this raster dataset

Returns:
   The number of rows in this raster dataset
END

METHOD nrOfBand()
Get number of bands in this raster dataset

Returns:
   The number of bands in this raster dataset
END

METHOD nrOfPlane()
Get number of planes in this raster dataset

Returns:
   The number of planes in this raster dataset
END

METHOD printNoDataValues()
Print the list of no data values of this raster dataset

Returns:
   This instance of Jim object (self)
END

METHOD pushNoDataValue()
Push a no data value for this raster dataset

Returns:
   This instance of Jim object (self)
END

METHOD setNoDataValue()
Set a single no data value for this raster dataset

Returns:
   This instance of Jim object (self)
END

METHOD setNoData(list)
Set a list of no data values for this raster dataset

Returns:
   This instance of Jim object (self)
END

METHOD clearNoData()
Clear the list of no data values for this raster dataset

Returns:
   This instance of Jim object (self)
END

METHOD getDataType()
Get the internal datatype for this raster dataset

Returns:
   The datatype id of this Jim object

   ========= ==
   datatype  id
   ========= ==
   Unknown   0
   Byte      1
   UInt16    2
   Int16     3
   UInt32    4
   Int32     5
   Float32   6
   Float64   7
   CInt16    8
   CInt32    9
   CFloat32  10
   CFloat64  11
   ========= ==
END

--------------------------
Get geospatial information
--------------------------

METHOD covers(*args)
Check if a geolocation is covered by this dataset. Only the coordinates of the point (variant 1) or region of interest (variant 2) are checked, irrespective of no data values. Set the additional flag to True if the region of interest must be entirely covered.

Args (variant 1):

    * ``x`` (float): x coordinate in spatial reference system of the raster dataset
    * ``y`` (float): y coordinate in spatial reference system of the raster dataset


Args (variant 2):

    * ``ulx`` (float): upper left x coordinate in spatial reference system of the raster dataset
    * ``uly`` (float): upper left y coordinate in spatial reference system of the raster dataset
    * ``lrx`` (float): lower right x coordinate in spatial reference system of the raster dataset
    * ``lry`` (float): lower right x coordinate in spatial reference system of the raster dataset
    * ``all`` (bool): set to True if the entire bounding box must be covered by the raster dataset


Returns:
   True if the raster dataset covers the point or region of interest.

END

METHOD getGeoTransform()
Get the geotransform data for this dataset as a list of floats.

Returns:
List of floats with geotransform data:
 * [0] top left x
 * [1] w-e pixel resolution
 * [2] rotation, 0 if image is "north up"
 * [3] top left y
 * [4] rotation, 0 if image is "north up"
 * [5] n-s pixel resolution

END

METHOD setGeoTransform(*args)
Set the geotransform data for this dataset.

Args:
List of floats with geotransform data:
 * [0] top left x
 * [1] w-e pixel resolution
 * [2] rotation, 0 if image is "north up"
 * [3] top left y
 * [4] rotation, 0 if image is "north up"
 * [5] n-s pixel resolution

Returns:
   This instance of Jim object (self)

END

METHOD copyGeoTransform(*args)
Copy geotransform information from another georeferenced image

Args:
 * A referenced Jim image

Returns:
   This instance of Jim object (self)

END

METHOD setProjection(*args)
Set the projection for this dataset in well known text (wkt) format

Args:
 * The projection string in well known text format (typically an EPSG code, e.g., 'epsg:3035')

Returns:
   This instance of Jim object (self)

END

METHOD getBoundingBox(*args)
Get the bounding box of this dataset in georeferenced coordinates.

Returns:
   A list with the bounding box of this dataset in georeferenced coordinates.

END

METHOD getCenterpos(*args)
Get the center position of this dataset in georeferenced coordinates

Returns:
   A list with the centre position of this dataset in georeferenced coordinates.

END

METHOD getUlx(*args)
Get the upper left corner x (georeferenced) coordinate of this dataset

Returns:
   The upper left corner x (georeferenced) coordinate of this dataset

END

METHOD getUly(*args)
Get the upper left corner y (georeferenced) coordinate of this dataset

Returns:
   The upper left corner y (georeferenced) coordinate of this dataset

END

METHOD getLrx(*args)
Get the lower left corner x (georeferenced) coordinate of this dataset

Returns:
   The lower left corner x (georeferenced) coordinate of this dataset

END

METHOD getLry(*args)
Get the lower left corner y (georeferenced) coordinate of this dataset

Returns:
   The lower left corner y (georeferenced) coordinate of this dataset

END

METHOD getDeltaX(*args)
Get the pixel cell spacing in x.

Returns:
   The pixel cell spacing in x.

END

METHOD getDeltaY(*args)
Get the piyel cell spacing in y.

Returns:
   The piyel cell spacing in y.

END


METHOD getRefPix(*args)
Calculate the reference pixel as the centre of gravity pixel (weighted average of all values not taking into account no data values) for a specific band (start counting from 0).

Returns:
   The reference pixel as the centre of gravity pixel (weighted average of all values not taking into account no data values) for a specific band (start counting from 0).

END

********************
Input/Output methods
********************

METHOD open(dict)
Open a raster dataset

Args:

    * ``dict`` (Python Dictionary) with key value pairs. Each key (a 'quoted' string) is separated from its value by a colon (:). The items are separated by commas and the dictionary is enclosed in curly braces. An empty dictionary without any items is written with just two curly braces, like this: {}. A value can be a list that is also separated by commas and enclosed in square brackets [].

Supported keys in the dict:

======== ===================================================
filename input filename to read from (GDAL supported format)
nodata   Nodata value to put in image
band     Bands to open, index starts from 0
ulx      Upper left x value bounding box
uly      Upper left y value bounding box
lrx      Lower right x value bounding box
lry      Lower right y value bounding box
dx       Resolution in x
dy       Resolution in y
resample Resample algorithm used for reading pixel data in case of interpolation (default: GRIORA_NearestNeighbour). Check http://www.gdal.org/gdal_8h.html#a640ada511cbddeefac67c548e009d5a for available options.
extent   get boundary from extent from polygons in vector dataset
noread   Set this flag to True to not read data when opening
======== ===================================================

# resample: (default: GRIORA_NearestNeighbour) Resample algorithm used for reading pixel data in case of interpolation GRIORA_NearestNeighbour | GRIORA_Bilinear | GRIORA_Cubic | GRIORA_CubicSpline | GRIORA_Lanczos | GRIORA_Average | GRIORA_Average | GRIORA_Gauss (check http://www.gdal.org/gdal_8h.html#a640ada511cbddeefac67c548e009d5a)

Supported keys when creating new Jim image object not read from file:

===== =================
ncol  Number of columns
nrow  Number of rows
nband (default: 1) Number of bands
otype (default: Byte) Data type ({Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64})
a_srs Assign the spatial reference for the output file, e.g., psg:3035 to use European projection and force to European grid
===== =================

Supported keys used to initialize random pixel values in new Jim image object:

======= ============================================
seed    (default: 0) seed value for random generator
mean    (default: 0) Mean value for random generator
stdev   (default: 0) Standard deviation for Gaussian random generator
uniform (default: 0) Start and end values for random value with uniform distribution
======= ============================================

Returns:
   This instance of Jim object (self)

Example:
See also :py:func:`createJim`

END

METHOD close()
Close a raster dataset, releasing resources such as memory and GDAL dataset handle

END


METHOD write(dict)
Write the raster dataset to file in a GDAL supporte format

Args:
    * ``dict`` (Python Dictionary) with key value pairs. Each key (a 'quoted' string) is separated from its value by a colon (:). The items are separated by commas and the dictionary is enclosed in curly braces. An empty dictionary without any items is written with just two curly braces, like this: {}. A value can be a list that is also separated by commas and enclosed in square brackets [].

Supported keys in the dict:

======== ===================================================
filename output filename to write to:
oformat  (default: GTiff) Output image (GDAL supported) format
co       Creation option for output file. Multiple options can be specified as a list
nodata   Nodata value to put in image
======== ===================================================

Returns:
   This instance of Jim object (self)

Note:
    Supported GDAL output formats are restricted to those that support creation (see http://www.gdal.org/formats_list.html#footnote1)
    The image data is kept in memory (unlike using method :py:func:`close``)

Example:
Create Jim image object by opening an existing file in jp2 format. Then write to a compressed and tiled file in the default GeoTIFF format::

    ifn='/eos/jeodpp/data/SRS/Copernicus/S2/scenes/source/L1C/2017/08/05/065/S2A_MSIL1C_20170805T102031_N0205_R065_T32TNR_20170805T102535.SAFE/GRANULE/L1C_T32TNR_A011073_20170805T102535/IMG_DATA/T32TNR_20170805T102031_B08.jp2'
    jim=jl.createJim({'filename':ifn})
    jim.write({'filename':'/tmp/test.tif','co':['COMPRESS=LZW','TILED=YES']})
    jim.close()

END

###########################################################################################################################################################################
# JimList
###########################################################################################################################################################################

CLASS JimList
JimList class represents a list of Jim images.

Notes:
A JimList can be created from a python list of Jim images or via the :py:method:`pushImage` method::

  ifn='/eos/jeodpp/data/SRS/Copernicus/S2/scenes/source/L1C/2017/08/05/065/S2A_MSIL1C_20170805T102031_N0205_R065_T32TNR_20170805T102535.SAFE/GRANULE/L1C_T32TNR_A011073_20170805T102535/IMG_DATA/T32TNR_20170805T102031_B08.jp2'
  jim0=createJim()
  jlist=jl.createJim()
  jlist.pushImage(jim0)
  #do stuff with jim ...
  jlist.close()

END

METHOD pushImage(Jim)
Push a Jim image to this JimList object

Args:

    * ``Jim`` :py:class:`Jim` object.
