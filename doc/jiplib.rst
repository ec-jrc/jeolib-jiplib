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

FUNC createJim(**kwargs)
Creates a Jim object as an instance of the basis image class of the Joint image processing library, using keyword arguments

..
   Args:
   * ``dict`` (Python Dictionary) with key value pairs. Each key (a 'quoted' string) is separated from its value by a colon (:). The items are separated by commas and the dictionary is enclosed in curly braces. An empty dictionary without any items is written with just two curly braces, like this: {}. A value can be a list that is also separated by commas and enclosed in square brackets [].

Supported keys as arguments:

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

.. note::
   You can specify a different spatial reference system to define the region of interest to read set with keys ulx, uly, lrx, and lry with the extra key 't_srs'. Notice this will not re-project the resulting image. You can use the function :py:func:Jim:`warp` for this.
..
   resample: (default: GRIORA_NearestNeighbour) Resample algorithm used for reading pixel data in case of interpolation GRIORA_NearestNeighbour | GRIORA_Bilinear | GRIORA_Cubic | GRIORA_CubicSpline | GRIORA_Lanczos | GRIORA_Average | GRIORA_Average | GRIORA_Gauss (check http://www.gdal.org/gdal_8h.html#a640ada511cbddeefac67c548e009d5a)

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
    jim=jl.createJim('filename'=ifn)
    #do stuff with jim ...
    jim.close()

The key 'filename' is optional::

    ifn='/eos/jeodpp/data/SRS/Copernicus/S2/scenes/source/L1C/2017/08/05/065/S2A_MSIL1C_20170805T102031_N0205_R065_T32TNR_20170805T102535.SAFE/GRANULE/L1C_T32TNR_A011073_20170805T102535/IMG_DATA/T32TNR_20170805T102031_B08.jp2'
    jim=jl.createJim(ifn)
    #do stuff with jim ...
    jim.close()

Create Jim image object by opening an existing file for specific region of interest and spatial resolution using cubic convolution resampling::

    ifn='/eos/jeodpp/data/SRS/Copernicus/S2/scenes/source/L1C/2017/08/05/065/S2A_MSIL1C_20170805T102031_N0205_R065_T32TNR_20170805T102535.SAFE/GRANULE/L1C_T32TNR_A011073_20170805T102535/IMG_DATA/T32TNR_20170805T102031_B08.jp2'
    jim0=jl.createJim(filename=ifn,'noread'=True)
    ULX=jim0.getUlx()
    ULY=jim0.getUly()
    LRX=jim0.getUlx()+100*jim0.getDeltaX()
    LRY=jim0.getUly()-100*jim0.getDeltaY()
    jim=jl.Jim.createImg(filename=ifn,ulx:ULX,'uly':ULY,'lrx':LRX,'lry':LRY,'dx':5,'dy':5,'resample':'GRIORA_Cubic'})
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

FUNC createJim(*args)
Creates an empty Jim object as an instance of the basis image class of the Joint image processing library.

Args:
* ``Jim``: A reference Jim object
* ``copyData`` (bool): Set to False if reference image is used as a template only, without copying actual pixel dat

Returns:
   This instance of Jim object (self)

END

FUNC createJimList()
Creates an empty JimList object.

Returns:
   This instance of Jim object (self)

END

FUNC createVector()
Creates an empty VectorOgr object as an instance of the basis vector class of the Joint image processing library.

Returns:
   This instance of VectorOgr object (self)

END

##########
#Jim class
##########

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

METHOD setGeoTransform()
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
Copy geotransform information from another georeferenced image.

Args:
* A referenced Jim image

Returns:
   This instance of Jim object (self)

END

METHOD getProjection()
Get the projection for this dataget in well known text (wkt) format.


Returns:
   The projection string in well known text format.

END

METHOD setProjection(*args)
Set the projection for this dataset in well known text (wkt) format.

Args:
* The projection string in well known text format (typically an EPSG code, e.g., 'epsg:3035')

Returns:
   This instance of Jim object (self)

END

METHOD getBoundingBox()
Get the bounding box of this dataset in georeferenced coordinates.

Returns:
   A list with the bounding box of this dataset in georeferenced coordinates.

END

METHOD getCenterPos()
Get the center position of this dataset in georeferenced coordinates

Returns:
   A list with the center position of this dataset in georeferenced coordinates.

END

METHOD getUlx()
Get the upper left corner x (georeferenced) coordinate of this dataset

Returns:
   The upper left corner x (georeferenced) coordinate of this dataset

END

METHOD getUly()
Get the upper left corner y (georeferenced) coordinate of this dataset

Returns:
   The upper left corner y (georeferenced) coordinate of this dataset

END

METHOD getLrx()
Get the lower left corner x (georeferenced) coordinate of this dataset

Returns:
   The lower left corner x (georeferenced) coordinate of this dataset

END

METHOD getLry()
Get the lower left corner y (georeferenced) coordinate of this dataset

Returns:
   The lower left corner y (georeferenced) coordinate of this dataset

END

METHOD getDeltaX()
Get the pixel cell spacing in x.

Returns:
   The pixel cell spacing in x.

END

METHOD getDeltaY()
Get the piyel cell spacing in y.

Returns:
   The piyel cell spacing in y.

END


METHOD getRefPix()
Calculate the reference pixel as the center of gravity pixel (weighted average of all values not taking into account no data values) for a specific band (start counting from 0).

Returns:
   The reference pixel as the centre of gravity pixel (weighted average of all values not taking into account no data values) for a specific band (start counting from 0).

END

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

 ..
    resample: (default: GRIORA_NearestNeighbour) Resample algorithm used for reading pixel data in case of interpolation GRIORA_NearestNeighbour | GRIORA_Bilinear | GRIORA_Cubic | GRIORA_CubicSpline | GRIORA_Lanczos | GRIORA_Average | GRIORA_Average | GRIORA_Gauss (check http://www.gdal.org/gdal_8h.html#a640ada511cbddeefac67c548e009d5a)

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
Close a raster dataset, releasing resources such as memory and GDAL dataset handle.

END


METHOD write(dict)
Write the raster dataset to file in a GDAL supported format

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

.. note::
    Supported GDAL output formats are restricted to those that support creation (see http://www.gdal.org/formats_list.html#footnote1)
    The image data is kept in memory (unlike using method :py:func:`Jim:close`)

Example:

Create Jim image object by opening an existing file in jp2 format. Then write to a compressed and tiled file in the default GeoTIFF format::

    ifn='/eos/jeodpp/data/SRS/Copernicus/S2/scenes/source/L1C/2017/08/05/065/S2A_MSIL1C_20170805T102031_N0205_R065_T32TNR_20170805T102535.SAFE/GRANULE/L1C_T32TNR_A011073_20170805T102535/IMG_DATA/T32TNR_20170805T102031_B08.jp2'
    jim=jl.createJim({'filename':ifn})
    jim.write({'filename':'/tmp/test.tif','co':['COMPRESS=LZW','TILED=YES']})
    jim.close()

END

METHOD dumpImg(dict)
Dump the raster dataset to output (screen or ASCII file).

Args:
* ``dict`` (Python Dictionary) with key value pairs. Each key (a 'quoted' string) is separated from its value by a colon (:). The items are separated by commas and the dictionary is enclosed in curly braces. An empty dictionary without any items is written with just two curly braces, like this: {}. A value can be a list that is also separated by commas and enclosed in square brackets [].

Supported keys in the dict:

=========  =============================================================
output     Output ascii file (Default is empty: dump to standard output)
oformat    Output format: matrix or list (x,y,z) form. Default is matrix
geo        (bool) Set to True to dump x and y in spatial reference system of raster dataset (for list form only). Default is to dump column and row index (starting from 0)
band       Band index to crop
srcnodata  Do not dump these no data values (for list form only)
force      (bool) Set to True to force full dump even for large images (above 100 rows and cols)
=========  =============================================================

Returns:
   This instance of Jim object (self)


Example:

Open resampled raster dataset in reduced spatial resolution of 20 km by 20 km and dump to screen (first in matrix then in list format)::

    ifn='/eos/jeodpp/data/SRS/Copernicus/S2/scenes/source/L1C/2017/08/05/065/S2A_MSIL1C_20170805T102031_N0205_R065_T32TNR_20170805T102535.SAFE/GRANULE/L1C_T32TNR_A011073_20170805T102535/IMG_DATA/T32TNR_20170805T102031_B08.jp2'
    jim=jl.createJim({'filename':ifn, 'dx':20000,'dy':20000,'resample':'GRIORA_Bilinear'})
    jim.dumpImg({'oformat':'matrix'})

    2503 2794 3148 3194 3042 2892
    2634 2792 2968 2864 2790 3171
    2335 2653 2723 2700 2703 2836
    2510 2814 3027 2946 2889 2814
    2972 2958 3014 2983 2900 2899
    2692 2711 2843 2755 2795 2823

    jim.dumpImg({'oformat':'list'})

    0 0 2503
    1 0 2794
    2 0 3148
    3 0 3194
    4 0 3042
    5 0 2892

    0 1 2634
    1 1 2792
    2 1 2968
    3 1 2864
    4 1 2790
    5 1 3171

    0 2 2335
    1 2 2653
    2 2 2723
    3 2 2700
    4 2 2703
    5 2 2836

    0 3 2510
    1 3 2814
    2 3 3027
    3 3 2946
    4 3 2889
    5 3 2814

    0 4 2972
    1 4 2958
    2 4 3014
    3 4 2983
    4 4 2900
    5 4 2899

    0 5 2692
    1 5 2711
    2 5 2843
    3 5 2755
    4 5 2795
    5 5 2823

    jim.close()

END

METHOD isEqual(*args)
Test raster dataset for equality.

Args:
* ``Jim``: A reference Jim object

Returns:
   True if raster dataset is equal to reference raster dataset, else False.

END

METHOD convert(dict)
Convert Jim image with respect to data type, creation options (compression, interleave, etc.).

Args:
* ``dict`` (Python Dictionary) with key value pairs. Each key (a 'quoted' string) is separated from its value by a colon (:). The items are separated by commas and the dictionary is enclosed in curly braces. An empty dictionary without any items is written with just two curly braces, like this: {}. A value can be a list that is also separated by commas and enclosed in square brackets [].

Supported keys in the dict:

+------------------+---------------------------------------------------------------------------------+
| key              | value                                                                           |
+==================+=================================================================================+
| otype            | Data type for output image                                                      |
+------------------+---------------------------------------------------------------------------------+
| scale            | Scale output: output=scale*input+offset                                         |
+------------------+---------------------------------------------------------------------------------+
| offset           | Apply offset: output=scale*input+offset                                         |
+------------------+---------------------------------------------------------------------------------+
| autoscale        | Scale output to min and max, e.g., [0,255]                                      |
+------------------+---------------------------------------------------------------------------------+
| a_srs            | Override the projection for the output file                                     |
+------------------+---------------------------------------------------------------------------------+

Returns:
   This converted Jim object

Example:

Convert data type of input image to byte, using autoscale and clipping respectively::

  jim_scaled=jim.convert({'otype':'Byte','autoscale':[0,255]})
  jim_clipped=jim.setThreshold({'min':0,'max':255,'nodata':0}).convert({'otype':'Byte'})

END

METHOD crop(dict)
Subset raster dataset according in spatial (subset region) or spectral/temporal domain (subset bands)

Args:
* ``dict`` (Python Dictionary) with key value pairs. Each key (a 'quoted' string) is separated from its value by a colon (:). The items are separated by commas and the dictionary is enclosed in curly braces. An empty dictionary without any items is written with just two curly braces, like this: {}. A value can be a list that is also separated by commas and enclosed in square brackets [].

Returns:
   This subset of Jim object

.. note::
   Spatial subsetting only supports nearest neighbor interpolation. Use :py:func:`createJim` for more flexible interpolation options

Supported keys in the dict:

.. note::
   In addition to the keys defined here, you can use all the keys defined in :py:func:`Jim:convert`

**Subset spatial region in coordinates of the image geospatial reference system**

+------------------+---------------------------------------------------------------------------------+
| key              | value                                                                           |
+==================+=================================================================================+
| extent           | Get boundary from extent from polygons in vector file                           |
+------------------+---------------------------------------------------------------------------------+
| eo               | Special extent options controlling rasterization                                |
+------------------+---------------------------------------------------------------------------------+
| ln               | Layer name of extent to crop                                                    |
+------------------+---------------------------------------------------------------------------------+
| crop_to_cutline  | True will crop the extent of the target dataset to the extent of the cutline    |
|                  | The outside area will be set to no data (the value defined by the key 'nodata') |
+------------------+---------------------------------------------------------------------------------+
| crop_in_cutline  | True: inverse operation to crop_to_cutline                                      |
|                  | The inside area will be set to no data (the value defined by the key 'nodata')  |
+------------------+---------------------------------------------------------------------------------+
| ulx              | Upper left x value of bounding box to crop                                      |
+------------------+---------------------------------------------------------------------------------+
| uly              | Upper left y value of bounding box to crop                                      |
+------------------+---------------------------------------------------------------------------------+
| lrx              | Lower right x value of bounding box to crop                                     |
+------------------+---------------------------------------------------------------------------------+
| lry              | Lower right y value of bounding box to crop                                     |
+------------------+---------------------------------------------------------------------------------+
| dx               | Output resolution in x (default: keep original resolution)                      |
+------------------+---------------------------------------------------------------------------------+
| dy               | Output resolution in y (default: keep original resolution)                      |
+------------------+---------------------------------------------------------------------------------+
| nodata           | Nodata value to put in image if out of bounds                                   |
+------------------+---------------------------------------------------------------------------------+
| align            | Align output bounding box to input image                                        |
+------------------+---------------------------------------------------------------------------------+

.. note::
   Possible values for the key 'eo' are: ATTRIBUTE|CHUNKYSIZE|ALL_TOUCHED|BURN_VALUE_FROM|MERGE_ALG. For instance you can use 'eo':'ATTRIBUTE=fieldname'

**Subset bands**

+------------------+---------------------------------------------------------------------------------+
| key              | value                                                                           |
+==================+=================================================================================+
| band             | List of band indices to crop (index is 0 based)                                 |
+------------------+---------------------------------------------------------------------------------+
| startband        | Start band sequence number (index is 0 based)                                   |
+------------------+---------------------------------------------------------------------------------+
| endband          | End band sequence number (index is 0 based)                                     |
+------------------+---------------------------------------------------------------------------------+

..
   | mask             | Data type for output image                                                      |
   +------------------+---------------------------------------------------------------------------------+
   | msknodata        | Scale output: output=scale*input+offset                                         |
   +------------------+---------------------------------------------------------------------------------+
   | mskband          | Apply offset: output=scale*input+offset                                         |
   +------------------+---------------------------------------------------------------------------------+

Example:

Convert data type of input image to byte, using autoscale and clipping respectively::

  jim_scaled=jim.convert({'otype':'Byte','autoscale':[0,255]})
  jim_clipped=jim.setThreshold({'min':0,'max':255,'nodata':0}).convert({'otype':'Byte'})

END

METHOD warp(dict)
Warp a raster dataset to a target spatial reference system

Args:
* ``dict`` (Python Dictionary) with key value pairs. Each key (a 'quoted' string) is separated from its value by a colon (:). The items are separated by commas and the dictionary is enclosed in curly braces. An empty dictionary without any items is written with just two curly braces, like this: {}. A value can be a list that is also separated by commas and enclosed in square brackets [].

Returns:
   This warped Jim object in the target spatial reference system

Supported keys in the dict:

+------------------+---------------------------------------------------------------------------------+
| key              | value                                                                           |
+==================+=================================================================================+
| s_srs            | Source spatial reference system (default is to read from input)                 |
+------------------+---------------------------------------------------------------------------------+
| t_srs            | Target spatial reference system                                                 |
+------------------+---------------------------------------------------------------------------------+
| resample         | Resample algorithm used for reading pixel data in case of interpolation         |
|                  | (default: GRIORA_NearestNeighbour).                                             |
|                  | Check http://www.gdal.org/gdal_8h.html#a640ada511cbddeefac67c548e009d5a         |
|                  | or available options.                                                           |
+------------------+---------------------------------------------------------------------------------+
| nodata           | Nodata value to put in image if out of bounds                                   |
+------------------+---------------------------------------------------------------------------------+
| otype            | Data type for output image                                                      |
+------------------+---------------------------------------------------------------------------------+

.. note::
   Possible values for the key 'otype' are: Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/CInt16/CInt32/CFloat32/CFloat64

Example:

Read a raster dataset from disk by selecting a bounding box in some target spatial reference system. Then warp the read raster dataset to the target spatial reference system::

  jim=jl.createJim({'filename':'/path/to/file.tif','t_srs':'epsg:3035','ulx':1000000,'uly':4000000','lrx':1500000,'lry':3500000})
  jim_warped=jim.warp({'t_srs':'epsg:3035})

END

METHOD filter1d(dict)
Filter Jim image in spectral/temporal domain performed on multi-band raster dataset.

Args:
* ``dict`` (Python Dictionary) with key value pairs. Each key (a 'quoted' string) is separated from its value by a colon (:). The items are separated by commas and the dictionary is enclosed in curly braces. An empty dictionary without any items is written with just two curly braces, like this: {}. A value can be a list that is also separated by commas and enclosed in square brackets [].


Returns:
   This filtered of Jim object (self)

Supported keys in the dict:


+------------------+---------------------------------------------------------------------------------+
| key              | value                                                                           |
+==================+=================================================================================+
| filter           | filter function (see values for different filter types in tables below)         |
+------------------+---------------------------------------------------------------------------------+
| dz               | filter kernel size in z (spectral/temporal dimension), must be odd (example: 3) |
+------------------+---------------------------------------------------------------------------------+
| pad              | Padding method for filtering (how to handle edge effects)                       |
|                  | Possible values are: symmetric (default), replicate, circular, zero (pad with 0)|
+------------------+---------------------------------------------------------------------------------+
| otype            | Data type for output image                                                      |
+------------------+---------------------------------------------------------------------------------+


**Morphological filters**

+---------------------+------------------------------------------------------+
| filter              | description                                          |
+=====================+======================================================+
| dilate              | morphological dilation                               |
+---------------------+------------------------------------------------------+
| erode               | morphological erosion                                |
+---------------------+------------------------------------------------------+
| close               | morpholigical closing (dilate+erode)                 |
+---------------------+------------------------------------------------------+
| open                | morpholigical opening (erode+dilate)                 |
+---------------------+------------------------------------------------------+

.. note::
   The morphological filter uses a linear structural element with a size defined by the key 'dz'

Example:

Perform a morphological dilation with a linear structural element of size 5::

  jim_filtered=jim.filter1d({'filter':'dilate','dz':5})


**Statistical filters**

+--------------+------------------------------------------------------+
| filter       | description                                          |
+==============+======================================================+
| smoothnodata | smooth nodata values (set nodata option!)            |
+--------------+------------------------------------------------------+
| nvalid       | report number of valid (not nodata) values in window |
+--------------+------------------------------------------------------+
| median       | perform a median filter                              |
+--------------+------------------------------------------------------+
| var          | calculate variance in window                         |
+--------------+------------------------------------------------------+
| min          | calculate minimum in window                          |
+--------------+------------------------------------------------------+
| max          | calculate maximum in window                          |
+--------------+------------------------------------------------------+
| sum          | calculate sum in window                              |
+--------------+------------------------------------------------------+
| mean         | calculate mean in window                             |
+--------------+------------------------------------------------------+
| stdev        | calculate standard deviation in window               |
+--------------+------------------------------------------------------+
| percentile   | calculate percentile value in window                 |
+--------------+------------------------------------------------------+
| proportion   | calculate proportion in window                       |
+--------------+------------------------------------------------------+

.. note::
   You can specify the no data value for the smoothnodata filter with the extra key 'nodata' and a list of no data values. The interpolation type can be set with the key 'interp' (check complete list of `values <http://www.gnu.org/software/gsl/manual/html_node/Interpolation-Types.html>`_, removing the leading "gsl_interp").

Example:

Smooth the 0 valued pixel values using a linear interpolation in a spectral/temporal neighborhood of 5 bands::

  jim_filtered=jim.filter1d({'filter':'smoothnodata','nodata':0,'interp':'linear','dz':5})

**Wavelet filters**

Perform a wavelet transform (or inverse) in spectral/temporal domain.

.. note::
   The wavelet coefficients can be positive and negative. If the input raster dataset has an unsigned data type, make sure to set the output to a signed data type using the key 'otype'.

You can use set the wavelet family with the key 'family' in the dictionary. The following wavelets are supported as values:

* daubechies
* daubechies_centered
* haar
* haar_centered
* bspline
* bspline_centered

+----------+--------------------------------------+
| filter   | description                          |
+==========+======================================+
| dwt      | discrete wavelet transform           |
+----------+--------------------------------------+
| dwti     | discrete inverse wavelet transform   |
+----------+--------------------------------------+
| dwt_cut  | DWT approximation in spectral domain |
+----------+--------------------------------------+

.. note::
   The filter 'dwt_cut' performs a forward and inverse transform, approximating the input signal. The approximation is performed by discarding a percentile of the wavelet coefficients that can be set with the key 'threshold'. A threshold of 0 (default) retains all and a threshold of 50 discards the lower half of the wavelet coefficients. 

Example:

Approximate the multi-temporal raster dataset by discarding the lower 20 percent of the coefficients after a discrete wavelet transform. The input dataset has a Byte data type. We wavelet transform is calculated using an Int16 data type. The approximated image is then converted to a Byte dataset, making sure all values below 0 and above 255 are set to 0::

  jim_approx=jim_multitemp.filter1d({'filter':'dwt_cut','threshold':20, 'otype':Int16})
  jim_approx=jim_approx({'min':0,'max':255,'nodata':0}).convert({'otype':'Byte'})

**Hyperspectral filters**

Hyperspectral filters assume the bands in the input raster dataset correspond to contiguous spectral bands. Full width half max (FWHM) and spectral response filters are supported. They converts an N band input raster dataset to an M (< N) band output raster dataset.

The full width half max (FWHM) filter expects a list of M center wavelenghts and a corresponding list of M FWHM values. The M center wavelenghts define the output wavelenghts and must be provided with the key 'wavelengthOut'. For the FHWM, use the key 'fwhm' and a list of M values. The algorithm needs to know the N wavelenghts that correspond to the N bands of the input raster dataset. Use the key 'wavelengthIn' and a list of N values. The units of input, output and FWHM are arbitrary, but should be identical (e.g., nm).

Example:

Covert the hyperspectral input raster dataset, with the wavelengths defined in the list wavelenghts_in to a multispectral raster dataset with three bands, corresponding to Red, Green, and Blue::

  wavelengths_in=[]
  #define the wavelenghts of the input raster dataset
  
  if len(wavelength_in) == jim_hyperspectral.nrOfBand():
     jim_rgb=jim_hyperspectral.filter1d({'wavelengthIn:wavelenghts_in,'wavelengthOut':[650,510,475],'fwhm':[50,50,50]})
  else:
     print("Error: number of input wavelengths must be equal to number of bands in input raster dataset")

.. note::
    The input wavelenghts are automatically interpolated. You can specify the interpolation using the key 'interp' and values as listed interpolation http://www.gnu.org/software/gsl/manual/html_node/Interpolation-Types.html

The spectral response filter (SRF) 

The input raster dataset is filtered with M of spectral response functions (SRF).  Each spectral response function must be provided by the user in an ASCII file that consists of two columns: wavelengths and response. Use the key 'srf' and a list of paths to the ASCII file(s). The algorithm automatically takes care of the normalization of the SRF.

Example:

Covert the hyperspectral input raster dataset, to a multispectral raster dataset with three bands, corresponding to Red, Green, and Blue as defined in the ASCII text files 'srf_red.txt', 'srf_green.txt', 'srf_blue.txt'::

  wavelengths_in=[]
  #specify the wavelenghts of the input raster dataset

  if len(wavelength_in) == jim_hyperspectral.nrOfBand():
     jim_rgb=jim_hyperspectral.filter1d({'wavelengthIn:wavelenghts_in,'srf':['srf_red.txt','srf_green.txt','srf_blue.txt']})
  else:
     print("Error: number of input wavelengths must be equal to number of bands in input raster dataset")

.. note::
    The input wavelenghts are automatically interpolated. You can specify the interpolation using the key 'interp' and values as listed interpolation http://www.gnu.org/software/gsl/manual/html_node/Interpolation-Types.html


**Custom filters**

For the custom filter, you can specify your own taps using the keyword 'tapz' and a list of filter tap values. The tap values are automatically normalized by the algorithm.

Example:

Perform a simple smoothing filter by defining three identical tap values::

  jim_filtered=jim.filter1d({'tapz':[1,1,1]})

END

METHOD filter2d(dict)
Filter Jim image in spatial domain performed on single or multi-band raster dataset.

Args:
* ``dict`` (Python Dictionary) with key value pairs. Each key (a 'quoted' string) is separated from its value by a colon (:). The items are separated by commas and the dictionary is enclosed in curly braces. An empty dictionary without any items is written with just two curly braces, like this: {}. A value can be a list that is also separated by commas and enclosed in square brackets [].


Returns:
   This filtered of Jim object (self)

Supported keys in the dict:


+------------------+---------------------------------------------------------------------------------+
| key              | value                                                                           |
+==================+=================================================================================+
| filter           | filter function (see values for different filter types in tables below)         |
+------------------+---------------------------------------------------------------------------------+
| dx               | filter kernel size in x, use odd values only (default is 3)                     |
+------------------+---------------------------------------------------------------------------------+
| dy               | filter kernel size in y, use odd values only (default is 3)                     |
+------------------+---------------------------------------------------------------------------------+
| pad              | Padding method for filtering (how to handle edge effects)                       |
|                  | Possible values are: symmetric (default), replicate, circular, zero (pad with 0)|
+------------------+---------------------------------------------------------------------------------+
| otype            | Data type for output image                                                      |
+------------------+---------------------------------------------------------------------------------+


**Edge detection**

+---------------------+-------------------------------------------------------------------------+
| filter              | description                                                             |
+=====================+=========================================================================+
| sobelx              | Sobel operator in x direction                                           |
+---------------------+-------------------------------------------------------------------------+
| sobely              | Sobel operator in y direction                                           |
+---------------------+-------------------------------------------------------------------------+
| sobelxy             | Sobel operator in x and y direction                                     |
+---------------------+-------------------------------------------------------------------------+
| homog               | binary value indicating if pixel is identical to all pixels in kernel   |
+---------------------+-------------------------------------------------------------------------+
| heterog             | binary value indicating if pixel is different than all pixels in kernel |
+---------------------+-------------------------------------------------------------------------+

Example:

Perform Sobel edge detection in both x and direction::

  jim_filtered=jim.filter2d({'filter':'sobelxy'})

**Morphological filters**

.. note::
   For a more comprehensive list morphological operators, please refer to :ref:`advanced spatial morphological operators <mia_morpho2d>`. 

+---------------------+------------------------------------------------------+
| filter              | description                                          |
+=====================+======================================================+
| dilate              | morphological dilation                               |
+---------------------+------------------------------------------------------+
| erode               | morphological erosion                                |
+---------------------+------------------------------------------------------+
| close               | morpholigical closing (dilate+erode)                 |
+---------------------+------------------------------------------------------+
| open                | morpholigical opening (erode+dilate)                 |
+---------------------+------------------------------------------------------+

.. note::
   You can use the optional key 'class' with a list value to take only these pixel values into account. For instance, use 'class':[255] to dilate clouds in the raster dataset that have been flagged with value 255. In addition, you can use a circular disc kernel (set the key 'circular' to True).

Example:

Perform a morphological dilation using a circular kernel with size (diameter) of 5 pixels::

  jim_filtered=jim.filter2d({'filter':'dilate','dx':5,'dy':5,'circular':True})

**Statistical filters**

+--------------+------------------------------------------------------+
| filter       | description                                          |
+==============+======================================================+
| smoothnodata | smooth nodata values (set nodata option!)            |
+--------------+------------------------------------------------------+
| nvalid       | report number of valid (not nodata) values in window |
+--------------+------------------------------------------------------+
| median       | perform a median filter                              |
+--------------+------------------------------------------------------+
| var          | calculate variance in window                         |
+--------------+------------------------------------------------------+
| min          | calculate minimum in window                          |
+--------------+------------------------------------------------------+
| max          | calculate maximum in window                          |
+--------------+------------------------------------------------------+
| ismin        | binary value indicating if pixel is minimum in kernel|
+--------------+------------------------------------------------------+
| ismax        | binary value indicating if pixel is maximum in kernel|
+--------------+------------------------------------------------------+
| sum          | calculate sum in window                              |
+--------------+------------------------------------------------------+
| mode         | calculate the mode (only for categorical values)     |
+--------------+------------------------------------------------------+
| mean         | calculate mean in window                             |
+--------------+------------------------------------------------------+
| stdev        | calculate standard deviation in window               |
+--------------+------------------------------------------------------+
| percentile   | calculate percentile value in window                 |
+--------------+------------------------------------------------------+
| proportion   | calculate proportion in window                       |
+--------------+------------------------------------------------------+

.. note::
   You can specify the no data value for the smoothnodata filter with the extra key 'nodata' and a list of no data values. The interpolation type can be set with the key 'interp' (check complete list of `values <http://www.gnu.org/software/gsl/manual/html_node/Interpolation-Types.html>`_, removing the leading "gsl_interp").

Example:

Perform a median filter with kernel size of 5x5 pixels::

  jim_filtered=jim.filter2d({'filter':'median','dz':5})

**Wavelet filters**

Perform a wavelet transform (or inverse) in spatial domain.

.. note::
   The wavelet coefficients can be positive and negative. If the input raster dataset has an unsigned data type, make sure to set the output to a signed data type using the key 'otype'.

You can use set the wavelet family with the key 'family' in the dictionary. The following wavelets are supported as values:

* daubechies
* daubechies_centered
* haar
* haar_centered
* bspline
* bspline_centered

+----------+--------------------------------------+
| filter   | description                          |
+==========+======================================+
| dwt      | discrete wavelet transform           |
+----------+--------------------------------------+
| dwti     | discrete inverse wavelet transform   |
+----------+--------------------------------------+
| dwt_cut  | DWT approximation in spectral domain |
+----------+--------------------------------------+

.. note::
   The filter 'dwt_cut' performs a forward and inverse transform, approximating the input signal. The approximation is performed by discarding a percentile of the wavelet coefficients that can be set with the key 'threshold'. A threshold of 0 (default) retains all and a threshold of 50 discards the lower half of the wavelet coefficients. 

Example:

Approximate the multi-temporal raster dataset by discarding the lower 20 percent of the coefficients after a discrete wavelet transform. The input dataset has a Byte data type. We wavelet transform is calculated using an Int16 data type. The approximated image is then converted to a Byte dataset, making sure all values below 0 and above 255 are set to 0::

  jim_approx=jim_multitemp.filter2d({'filter':'dwt_cut','threshold':20, 'otype':Int16})
  jim_approx=jim_approx({'min':0,'max':255,'nodata':0}).convert({'otype':'Byte'})

END

METHOD classify(dict)
Supervised classification of a raster dataset. The classifier must have been trained via the :py:func:`VectorOgr:train` method.
The classifier can be selected with the key 'method' and possible values 'svm' and 'ann':

Args:
* ``dict`` (Python Dictionary) with key value pairs. Each key (a 'quoted' string) is separated from its value by a colon (:). The items are separated by commas and the dictionary is enclosed in curly braces. An empty dictionary without any items is written with just two curly braces, like this: {}. A value can be a list that is also separated by commas and enclosed in square brackets [].


Returns:
   The classified raster dataset.

Supported keys in the dict (with more keys defined for the respective classication methods):

+------------------+------------------------------------------------------------------------------------------------------+
| key              | value                                                                                                |
+==================+======================================================================================================+
| method           | Classification method (svm or ann)                                                                   |
+------------------+------------------------------------------------------------------------------------------------------+
| model            | Model filename to save trained classifier                                                            |
+------------------+------------------------------------------------------------------------------------------------------+
| band             | Band index (starting from 0). The band order must correspond to the band names defined in the model. |
|                  | Leave empty to use all bands                                                                         |
+------------------+------------------------------------------------------------------------------------------------------+

The support vector machine (SVM) supervised classifier is described `here <http://dx.doi.org/10.1007/BF00994018>`_. The implementation in JIPlib is based on the open source `libsvm <https://www.csie.ntu.edu.tw/~cjlin/libsvm/>`_.

The artificial neural network (ANN) supervised classifier is based on the back propagation model as introduced by D. E. Rumelhart, G. E. Hinton, and R. J. Williams (Nature, vol. 323, pp. 533-536, 1986). The implementation is based on the open source C++ library fann (http://leenissen.dk/fann/wp/).

**Prior probabilities**

Prior probabilities can be set for each of the classes. The prior probabilities can be provided with the key 'prior' and a list of values for each of the (in ascending order). The priors are automatically normalized by the algorithm. Alternatively, a prior probability can be provided for each pixel, using the key 'priorimg' and a value pointing to the path of multi-band raster dataset. The bands of the raster dataset represent the prior probabilities for each of the classes.

**Classifying parts of the input raster dataset**

Parts of the input raster dataset can be classified only by using a vector or raster mask. To apply a vector mask, use the key 'extent' with the path of the vector dataset as a value. Optionally, a spatial extent option can be provided with the key 'eo' that controlls the rasterization process (values can be either one of: ATTRIBUTE|CHUNKYSIZE|ALL_TOUCHED|BURN_VALUE_FROM|MERGE_ALG). For instance, you can define 'eo':'ATTRIBUTE=fieldname' to rasterize only those features with an attribute equal to fieldname.

To apply a raster mask, use the key 'mask' with the path of the raster dataset as a value. Mask value(s) not to consider for classification can be set as a list value with the key 'msknodata'.

+------------------+---------------------------------------------------------------------------------+
| key              | value                                                                           |
+==================+=================================================================================+
| extent           | Data type for output image                                                      |
+------------------+---------------------------------------------------------------------------------+
| eo               | Special extent options controlling rasterization                                |
+------------------+---------------------------------------------------------------------------------+
| mask             | Only classify within specified mask                                             |
+------------------+---------------------------------------------------------------------------------+
| msknodata        | Mask value(s) in mask not to consider for classification                        |
+------------------+---------------------------------------------------------------------------------+
| nodata           | Nodata value to put where image is masked as no data                            |
+------------------+---------------------------------------------------------------------------------+

END

METHOD classifySML(dict)
Supervised classification of a raster dataset using the symbolic machine learning algorithm `sml <https://doi.org/10.3390/rs8050399>`_. For training, one or more reference raster datasets with categorical values is expected as a JimList. The reference raster dataset is typically at a lower spatial resolution than the input raster dataset to be classified. Unlike the :py:func:`Jim:classify`, the training is performed not prior to the classification, but in the same process as the classification.

Args:
* ``dict`` (Python Dictionary) with key value pairs. Each key (a 'quoted' string) is separated from its value by a colon (:). The items are separated by commas and the dictionary is enclosed in curly braces. An empty dictionary without any items is written with just two curly braces, like this: {}. A value can be a list that is also separated by commas and enclosed in square brackets [].

Returns:
   A multiband raster dataset with one band for each class. The pixel values represent the respective frequencies of the classes (scaled to Byte). To create a hard classified output, obtain the maxindex of this output. The result will then contains the class indices (0-nclass-1). To obtain the same class numbers as defined in the reference dataset, use the :py:func:`Jim:reclass` method (see example below).

Supported keys in the dict:

+------------------+---------------------------------------------------------------------------------+
| key              | value                                                                           |
+==================+=================================================================================+
| band             | List of band indices (starting from 0). Leave empty to use all bands            |
+------------------+---------------------------------------------------------------------------------+
| class            | List of classes to extract from the reference. Leave empty to extract two       |
|                  | classes only (1 against rest)                                                   |
+------------------+---------------------------------------------------------------------------------+
| otype            | Data type for output image                                                      |
+------------------+---------------------------------------------------------------------------------+


**Classifying parts of the input raster dataset**

See :py:func:`Jim:classify`.

Example:

Use the Corine land cover product as a reference to perform an SML classification of a Sentinel-2 image using the 10 m bands (B02, B03, B04 and B08).

Import modules::

  import os, sys
  from osgeo import gdal
  from osgeo import gdalconst
  from osgeo import ogr
  from osgeo import osr
  import fnmatch
  import time
  import numpy as np
  from scipy import misc
  import operator
  import jiplib as jl
  from osgeo import gdal

Preparation of input. Stack all input bands to single multiband input raster dataset. Scale input to Byte and adapt the dynamic range to chosen number of bits::
 
  NBIT=7
  jimlist=jl.createJimList()
  for file in sorted(fnmatch.filter(os.listdir(infolder), '*_B0[2348].jp2')):
      file=os.path.join(infolder,file)
      jim=jl.createJim({'filename':file,'dx':100,'dy':100})
      jim_convert=jim.convert({'autoscale':[2**(8-NBIT),2**8-1],'otype':'GDT_Byte'}).pointOpBitShift(8-NBIT)
      jim.close()
      jimlist.pushImage(jim_convert)
  jim=jimlist.stack()
  jimlist.close()

Then prepare reference dataset. The reference Corine land cover is in the LAEA (EPSG:3035) coordinate reference system. We will only read the area corresponding to the input image Therefore, we need to calculate the transformed bounding box of the input image in LAEA::

  corinefn='/eos/jeodpp/data/base/Landcover/EUROPE/CorineLandCover/CLC2012/VER18-5/Data/GeoTIFF/250m/g250_clc12_V18_5.tif'
  jim_ref=jl.createJim({'filename':corinefn,'noread':True,'a_srs':'EPSG:3035'})
  print("bounding box input image:",jim.getUlx(), jim.getUly(), jim.getLrx(), jim.getLry())
  pointUL = ogr.Geometry(ogr.wkbPoint)
  pointUL.AddPoint(jim.getUlx(), jim.getUly())
  pointLR = ogr.Geometry(ogr.wkbPoint)
  pointLR.AddPoint(jim.getLrx(), jim.getLry())
  source = osr.SpatialReference()
  source.ImportFromEPSG(32632)
  target = osr.SpatialReference()
  target.ImportFromEPSG(3035)
  transform = osr.CoordinateTransformation(source, target)
  pointUL.Transform(transform)
  pointLR.Transform(transform)

Now we can open the reference image for the region of interest. We will open it in a reduced spatial resolution of 500 m::

   jim_ref=jl.createJim({'filename':corinefn,'dx':500,'dy':500.0,'ulx':pointUL.GetX(),'uly':pointUL.GetY(),'lrx':pointLR.GetX(),'lry':pointLR.GetY(),'a_srs':'EPSG:3035'})

Create a dictionary with the class names and corresponding values used in the classified raster map::

  classDict={}
  classDict['urban']=2
  classDict['agriculture']=12
  classDict['forest']=25
  classDict['water']=41
  classDict['rest']=50
  sorted(classDict.values())

Reclass the reference to the selected classes::

  classFrom=range(0,50)
  classTo=[50]*50
  for i in range(0,50):
  if i>=1 and i<10:
  classTo[i]=classDict['urban']
  elif i>=11 and i<22:
  classTo[i]=classDict['agriculture']
  elif i>=23 and i<25:
  classTo[i]=classDict['forest']
  elif i>=40 and i<45:
  classTo[i]=classDict['water']
  else:
  classTo[i]=classDict['rest']

  jim_ref=jim_ref.reclass({'class':classFrom,'reclass':classTo})

The SML algorithm uses a JimList of reference raster datasets. Here we will create a list of a single reference only::

  reflist=jl.createJimList([jim_ref])

For a multi-class problem, we must define the list of classes that should be taken into account by the SML algorithm::

  sml=jim.classifySML(reflist,{'class':sorted(classDict.values())}).setNoData([0])

Preparation of output. The output is a multiband raster dataset with one band for each class. The pixels represent the respective frequencies of the classes (scaled to Byte)

We can create a hard classified output by obtaining the maxindex of this output. The result contains the class indices (0-nclass-1).
To obtain the same class numbers as defined in the reference dataset, we can reclass accordingly::

  sml_class=sml.statProfile({'function':'maxindex'}).reclass({'class':range(0,sml.nrOfBand()),'reclass':sorted(classDict.values())})

END

METHOD reclass(dict)
Replace categorical pixel values in raster dataset

Args:
* ``dict`` (Python Dictionary) with key value pairs. Each key (a 'quoted' string) is separated from its value by a colon (:). The items are separated by commas and the dictionary is enclosed in curly braces. An empty dictionary without any items is written with just two curly braces, like this: {}. A value can be a list that is also separated by commas and enclosed in square brackets [].

Returns:
   Raster dataset with class values replaced according to corresponding class and reclass list values.

Supported keys in the dict:

+------------------+---------------------------------------------------------------------------------+
| key              | value                                                                           |
+==================+=================================================================================+
| class            | List of input classes to reclass from                                           |
+------------------+---------------------------------------------------------------------------------+
| reclass          | List of output classes to reclass to                                            |
+------------------+---------------------------------------------------------------------------------+
| otype            | Data type for output image (default is type of input raster dataset)            |
+------------------+---------------------------------------------------------------------------------+

.. note::
   The list size of the class and reclass should be identical. The value class[index] will be replaced with the value reclass[index].

Example:

Reclass all pixel values 0 to 255::

  jim_reclass=jim.reclass({'class':[0],'reclass':[255]})

END

METHOD setThreshold(dict)
Apply minimum and maximum threshold to pixel values in raster dataset

Args:
* ``dict`` (Python Dictionary) with key value pairs. Each key (a 'quoted' string) is separated from its value by a colon (:). The items are separated by commas and the dictionary is enclosed in curly braces. An empty dictionary without any items is written with just two curly braces, like this: {}. A value can be a list that is also separated by commas and enclosed in square brackets [].

Supported keys in the dict:

+------------------+---------------------------------------------------------------------------------+
| key              | value                                                                           |
+==================+=================================================================================+
| min              | Minimum threshold value (if pixel value < min set pixel value to no data)       |
+------------------+---------------------------------------------------------------------------------+
| max              | Maximum threshold value (if pixel value < max set pixel value to no data)       |
+------------------+---------------------------------------------------------------------------------+
| value            | value to be set if within min and max                                           |
|                  | (if not set, valid pixels will remain their input value)                        |
+------------------+---------------------------------------------------------------------------------+
| abs              | Set to True to perform threshold test to absolute pixel values                  |
+------------------+---------------------------------------------------------------------------------+
| nodata           | Set pixel value to this no data if pixel value < min or > max                   |
+------------------+---------------------------------------------------------------------------------+

Returns:
   Raster dataset with pixel threshold applied.

Example:

Mask all values not within [0,250] and set to 255 (no data)::

  jim_threshold=jim.setThreshold({'min':0,'max':250,'nodata':255})

END

METHOD getMask(dict)
Create mask image based on values in input raster dataset.

Args:
* ``dict`` (Python Dictionary) with key value pairs. Each key (a 'quoted' string) is separated from its value by a colon (:). The items are separated by commas and the dictionary is enclosed in curly braces. An empty dictionary without any items is written with just two curly braces, like this: {}. A value can be a list that is also separated by commas and enclosed in square brackets [].

Supported keys in the dict (more keys defined depending on the mask type)

+------------------+---------------------------------------------------------------------------------+
| key              | value                                                                           |
+==================+=================================================================================+
| band             | List of bands (0 indexed) user for mask.                                        |
+------------------+---------------------------------------------------------------------------------+
| min              | List of minimum threshold values.                                               |
+------------------+---------------------------------------------------------------------------------+
| min              | List of maximum threshold values.                                               |
+------------------+---------------------------------------------------------------------------------+
| operator         | Boolean operator ("AND" or "OR") used to combine tests applied to list of bands |
|                  | or min/max thresholds. Default is OR.                                           |
+------------------+---------------------------------------------------------------------------------+
| data             | List of pixel values to set if pixel value is within min and max.               |
|                  | List of values correspond to the list of min/max values in min/max values       |
+------------------+---------------------------------------------------------------------------------+
| data             | List of pixel values to set if pixel value is not within min and max.           |
|                  | List of values correspond to the list of min/max values in min/max values       |
+------------------+---------------------------------------------------------------------------------+

Returns:
   Raster mask dataset.

Example:

Create a binary mask from a raster dataset. The mask will get a value 1 (defined by the key 'data') if pixels in the input image are between 1 and 20. Otherwise, the mask will have a 0 (defined by the key 'nodata') value::

  jim_threshold=jim.setThreshold({'min':0,'max':250,'nodata':255})

END

METHOD setMask(mask, dict)
Apply mask image based on values in vector or raster dataset.

Args:
* ``mask`` Either a list of raster datasets (:py:class:`JimList`) or a vector dataset (:py:class:`VectorOgr`)
* ``dict`` (Python Dictionary) with key value pairs. Each key (a 'quoted' string) is separated from its value by a colon (:). The items are separated by commas and the dictionary is enclosed in curly braces. An empty dictionary without any items is written with just two curly braces, like this: {}. A value can be a list that is also separated by commas and enclosed in square brackets [].

Returns:
   Raster dataset with pixel mask applied.

Supported keys in the dict (more keys defined depending on the mask type)

+------------------+---------------------------------------------------------------------------------+
| key              | value                                                                           |
+==================+=================================================================================+
| otype            | Data type for output image                                                      |
+------------------+---------------------------------------------------------------------------------+
| nodata           | Set pixel value to this no data if pixel value not valid according to mask      |
+------------------+---------------------------------------------------------------------------------+

Mask is a :py:class:`JimList`

+------------------+---------------------------------------------------------------------------------+
| key              | value                                                                           |
+==================+=================================================================================+
| msknodata        | List of mask values where raster dataset should be set to nodata.               |
|                  | Use one value for each mask, or multiple values for a single mask.              |
+------------------+---------------------------------------------------------------------------------+
| mskband          | List of mask bands to read (0 indexed). Provide band for each mask.             |
+------------------+---------------------------------------------------------------------------------+
| operator         | List of operators used for testing pixel values against mask.                   |
|                  | Provide one operator for each msknodata value.                                  |
+------------------+---------------------------------------------------------------------------------+

.. note::
   The mask raster datasets in the :py:class:`JimList` can be of a different spatial resolution than the input raster dataset to be masked. A nearest neighbor resampling is used.

Mask is a :py:class:`VectorOgr`

+------------------+---------------------------------------------------------------------------------+
| key              | value                                                                           |
+==================+=================================================================================+
| eo               | Special extent options controlling rasterization                                |
+------------------+---------------------------------------------------------------------------------+
| ln               | List of layer names.                                                            |
+------------------+---------------------------------------------------------------------------------+

.. note::
   Possible values for the key 'eo' are: ATTRIBUTE|CHUNKYSIZE|ALL_TOUCHED|BURN_VALUE_FROM|MERGE_ALG. For instance you can use 'eo':'ATTRIBUTE=fieldname'

Example:

Apply vector mask to a raster dataset, masking all pixels that are touched by the vector to a value 255 (no data). You can reduce the memory footprint by not reading the vector dataset::

  v0=jl.createVector()
  v0.open({'filename':args.vm,'noread':True})
  jim1=jim0.setMask(v0,{'nodata':255,'eo':'ALL_TOUCHED'})

Apply list of raster masks that consists of a single raster dataset jim_mask (created from jim1 with :py:func:`Jim:getMask`) to a raster dataset jim. Set a value 255 (no data) to all values where the mask has a value 0 (msknodata)::

  jim_mask=jim1.getMask({'min':1,'max':20,'nodata':0,'data':1})
  jlist=jl.JimList([jim_mask])
  jim_masked=jim.setMask(jlist,{'nodata':255,'msknodata':0})

END

METHOD getStats(dict)
Calculate statistics of a raster dataset.

Args:
* ``dict`` (Python Dictionary) with key value pairs. Each key (a 'quoted' string) is separated from its value by a colon (:). The items are separated by commas and the dictionary is enclosed in curly braces. An empty dictionary without any items is written with just two curly braces, like this: {}. A value can be a list that is also separated by commas and enclosed in square brackets [].

Returns:
   A dictionary with the results of the statistics, using the same keys as for the functions.

Supported keys in the dict:

+------------------+---------------------------------------------------------------------------------+
| key              | value                                                                           |
+==================+=================================================================================+
| function         | Statistical function (see values for different functions in tables below)       |
+------------------+---------------------------------------------------------------------------------+
| cband            | List of bands on which to calculate the statistics                              |
+------------------+---------------------------------------------------------------------------------+
| down             | Down sampling factor (in pixels x and y) to calculate the statistics on a subset|
+------------------+---------------------------------------------------------------------------------+
| src_min          | Do not take smaller values into account when calculating statistics             |
+------------------+---------------------------------------------------------------------------------+
| src_max          | Do not take higher values into account when calculating statistics              |
+------------------+---------------------------------------------------------------------------------+
| nodata           | Do not take these values into account when calculating statistics               |
+------------------+---------------------------------------------------------------------------------+
| otype            | Data type for output image                                                      |
+------------------+---------------------------------------------------------------------------------+

.. note::
   For statistical functions requiring two sets of inputs, use a list of two values for cband (e.g., regression and histogram2d)

**Supported statistical functions**

+--------------+------------------------------------------------------+
| function     | description                                          |
+=====================+===============================================+
| invalid      | report number of invalid (nodata) values             |
+--------------+------------------------------------------------------+
| nvalid       | report number of valid (not nodata) values           |
+--------------+------------------------------------------------------+
| basic        | Shows basic statistics                               |
|              | (min,max, mean and stdDev of the raster datasets)    |
+--------------+------------------------------------------------------+
| gdal         | Use the GDAL calculation of basic statistics         |
+--------------+------------------------------------------------------+
| mean         | calculate the mean value                             |
+--------------+------------------------------------------------------+
| median       | calculate the median value                           |
+--------------+------------------------------------------------------+
| var          | calculate variance value                             |
+--------------+------------------------------------------------------+
| stdev        | calculate standard deviation                         |
+--------------+------------------------------------------------------+
| skewness     | calculate the skewness                               |
+--------------+------------------------------------------------------+
| kurtosis     | calculate the kurtosis                               |
+--------------+------------------------------------------------------+
| sum          | calculate sum of all values                          |
+--------------+------------------------------------------------------+
| minmax       | calculate minimum and maximum value                  |
+--------------+------------------------------------------------------+
| min          | calculate minimum value                              |
+--------------+------------------------------------------------------+
| max          | calculate maximum value                              |
+--------------+------------------------------------------------------+
| histogram    | calculate the histogram                              |
+--------------+------------------------------------------------------+
| histogram2d  | calculate the two-dimensional histogram for two bands|
+--------------+------------------------------------------------------+
| rmse         | calculate root mean square error for two bands       |
+--------------+------------------------------------------------------+
| regresssion  | calculate the regression between two bands           |
+--------------+------------------------------------------------------+

For the histogram function, the following key values can be set:

+--------------+------------------------------------------------------+
| key          | description                                          |
+=====================+===============================================+
| nbin         | Number of bins for the histogram                     |
+--------------+------------------------------------------------------+
| relative     | Set to True to report percentage values              |
+--------------+------------------------------------------------------+
| kde          | Set to True to use Kernel density estimation when    |
|              | producing histogram. The standard deviation is       |
|              | estimated based on Silverman's rule of thumb         |
+--------------+------------------------------------------------------+

Example:

Get the histogram of the input raster dataset using 10 bins::

  jim.getStats({'function':['histogram','nbin':10})

END

METHOD statProfile(dict)
Obtain a statistical profile per pixel based on a multi-band input raster dataset. Multiple functions can be set, resulting in a multi-band raster dataset (one output band for each function).

Args:
* ``dict`` (Python Dictionary) with key value pairs. Each key (a 'quoted' string) is separated from its value by a colon (:). The items are separated by commas and the dictionary is enclosed in curly braces. An empty dictionary without any items is written with just two curly braces, like this: {}. A value can be a list that is also separated by commas and enclosed in square brackets [].

Returns:
   The statistical profile of the input raster dataset

Supported keys in the dict:


+------------------+---------------------------------------------------------------------------------+
| key              | value                                                                           |
+==================+=================================================================================+
| function         | Statistical function (see values for different functions in tables below)       |
+------------------+---------------------------------------------------------------------------------+
| nodata           | Do not take these values into account when calculating statistics               |
+------------------+---------------------------------------------------------------------------------+
| otype            | Data type for output image                                                      |
+------------------+---------------------------------------------------------------------------------+


**Statistical profile functions**

+--------------+------------------------------------------------------+
| function     | description                                          |
+=====================+===============================================+
| nvalid       | report number of valid (not nodata) values in window |
+--------------+------------------------------------------------------+
| median       | perform a median filter                              |
+--------------+------------------------------------------------------+
| var          | calculate variance in window                         |
+--------------+------------------------------------------------------+
| min          | calculate minimum in window                          |
+--------------+------------------------------------------------------+
| max          | calculate maximum in window                          |
+--------------+------------------------------------------------------+
| sum          | calculate sum in window                              |
+--------------+------------------------------------------------------+
| mode         | calculate the mode (only for categorical values)     |
+--------------+------------------------------------------------------+
| mean         | calculate mean in window                             |
+--------------+------------------------------------------------------+
| stdev        | calculate standard deviation in window               |
+--------------+------------------------------------------------------+
| percentile   | calculate percentile value in window                 |
+--------------+------------------------------------------------------+
| proportion   | calculate proportion in window                       |
+--------------+------------------------------------------------------+

.. note::
   The 'percentile' function calculates the percentile value based on the pixel values in the multi-band input raster dataset. A number of percentiles can be calculated, e.g., 10th and 50th percentile, resulting in a multi-band output raster dataset (one band for each calculated percentile). The percentiles to be calculated can be set with the key 'perc' and a list of values.

Example:

Calculated the 10th and 50th percentiles for the multi-band input raster dataset jim::

  jim_percentiles=jim.statProfile({'function':args.function,'perc':[10,50]})

END

METHOD stretch(dict)
Stretch the input raster dataset.

Args:
* ``dict`` (Python Dictionary) with key value pairs. Each key (a 'quoted' string) is separated from its value by a colon (:). The items are separated by commas and the dictionary is enclosed in curly braces. An empty dictionary without any items is written with just two curly braces, like this: {}. A value can be a list that is also separated by commas and enclosed in square brackets [].

Returns:
   A dictionary with the results of the statistics, using the same keys as for the functions.

Supported keys in the dict:

+------------------+---------------------------------------------------------------------------------+
| key              | value                                                                           |
+==================+=================================================================================+
| function         | Statistical function (see values for different functions in tables below)       |
+------------------+---------------------------------------------------------------------------------+
| down             | Down sampling factor (in pixels x and y) to calculate the statistics on a subset|
+------------------+---------------------------------------------------------------------------------+
| src_min          | Clip source below this minimum value                                            |
+------------------+---------------------------------------------------------------------------------+
| src_max          | Clip source above this minimum value                                            |
+------------------+---------------------------------------------------------------------------------+
| dst_min          | Mininum value in output image                                                   |
+------------------+---------------------------------------------------------------------------------+
| dst_max          | Maximum value in output image                                                   |
+------------------+---------------------------------------------------------------------------------+
| cc_min           | Cumulative count cut from                                                       |
+------------------+---------------------------------------------------------------------------------+
| cc_max           | Cumulative count cut to                                                         |
+------------------+---------------------------------------------------------------------------------+
| band             | List of bands to stretch                                                        |
+------------------+---------------------------------------------------------------------------------+
| eq               | Set to True to perform histogram equalization                                   |
+------------------+---------------------------------------------------------------------------------+
| nodata           | List of values not to take into account when stretching                         |
+------------------+---------------------------------------------------------------------------------+
| otype            | Data type for output image                                                      |
+------------------+---------------------------------------------------------------------------------+

Example:

Stretch the input raster dataset using the cumulative counts of 5 and 95 percent. Then, the output is converted to Byte with a dynamic range that is calculated based on the number of user defined bits (NBIT=[1:8])::

  CCMIN=5
  CCMAX=95
  NBIT=7
  jim_stretched=jim.({'cc_min':CCMIN,'cc_max':CCMAX,'dst_min':2**(8-NBIT),'dst_max':2**8-1,'otype':'GDT_Float32'})
  jim_byte=jim_stretched.convert({'otype':'GDT_Byte'}).pointOpBitShift(8-NBIT)

END

METHOD extractOgr(*args)
Extract pixel values from raster image using a vector dataset sample.

Args:
* ``dict`` (Python Dictionary) with key value pairs. Each key (a 'quoted' string) is separated from its value by a colon (:). The items are separated by commas and the dictionary is enclosed in curly braces. An empty dictionary without any items is written with just two curly braces, like this: {}. A value can be a list that is also separated by commas and enclosed in square brackets [].

Returns:
   A :py:class:`VectorOgr` with the same geometry as the sample vector dataset and an extra field for each of the calculated raster value (zonal) statistics. The same layer name(s) of the sample will be used for the output vector dataset.

Supported keys in the dict:

+------------------+---------------------------------------------------------------------------------+
| key              | value                                                                           |
+==================+=================================================================================+
| rule             | Rule how to calculate zonal statistics per feature                              |
+------------------+---------------------------------------------------------------------------------+
| copy             | Copy these fields from the sample vector dataset (default is to copy all fields)|
+------------------+---------------------------------------------------------------------------------+
| label            | Create extra field named 'label' with this value                                |
+------------------+---------------------------------------------------------------------------------+
| fid              | Create extra field named 'fid' with this field identifier (sequence of features)|
+------------------+---------------------------------------------------------------------------------+
| band             | List of bands to extract (0 indexed). Default is to use extract all bands       |
+------------------+---------------------------------------------------------------------------------+
| bandname         | List of band name corresponding to list of bands to extract                     |
+------------------+---------------------------------------------------------------------------------+
| startband        | Start band sequence number (0 indexed)                                          |
+------------------+---------------------------------------------------------------------------------+
| endband          | End band sequence number (0 indexed)                                            |
+------------------+---------------------------------------------------------------------------------+
| output           | Name of the output vector dataset in which the zonal statistics are saved       |
+------------------+---------------------------------------------------------------------------------+
| oformat          | Output vector dataset format                                                    |
+------------------+---------------------------------------------------------------------------------+
| co               | Creation option for output vector dataset                                       |
+------------------+---------------------------------------------------------------------------------+

**Supported rules for extraction**

+------------------+---------------------------------------------------------------------------------------------------+
| rule             | description                                                                                       |
+==================+===================================================================================================+
| point            | extract a single pixel within the polygon or on each point feature                                |
+------------------+---------------------------------------------------------------------------------------------------+
| allpoints        | Extract all pixel values covered by the polygon                                                   |
+------------------+---------------------------------------------------------------------------------------------------+
| centroid         | Extract pixel value at the centroid of the polygon                                                |
+------------------+---------------------------------------------------------------------------------------------------+
| mean             | Extract average of all pixel values within the polygon                                            |
+------------------+---------------------------------------------------------------------------------------------------+
| stdev            | Extract standard deviation of all pixel values within the polygon                                 |
+------------------+---------------------------------------------------------------------------------------------------+
| median           | Extract median of all pixel values within the polygon                                             |
+------------------+---------------------------------------------------------------------------------------------------+
| min              | Extract minimum value of all pixels within the polygon                                            |
+------------------+---------------------------------------------------------------------------------------------------+
| max              | Extract maximum value of all pixels within the polygon                                            |
+------------------+---------------------------------------------------------------------------------------------------+
| sum              | Extract sum of the values of all pixels within the polygon                                        |
+------------------+---------------------------------------------------------------------------------------------------+
| mode             | Extract the mode of classes within the polygon (classes must be set with the option class)        |
+------------------+---------------------------------------------------------------------------------------------------+
| proportion       | Extract proportion of class(es) within the polygon (classes must be set with the option class)    |
+------------------+---------------------------------------------------------------------------------------------------+
| count            | Extract count of class(es) within the polygon (classes must be set with the option class)         |
+------------------+---------------------------------------------------------------------------------------------------+
| percentile       | Extract percentile as defined by option perc (e.g, 95th percentile of values covered by polygon)  |
+------------------+---------------------------------------------------------------------------------------------------+

**Masking values from extract**

To mask some pixels from the extraction process, there are some keys that can be used:

+------------------+---------------------------------------------------------------------------------+
| key              | value                                                                           |
+==================+=================================================================================+
| srcnodata        | List of nodata values not to extract                                            |
+------------------+---------------------------------------------------------------------------------+
| bndnodata        | List of band in input image to check if pixel is valid (used for srcnodata)     |
+------------------+---------------------------------------------------------------------------------+
| mask             | Use the the specified file as a validity mask                                   |
+------------------+---------------------------------------------------------------------------------+
| mskband          | Use the the specified band of the mask file defined                             |
+------------------+---------------------------------------------------------------------------------+
| msknodata        | List of mask values not to extract                                              |
+------------------+---------------------------------------------------------------------------------+
| threshold        | Maximum number of features to extract (use positive values for percentage value |
|                  | and negative value for absolute threshold)                                      |
+------------------+---------------------------------------------------------------------------------+

Example:

Open a raster sample dataset based on land cover map (e.g., Corine) and use it to extract a stratified sample of 100 points from an input raster dataset with four spectral bands ('B02', 'B03', 'B04', 'B08'). Only sample classes 2 (urban), 12 (agriculture), 25 (forest), 41 (water) and an aggregated (rest) class 50::

  jim_ref=jl.createJim({'filename':'/path/to/landcovermap.tif'})

  samplefn='path/to/sample.sqlite'
  outputfn='path/to/output.sqlite'

  classDict={}
  classDict['urban']=2
  classDict['agriculture']=12
  classDict['forest']=25
  classDict['water']=41
  classDict['rest']=50
  classFrom=range(0,50)
  classTo=[50]*50
  for i in range(0,50):
     if i>=1 and i<10:
        classTo[i]=classDict['urban']
     elif i>=11 and i<22:
        classTo[i]=classDict['agriculture']
     elif i>=23 and i<25:
        classTo[i]=classDict['forest']
     elif i>=40 and i<45:
        classTo[i]=classDict['water']
     else:
        classTo[i]=classDict['rest']


  jim_ref=jl.createJim({'filename':args.reference,'dx':jim.getDeltaX(),'dy':jim.getDeltaY(),'ulx':jim.getUlx(),'uly':jim.getUly(),'lrx':jim.getLrx(),'lry':jim.getLry()})
  jim_ref=jim_ref.reclass({'class':classFrom,'reclass':classTo})

  srcnodata=[0]
  dict={'srcnodata':srcnodata}
  dict.update({'output':output})
  dict.update({'class':sorted(classDict.values())})
  sampleSize=-100 #use negative values for absolute and positive values for percentage values
  dict.update({'threshold':sampleSize})
  dict.update({'bandname':['B02','B03','B04','B08']})
  dict.update({'band':[0,1,2,3]})

  sample=jim.extractImg(jim_ref,dict)

END

METHOD extractSample(dict)
Extract a random or grid sample from raster image.

Args:
* ``dict`` (Python Dictionary) with key value pairs. Each key (a 'quoted' string) is separated from its value by a colon (:). The items are separated by commas and the dictionary is enclosed in curly braces. An empty dictionary without any items is written with just two curly braces, like this: {}. A value can be a list that is also separated by commas and enclosed in square brackets [].

Returns:
   A :py:class:`VectorOgr` with fields for each of the calculated raster value (zonal) statistics.

Supported keys in the dict:

+------------------+---------------------------------------------------------------------------------+
| key              | value                                                                           |
+==================+=================================================================================+
| rule             | Rule how to calculate zonal statistics per feature                              |
+------------------+---------------------------------------------------------------------------------+
| buffer           | Buffer for calculating statistics for point features (in number of pixels)      |
+------------------+---------------------------------------------------------------------------------+
| label            | Create extra field named 'label' with this value                                |
+------------------+---------------------------------------------------------------------------------+
| fid              | Create extra field named 'fid' with this field identifier (sequence of features)|
+------------------+---------------------------------------------------------------------------------+
| band             | List of bands to extract (0 indexed). Default is to use extract all bands       |
+------------------+---------------------------------------------------------------------------------+
| bandname         | List of band name corresponding to list of bands to extract                     |
+------------------+---------------------------------------------------------------------------------+
| startband        | Start band sequence number (0 indexed)                                          |
+------------------+---------------------------------------------------------------------------------+
| endband          | End band sequence number (0 indexed)                                            |
+------------------+---------------------------------------------------------------------------------+
| output           | Name of the output vector dataset in which the zonal statistics are saved       |
+------------------+---------------------------------------------------------------------------------+
| ln               | Layer name of output vector dataset                                             |
+------------------+---------------------------------------------------------------------------------+
| oformat          | Output vector dataset format                                                    |
+------------------+---------------------------------------------------------------------------------+
| co               | Creation option for output vector dataset                                       |
+------------------+---------------------------------------------------------------------------------+

**Supported rules for extraction**

+------------------+---------------------------------------------------------------------------------------------------+
| rule             | description                                                                                       |
+==================+===================================================================================================+
| point            | extract a single pixel within the polygon or on each point feature                                |
+------------------+---------------------------------------------------------------------------------------------------+
| allpoints        | Extract all pixel values covered by the polygon                                                   |
+------------------+---------------------------------------------------------------------------------------------------+
| centroid         | Extract pixel value at the centroid of the polygon                                                |
+------------------+---------------------------------------------------------------------------------------------------+
| mean             | Extract average of all pixel values within the polygon                                            |
+------------------+---------------------------------------------------------------------------------------------------+
| stdev            | Extract standard deviation of all pixel values within the polygon                                 |
+------------------+---------------------------------------------------------------------------------------------------+
| median           | Extract median of all pixel values within the polygon                                             |
+------------------+---------------------------------------------------------------------------------------------------+
| min              | Extract minimum value of all pixels within the polygon                                            |
+------------------+---------------------------------------------------------------------------------------------------+
| max              | Extract maximum value of all pixels within the polygon                                            |
+------------------+---------------------------------------------------------------------------------------------------+
| sum              | Extract sum of the values of all pixels within the polygon                                        |
+------------------+---------------------------------------------------------------------------------------------------+
| mode             | Extract the mode of classes within the polygon (classes must be set with the option class)        |
+------------------+---------------------------------------------------------------------------------------------------+
| proportion       | Extract proportion of class(es) within the polygon (classes must be set with the option class)    |
+------------------+---------------------------------------------------------------------------------------------------+
| count            | Extract count of class(es) within the polygon (classes must be set with the option class)         |
+------------------+---------------------------------------------------------------------------------------------------+
| percentile       | Extract percentile as defined by option perc (e.g, 95th percentile of values covered by polygon)  |
+------------------+---------------------------------------------------------------------------------------------------+

.. note::
   For the rules mode, proportion and count, set the extra key 'class' with the list of class values in the input raster image to use.

**Masking values from extract**

To mask some pixels from the extraction process, there are some keys that can be used:

+------------------+---------------------------------------------------------------------------------+
| key              | value                                                                           |
+==================+=================================================================================+
| srcnodata        | List of nodata values not to extract                                            |
+------------------+---------------------------------------------------------------------------------+
| bndnodata        | List of band in input image to check if pixel is valid (used for srcnodata)     |
+------------------+---------------------------------------------------------------------------------+
| mask             | Use the the specified file as a validity mask                                   |
+------------------+---------------------------------------------------------------------------------+
| mskband          | Use the the specified band of the mask file defined                             |
+------------------+---------------------------------------------------------------------------------+
| msknodata        | List of mask values not to extract                                              |
+------------------+---------------------------------------------------------------------------------+
| threshold        | Maximum number of features to extract (use positive values for percentage value |
|                  | and negative value for absolute threshold)                                      |
+------------------+---------------------------------------------------------------------------------+

Example:

Extract a random sample of 100 points, calculating the mean value based on a 3x3 window (buffer value of 1 pixel neighborhood) in a vector dataset in memory::

  v01=jim0.extractSample({'random':100,'buffer':1,'rule':['mean'],'output':'mem01','oformat':'Memory'})
  v01.close()

Extract a sample of 100 points using a regular grid sampling scheme. For each grid point, calculate the median value based on a 3x3 window (buffer value of 1 pixel neighborhood). Write the result in a SQLite vector dataset on disk::

  outputfn='/path/to/output.sqlite'
  npoint=100
  gridsize=int(jim.nrOfCol()*jim.getDeltaX()/math.sqrt(npoint))
  v=jim.extractSample({'grid':gridsize,'buffer':1,'rule':['median'],'output':output,'oformat':'SQLite'})
  v.write()
  v.close()

END

METHOD extractImg(dict)
Extract a pixel values from an input raster dataset based on a raster sample dataset.

Args:
* ``dict`` (Python Dictionary) with key value pairs. Each key (a 'quoted' string) is separated from its value by a colon (:). The items are separated by commas and the dictionary is enclosed in curly braces. An empty dictionary without any items is written with just two curly braces, like this: {}. A value can be a list that is also separated by commas and enclosed in square brackets [].

Returns:
   A :py:class:`VectorOgr` with fields for each of the calculated raster value (zonal) statistics.

Supported keys in the dict:

+------------------+---------------------------------------------------------------------------------+
| key              | value                                                                           |
+==================+=================================================================================+
| rule             | Rule how to calculate zonal statistics per feature                              |
+------------------+---------------------------------------------------------------------------------+
| class            | List of classes to extract from the raster sample dataset.                      |
|                  | Leave empty to extract all valid data pixels from thee sample                   |
+------------------+---------------------------------------------------------------------------------+
| cname            | Name of the class label in the output vector dataset (default is 'label')       |
+------------------+---------------------------------------------------------------------------------+
| fid              | Create extra field named 'fid' with this field identifier (sequence of features)|
+------------------+---------------------------------------------------------------------------------+
| band             | List of bands to extract (0 indexed). Default is to use extract all bands       |
+------------------+---------------------------------------------------------------------------------+
| bandname         | List of band name corresponding to list of bands to extract                     |
+------------------+---------------------------------------------------------------------------------+
| startband        | Start band sequence number (0 indexed)                                          |
+------------------+---------------------------------------------------------------------------------+
| endband          | End band sequence number (0 indexed)                                            |
+------------------+---------------------------------------------------------------------------------+
| down             | Down sampling factor to extract a subset of the sample based on a grid          |
+------------------+---------------------------------------------------------------------------------+
| output           | Name of the output vector dataset in which the zonal statistics are saved       |
+------------------+---------------------------------------------------------------------------------+
| ln               | Layer name of output vector dataset                                             |
+------------------+---------------------------------------------------------------------------------+
| oformat          | Output vector dataset format                                                    |
+------------------+---------------------------------------------------------------------------------+
| co               | Creation option for output vector dataset                                       |
+------------------+---------------------------------------------------------------------------------+

**Supported rules for extraction**

+------------------+---------------------------------------------------------------------------------------------------+
| rule             | description                                                                                       |
+==================+===================================================================================================+
| point            | extract a single pixel within the polygon or on each point feature                                |
+------------------+---------------------------------------------------------------------------------------------------+
| allpoints        | Extract all pixel values covered by the polygon                                                   |
+------------------+---------------------------------------------------------------------------------------------------+
| centroid         | Extract pixel value at the centroid of the polygon                                                |
+------------------+---------------------------------------------------------------------------------------------------+
| mean             | Extract average of all pixel values within the polygon                                            |
+------------------+---------------------------------------------------------------------------------------------------+
| stdev            | Extract standard deviation of all pixel values within the polygon                                 |
+------------------+---------------------------------------------------------------------------------------------------+
| median           | Extract median of all pixel values within the polygon                                             |
+------------------+---------------------------------------------------------------------------------------------------+
| min              | Extract minimum value of all pixels within the polygon                                            |
+------------------+---------------------------------------------------------------------------------------------------+
| max              | Extract maximum value of all pixels within the polygon                                            |
+------------------+---------------------------------------------------------------------------------------------------+
| sum              | Extract sum of the values of all pixels within the polygon                                        |
+------------------+---------------------------------------------------------------------------------------------------+
| mode             | Extract the mode of classes within the polygon (classes must be set with the option class)        |
+------------------+---------------------------------------------------------------------------------------------------+
| proportion       | Extract proportion of class(es) within the polygon (classes must be set with the option class)    |
+------------------+---------------------------------------------------------------------------------------------------+
| count            | Extract count of class(es) within the polygon (classes must be set with the option class)         |
+------------------+---------------------------------------------------------------------------------------------------+
| percentile       | Extract percentile as defined by option perc (e.g, 95th percentile of values covered by polygon)  |
+------------------+---------------------------------------------------------------------------------------------------+

.. note::
   For the rules mode, proportion and count, set the extra key 'class' with the list of class values in the input raster image to use.

**Masking values from extract**

To mask some pixels from the extraction process, there are some keys that can be used:

+------------------+---------------------------------------------------------------------------------+
| key              | value                                                                           |
+==================+=================================================================================+
| srcnodata        | List of nodata values not to extract                                            |
+------------------+---------------------------------------------------------------------------------+
| bndnodata        | List of band in input image to check if pixel is valid (used for srcnodata)     |
+------------------+---------------------------------------------------------------------------------+
| mask             | Use the the specified file as a validity mask                                   |
+------------------+---------------------------------------------------------------------------------+
| mskband          | Use the the specified band of the mask file defined                             |
+------------------+---------------------------------------------------------------------------------+
| msknodata        | List of mask values not to extract                                              |
+------------------+---------------------------------------------------------------------------------+
| threshold        | Maximum number of features to extract (use positive values for percentage value |
|                  | and negative value for absolute threshold)                                      |
+------------------+---------------------------------------------------------------------------------+

Example:

Extract a random sample of 100 points, calculating the mean value based on a 3x3 window (buffer value of 1 pixel neighborhood) in a vector dataset in memory::

  v01=jim0.extractSample({'random':100,'buffer':1,'rule':['mean'],'output':'mem01','oformat':'Memory'})
  v01.close()

Extract a sample of 100 points using a regular grid sampling scheme. For each grid point, calculate the median value based on a 3x3 window (buffer value of 1 pixel neighborhood). Write the result in a SQLite vector dataset on disk::

  outputfn='/path/to/output.sqlite'
  npoint=100
  gridsize=int(jim.nrOfCol()*jim.getDeltaX()/math.sqrt(npoint))
  v=jim.extractSample({'grid':gridsize,'buffer':1,'rule':['median'],'output':output,'oformat':'SQLite'})
  v.write()
  v.close()

END

#########
# JimList
#########

CLASS JimList
JimList class represents a list of Jim images.

Notes:
A JimList can be created from a python list of Jim images::

  ifn='/eos/jeodpp/data/SRS/Copernicus/S2/scenes/source/L1C/2017/08/05/065/S2A_MSIL1C_20170805T102031_N0205_R065_T32TNR_20170805T102535.SAFE/GRANULE/L1C_T32TNR_A011073_20170805T102535/IMG_DATA/T32TNR_20170805T102031_B08.jp2'
  jim0=createJim()
  jlist=jl.createJimList([jim0])
  #do stuff with jim ...
  jlist.close()

END

METHOD pushImage(Jim)
Push a Jim image to this JimList object

Args:
* A :py:class:`Jim` object.

Returns:
   The :py:class:`JimList` (self) with the extra image pushed to the end

Push a :py:class:`Jim` image object to an empty :py:class:`JimList`::

  ifn='/eos/jeodpp/data/SRS/Copernicus/S2/scenes/source/L1C/2017/08/05/065/S2A_MSIL1C_20170805T102031_N0205_R065_T32TNR_20170805T102535.SAFE/GRANULE/L1C_T32TNR_A011073_20170805T102535/IMG_DATA/T32TNR_20170805T102031_B08.jp2'
  jim0=createJim()
  jlist=jl.createJimList()
  jlist.pushImage(jim0)
  #do stuff with jim ...
  jlist.close()

END

METHOD popImage(Jim)
Pop a Jim image from this JimList

Returns:
   The :py:class:`JimList` (self) without the last image (that has been removed) 

Pop a :py:class`Jim` image object to an empty :py:class:`JimList`::

  ifn='/eos/jeodpp/data/SRS/Copernicus/S2/scenes/source/L1C/2017/08/05/065/S2A_MSIL1C_20170805T102031_N0205_R065_T32TNR_20170805T102535.SAFE/GRANULE/L1C_T32TNR_A011073_20170805T102535/IMG_DATA/T32TNR_20170805T102031_B08.jp2'
  jim0=createJim()
  jlist=jl.createJimList()
  jlist.pushImage(jim0)
  jlist.popImage()
  jlist.close()

END

METHOD getImage(integer)
Get an image at the specified index (0 based)

Args:
* ``Integer`` the index of the index to get (0 based).

Returns:
   The :py:class:`Jim` object at the specified index

Push an image to an empty list and get it back::

  ifn='/eos/jeodpp/data/SRS/Copernicus/S2/scenes/source/L1C/2017/08/05/065/S2A_MSIL1C_20170805T102031_N0205_R065_T32TNR_20170805T102535.SAFE/GRANULE/L1C_T32TNR_A011073_20170805T102535/IMG_DATA/T32TNR_20170805T102031_B08.jp2'
  jim0=createJim()
  jlist=jl.createJimList()
  jlist.pushImage(jim0)
  jim1=jlist.getImage(0)
  #jim1 is a reference to jim0

END

METHOD getSize()
Get number of images in list

Returns:
   The number of images in the list

Push an image to an empty list and get it back::

  ifn='/eos/jeodpp/data/SRS/Copernicus/S2/scenes/source/L1C/2017/08/05/065/S2A_MSIL1C_20170805T102031_N0205_R065_T32TNR_20170805T102535.SAFE/GRANULE/L1C_T32TNR_A011073_20170805T102535/IMG_DATA/T32TNR_20170805T102031_B08.jp2'
  jim0=createJim()
  jlist=jl.createJimList()
  jlist.pushImage(jim0)
  if jlist.getSize() != 1:
     print("Error: size of list should be 1")

END

METHOD pushNoDataValue(float)
Push a no data value to this :py:class:`JimList` object.

Args:
* ``Float`` the no data value

Returns:
   The :py:class:`JimList` (self)

END


METHOD clearNoData(float)
Clear all no data values from this :py:class:`JimList` object.

Returns:
   The :py:class:`JimList` (self)

END

METHOD covers(*args)
Check if a geolocation is covered by this :py:class:`JimList` object. Only the coordinates of the point (variant 1) or region of interest (variant 2) are checked, irrespective of no data values. Set the additional flag to True if the region of interest must be entirely covered.

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

METHOD selectGeo(*args)
Removes all images in this :py:class:`JimList` object if not covered by the coordinates of the point (variant 1) or region of interest (variant 2).

Args (variant 1):

* ``x`` (float): x coordinate in spatial reference system of the this :py:class:`JimList` object
* ``y`` (float): y coordinate in spatial reference system of the this :py:class:`JimList` object


Args (variant 2):

* ``ulx`` (float): upper left x coordinate in spatial reference system of the this :py:class:`JimList` object
* ``uly`` (float): upper left y coordinate in spatial reference system of the this :py:class:`JimList` object
* ``lrx`` (float): lower right x coordinate in spatial reference system of the this :py:class:`JimList` object
* ``lry`` (float): lower right x coordinate in spatial reference system of the this :py:class:`JimList` object


Returns:
   A subset of the :py:class:`JimList` object that covers the point or region of interest.

END

METHOD getBoundingBox()
Get the bounding box of this :py:class:`JimList` object in georeferenced coordinates.

Returns:
   A list with the bounding box of this :py:class:`JimList` object in georeferenced coordinates.

END

METHOD getUlx()
Get the upper left corner x (georeferenced) coordinate of this :py:class:`JimList` object

Returns:
   The upper left corner x (georeferenced) coordinate of this :py:class:`JimList` object

END

METHOD getUly()
Get the upper left corner y (georeferenced) coordinate of this :py:class:`JimList` object

Returns:
   The upper left corner y (georeferenced) coordinate of this :py:class:`JimList` object

END

METHOD getLrx()
Get the lower left corner x (georeferenced) coordinate of this :py:class:`JimList` object

Returns:
   The lower left corner x (georeferenced) coordinate of this :py:class:`JimList` object

END

METHOD getLry()
Get the lower left corner y (georeferenced) coordinate of this :py:class:`JimList` object

Returns:
   The lower left corner y (georeferenced) coordinate of this :py:class:`JimList` object

END

METHOD composite(dict)
Composite overlapping :py:class:`Jim` raster datasets according to a composite rule.
This method can be used to mosaic and composite multiple (georeferenced) :py:class:`Jim` raster datasets. A mosaic can merge images with different geographical extents into a single larger image. Compositing resolves the overlapping pixels according to some rule (e.g, the median of all overlapping pixels). Input datasets can have a different bounding boxes and spatial resolution.

Args:
* ``dict`` (Python Dictionary) with key value pairs. Each key (a 'quoted' string) is separated from its value by a colon (:). The items are separated by commas and the dictionary is enclosed in curly braces. An empty dictionary without any items is written with just two curly braces, like this: {}. A value can be a list that is also separated by commas and enclosed in square brackets [].

Supported keys in the dict:

+------------------+---------------------------------------------------------------------------------+
| key              | value                                                                           |
+==================+=================================================================================+
| crule            | Composite rule                                                                  |
+------------------+---------------------------------------------------------------------------------+
| band             | band index(es) to crop (leave empty if all bands must be retained)              | 
+------------------+---------------------------------------------------------------------------------+
| resample         | Resampling method (near or bilinear)                                            |
+------------------+---------------------------------------------------------------------------------+
| otype            | Data type for output image. Default is to inherit type from input image         |
+------------------+---------------------------------------------------------------------------------+
| a_srs            | Override the projection for the output file                                     |
+------------------+---------------------------------------------------------------------------------+
| file             | Create extra band in output representing number of observations (1) and/or      |
|                  | sequence number of selected raster dataset in the list (2) for each pixel       |
+------------------+---------------------------------------------------------------------------------+

Returns:
   The composite :py:class:`Jim` raster dataset object

**Managing no data values in input and output**

+------------------+---------------------------------------------------------------------------------------------+
| key              | value                                                                                       |
+==================+=============================================================================================+
| bndnodata        | Band(s) in input image to check if pixel is valid (used for srcnodata, min and max options) |
+------------------+---------------------------------------------------------------------------------------------+
| srcnodata        | invalid value(s) for input raster dataset                                                   |
+------------------+---------------------------------------------------------------------------------------------+
| bndnodata        | Band(s) in input image to check if pixel is valid (used for srcnodata, min and max options) |
+------------------+---------------------------------------------------------------------------------------------+
| min              | flag values smaller or equal to this value as invalid                                       |
+------------------+---------------------------------------------------------------------------------------------+
| max              | flag values larger or equal to this value as invalid                                        |
+------------------+---------------------------------------------------------------------------------------------+
| dstnodata        | nodata value to put in output raster dataset if not valid or out of bounds                  |
+------------------+---------------------------------------------------------------------------------------------+

**Subset spatial region in coordinates of the image geospatial reference system**

+------------------+---------------------------------------------------------------------------------+
| key              | value                                                                           |
+==================+=================================================================================+
| extent           | Get boundary from extent from polygons in vector file                           |
+------------------+---------------------------------------------------------------------------------+
| eo               | Special extent options controlling rasterization                                |
+------------------+---------------------------------------------------------------------------------+
| ln               | Layer name of extent to crop                                                    |
+------------------+---------------------------------------------------------------------------------+
| crop_to_cutline  | True will crop the extent of the target dataset to the extent of the cutline    |
|                  | The outside area will be set to no data (the value defined by the key 'nodata') |
+------------------+---------------------------------------------------------------------------------+
| ulx              | Upper left x value of bounding box to crop                                      |
+------------------+---------------------------------------------------------------------------------+
| uly              | Upper left y value of bounding box to crop                                      |
+------------------+---------------------------------------------------------------------------------+
| lrx              | Lower right x value of bounding box to crop                                     |
+------------------+---------------------------------------------------------------------------------+
| lry              | Lower right y value of bounding box to crop                                     |
+------------------+---------------------------------------------------------------------------------+
| dx               | Output resolution in x (default: keep original resolution)                      |
+------------------+---------------------------------------------------------------------------------+
| dy               | Output resolution in y (default: keep original resolution)                      |
+------------------+---------------------------------------------------------------------------------+
| align            | Align output bounding box to first input raster dataset in list                 |
+------------------+---------------------------------------------------------------------------------+

.. note::
   Possible values for the key 'eo' are: ATTRIBUTE|CHUNKYSIZE|ALL_TOUCHED|BURN_VALUE_FROM|MERGE_ALG. For instance you can use 'eo':'ATTRIBUTE=fieldname'

**Supported composite rules**

+-----------------+---------------------------------------------------------------------------------+
| composite rule  | composite output                                                                | 
+=================+=================================================================================+
| overwrite       | Overwrite overlapping pixels                                                    |
+-----------------+---------------------------------------------------------------------------------+
| maxndvi         | Create a maximum NDVI (normalized difference vegetation index) composite        |
+-----------------+---------------------------------------------------------------------------------+
| maxband         | Select the pixel with a maximum value in the band specified by option cband     |
+-----------------+---------------------------------------------------------------------------------+
| minband         | Select the pixel with a minimum value in the band specified by option cband     |
+-----------------+---------------------------------------------------------------------------------+
| mean            | Calculate the mean (average) of overlapping pixels                              |
+-----------------+---------------------------------------------------------------------------------+
| stdev           | Calculate the standard deviation of overlapping pixels                          |
+-----------------+---------------------------------------------------------------------------------+
| median          | Calculate the median of overlapping pixels                                      |
+-----------------+---------------------------------------------------------------------------------+
| mode            | Select the mode of overlapping pixels (maximum voting): use for Byte images only|
+-----------------+---------------------------------------------------------------------------------+
| sum             | Calculate the arithmetic sum of overlapping pixels                              |
+-----------------+---------------------------------------------------------------------------------+
| maxallbands     | For each individual band, assign the maximum value found in all overlapping     |
|                 | pixels. Unlike maxband, output band values cannot be attributed to a single     |
|                 | (date) pixel in the input time series                                           |
+-----------------+---------------------------------------------------------------------------------+
| minallbands     | For each individual band, assign the minimum value found in all overlapping     |
|                 | pixels. Unlike minband, output band values cannot be attributed to a single     |
|                 | (date) pixel in the input time series                                           |
+-----------------+---------------------------------------------------------------------------------+

.. note::
   Some rules require multiple input bands. For instance, the maxndvi rule calculates the NDVI per pixel based on two input bands. Use the extra key 'cband' to indicate the list of bands representing the red and near infrared band respectively.

END

METHOD stack(dict)
Stack all raster datasets in the list to a single multi-band raster dataset.

Args:
* ``dict`` (Python Dictionary) with key value pairs. Each key (a 'quoted' string) is separated from its value by a colon (:). The items are separated by commas and the dictionary is enclosed in curly braces. An empty dictionary without any items is written with just two curly braces, like this: {}. A value can be a list that is also separated by commas and enclosed in square brackets [].

Supported keys in the dict:

+------------------+---------------------------------------------------------------------------------+
| key              | value                                                                           |
+==================+=================================================================================+
| band             | band index(es) to crop (leave empty if all bands must be retained)              | 
+------------------+---------------------------------------------------------------------------------+
| otype            | Data type for output image. Default is to inherit type from input image         |
+------------------+---------------------------------------------------------------------------------+
| a_srs            | Override the projection for the output file                                     |
+------------------+---------------------------------------------------------------------------------+

Returns:
   Multi-band :py:class:`Jim` raster dataset object.

END

METHOD getStats(dict)
Calculate statistics of a raster dataset.

Args:
* ``dict`` (Python Dictionary) with key value pairs. Each key (a 'quoted' string) is separated from its value by a colon (:). The items are separated by commas and the dictionary is enclosed in curly braces. An empty dictionary without any items is written with just two curly braces, like this: {}. A value can be a list that is also separated by commas and enclosed in square brackets [].

Returns:
   A dictionary with the results of the statistics, using the same keys as for the functions.

Supported keys in the dict:

+------------------+---------------------------------------------------------------------------------+
| key              | value                                                                           |
+==================+=================================================================================+
| function         | Statistical function (see values for different functions in tables below)       |
+------------------+---------------------------------------------------------------------------------+
| cband            | List of bands on which to calculate the statistics                              |
+------------------+---------------------------------------------------------------------------------+
| down             | Down sampling factor (in pixels x and y) to calculate the statistics on a subset|
+------------------+---------------------------------------------------------------------------------+
| src_min          | Do not take smaller values into account when calculating statistics             |
+------------------+---------------------------------------------------------------------------------+
| src_max          | Do not take higher values into account when calculating statistics              |
+------------------+---------------------------------------------------------------------------------+
| nodata           | Do not take these values into account when calculating statistics               |
+------------------+---------------------------------------------------------------------------------+
| otype            | Data type for output image                                                      |
+------------------+---------------------------------------------------------------------------------+

.. note::
   For statistical functions requiring two sets of inputs, use a list of two values for cband (e.g., regression and histogram2d)

**Supported statistical functions**

+--------------+------------------------------------------------------+
| function     | description                                          |
+==============+======================================================+
| invalid      | report number of invalid (nodata) values             |
+--------------+------------------------------------------------------+
| nvalid       | report number of valid (not nodata) values           |
+--------------+------------------------------------------------------+
| basic        | Shows basic statistics                               |
|              | (min,max, mean and stdDev of the raster datasets)    |
+--------------+------------------------------------------------------+
| gdal         | Use the GDAL calculation of basic statistics         |
+--------------+------------------------------------------------------+
| mean         | calculate the mean value                             |
+--------------+------------------------------------------------------+
| median       | calculate the median value                           |
+--------------+------------------------------------------------------+
| var          | calculate variance value                             |
+--------------+------------------------------------------------------+
| stdev        | calculate standard deviation                         |
+--------------+------------------------------------------------------+
| skewness     | calculate the skewness                               |
+--------------+------------------------------------------------------+
| kurtosis     | calculate the kurtosis                               |
+--------------+------------------------------------------------------+
| sum          | calculate sum of all values                          |
+--------------+------------------------------------------------------+
| minmax       | calculate minimum and maximum value                  |
+--------------+------------------------------------------------------+
| min          | calculate minimum value                              |
+--------------+------------------------------------------------------+
| max          | calculate maximum value                              |
+--------------+------------------------------------------------------+
| histogram    | calculate the histogram                              |
+--------------+------------------------------------------------------+
| histogram2d  | calculate the two-dimensional histogram for two bands|
+--------------+------------------------------------------------------+
| rmse         | calculate root mean square error for two bands       |
+--------------+------------------------------------------------------+
| regresssion  | calculate the regression between two bands           |
+--------------+------------------------------------------------------+

For the histogram function, the following key values can be set:

+--------------+------------------------------------------------------+
| key          | description                                          |
+==============+======================================================+
| nbin         | Number of bins for the histogram                     |
+--------------+------------------------------------------------------+
| relative     | Set to True to report percentage values              |
+--------------+------------------------------------------------------+
| kde          | Set to True to use Kernel density estimation when    |
|              | producing histogram. The standard deviation is       |
|              | estimated based on Silverman's rule of thumb         |
+--------------+------------------------------------------------------+

Example:

Get the histogram of the input raster dataset using 10 bins::

  jlist.getStats({'function':['histogram','nbin':10})

END

METHOD statProfile(dict)
Obtain a statistical profile per pixel based on the data available in a :py:class:`JimList` object. Multiple functions can be set, resulting in a multi-band raster dataset (one output band for each function).

Args:
* ``dict`` (Python Dictionary) with key value pairs. Each key (a 'quoted' string) is separated from its value by a colon (:). The items are separated by commas and the dictionary is enclosed in curly braces. An empty dictionary without any items is written with just two curly braces, like this: {}. A value can be a list that is also separated by commas and enclosed in square brackets [].

Returns:
   The statistical profile of the input raster dataset

Supported keys in the dict:


+------------------+---------------------------------------------------------------------------------+
| key              | value                                                                           |
+==================+=================================================================================+
| function         | Statistical function (see values for different functions in tables below)       |
+------------------+---------------------------------------------------------------------------------+
| nodata           | Do not take these values into account when calculating statistics               |
+------------------+---------------------------------------------------------------------------------+
| otype            | Data type for output image                                                      |
+------------------+---------------------------------------------------------------------------------+


**Statistical profile functions**

+--------------+------------------------------------------------------+
| function     | description                                          |
+==============+======================================================+
| nvalid       | report number of valid (not nodata) values in window |
+--------------+------------------------------------------------------+
| median       | perform a median filter                              |
+--------------+------------------------------------------------------+
| var          | calculate variance in window                         |
+--------------+------------------------------------------------------+
| min          | calculate minimum in window                          |
+--------------+------------------------------------------------------+
| max          | calculate maximum in window                          |
+--------------+------------------------------------------------------+
| sum          | calculate sum in window                              |
+--------------+------------------------------------------------------+
| mode         | calculate the mode (only for categorical values)     |
+--------------+------------------------------------------------------+
| mean         | calculate mean in window                             |
+--------------+------------------------------------------------------+
| stdev        | calculate standard deviation in window               |
+--------------+------------------------------------------------------+
| percentile   | calculate percentile value in window                 |
+--------------+------------------------------------------------------+
| proportion   | calculate proportion in window                       |
+--------------+------------------------------------------------------+

.. note::
   The 'percentile' function calculates the percentile value based on the pixel values in the multi-band input raster dataset. A number of percentiles can be calculated, e.g., 10th and 50th percentile, resulting in a multi-band output raster dataset (one band for each calculated percentile). The percentiles to be calculated can be set with the key 'perc' and a list of values.

Example:

Calculated the 10th and 50th percentiles for the multi-band input raster dataset jim::

  jim_percentiles=jlist.statProfile({'function':args.function,'perc':[10,50]})

END

###########
# VectorOgr
###########

CLASS VectorOgr
VectorOgr class is the basis vector dataset class of the Joint image processing library.


END

METHOD getLayerCount()
Get number of layers in this vector dataset

Returns:
   The number of layers in this vector dataset
END

METHOD getFeatureCount()
Get number of features in this vector dataset

Returns:
   The number of features in this vector dataset
END

METHOD getBoundingBox()
Get the bounding box of this dataset in georeferenced coordinates.

Returns:
   A list with the bounding box of this dataset in georeferenced coordinates.

END

METHOD getUlx()
Get the upper left corner x (georeferenced) coordinate of this dataset

Returns:
   The upper left corner x (georeferenced) coordinate of this dataset

END

METHOD getUly()
Get the upper left corner y (georeferenced) coordinate of this dataset

Returns:
   The upper left corner y (georeferenced) coordinate of this dataset

END

METHOD getLrx()
Get the lower left corner x (georeferenced) coordinate of this dataset

Returns:
   The lower left corner x (georeferenced) coordinate of this dataset

END

METHOD getLry()
Get the lower left corner y (georeferenced) coordinate of this dataset

Returns:
   The lower left corner y (georeferenced) coordinate of this dataset

END

METHOD open(dict)
Open a vector dataset

Args:

* ``dict`` (Python Dictionary) with key value pairs. Each key (a 'quoted' string) is separated from its value by a colon (:). The items are separated by commas and the dictionary is enclosed in curly braces. An empty dictionary without any items is written with just two curly braces, like this: {}. A value can be a list that is also separated by commas and enclosed in square brackets [].

Supported keys in the dict:

======== ===================================================
filename Filename of the vector dataset
ln       Layer name
======== ===================================================

Returns:
   This instance of VectorOgr object (self)

**keys specific for reading vector datasets**

=============== ===================================================
attributeFilter Set an attribute filter 
noread          Set this flag to True to not read data when opening
=============== ===================================================

**keys specific for writing vector datasets**

======== ===================================================================
a_srs    Assign this projection (e.g., epsg:3035)
gtype    Geometry type (default is wkbUnknown)
co       Format dependent options controlling creation of the output file
oformat  Output sample dataset format supported by OGR (default is "SQLite")
======== ===================================================================

Example:

Create a vector and open a dataset::

  v0=jl.createVector()
  v0.open({'filename':'/path/to/vector.sqlite'})

END

METHOD close()
Close a vector dataset, releasing resources such as memory and OGR dataset handle.

END

METHOD write()
Write the vector dataset to file

Returns:
   This instance of Jim object (self)

.. note::
   Unlike writing a raster dataset :py:class:`Jim` where the output filename and type can be defined at the time of writing, these parameters have already been set when opening the :py:class:`VectorOgr`.

END

METHOD train(dict)
Train a supervised classifier based on extracted data including label information (typically obtained via :py:func:`Jim:extractOgr`).

Args:

* ``dict`` (Python Dictionary) with key value pairs. Each key (a 'quoted' string) is separated from its value by a colon (:). The items are separated by commas and the dictionary is enclosed in curly braces. An empty dictionary without any items is written with just two curly braces, like this: {}. A value can be a list that is also separated by commas and enclosed in square brackets [].

Supported keys in the dict:

======== =====================================================================================================================
method   Classification method: 'svm' (support vector machine), 'ann' (artificial neural network)
model    Model filename to save trained classifier
label    Attribute name for class label in training vector file (default: 'label')
bandname List of band names to use that correspond to the fields in the vector dataset. Leave empty to use all bands
class    List of alpha numeric class names as defined in the label attribute (use only if labels contain not numerical values)
reclass  List of numeric class values corresponding to the list defined by the class key
======== =====================================================================================================================

Returns:

   This instance of VectorOgr object (self)

**Balancing the training sample**

Keys used to balance the training sample:

======== ================================================================================================
balance  Balance the input data to this number of samples for each class
random   Randomize training data for balancing
min      Set to a value to not take classes into account with a sample size that is lower than this value
======== ================================================================================================

**Support vector machine**

The support vector machine (SVM) supervised classifier is described `here <http://dx.doi.org/10.1007/BF00994018>`_. The implementation in JIPlib is based on the open source `libsvm <https://www.csie.ntu.edu.tw/~cjlin/libsvm/>`_.

Keys specific to the SVM:

========== ======================================================================
svmtype    Type of SVM (C_SVC, nu_SVC,one_class, epsilon_SVR, nu_SVR)","C_SVC")
kerneltype Type of kernel function (linear,polynomial,radial,sigmoid) ","radial")
kd         Degree in kernel function",3)
gamma      Gamma in kernel function",1.0)
coef0      Coef0 in kernel function",0)
ccost      The parameter C of C_SVC, epsilon_SVR, and nu_SVR",1000)
nu         The parameter nu of nu_SVC, one_class SVM, and nu_SVR",0.5)
eloss      The epsilon in loss function of epsilon_SVR",0.1)
cache      Cache memory size in MB",100)
etol       The tolerance of termination criterion",0.001)
shrink     Whether to use the shrinking heuristics",false)
probest    Whether to train a SVC or SVR model for probability estimates",true,2)
========== ======================================================================

**Artificial neural network**

The artificial neural network (ANN) supervised classifier is based on the back propagation model as introduced by D. E. Rumelhart, G. E. Hinton, and R. J. Williams (Nature, vol. 323, pp. 533-536, 1986). The implementation is based on the open source C++ library fann (http://leenissen.dk/fann/wp/).


Keys specific to the ANN:

========== ==========================================================================
nneuron    List defining the number of neurons in each hidden layer in the neural network 
connection Connection rate (default: 1.0 for a fully connected network
learning   Learning rate (default: 0.7)
weights    Weights for neural network. Apply to fully connected network only, starting from first input neuron to last output neuron, including the bias neurons (last neuron in each but last layer)
maxit      Maximum epochs used for training the neural network (default: 500)
========== ==========================================================================

.. note::
   To define two hidden layers with 3 and 5 neurons respectively, define a list of two values for the key 'nneuron': [3, 5].

END
