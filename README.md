# jiplib

jiplib is a C++ library with a Python wrapper for image processing for geospatial data implemented in JRC Ispra. Python users are encouraged to use [pyjeo](https://github.com/ec-jrc/jeolib-pyjeo) that is built upon this library.

# License

jiplib is released under the [GPLv3](https://www.gnu.org/licenses) license.

# Dependencies
 ## libraries: 

* gdal: MIT/X style https://gdal.org/license.html
* PROJ: MIT https://proj.org/about.html
* GNU compiler selection (gcc/g++): GPL v.3 https://gcc.gnu.org/onlinedocs/libstdc++/manual/license.html
* cmake: BSD 3 https://cmake.org/licensing/
* GNU Scientific library: GPL https://www.gnu.org/software/gsl/
* FANN: LGPL http://leenissen.dk/fann/wp/
* libsvm: modified BSD license https://www.csie.ntu.edu.tw/~cjlin/libsvm/COPYRIGHT
* libLAS: BSD https://liblas.org/
* jsoncpp: MIT https://github.com/open-source-parsers/jsoncpp/blob/master/LICENSE
* doxygen: GPL v.2 https://github.com/doxygen/doxygen/blob/master/LICENSE
* boost: BSD/MIT like https://www.boost.org/users/license.html
* SWIG: GPL v.3 http://www.swig.org/Release/LICENSE
* Python: Python Software Foundation License https://docs.python.org/3/license.html
* numpy: BSD https://numpy.org/license.html
* scipy: BSD https://www.scipy.org/scipylib/license.html
* Sphinx: BSD http://www.sphinx-doc.org/en/master/


## Install dependency libraries (example for Debian based system using apt)

```
sudo apt install -yq \
  build-essential \
  cmake \
  libgsl-dev \
  libfann-dev \
  libgdal-dev \
  libjsoncpp-dev \
  libpython3-dev \
  python3-numpy \
  libboost-filesystem-dev  \
  libboost-serialization-dev \
  swig
```

 ## miallib

See more information at [miallib](https://github.com/ec-jrc/jeolib-miallib)

# Install

```
mkdir build
cd build
cmake ..
make
sudo make install
```

# Install without sudo rights

Download the source code from [miallib](https://github.com/ec-jrc/jeolib-miallib) to a local directory:
```
/local/miallib/dir/jeolib-miallib
```

and install [miallib](https://github.com/ec-jrc/jeolib-miallib) locally:

```
make install prefix=/local/miallib/install/dir
```

Then build and install jiplib locally (replacing the same directories for `MIAL_INCLUDE_DIR` for `MIAL_LIBRARY` appropriately as used above):

```
mkdir build
cd build
cmake -DMIAL_INCLUDE_DIR=/local/miallib/dir/jeolib-miallib/core/c/ -DMIAL_LIBRARY=/local/miallib/install/dir/libmiallib_generic.so -DCMAKE_INSTALL_PREFIX=/local/jiplib/install/dir -DPYTHON_INSTALL_DIR=/local/jiplib/python/dist/dir ..
make -j
make install
```

For `CMAKE_INSTALL_PREFIX` and `PYTHON_INSTALL_DIR` use any directory where you have write access.

Then export the `LD_LIBRARY_PATH` environment variable so that the libraries can be found:

```
export LD_LIBRARY_PATH=/local/miallib/install/dir/:/local/jiplib/install/dir
```

Finally adapt the `PYTHONPATH`:

```
export PYTHONPATH=/local/jiplib/python/dist/dir:$PYTHONPATH
```

# Test the installation

From the build directory, run:
```
ctest
```

# Build documentation (deprecated, users are encouraged to use pyjeo documentation)

Go to directory `doc` and run `make html`.
```
cd doc
make html
```
