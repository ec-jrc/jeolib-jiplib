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
