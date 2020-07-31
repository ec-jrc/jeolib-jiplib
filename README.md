# jiplib

jiplib is a library for image processing for geospatial data implemented in JRC Ispra. 

# License

jiplib is released under an
[EUPL](https://joinup.ec.europa.eu/collection/eupl) license (see
[LICENSE.txt](LICENSE.txt))

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


## install dependency libraries in Debian

```
g++ make cmake swig libgsl-dev libfann-dev libgdal-dev libjsoncpp-dev libboost-serialization-dev libboost-filesystem-dev libpython3-dev python3-numpy  
apt install build-essential cmake
apt install libgsl-dev
apt install libfann-dev
apt install libgdal-dev
apt install libjsoncpp-dev
apt install libpython3-dev
apt install python3-numpy
apt install libboost-filesystem-dev libboost-serialization-dev
apt install swig
```

 ## mialib

See more infor at [mia](https://jeodpp.jrc.ec.europa.eu/apps/gitlab/jeodpp/JIPlib/mia)

# Install
## Python3

From the directory of the repository, run:
```
mkdir build
cd build
cmake ..
make -j
sudo make install
```

## Python2

From the directory of the repository, run:
```
mkdir build
cd build
cmake -DPYTHON3=OFF ..
make -j
sudo make install
sudo ldconfig
```

## Create wheel

After `make`, go to directory `utils` and run the bash script in the following
syntax:

```bash
bash create_wheel.sh path_to_your_libmialib_generic.so
```

The wheel will be created in the `build` directory and will contain also
`mialib` necessary tools inside (no need for the user to install `mialib`
when installing the `jiplib` wheel).

> NOTE: The common path to `libmialib_generic.so` is `/usr/local/lib`

# Test the installation

From the directory of the repository, run:
```
ctest
```

# Build documentation

Go to directory `doc` and run `make html`.
```
cd doc
make html
```
