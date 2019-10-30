# jiplib

jiplib is a library for image processing for geospatial data implemented in JRC Ispra. 

# License

jiplib is released under an
[EUPL](https://joinup.ec.europa.eu/collection/eupl) license (see
[LICENSE.txt](LICENSE.txt))

# Dependencies

 * mialib

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
