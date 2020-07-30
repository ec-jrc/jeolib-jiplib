PWD=$(pwd -P)

# copy the setup.py and change dir to build
cp setup.py ../build/
cd ../build/

# set up the necessary environment
mkdir jiplib
cp *py jiplib/
cp *.so* jiplib/

# create wheel and move it to build
pip3 wheel .
#mv jiplib-* ../build/

# remove the intermediate directory
rm setup.py
rm -r jiplib

# get to the original directory
cd $PWD
