PWD=$(pwd -P)

mialib_generic_path=$1

# copy the setup.py and change dir to build
cp setup.py ../build/
cd ../build/

# set up the necessary environment
mkdir jiplib
cp __init__.py jiplib/
cp jiplib.py jiplib/
cp *.so* jiplib/
cp $mialib_generic_path jiplib/

# create wheel and move it to build
pip3 wheel .
#mv jiplib-* ../build/

# remove the intermediate directory
rm setup.py
rm -r jiplib

# get to the original directory
cd $PWD
