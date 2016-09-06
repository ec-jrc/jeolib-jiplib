# Makefile adapted by Pierre.Soille@jrc.ec.europa.eu
# for testing jipl C library binding to python

# Adpated from mia on 20160906 by Pierre.Soille@jrc.ec.europa.eu

# ------------------------------------------------------------
# SWIG Examples Makefile
# http://crosswire.org/svn/sword/trunk/bindings/swig/oldmake/Makefile.swig
# [accessed on 20160127]
#
# This file is used by the examples to build modules.  Assuming
# you ran configure, this file will probably work.  However,
# it's not perfect so you might need to do some hand tweaking.
#
# Other notes:
#
# 1.   Take a look at the prefixes below.   Since SWIG works with
#      multiple target languages, you may need to find out where
#      certain packages have been installed.   Set the prefixes
#      accordingly.
#
# 2.   To use this makefile, simply set SRCS, INTERFACE, INCLUDE, LIBS,
#      TARGET, and do a
#           $(MAKE) -f Makefile.template.in SRCS='$(SRCS)' \
#           INCLUDE='$(INCLUDE) LIBS='$(LIBS)' INTERFACE='$(INTERFACE)' \
#           TARGET='$(TARGET)' method
#
#      'method' describes what is being built.
#---------------------------------------------------------------

ifndef PYTHON
	PYTHON=python
endif

BINDING = python

#GDAL include ../SWIGmake.base
WRAPPERS = jiplib_wrap.cc

#PACKAGE_DIR=osmial

SWIGOUTPUTDIR = extensions/
JIPL_ROOT = ../../core/c/
BUILDBASEDIR = ../build
BUILDDIR = ../build/python/

SWIGOUTPUTDIR = extensions/
JIPL_ROOT = ../../core/c/
BUILDDIR = ../build/python/

PY_MODULES      =       ${PACKAGE_DIR}/jiplib.py


TARGET     = jiplib
CC         = g++
CXX        = c++
prefix     = /usr/local
exec_prefix= ${prefix}
# skipped -DNNI


SWIGARGS = -c++ -Wall -I../include -I../include/$(BINDING) -I/usr/local/include/pktools/ -I../../
CFLAGS     = -fPIC -std=c++11 -DPYTHON -DOPENMP -DLIIAR -DLIBPROJ -DTESTING -DTEST2 -DEERIE -DUNIX -DMARCIN -DDOMINIK -std=c99 -m64 -O3 -Wall -fopenmp -c -Wno-unknown-pragmas  
INCLUDE    = -I./ -I.. -I../../ -I../../../build -I/usr/local/include/mia -I/usr/local/include/pktools $(python-config --cflags)
LIBS       = -ldl -lmia_generic -ljiplib -limageClasses -lalgorithms
LIBDIR	   = 
INTERFACE  = jiplib.i
SWIGOPT    =
SWIG       = swig
RUNTIMEDIR = $(exec_prefix)/lib

LIBM       = -lieee -lm
LIBC       = 
LIBCRYPT   = -lcrypt
SYSLIBS    = $(LIBM) $(LIBC) $(LIBCRYPT)

libtool_comp = $(TOP)/../Tools/libtool --mode compile
libtool_link = $(TOP)/../Tools/libtool --mode link

# X11 options

XLIB       = -L/usr/X11R6/lib -lX11
XINCLUDE   = -I/usr/X11R6/include

IWRAP      = $(INTERFACE:.i=_wrap.i)
ISRCS      = $(IWRAP:%.i=$(SWIGOUTPUTDIR)/%.cc)
ICXXSRCS   = $(IWRAP:.i=.cxx)
IOBJS      = $(IWRAP:%.i=$(BUILDDIR)/temp/$(SWIGOUTPUTDIR)/%.o)


# Symbols used for using shared libraries
SO=		.so
LDSHARED=	gcc -shared
CCSHARED=	-fpic
CXXSHARED=      gcc -shared




##################################################################
#####                       PYTHON                          ######
##################################################################

# Make sure these locate your Python installation
PYTHON_INCLUDE= -DHAVE_CONFIG_H -I/usr/include/python2.7
PYTHON_LIB    = 

# PYTHON_PACKAGES_DIR = /home/soillpi/.local/lib/python2.7/site-packages/
PYTHON_PACKAGES_DIR = /usr/local/lib/python2.7/site-packages/

# Extra Python specific dynamic linking options
PYTHON_DLNK   = 

# ----------------------------------------------------------------
# Build a C dynamically loadable module
# ----------------------------------------------------------------


#$(LIBDIR)/libjip_python.so: 
#	cd ../ljip/src && make $@



jiplib_wrap.cc: ../include/jiplib.i ../include/$(BINDING)/jiplib_$(BINDING).i # ../include/$(BINDING)/typemaps_$(BINDING).i
	@mkdir -p $(SWIGOUTPUTDIR)$(@D)
	@mkdir -p $(BUILDDIR)$(@D)
	$(SWIG) $(SWIGARGS) $(SWIGDEFINES) -I$(JIPL_ROOT) -$(BINDING) -o $(SWIGOUTPUTDIR)$@ ../include/jiplib.i


jiplib_wrap.o: jiplib_wrap.cc
	@mkdir -p $(BUILDDIR)/temp/$(SWIGOUTPUTDIR)/$(@D)
	$(CC) -c $(CCSHARED) $(CFLAGS) $(ISRCS) $(INCLUDE) $(PYTHON_INCLUDE) -o  $(BUILDDIR)/temp/$(SWIGOUTPUTDIR)/$@ 

jiplib: jiplib_wrap.o
	@mkdir -p $(BUILDDIR)/$(@D)
	$(LDSHARED) -Wall -fopenmp  -m64 $(IOBJS) $(PYTHON_DLNK) $(LIBS) -o $(BUILDDIR)/_$(TARGET)$(SO)

doc:
	@mkdir -p $(BUILDDIR)/doc/$(@D)
	#epydoc -v --name jiplib --html --graph all $(BUILDDIR)/jipl.py -o $(BUILDDIR)/doc/jiplib_docs_html
	#epydoc -v --name jiplib --pdf  --graph all $(BUILDDIR)/jipl.py -o $(BUILDDIR)/doc/jiplib_docs_tex


all: jiplib doc


install:
	test -f $(SWIGOUTPUTDIR)/$(TARGET).py && mv $(SWIGOUTPUTDIR)/$(TARGET).py $(BUILDDIR)
	cp -a $(BUILDDIR)/_$(TARGET).py $(PYTHON_PACKAGES_DIR)
	cp -a $(BUILDDIR)/_$(TARGET)$(SO) $(PYTHON_PACKAGES_DIR)


clean:
	-rm -f ${PACKAGE_DIR}/*.pyc
	-rm -rf $(BUILDDIR)
	-rm -rf $(BUILDBASEDIR)
	-rm -f *.pyc
	-rm -rf *.egg-info
	-rm -f *.so ./osgeo/*.so
	-rm -rf dist

veryclean: clean
	-rm -f ${WRAPPERS} ${PY_MODULES}
	-rm -f ${SWIGOUTPUTDIR}/jiplib_wrap.c


