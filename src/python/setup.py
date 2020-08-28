"""
Install the jiplib package.
Author(s): Pieter.Kempeneers@ec.europa.eu,
           Ondrej Pesek,
           Pierre.Soille@ec.europa.eu
Copyright (C) 2018-2020 European Union (Joint Research Centre)

This file is part of jiplib.

jiplib is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

jiplib is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
"""

try:
    from setuptools import setup, Distribution
except ImportError:
    from distutils.core import setup

from setuptools import find_packages


setup(
    name='jiplib',
    version='0.6.0',
    author_email='pieter.kempeneers@.ec.europa.eu',
    url='https://cidportal.jrc.ec.europa.eu/apps/gitlab/JIPlib/jiplib',
    license='GPLv3',
    packages=find_packages(),
    package_data={'jiplib': ['_jiplib.so', 'libjiplib.so.1', 'libjiplib.so',
                             'libjiplib.so.1.0.0', 'libmiallib_generic.so']},
    include_package_data=True
)
