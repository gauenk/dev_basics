#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
from setuptools import setup, find_packages
# from distutils.core import setup
import os
import stat
import shutil
import platform
import sys
import site
import glob


# -- file paths --
long_description="""A Python implementation of some shared basic functions"""
setup(
    name='dev_basics',
    version='100.100.100',
    description='Some shared basics',
    long_description=long_description,
    url='https://github.com/gauenk/dev_basics',
    author='Kent Gauen',
    author_email='gauenk@purdue.edu',
    license='MIT',
    keywords='neural network',
    install_requires=[],
    package_dir={"": "lib"},
    packages=find_packages("lib"),
    entry_points = {
        'console_scripts': ['timed_launch=dev_basics.timed_launch:main',
                            'named_launch=dev_basics.named_launch:main',
                        ],
    }
)
