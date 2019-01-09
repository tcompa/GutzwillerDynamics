'''
Program: setup_cython.py
Created: Tue Aug 22 22:06:12 CEST 2017
Author: Tommaso Comparin
Description: Compiles a cython module
Notes: To be executed through
               $ python setup_cython.py build_ext --inplace
'''

import glob
import os
from distutils.extension import Extension
from distutils.core import setup
from Cython.Distutils import build_ext
from Cython.Compiler import Options


#libraries = ['gsl', 'gslcblas']
libraries = []

#lib = 'lib_mooreread_metropolis.pyx'
for lib in glob.glob('*.pyx'):
    basename = lib[:-4]
    ext_modules = [Extension(basename, [basename + '.pyx'], libraries=libraries)]
    setup(cmdclass={'build_ext': build_ext}, ext_modules=ext_modules)
    os.system('chmod -x %s.so' % basename)
