# GutzwillerDynamics

[![Build Status](https://travis-ci.org/tcompa/GutzwillerDynamics.svg?branch=master)](https://travis-ci.org/tcompa/GutzwillerDynamics)

[![DOI](https://zenodo.org/badge/164842990.svg)](https://zenodo.org/badge/latestdoi/164842990)



## Important
This code is not yet complete, use it at your own risk!

## What is this?
This program implements the Gutzwiller variational wave function for the [Bose-Hubbard model](https://en.wikipedia.org/wiki/Bose%E2%80%93Hubbard_model). The Gutzwiller coefficients are site-dependent (so that, for instance, one can add an additional confining potential) and complex-valued (so that this code can be used both for imaginary- and real-time evolution).

## How to use it?
This is a python code, which requires the [numpy](http://www.numpy.org/), [scipy](https://www.scipy.org/scipylib/download.html) and
[cython](http://cython.org/) libraries. You may also need the [future](https://pypi.python.org/pypi/future) library (if you are on Python 2) and [matplotlib](https://matplotlib.org/) or [gnuplot](http://www.gnuplot.info/) (for some of the examples).

Before being imported in a python script, the module
`lib_inhomogeneous_gutzwiller.pyx` has to be compiled through the command

    $ python setup_cython.py build_ext --inplace
to be issued in the `Code` folder.
After this step, `lib_inhomogeneous_gutzwiller` can be imported in ordinary python scripts (have a look at the `Examples` folder).

Elementary tests are available in the `Code/Tests` folder, and they are performed at each commit - see the current status on https://travis-ci.org/tcompa/GutzwillerDynamics.
