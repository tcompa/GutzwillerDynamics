# Time_dependent_Gutzwiller
Complex-coefficient Gutzwiller state for the Bose-Hubbard model on an extended lattice, suitable for both imaginary-time and real-time evolution.

## Important
This code is not yet complete, use it at your own risk!

## What is this?
This is a simple python/cython code implementing the inhomogeneous Gutzwiller
variational wave function for the [Bose-Hubbard
model](https://en.wikipedia.org/wiki/Bose%E2%80%93Hubbard_model).
The Gutzwiller coefficients are complex-valued, so that this code can be used both for imaginary-time and real-time evolution.

## How to use it?
This code requires the [numpy](http://www.numpy.org/), [scipy](https://www.scipy.org/scipylib/download.html) and
[cython](http://cython.org/) libraries (plus the
[future](https://pypi.python.org/pypi/future) library, if you are on python 2).
Elementary tests are available in the `Code/Tests` folder, and they are performed at each commit - see
the current status on https://travis-ci.org/tcompa/...).

Before being imported in a python script, the module
`lib_inhomogeneous_gutzwiller.pyx` has to be compiled through the command

    $ python setup_cython.py build_ext --inplace
to be issued in the `Code` folder.

After this step, it can be imported in ordinary python scripts.
Have a look at the example files in the `Examples` folder.
