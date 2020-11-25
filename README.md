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


## Relevant parameters

Here are the relevant parameters of the Gutzwiller ansatz and of the Bose-Hubbard Hamiltonian:

    Gutzwiller ansatz for a Bose-Hubbard model.

    Parameters:
        D               Lattice dimensionality (allowed values: 1, 2)
        L               Linear size of lattice
        OBC             Boundary conditions (OBC=1 for OBC, OBC=0 for PBC)
        J               Nearest-neighbor hopping parameter (see Hamiltonian)
        U               On-site interaction parameter (see Hamiltonian)
        mu              Chemical potential (see Hamiltonian)
        Vnn             Nearest-neighbor interaction parameter (see Hamiltonian)
        VT              Trap prefactor (see Hamiltonian)
        alphaT          Trap exponent (see Hamiltonian)
        trap_center     Trap center (see Hamiltonian)
        nmax            Cutoff on the local occupation number (state indices go from 0 to nmax)

    The Hamiltonian includes several terms.
    Here we denote the sum over the neighbors of site i as sum_{j~i}.
        - J * sum_{i} sum_{j~i} b_i^dagger b_j
        + U * sum_{i} n_i * (n_i - 1)
        + sum_{i} n_i * ( -mu + VT * |x_i - trap_center|^alpha )
        + (Vnn/2) sum_{i} sum_{j~i} n_i * n_j

    NOTE:
    1) In the nearest-neighbor-interaction term, I use (Vnn/2) because each pair of neighbors is counted twice.
    2) In the hopping term, I use J (and not J/2) because pairs (i,j) and (j,i) are not equivalent (due to the Hermitian conjugate).
