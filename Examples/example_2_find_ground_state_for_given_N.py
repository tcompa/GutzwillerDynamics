#!/usr/bin/env python

from __future__ import print_function

# Comment these lines if you want to use more than one core
import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import numpy

# Replace the following line with the actual code path
codedir = '../Code/'
sys.path.append(codedir)
from lib_inhomogeneous_gutzwiller import Gutzwiller


# Physical parameters
D = 1           # Dimensions
L = 10          # Linear lattice size
U = 1.0         # On-site repulsion
J = 0.08        # Hopping amplitude
nmax = 4        # Cutoff on the local occupation
OBC = 0         # Whether to impose open (OBC=1) or periodic (OBC=0) boundary conditions
VT = 0.0        # Trap prefactor, as in V(r) = VT * r^alpha
alphaT = 0.0    # Trap exponent, as in V(r) = VT * r^alpha

# Parameters for the bisection algorithm (that is, to look for the chemical potential giving the right number of particles)
Ntarget = 17    # Expected number of particles
tol_N = 0.01    # Tolerance on the average N
mu_min = 0.0    # The bisection algorithm will search the optimal chemical potential in the interval [mu_min, mu_max]
mu_max = 2.0    # The bisection algorithm will search the optimal chemical potential in the interval [mu_min, mu_max]

# Initialize the state
G = Gutzwiller(seed=12312311, J=J, U=U, mu=-20.0, D=D, L=L, VT=VT, alphaT=alphaT, nmax=nmax, OBC=OBC)

# Perform bisection to find the right chemical potential
n_iterations = G.set_mu_via_bisection(Ntarget=Ntarget, mu_min=mu_min, mu_max=mu_max, tol_N=tol_N)


print()
print('Final chemical potential:   mu/U = %.6f' % (G.mu / G.U))
print('Final number of particles:  E/U =  %.6f' % G.N)
print('Final density:              <n> =  %.6f' % numpy.mean(G.density))
print('Final energy:               E/U =  %.6f' % (G.E / G.U))
print()
