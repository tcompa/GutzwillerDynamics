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
mu = 1.2        # Chemical potential

# Parameters for imaginary-time evolution
dt = 0.1 / J    # Time interval, in units of 
nsteps = 200    # Number of time-evolution steps

# Initialize the state
G = Gutzwiller(seed=12312311, J=J, U=U, mu=mu, D=D, L=L, VT=VT, alphaT=alphaT, nmax=nmax, OBC=OBC)

# Imaginary-time dynamics
G.many_time_steps(1.0j * dt, nsteps=nsteps)

print()
print('Chemical potential:         mu/U = %.6f' % (G.mu / G.U))
print('Final number of particles:  E/U =  %.6f' % G.N)
print('Final density:              <n> =  %.6f' % numpy.mean(G.density))
print('Final energy:               E/U =  %.6f' % (G.E / G.U))
print()

# Print ground state densities
print('*' * 80)
print('Ground state densities (site, density, condensed density):')
for i_site in range(G.L):
    print('%2i %.10f %.10f' % (i_site, G.density[i_site], abs(G.bmean[i_site]) ** 2))
print('*' * 80)
print()

# Print local state at site
print('*' * 80)
print('Fock coeffients at site 0:')
G.print_gutzwiller_coefficients_at_one_site(0)
print('*' * 80)
print()
