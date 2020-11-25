#!/usr/bin/env python

from __future__ import print_function

# Comment these lines if you want to use more than one core
import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import numpy as np
import matplotlib.pyplot as plt

# Replace the following line with the actual code path
codedir = '../Code/'
sys.path.append(codedir)
from lib_inhomogeneous_gutzwiller import Gutzwiller


# Physical parameters
D = 2           # Dimensions
L = 4           # Linear lattice size
U = 1.0         # On-site repulsion
J = 0.08        # Hopping amplitude
Vnn = 0.6       # Nearest-neighbor interaction
nmax = 6        # Cutoff on the local occupation
OBC = 0         # Whether to impose open (OBC=1) or periodic (OBC=0) boundary conditions
VT = 0.0        # Trap prefactor, as in V(r) = VT * r^alpha
alphaT = 0.0    # Trap exponent, as in V(r) = VT * r^alpha
mu = 1.2        # Chemical potential

# Parameters for imaginary-time evolution
dt = 0.1 / J    # Time interval, in units of 
nsteps = 200    # Number of time-evolution steps

# Initialize the state
G = Gutzwiller(seed=12312311, J=J, U=U, mu=mu, D=D, L=L, Vnn=Vnn, VT=VT, alphaT=alphaT, nmax=nmax, OBC=OBC)
G.print_parameters()

# Imaginary-time dynamics
G.many_time_steps(1.0j * dt, nsteps=nsteps)
print('Optimization completed')
print('Number of particles:  E/U =  %.6f' % G.N)
print('Energy:               E/U =  %.6f' % (G.E / G.U))
print('Densities:')
for i_site in range(G.N_sites):
    print('    site %2i, n=%.6f' % (i_site, G.density[i_site]))
print()

# Plot Gutzwiller coefficients
for i_site in range(G.N_sites):
    f = G.get_gutzwiller_coefficients_at_one_site(i_site)
    plt.plot(np.absolute(f) ** 2, '.-', label='site %i' % i_site)
plt.xlabel('State $n$', fontsize=14)
plt.ylabel('Coefficient $|f_n|^2$', fontsize=14)
plt.legend()
plt.savefig('fig_ex6_gutzwiller_coefficients.pdf', bbox_inches='tight')
