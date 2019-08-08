#!/usr/bin/env python

from __future__ import print_function

# Comment these lines if you want to use more than one core
import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import numpy
import matplotlib.pyplot as plt

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
OBC = 1         # Whether to impose open (OBC=1) or periodic (OBC=0) boundary conditions
VT = 0.2        # Trap prefactor, as in V(r) = VT * r^alpha
alphaT = 2.0    # Trap exponent, as in V(r) = VT * r^alpha
mu = 1.2        # Chemical potential

# Parameters for imaginary-time evolution
dt = 0.1 / J    # Time interval, in units of 
nsteps = 300    # Number of time-evolution steps

# Initialize the state
G = Gutzwiller(seed=12312311, J=J, U=U, mu=mu, D=D, L=L, VT=VT, alphaT=alphaT, nmax=nmax, OBC=OBC)

# Imaginary-time dynamics
G.many_time_steps(1.0j * dt, nsteps=nsteps)

# Print some output and save the density profile
print('End of ground-state search (via imaginary-time evolution)')
print('Chemical potential:         mu/U = %.6f' % (G.mu / G.U))
print('Final number of particles:  E/U =  %.6f' % G.N)
print('Final density:              <n> =  %.6f' % numpy.mean(G.density))
print('Final energy:               E/U =  %.6f' % (G.E / G.U))
print()
G.save_densities('data_ex3_densities_before_time_evolution.dat')

# Change some parameters
VT_new = VT * 0.1
G.update_VT(VT_new)

# Parameters for real-time evolution
dt = 0.01 / J
nsteps = 200

# Real-time evolution
alldata = []
for step in range(nsteps):
    alldata.append([step * dt * U] + numpy.array(G.density).tolist())
    G.many_time_steps(dt, nsteps=1, normalize_at_each_step=0, update_variables=1)
alldata.append([nsteps * dt * U] + numpy.array(G.density).tolist())

# Print some output and save the density profile
print('End of real-time evolution')
print('Chemical potential:         mu/U = %.6f' % (G.mu / G.U))
print('Final number of particles:  E/U =  %.6f' % G.N)
print('Final density:              <n> =  %.6f' % numpy.mean(G.density))
print('Final energy:               E/U =  %.6f' % (G.E / G.U))
print()
G.save_densities('data_ex3_densities_after_time_evolution.dat')

# Do plot of densities
alldata = numpy.array(alldata).T
for i in range(L):
    plt.plot(alldata[0], alldata[i + 1], alpha=0.8, label='Site %i' % i)
plt.plot(alldata[0], alldata[1:].mean(axis=0), ls='--', c='k', lw=2)
plt.xlabel('Time $t \\qquad [1/U]$', fontsize=14)
plt.ylabel('Local density $n_i(t)$', fontsize=14)
plt.legend(framealpha=1, ncol=2)
plt.show()
