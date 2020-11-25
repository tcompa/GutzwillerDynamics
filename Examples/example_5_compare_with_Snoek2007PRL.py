#!/usr/bin/env python

from __future__ import print_function

# Comment these lines if you want to use more than one core
import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import numpy as np

# Replace the following line with the actual code path
codedir = '../Code/'
sys.path.append(codedir)
from lib_inhomogeneous_gutzwiller import Gutzwiller


print()
print('WARNING: with the current parameters, this program takes ' +
      'relatively long to run (2-3 minutes on a laptop)')
print()

# Physical parameters
D = 1             # Dimensions
L = 22            # Linear lattice size
U = 0.2           # On-site repulsion
J = 1.0           # Hopping amplitude
mu = -0.49260342  # Chemical potential
nmax = 32         # Cutoff on the local occupation
OBC = 1           # Open (OBC=1) or periodic (OBC=0) boundary conditions
Ntarget = 20      # Target number of particles

VT = 0.3 * J   # Trap prefactor, as in V(r) = VT * r^alpha
alphaT = 2.0    # Trap exponent, as in V(r) = VT * r^alpha

# Initialize the state
G = Gutzwiller(seed=12312311, J=J, U=U, mu=mu, D=D, L=L,
               VT=VT, alphaT=alphaT, nmax=nmax, OBC=OBC)

# Find ground state via imaginary-time dynamics
dt = 0.1 / J    # Time interval
nsteps = 200    # Number of time-evolution steps
G.many_time_steps(1.0j * dt, nsteps=nsteps)

# Print some output and save the density profile
G.save_densities('data_ex5_densities_pre_quench.dat')
print()
print('End of initial-state preparation (via imaginary-time dynamics)')
print('Chemical potential:   mu/U = %.8f, mu/J = %.8f' % (G.mu / G.U, G.mu / G.J))
print('Number of particles:  <N> =  %.6f' % G.N)
print('Energy:               E/U =  %.6f' % (G.E / G.U))
print()

# Save local state at site
site = L // 2
G.save_gutzwiller_coefficients_at_one_site(site, 'data_ex5_coefficients_pre_quench.dat')

# Quench
A = 2.0
prequench_trap_center = G.trap_center
G.update_trap_center(G.trap_center + A)
G.update_energy()
print('Now performing a quench, shifting the trap center from %.2f to %.2f' % (prequench_trap_center, G.trap_center))
print('Number of particles:  <N> =  %.6f' % G.N)
print('Energy:               E/U =  %.6f' % (G.E / G.U))
print()

# Parameters for real-time evolution
dt = 0.001 / J
tmax = 15.0 / J
nsteps = int(tmax / dt + 0.5)
times = np.arange(nsteps) * dt

# Real-time evolution
out = open('data_ex5_dynamics.dat', 'w')
out.write('# t*J, E/U, N, C.O.M. - trap_center\n')
skip = 10
for step in range(nsteps):
    G.many_time_steps(dt, nsteps=1, normalize_at_each_step=0, update_variables=1)
    if step % skip == 0:
        out.write('%.8f %.8f %.8f %.8f\n' % (step * dt * J, G.E / G.U, G.N, G.compute_center_of_mass()[0] - G.trap_center))
        out.flush()
out.close()

# Print some output and save the density profile
print('End of real-time evolution')
print('Number of particles:  <N> =  %.6f' % G.N)
print('Energy:               E/U =  %.6f' % (G.E / G.U))
print()
