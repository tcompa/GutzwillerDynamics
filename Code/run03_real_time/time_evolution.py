#!/usr/bin/env python

import os
import sys
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy

import matplotlib.pyplot as plt

sys.path.append('..')
from lib_inhomogeneous_gutzwiller import Gutzwiller

G = Gutzwiller(seed=1123123, J=1.0, U=0.2, D=1, mu=-0.4926, L=17, VT=0.3, alphaT=2, nmax=25, OBC=1)

G.save_config('data_00_config.dat')
G.save_densities('data_00_densities.dat')

# Imaginary-time evolution
tmax = 20.0
dt = G.J / 40.0
nsteps = int(tmax / dt + 0.5)
print '--- Imaginary-time evolution ---'
print '  tmax:   %f' % tmax
print '  dt:     %f' % dt
print '  nsteps: %i' % nsteps
out = open('log_00_imaginary_time_evolution.dat', 'w')
for i_t in xrange(nsteps + 1):
    out.write('%10.6f %.8f %.8f\n' % (i_t * dt, G.E, G.N))
    G.one_sequential_time_step(1.0j * dt)
G.save_config('data_01_config.dat')
G.save_densities('data_01_densities.dat')
print '---------------------------------'

print('mu=%.8f | N=%.8f' % (G.mu, G.N))

G.print_basic_info()

# Shift the trap center
A = 2
G.initialize_trap(G.trap_center + A)

tmax = 16.0
dt = G.J / 500.0
nsteps = int(tmax / dt + 0.5)
print '--- Real-time evolution ---------'
print '  tmax:   %f' % tmax
print '  dt:     %f' % dt
print '  nsteps: %i' % nsteps
out = open('log_01_real_time_evolution.dat', 'w')
for i_t in xrange(nsteps + 1):
    out.write('%10.6f %.8f %.8f %.8f\n' % (i_t * dt, G.E, G.N, G.compute_center_of_mass()))
    G.one_sequential_time_step(dt, normalize_at_each_step=0)
    if i_t % 50 == 0 and i_t < 500:
        G.save_densities('data_02_densities_%08i.dat' % i_t)
print '---------------------------------'
