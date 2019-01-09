#!/usr/bin/env python

from lib_inhomogeneous_gutzwiller import Gutzwiller
import matplotlib.pyplot as plt

G = Gutzwiller(seed=11232123, J=0.05, U=1.0, D=1, mu=0.5, L=7, VT=0.1, alphaT=2, nmax=4)

dt = 1e-2j

out = open('output.dat', 'w')

print 'dtau: ', dt
print
G.print_basic_info()
print
for i_t in xrange(1000):
    G.one_real_time_step(dt)
    if i_t % 100 == 0:
        print 't=%06i, E=%+.6f, N=%.6f, N/N_sites=%.6f' % (i_t + 1, G.E, G.N, G.N / G.N_sites)
        out.write('%06i %.6f %.6f %.6f\n' % (i_t + 1, G.E, G.N, G.N / G.N_sites))
        out.flush()
print
G.print_basic_info()
print
