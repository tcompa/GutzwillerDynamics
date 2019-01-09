#!/usr/bin/env python

from lib_inhomogeneous_gutzwiller import Gutzwiller

G = Gutzwiller(seed=11232123, J=0.025, U=1.0, D=1, mu=0.5, L=3, nmax=2)

dt = 1e-1j

out = open('output.dat', 'w')

print 'dtau: ', dt
G.print_basic_info()
for i_t in xrange(200000):
    G.one_real_time_step(1e-3)
    if i_t % 250 == 0:
        print 't=%06i, E=%+.6f, N=%.6f, N/N_sites=%.6f' % (i_t + 1, G.E, G.N, G.N / G.N_sites)
        out.write('%06i %.6f %.6f %.6f\n' % (i_t + 1, G.E, G.N, G.N / G.N_sites))
        out.flush()
