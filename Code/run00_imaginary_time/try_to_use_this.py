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

G = Gutzwiller(seed=11232123, J=0.04, U=1.0, D=1, mu=1.5, L=19, VT=0.035, alphaT=2, nmax=3, OBC=0)
G.print_nbr_table()

dt = 2.0e-3j
nsteps = 20000

skip_steps = 100
skip_steps_plot = 800

out = open('output.dat', 'w')

print 'dtau: ', dt
print
for i_t in xrange(nsteps):
    G.one_real_time_step(dt)
    if i_t % skip_steps == 0:
        print 't=%06i, E=%+.6f, N=%.6f, N/N_sites=%.6f' % (i_t + 1, G.E, G.N, G.N / G.N_sites)
        out.write('%06i %.6f %.6f %.6f\n' % (i_t + 1, G.E, G.N, G.N / G.N_sites))
        out.flush()
    if i_t % skip_steps_plot == 0:
        G.print_basic_info()
        n = numpy.array(G.density)
        bmean_sq = numpy.absolute(numpy.array(G.bmean)) ** 2
        plt.plot(n, 'o-', c='C1', label='$\\langle \\hat{n}_i \\rangle$', mec='k', lw=1)
        plt.plot(bmean_sq, '.--', c='C1', label='$\\left| \\langle \\hat{b}_i \\rangle \\right|^2$')
        print
        plt.legend(framealpha=1)
        plt.grid()
        plt.ylim(bottom=0.0)
        plt.ylim(top=2.5)
        plt.xlabel('Site $i$')
        plt.title('$t = %i$' % i_t)
        plt.savefig('Figs/fig_profile_t%06i.png' % i_t, bbox_inches='tight', dpi=192)
        plt.close()
