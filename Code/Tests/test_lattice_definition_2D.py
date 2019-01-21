from __future__ import print_function
import os
import sys
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy
import itertools

repo = 'Time_dependent_Gutzwiller/'
rootdir = os.path.abspath(os.getcwd()).split(repo)[0] + repo
codedir = rootdir + 'Code'
sys.path.append(codedir)
from lib_inhomogeneous_gutzwiller import Gutzwiller


def test_lattice_definition_2D():
    xy = numpy.empty(2, dtype=numpy.int32)
    for L in [2, 3, 10, 11]:
        G = Gutzwiller(D=2, L=L)
        for i_site in xrange(G.N_sites):
            x, y = G.site_coords[i_site, :]
            assert i_site == G.xy2i(x, y)
        for x, y in itertools.product(range(L), repeat=2):
            i_site = G.xy2i(x, y)
            G.i2xy(i_site, xy)
            assert x == xy[0]
            assert y == xy[1]


if __name__ == '__main__':
    test_lattice_definition_2D()
