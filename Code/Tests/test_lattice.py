from __future__ import print_function
import itertools
import numpy
import sys

sys.path.append('..')
from lib_inhomogeneous_gutzwiller import Gutzwiller


def test_lattice_definition_2D():
    xy = numpy.empty(2, dtype=numpy.int32)
    for L in [2, 3, 10, 11]:
        G = Gutzwiller(D=2, L=L)
        for i_site in range(G.N_sites):
            x, y = G.site_coords[i_site, :]
            assert i_site == G.xy2i(x, y)
        for x, y in itertools.product(range(L), repeat=2):
            i_site = G.xy2i(x, y)
            G.i2xy(i_site, xy)
            assert x == xy[0]
            assert y == xy[1]


def test_neighbors_reciprocity():
    for OBC in [0, 1]:
        for D in [1, 2]:
            for L in [5, 10]:
                G = Gutzwiller(D=D, L=L, OBC=OBC)
                for i_site in range(G.N_sites):
                    for j_nbr in range(G.N_nbr[i_site]):
                        j_site = G.nbr[i_site, j_nbr]
                        assert i_site in list(G.nbr[j_site, :])


if __name__ == '__main__':
    test_lattice_definition_2D()
    test_neighbors_reciprocity()
