from __future__ import print_function
import sys

sys.path.append('..')
from lib_inhomogeneous_gutzwiller import Gutzwiller


def perform_imaginary_time_evolution_1D(J):
    if J == 0.15:
        E_expected = -3.9412627297
    elif J == 0.05:
        E_expected = -3.317526309839
    else:
        sys.exit('ERROR: test_imaginary_time_evolution_1D called with J=%s.' % J)

    G = Gutzwiller(seed=12311123, J=J, U=1.0, D=1, mu=0.5, L=19, VT=0.02, alphaT=2, nmax=4, OBC=1)
    G.many_time_steps(0.5j, nsteps=1000, normalize_at_each_step=1, update_variables=0)
    print('J/U: %s' % (G.J / G.U))
    print('E:   %.12f' % G.E)
    print('E:   %.12f (expexted)' % E_expected)
    print()
    assert abs(G.E - E_expected) < 1e-4


def test_imaginary_time_evolution_1D_JbyU015():
    perform_imaginary_time_evolution_1D(0.15)


def test_imaginary_time_evolution_1D_JbyU005():
    perform_imaginary_time_evolution_1D(0.05)


if __name__ == '__main__':
    test_imaginary_time_evolution_1D_JbyU005()
    test_imaginary_time_evolution_1D_JbyU015()
