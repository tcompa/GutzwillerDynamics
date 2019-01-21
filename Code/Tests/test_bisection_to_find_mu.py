from __future__ import print_function
import os
import sys
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy

repo = 'Time_dependent_Gutzwiller/'
rootdir = os.path.abspath(os.getcwd()).split(repo)[0] + repo
codedir = rootdir + 'Code'
sys.path.append(codedir)
from lib_inhomogeneous_gutzwiller import Gutzwiller


def test_bisection_to_find_mu():
    Ntarget = 10.0
    tol = 0.1
    G = Gutzwiller(J=0.05, U=1.0, D=1, mu=0.0, L=15, VT=0.05, alphaT=2, nmax=4, OBC=1)
    n_iterations = G.set_mu_via_bisection(Ntarget=Ntarget, mu_min=-0.5, mu_max=1.5, tol_N=tol)
    print('#iterations: %i' % n_iterations)
    print('Ntarget:     %f' % Ntarget)
    print('G.mu:        %.8f   (N=%f)' % (G.mu, G.N))
    assert (G.N - Ntarget) < tol

if __name__ == '__main__':
    test_bisection_to_find_mu()
