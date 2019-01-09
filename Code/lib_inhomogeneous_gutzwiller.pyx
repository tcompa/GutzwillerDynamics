# cython: language_level=2

from __future__ import print_function

import sys
import math
import random

import numpy
import cython

cdef extern from "math.h" nogil:
    double c_sqrt 'sqrt' (double)
    double c_pow 'pow' (double, double)

cdef extern from "complex.h" nogil:
    double complex c_conj 'conj' (double complex)
    double c_abs 'cabs' (double complex)
    double c_real 'creal' (double complex)
    double c_imag 'cimag' (double complex)


@cython.wraparound(False)  #FIXME
@cython.boundscheck(False)
@cython.initializedcheck(False)
cdef class Gutzwiller:
    '''
    '''

    # Lattice parameters
    cdef int D, L, z, N_sites
    cdef int [:, :] nbr
    cdef int [:] N_nbr
    cdef int [:, :] site_coords

    # Bose-Hubbard parameters
    cdef double J, U, mu, VT
    cdef double [:] mu_local
    cdef int alphaT

    # State properties
    cdef int nmax
    cdef double complex [:, :] f
    cdef double complex [:] bmean
    cdef double [:] density
    cdef double E

    def __init__(self,
                 int nmax=3,                                    # cutoff on occupation (index goes from 0 to nmax)
                 int D=1, int L=11,                             # lattice parameters (site index goes from 0 to L-1 in each dimensions)
                 double J=0.0, double U=0.0, double mu=0.0,     # homogeneous-model parameters
                 double VT=0.0, int alphaT=0,                   # trap parameters
                 ):

        self.nmax = nmax
        self.D = D
        self.L = L
        self.J = J
        self.U = U
        self.mu = mu
        self.VT = VT
        self.alphaT = alphaT

        cdef int i_site, i_dim, n

        # Define lattice and neighbor table
        self.N_sites = self.L ** self.D
        self.z = 2 * self.D  # hypercubic lattice
        self.nbr = numpy.zeros((self.N_sites, self.z), dtype=numpy.int32)
        self.site_coords = numpy.zeros((self.N_sites, self.D), dtype=numpy.int32)
        self.N_nbr = numpy.zeros(self.N_sites, dtype=numpy.int32)
        if D == 1:
            for i_site in range(self.N_sites):
                self.site_coords[i_site] = i_site
                self.N_nbr[i_site] = 2
                self.nbr[i_site, 0] = (i_site - 1 + self.L) % self.L
                self.nbr[i_site, 1] = (i_site + 1) % self.L
        else:
            sys.exit('ERROR: By now only D=1 is accepted.')

        # Define trapping potential
        if L % 2 == 0:
            sys.exit('ERROR: If L is even, where should I put the trap center?')
        cdef int trap_center = (L - 1) // 2
        cdef double r, r_sq
        self.mu_local = numpy.zeros(self.N_sites)
        for i_site in range(self.N_sites):
            r_sq = 0.0
            for i_dim in range(self.D):
                r_sq += c_pow(self.site_coords[i_site, i_dim] - trap_center, 2)
            r = c_sqrt(r_sq)
            self.mu_local[i_site] = self.mu - self.VT * c_pow(r, self.alphaT)

        # Declare useful variables
        self.f = numpy.zeros((self.N_sites, self.nmax + 1)) + 0.0j
        self.bmean = numpy.zeros(self.N_sites) + 0.0j
        self.density = numpy.zeros(self.N_sites)

        # Initialize Gutzwiller coefficients
        for i_site in range(self.N_sites):
            for n in range(self.nmax + 1):
                self.f[i_site, n] = random.random() + 1.0j * random.random()
        self.normalize_gutzwiller_coefficients()

        self.update_bmean()
        self.update_density()
        self.E = self.compute_energy()

        for i_site in range(self.N_sites):
            print('i_site=%02i' % i_site)
            print(' <b>=(%+.4f, %+.4f), |<b>|=%.4f, <n>=%.4f' % (i_site,
                        c_real(self.bmean[i_site]), c_imag(self.bmean[i_site]), c_abs(self.bmean[i_site]), self.density[i_site]))

    cdef void normalize_gutzwiller_coefficients(self):
        cdef int i_site, n
        cdef double norm_sq, inv_norm
        for i_site in range(self.N_sites):
            norm_sq = 0.0
            for n in range(self.nmax + 1):
                norm_sq += c_pow(c_abs(self.f[i_site, n]), 2)
            inv_norm = 1.0 / c_sqrt(norm_sq)
            for n in range(self.nmax + 1):
                self.f[i_site, n] *= inv_norm

    cdef void update_bmean(self):
        cdef int i_site, n
        for i_site in range(self.N_sites):
            self.bmean[i_site] = 0.0 + 0.0j
            for n in range(0, self.nmax):
                self.bmean[i_site] += c_conj(self.f[i_site, n]) * self.f[i_site, n + 1] * c_sqrt(n + 1)

    cdef void update_density(self):
        cdef int i_site, n
        for i_site in range(self.N_sites):
            self.density[i_site] = 0.0
            for n in range(0, self.nmax + 1):
                self.density[i_site] += c_pow(c_abs(self.f[i_site, n]), 2) * n

    cdef double compute_energy(self):
        cdef int i_site, j_site, n
        cdef double E = 0.0
        for i_site in range(self.N_sites):

            # 1/3 - Hopping
            for j_site in range(i_site):
                E -= 2.0 * self.J * c_real(c_conj(self.bmean[i_site]) * self.bmean[j_site])

            # 2/3 - On-site repulsion
            for n in range(self.nmax + 1):
                E += 0.5 * self.U * c_pow(c_abs(self.f[i_site, n]), 2) * n * (n - 1)

            # 3/3 - On-site chemical potential (trap included)
            E -= self.mu_local[i_site] * self.density[i_site]
        return E
