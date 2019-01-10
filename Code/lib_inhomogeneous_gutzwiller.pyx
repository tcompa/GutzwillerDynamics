# cython: language_level=2

from __future__ import print_function

import sys
import math
import random

import numpy
import scipy
import cython

import scipy.linalg
import scipy.sparse.linalg


cdef extern from "math.h" nogil:
    double c_sqrt 'sqrt' (double)
    double c_pow 'pow' (double, double)

cdef extern from "complex.h" nogil:
    double complex c_conj 'conj' (double complex)
    double c_abs 'cabs' (double complex)
    double c_real 'creal' (double complex)
    double c_imag 'cimag' (double complex)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
cdef class Gutzwiller:

    # Lattice parameters
    cdef public int N_sites
    cdef int D, L, z
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
    cdef double complex [:, :] f_new
    cdef public double complex [:] bmean
    cdef double complex [:] sum_bmeans
    cdef public double [:] density
    cdef public double E
    cdef public double N

    # Time evolution
    cdef int size_M
    cdef double complex [:, :] M
    cdef double complex [:, :] exp_M

    def __init__(self,
                 int nmax=3,                                    # cutoff on occupation (index goes from 0 to nmax)
                 int D=1, int L=7, int OBC=0,                   # lattice parameters (site index goes from 0 to L-1 in each dimensions)
                 double J=0.1, double U=1.0, double mu=0.5,     # homogeneous-model parameters
                 double VT=0.0, int alphaT=0,                   # trap parameters
                 int seed=-1,
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

        # Random seed
        if seed > -1:
            random.seed(seed)

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
            if OBC == 1:
                if L < 2 or D > 1:
                    sys.exit('ERROR: Trying to implement OBC with D=%i and L=%i. Exit.' % (self.D, self.L))
                self.N_nbr[0] = 1
                self.nbr[0, 0] = 1
                self.N_nbr[self.L - 1] = 1
                self.nbr[self.L - 1, 0] = self.L - 2
        else:
            sys.exit('ERROR: By now only D=1 is implemented. Exit.')

        # Define things for time evolution
        self.size_M = self.nmax + 1
        print('size_M: %i' % self.size_M)
        self.M = numpy.zeros((self.size_M, self.size_M)) + 0.0j
        self.exp_M = numpy.zeros((self.size_M, self.size_M)) + 0.0j

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
        self.f_new = numpy.zeros((self.N_sites, self.nmax + 1)) + 0.0j
        self.bmean = numpy.zeros(self.N_sites) + 0.0j
        self.sum_bmeans = numpy.zeros(self.N_sites) + 0.0j
        self.density = numpy.zeros(self.N_sites)

        # Initialize Gutzwiller coefficients
        for i_site in range(self.N_sites):
            for n in range(self.nmax + 1):
                self.f[i_site, n] = random.random() + 1.0j * random.random()
        self.normalize_gutzwiller_coefficients()

        self.update_bmean()
        self.update_sum_bmeans()
        self.update_density()
        self.E = self.compute_energy()

    def print_nbr_table(self):
        print('----- Neighbor table -----')
        for i_site in range(self.N_sites):
            print('i_site=%3i, nbr:' % i_site, end='')
            for k in range(self.N_nbr[i_site]):
                print(' %i' % self.nbr[i_site, k], end='')
            print('\n', end='')
        print('--------------------------')

    def print_f_coefficients(self, int i_site):
        print('------- f(site=%i) -------' % i_site)
        for n in range(self.nmax + 1):
            print(' (%+.4f, %+.4f)' % (c_real(self.f[i_site, n]), c_imag(self.f[i_site, n])), end='')
        print('\n', end='')
        print('--------------------------')

    def print_basic_info(self):
        cdef int i_site

        for i_site in range(self.N_sites):
            print('i_site=%02i' % i_site, end='')
            print(' <b>=(%+.4f, %+.4f), |<b>|=%.4f, <n>=%.4f' % (c_real(self.bmean[i_site]), c_imag(self.bmean[i_site]),
                                                                 c_abs(self.bmean[i_site]), self.density[i_site]))
        print('-' * 80)

    #@cython.cdivision(True)
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
        self.bmean[:] = 0.0
        for i_site in range(self.N_sites):
            for n in range(0, self.nmax):
                self.bmean[i_site] += c_conj(self.f[i_site, n]) * self.f[i_site, n + 1] * c_sqrt(n + 1)

    cdef void update_sum_bmeans(self):
        cdef int i_site, j_nbr, j_site
        self.sum_bmeans[:] = 0.0
        for i_site in range(self.N_sites):
            for j_nbr in range(self.N_nbr[i_site]):
                j_site = self.nbr[i_site, j_nbr]
                self.sum_bmeans[i_site] += self.bmean[j_site]

    cdef void update_density(self):
        cdef int i_site, n
        self.N = 0.0
        self.density[:] = 0.0
        for i_site in range(self.N_sites):
            for n in range(0, self.nmax + 1):
                self.density[i_site] += c_pow(c_abs(self.f[i_site, n]), 2) * n
            self.N += self.density[i_site]

    cdef double compute_energy(self):
        cdef int i_site, j_site, j_nbr, n, m
        cdef double E = 0.0
        for i_site in range(self.N_sites):

            # 1/2 - Hopping
            for j_nbr in range(self.N_nbr[i_site]):
                j_site = self.nbr[i_site, j_nbr]
                E -= self.J * c_real(c_conj(self.bmean[i_site]) * self.bmean[j_site])

            # 2/3 - On-site repulsion
            for n in range(self.nmax + 1):
                E += 0.5 * self.U * c_pow(c_abs(self.f[i_site, n]), 2) * n * (n - 1)
 
            # 3/3 - On-site chemical potential (trap included)
            E -= self.mu_local[i_site] * self.density[i_site]

        return E

    cpdef void one_real_time_step(self, double complex dtau):
        cdef int i_site, n, m

        self.f_new[:, :] = 0.0
        for i_site in range(self.N_sites):

            # Build matrix
            self.M[:, :] = 0.0
            for m in range(self.nmax + 1):
                self.M[m, m] += 0.5 * self.U * m * (m - 1.0) - self.mu_local[i_site] * m
            for m in range(0, self.nmax):
                self.M[m + 1, m] -= self.J * self.sum_bmeans[i_site] * c_sqrt(m + 1)
            for m in range(1, self.nmax + 1):
                self.M[m - 1, m] -= self.J * c_conj(self.sum_bmeans[i_site]) * c_sqrt(m)

            # Update on-site coefficients
            self.exp_M = scipy.linalg.expm(1.0j * dtau * numpy.array(self.M[:, :]))   #FIXME: SLOW!
            #self.exp_M = scipy.sparse.linalg.expm(1.0j * dtau * numpy.array(self.M[:, :]).T)   #FIXME: SLOW!
            for n in range(self.nmax + 1):
                for m in range(self.nmax + 1):
                    self.f_new[i_site, n] += self.exp_M[n, m] * self.f[i_site, m]

        # Update all coefficients
        self.f[:, :] = self.f_new[:, :]

        self.normalize_gutzwiller_coefficients()

        self.update_bmean()
        self.update_sum_bmeans()
        self.update_density()
        self.E = self.compute_energy()




