# cython: language_level=2

from __future__ import print_function
import cython
import sys
import random
import numpy
import scipy.linalg


cdef extern from "math.h" nogil:
    double c_sqrt 'sqrt' (double)
    double c_pow 'pow' (double, double)

cdef extern from "complex.h" nogil:
    double complex c_conj 'conj' (double complex)
    double c_abs 'cabs' (double complex)
    double c_real 'creal' (double complex)
    double c_imag 'cimag' (double complex)


#@cython.profile(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
cdef class Gutzwiller:

    # Lattice parameters
    cdef public int N_sites
    cdef public int D, L, z, OBC
    cdef int [:, :] nbr
    cdef int [:] N_nbr
    cdef public int [:, :] site_coords

    # Bose-Hubbard parameters
    cdef public double J, U, mu, VT, trap_center
    cdef double [:] mu_local
    cdef int alphaT

    # State properties
    cdef int nmax
    cdef double complex [:, :] f
    cdef double complex [:] f_new
    cdef public double complex [:] bmean
    cdef double complex [:] sum_bmeans
    cdef public double [:] density
    cdef public double E
    cdef public double N
    cdef public double N_cond
    cdef double [:] x_com

    # Time evolution
    cdef int size_M
    cdef double complex [:, :] M
    cdef double complex [:, :] exp_M

    def __init__(self,
                 int nmax=3,                                          # cutoff on occupation (index goes from 0 to nmax)
                 int D=1, int L=7, int OBC=0,                         # lattice parameters (site index goes from 0 to L-1 in each dimensions)
                 double J=0.1, double U=1.0, double mu=0.5,           # homogeneous-model parameters
                 double VT=0.0, int alphaT=0, double trap_center=-1,  # trap parameters
                 int seed=-1,
                 ):

        self.nmax = nmax
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
        self.D = D
        self.L = L
        self.OBC = OBC
        self.N_sites = self.L ** self.D
        self.z = 2 * self.D  # hypercubic lattice
        self.nbr = numpy.zeros((self.N_sites, self.z), dtype=numpy.int32)
        self.site_coords = numpy.zeros((self.N_sites, self.D), dtype=numpy.int32)
        self.N_nbr = numpy.zeros(self.N_sites, dtype=numpy.int32)
        self.initialize_lattice()

        # Declare center of mass
        self.x_com = numpy.zeros(D)

        # Define things for time evolution
        self.size_M = self.nmax + 1
        self.M = numpy.zeros((self.size_M, self.size_M)) + 0.0j
        self.exp_M = numpy.zeros((self.size_M, self.size_M)) + 0.0j

        # Define trapping potential
        self.trap_center = trap_center
        if trap_center == -1:
            if self.L % 2 == 0:
                self.trap_center = (self.L - 1) // 2 + 0.5
            else:
                self.trap_center = (self.L - 1) // 2
        self.initialize_trap()

        # Declare useful variables
        self.f = numpy.zeros((self.N_sites, self.nmax + 1)) + 0.0j
        self.f_new = numpy.zeros(self.nmax + 1) + 0.0j
        self.bmean = numpy.zeros(self.N_sites) + 0.0j
        self.sum_bmeans = numpy.zeros(self.N_sites) + 0.0j
        self.density = numpy.zeros(self.N_sites)

        # Initialize Gutzwiller coefficients
        self.initialize_gutzwiller_coefficients_random()

        self.update_bmeans()
        self.update_density()
        self.update_energy()

    cdef void initialize_gutzwiller_coefficients_LDA(self):
        # For each site, define a new instance of the Gutzwiller class with the local mu, and perform imaginary-time to find the best coefficients.
        sys.exit('ERROR: initialize_gutzwiller_coefficients_LDA is not implemented. Exit.')
        cdef int i_site
        for i_site in range(self.N_sites):
            # do things...
            self.normalize_coefficients_single_site(i_site)

    cdef void initialize_gutzwiller_coefficients_random(self):
        for i_site in range(self.N_sites):
            for n in range(self.nmax + 1):
                self.f[i_site, n] = random.uniform(-1.0, 1.0) + 1.0j * random.uniform(-1.0, 0.1)
        self.normalize_coefficients_all_sites()

    @cython.cdivision(True)
    cdef void initialize_lattice(self):
        cdef int i_site, x, y
        cdef int [:] xy = numpy.zeros(2, dtype=numpy.int32)
        if self.D == 1:
            for i_site in range(self.N_sites):
                self.site_coords[i_site] = i_site
                self.N_nbr[i_site] = 2
                self.nbr[i_site, 0] = (i_site - 1 + self.L) % self.L
                self.nbr[i_site, 1] = (i_site + 1) % self.L
            if self.OBC == 1:
                if self.L < 2 or self.D > 1:
                    sys.exit('ERROR: Trying to implement OBC with D=%i and L=%i. Exit.' % (self.D, self.L))
                self.N_nbr[0] = 1
                self.nbr[0, 0] = 1
                self.N_nbr[self.L - 1] = 1
                self.nbr[self.L - 1, 0] = self.L - 2
        elif self.D == 2:
            for i_site in range(self.N_sites):
                self.i2xy(i_site, xy)
                x = xy[0]
                y = xy[1]
                self.site_coords[i_site, :] = xy[:]
                if self.OBC == 0:     # PBC
                    self.N_nbr[i_site] = 4
                    self.nbr[i_site, 0] = self.xy2i((x - 1 + self.L) % self.L, y)
                    self.nbr[i_site, 1] = self.xy2i(x, (y - 1 + self.L) % self.L)
                    self.nbr[i_site, 2] = self.xy2i((x + 1) % self.L, y)
                    self.nbr[i_site, 3] = self.xy2i(x, (y + 1) % self.L)
                else:                 # OBC
                    self.N_nbr[i_site] = 0
                    if x > 0:
                        self.nbr[i_site, self.N_nbr[i_site]] = self.xy2i((x - 1 + self.L) % self.L, y)
                        self.N_nbr[i_site] += 1
                    if y > 0:
                        self.nbr[i_site, self.N_nbr[i_site]] = self.xy2i(x, (y - 1 + self.L) % self.L)
                        self.N_nbr[i_site] += 1
                    if x < self.L - 1:
                        self.nbr[i_site, self.N_nbr[i_site]] = self.xy2i((x + 1) % self.L, y)
                        self.N_nbr[i_site] += 1
                    if y < self.L - 1:
                        self.nbr[i_site, self.N_nbr[i_site]] = self.xy2i(x, (y + 1) % self.L)
                        self.N_nbr[i_site] += 1
        else:
            sys.exit('ERROR: By now only D=1 and D=2 are implemented, not D=%i Exit.' % self.D)

    @cython.cdivision(True)
    cpdef void i2xy(self, int _i, int [:] xy):
        xy[0] = _i // self.L
        xy[1] = _i - (_i // self.L) * self.L

    cpdef int xy2i(self, int _x, int _y):
        return  _x * self.L + _y

    cpdef void initialize_trap(self):
        cdef double r_sq, r
        cdef int i_site, i_dim
        self.mu_local = numpy.zeros(self.N_sites)
        for i_site in range(self.N_sites):
            r_sq = 0.0
            for i_dim in range(self.D):
                r_sq += c_pow(self.site_coords[i_site, i_dim] - self.trap_center, 2)
            r = c_sqrt(r_sq)
            self.mu_local[i_site] = self.mu - self.VT * c_pow(r, self.alphaT)

    def load_config(self, datafile):
        new_f = numpy.loadtxt(datafile).view(complex)
        new_Nsites = new_f.shape[0]
        new_nmax = new_f.shape[1] - 1
        print('Loading %s. N_sites=%i, nmax=%i' % (datafile, new_Nsites, new_nmax))
        if (new_nmax != self.nmax) or (new_Nsites != self.N_sites):
            sys.exit('ERROR: Loaded %s, but (%i,%i) != (%i, %i). Exit.' % (datafile, new_Nsites, new_nmax + 1, self.N_sites, self.nmax + 1))
        for i_site in range(self.N_sites):
            for n in range(self.nmax + 1):
                self.f[i_site, n] = new_f[i_site, n]
        self.normalize_coefficients_all_sites()
        self.update_bmeans()
        self.update_density()
        self.update_energy()

    def save_config(self, datafile):
        numpy.savetxt(datafile, numpy.array(self.f).view(float))

    def save_densities(self, datafile):
        cdef int i_site, i_dim
        with open(datafile, 'w') as out:
            out.write('# site, density, |<b>|^2\n')
            for i_site in range(self.N_sites):
                for i_dim in range(self.D):
                    out.write('%i ' % self.site_coords[i_site, i_dim])
                out.write('%.8f %.8f\n' % (self.density[i_site], c_pow(c_abs(self.bmean[i_site]), 2)))

    def print_nbr_table(self):
        cdef int i_site, k
        print('----- Neighbor table -----')
        for i_site in range(self.N_sites):
            print('i_site=%3i, nbr:' % i_site, end='')
            for k in range(self.N_nbr[i_site]):
                print(' %i' % self.nbr[i_site, k], end='')
            print('\n', end='')
        print('--------------------------')

    def print_basic_info(self):
        cdef int i_site
        print('-' * 80)
        print('E = %f' % self.E)
        print('N = %f' % self.N)
        for i_site in range(self.N_sites):
            print('i_site=%02i' % i_site, end='')
            print(' <b>=(%+.4f, %+.4f), |<b>|^2=%.4f, <n>=%.4f' % (c_real(self.bmean[i_site]), c_imag(self.bmean[i_site]),
                                                                   c_abs(self.bmean[i_site]), self.density[i_site]))
        print('-' * 80)

    cpdef double [:] compute_center_of_mass(self):
        self.x_com[:] = 0.0
        cdef int i_site, i_dim
        for i_dim in range(self.D):
            for i_site in range(self.N_sites):
                self.x_com[i_dim] += self.site_coords[i_site, i_dim] * self.density[i_site]
            self.x_com[i_dim] /= self.N
        return self.x_com

    @cython.cdivision(True)
    cdef void normalize_coefficients_single_site(self, int i_site):
        cdef int n
        cdef double norm_sq, inv_norm
        norm_sq = 0.0
        for n in range(self.nmax + 1):
            norm_sq += c_pow(c_abs(self.f[i_site, n]), 2)
        inv_norm = 1.0 / c_sqrt(norm_sq)
        for n in range(self.nmax + 1):
            self.f[i_site, n] *= inv_norm

    cdef void normalize_coefficients_all_sites(self):
        cdef int i_site
        for i_site in range(self.N_sites):
            self.normalize_coefficients_single_site(i_site)

    cdef void update_bmeans(self):
        cdef int i_site, n, j_nbr, j_site
        self.bmean[:] = 0.0
        self.sum_bmeans[:] = 0.0
        self.N_cond = 0.0
        for i_site in range(self.N_sites):
            for n in range(0, self.nmax):
                self.bmean[i_site] += c_conj(self.f[i_site, n]) * self.f[i_site, n + 1] * c_sqrt(n + 1)
            for j_nbr in range(self.N_nbr[i_site]):
                self.sum_bmeans[self.nbr[i_site, j_nbr]] += self.bmean[i_site]
            self.N_cond += c_pow(c_abs(self.bmean[i_site]), 2.0)

    cpdef void update_density(self):
        cdef int i_site, n
        self.N = 0.0
        self.density[:] = 0.0
        for i_site in range(self.N_sites):
            for n in range(0, self.nmax + 1):
                self.density[i_site] += c_pow(c_abs(self.f[i_site, n]), 2) * n
            self.N += self.density[i_site]

    cpdef void update_energy(self):
        cdef int i_site, j_site, j_nbr, n, m
        self.E = 0.0
        for i_site in range(self.N_sites):

            # 1/2 - Hopping
            for j_nbr in range(self.N_nbr[i_site]):
                j_site = self.nbr[i_site, j_nbr]
                self.E -= self.J * c_real(c_conj(self.bmean[i_site]) * self.bmean[j_site])

            # 2/3 - On-site repulsion
            for n in range(self.nmax + 1):
                self.E += 0.5 * self.U * c_pow(c_abs(self.f[i_site, n]), 2) * n * (n - 1)
 
            # 3/3 - On-site chemical potential (trap included)
            self.E -= self.mu_local[i_site] * self.density[i_site]

    cdef void one_sequential_time_step(self, double complex dtau, int normalize_at_each_step=1, int update_variables=1):
        cdef int i_site, n, m, j_nbr
        cdef double complex old_bmean, diff_bmean

        for i_site in range(self.N_sites):

            # Build matrix
            self.M[:, :] = 0.0
            for m in range(self.nmax + 1):
                self.M[m, m] += 0.5 * self.U * m * (m - 1.0) - self.mu_local[i_site] * m
                if m < self.nmax:
                    self.M[m + 1, m] -= self.J * self.sum_bmeans[i_site] * c_sqrt(m + 1)
                    self.M[m, m + 1] = c_conj(self.M[m + 1, m])

            # Update on-site coefficients
            self.exp_M = scipy.linalg.expm(1.0j * dtau * numpy.asarray(self.M))
            self.f_new[:] = 0.0
            for n in range(self.nmax + 1):
                for m in range(self.nmax + 1):
                    self.f_new[n] += self.exp_M[n, m] * self.f[i_site, m]
            self.f[i_site, :] = self.f_new[:]
            if normalize_at_each_step == 1:
                self.normalize_coefficients_single_site(i_site)

            # Update on-site bmean and off-site sum of bmean (and N_cond).
            self.N_cond -= c_pow(c_abs(self.bmean[i_site]), 2)
            old_bmean = self.bmean[i_site]
            self.bmean[i_site] = 0.0
            for n in range(0, self.nmax):
                self.bmean[i_site] += c_conj(self.f[i_site, n]) * self.f[i_site, n + 1] * c_sqrt(n + 1)
            self.N_cond += c_pow(c_abs(self.bmean[i_site]), 2)
            diff_bmean = self.bmean[i_site] - old_bmean
            for j_nbr in range(self.N_nbr[i_site]):
                self.sum_bmeans[self.nbr[i_site, j_nbr]] += diff_bmean

        if update_variables == 1:
            self.update_bmeans()
            self.update_density()
            self.update_energy()

    cpdef void many_time_steps(self, double complex dtau, int nsteps=1, int normalize_at_each_step=1, int update_variables=0):
        cdef int i_step
        for i_site in range(nsteps):
            self.one_sequential_time_step(dtau, normalize_at_each_step=normalize_at_each_step, update_variables=update_variables)

        self.update_density()
        self.update_energy()

    cpdef void update_J(self, double J):
        self.J = J

    cpdef void update_U(self, double U):
        self.U = U

    cpdef void update_mu(self, double mu):
        self.mu = mu
        self.initialize_trap()    #NOTE: If this line is removed, set_mu_via_bisection() should be changed.

    cpdef void update_VT(self, double VT):
        self.VT = VT
        self.initialize_trap()

    cpdef void update_trap_center(self, double trap_center):
        self.trap_center = trap_center
        self.initialize_trap()

    cpdef int set_mu_via_bisection(self,
                                    double Ntarget=0.0, double mu_min=-3.0, double mu_max=3.0, double tol_N=0.1,
                                    double complex dtau_times_J=0.5):
        cdef double f_min, f_mid, f_max, df
        cdef double mu_mid = 0.5 * (mu_min + mu_max)   # This is only to avoid a compile-time warning ("mu_mid’ may be used uninitialized..")
        cdef double complex dtau = dtau_times_J / self.J * 1.0j
        cdef int nsteps = 200
        cdef int bisection_iterations = 0

        self.update_mu(mu_min)
        self.initialize_gutzwiller_coefficients_random()
        self.many_time_steps(dtau, nsteps=nsteps)
        f_min = self.N - Ntarget
        if f_min > 0.0:
            sys.exit('[bisection] ERROR: mu_min=%f, Ntarget=%f, N_min=%f' % (mu_min, Ntarget, self.N))

        self.update_mu(mu_max)
        self.initialize_gutzwiller_coefficients_random()
        self.many_time_steps(dtau, nsteps=nsteps)
        f_max = self.N - Ntarget
        if f_max < 0.0:
            sys.exit('[bisection] ERROR: mu_max=%f, Ntarget=%f, N_max=%f' % (mu_max, Ntarget, self.N))

        df = 10.0 * tol_N
        while df > tol_N:
            bisection_iterations += 1
            mu_mid = 0.5 * (mu_min + mu_max)
            self.update_mu(mu_mid)
            self.initialize_gutzwiller_coefficients_random()
            self.many_time_steps(dtau, nsteps=nsteps)
            f_mid = self.N - Ntarget

            if f_mid < 0.0:
                mu_min = mu_mid
                f_min = f_mid
            else:
                mu_max = mu_mid
                f_max = f_mid
            df = f_max - f_min

        self.update_mu(mu_mid)
        self.initialize_gutzwiller_coefficients_random()
        self.many_time_steps(dtau, nsteps=nsteps)
        if abs(self.N - Ntarget) > tol_N:
            sys.exit('[bisection] ERROR: mu=%f, Ntarget=%f, N=%f' % (self.mu, Ntarget, self.N))

        return bisection_iterations
