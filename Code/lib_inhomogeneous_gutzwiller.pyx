# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, embedsignature=True, linetrace=True


from __future__ import print_function
import cython
import sys
import random
import numpy as np
import scipy.linalg


cdef extern from "math.h" nogil:
    double c_sqrt 'sqrt' (double)
    double c_pow 'pow' (double, double)

cdef extern from "complex.h" nogil:
    double complex c_conj 'conj' (double complex)
    double c_abs 'cabs' (double complex)
    double c_real 'creal' (double complex)
    double c_imag 'cimag' (double complex)


cdef class Gutzwiller:
    '''
    Gutzwiller ansatz for a Bose-Hubbard model.

    Parameters:
        D               Lattice dimensionality (allowed values: 1, 2)
        L               Linear size of lattice
        OBC             Boundary conditions (OBC=1 for OBC, OBC=0 for PBC)
        J               Nearest-neighbor hopping parameter (see Hamiltonian)
        U               On-site interaction parameter (see Hamiltonian)
        mu              Chemical potential (see Hamiltonian)
        Vnn             Nearest-neighbor interaction parameter (see Hamiltonian)
        VT              Trap prefactor (see Hamiltonian)
        alphaT          Trap exponent (see Hamiltonian)
        trap_center     Trap center (see Hamiltonian)
        nmax            Cutoff on the local occupation number (state indices go from 0 to nmax)

    The Hamiltonian includes several terms.
    Here we denote the sum over the neighbors of site i as sum_{j~i}.
        - J * sum_{i} sum_{j~i} b_i^dagger b_j
        + U * sum_{i} n_i * (n_i - 1)
        + sum_{i} n_i * ( -mu + VT * |x_i - trap_center|^alpha )
        + (Vnn/2) sum_{i} sum_{j~i} n_i * n_j

    NOTE:
    1) In the nearest-neighbor-interaction term, I use (Vnn/2) because each pair of neighbors is counted twice.
    2) In the hopping term, I use J (and not J/2) because pairs (i,j) and (j,i) are not equivalent (due to the Hermitian conjugate).
    '''


    # Lattice parameters
    cdef public int N_sites
    cdef public int D, L, z, OBC
    cdef public int [:, :] nbr
    cdef public int [:] N_nbr
    cdef public int [:, :] site_coords

    # Bose-Hubbard parameters
    cdef public double J, U, Vnn, mu, VT, trap_center, alphaT
    cdef double [:] mu_local

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
                 int nmax=3,                                                 # cutoff on occupation (index goes from 0 to nmax)
                 int D=1, int L=7, int OBC=0,                                # lattice parameters (sites go from 0 to L-1 in each dimensions)
                 double J=0.1, double U=1.0, double mu=0.5, double Vnn=0.0,  # homogeneous-model parameters
                 double VT=0.0, double alphaT=0, double trap_center=-1,      # trap parameters
                 int seed=-1,
                 ):

        self.nmax = nmax
        self.J = J
        self.U = U
        self.mu = mu
        self.VT = VT
        self.Vnn = Vnn
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
        self.nbr = np.zeros((self.N_sites, self.z), dtype=np.int32)
        self.site_coords = np.zeros((self.N_sites, self.D), dtype=np.int32)
        self.N_nbr = np.zeros(self.N_sites, dtype=np.int32)
        self.initialize_lattice()

        # Declare center of mass
        self.x_com = np.zeros(D)

        # Declare variables for time evolution
        self.size_M = self.nmax + 1
        self.M = np.zeros((self.size_M, self.size_M)) + 0.0j
        self.exp_M = np.zeros((self.size_M, self.size_M)) + 0.0j

        # Define trapping potential [NOTE: a call to self.initialize_trap() is included in self.update_trap_center()]
        self.mu_local = np.zeros(self.N_sites)
        self.update_trap_center(trap_center)

        # Declare useful variables
        self.f = np.zeros((self.N_sites, self.nmax + 1)) + 0.0j
        self.f_new = np.zeros(self.nmax + 1) + 0.0j
        self.bmean = np.zeros(self.N_sites) + 0.0j
        self.sum_bmeans = np.zeros(self.N_sites) + 0.0j
        self.density = np.zeros(self.N_sites)

        # Initialize Gutzwiller coefficients
        self.initialize_gutzwiller_coefficients_random()

        self.update_bmeans()
        self.update_density()
        self.update_energy()

    cdef void initialize_gutzwiller_coefficients_LDA(self):
        ''' Initialize local Gutzwiller coefficients with their optimal LDA values. '''
        sys.exit('ERROR: initialize_gutzwiller_coefficients_LDA is not implemented. Exit.')

    cdef void initialize_gutzwiller_coefficients_random(self):
        ''' Initialize local Gutzwiller coefficients with random complex numbers. '''
        for i_site in range(self.N_sites):
            for n in range(self.nmax + 1):
                self.f[i_site, n] = random.uniform(-1.0, 1.0) + 1.0j * random.uniform(-1.0, 0.1)
        self.normalize_coefficients_all_sites()

    @cython.cdivision(True)
    cdef void initialize_lattice(self):
        ''' Define lattice (site coordinates and table of neighbors). '''
        cdef int i_site, x, y
        cdef int [:] xy = np.zeros(2, dtype=np.int32)
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
                # Case 1: Periodic boundary conditions (PBC):
                if self.OBC == 0:
                    self.N_nbr[i_site] = 4
                    self.nbr[i_site, 0] = self.xy2i((x - 1 + self.L) % self.L, y)
                    self.nbr[i_site, 1] = self.xy2i(x, (y - 1 + self.L) % self.L)
                    self.nbr[i_site, 2] = self.xy2i((x + 1) % self.L, y)
                    self.nbr[i_site, 3] = self.xy2i(x, (y + 1) % self.L)
                # Case 2: Open boundary conditions (OBC):
                else:
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
            sys.exit('ERROR: By now only D=1 and D=2 are implemented, not D=%i. Exit.' % self.D)

    @cython.cdivision(True)
    cpdef void i2xy(self, int _i, int [:] xy):
        ''' For a 2D lattice, convert between 1D and 2D representation of site. '''
        xy[0] = _i // self.L
        xy[1] = _i - (_i // self.L) * self.L

    cpdef int xy2i(self, int _x, int _y):
        ''' For a 2D lattice, convert between 2D and 1D representation of site. '''
        return  _x * self.L + _y

    cpdef void initialize_trap(self):
        cdef double r_sq, r
        cdef int i_site, i_dim
        self.mu_local[:] = 0.0
        for i_site in range(self.N_sites):
            r_sq = 0.0
            for i_dim in range(self.D):
                r_sq += c_pow(self.site_coords[i_site, i_dim] - self.trap_center, 2)
            r = c_sqrt(r_sq)
            self.mu_local[i_site] = self.mu - self.VT * c_pow(r, self.alphaT)

    def print_parameters(self):
        print()
        print('PARAMETERS')
        print('  # Lattice:')
        print('    D:              %i' % self.D)
        print('    L:              %i' % self.L)
        print('    OBC:            %i' % self.OBC)
        print('  # Trap:')
        print('    VT:             %.8f' % self.VT)
        print('    alphaT:         %.8f' % self.alphaT)
        print('    trap_center:    %.8f' % self.trap_center)
        print('  # Bose-Hubbard:')
        print('    J:              %.8f' % self.J)
        print('    U:              %.8f' % self.U)
        print('    Vnn:            %.8f' % self.Vnn)
        print('  # Gutzwiller')
        print('    nmax:           %i' % self.nmax)
        print()

    def get_gutzwiller_coefficients_at_one_site(self, int i_site):
        return np.array(self.f[i_site, :])

    def print_gutzwiller_coefficients_at_one_site(self, int i_site):
        for n in range(self.nmax + 1):
            print('%i Re(f)=%+.10f Im(f)=%+.10f abs(f)=%.12f' % (n, c_real(self.f[i_site, n]), c_imag(self.f[i_site, n]), c_abs(self.f[i_site, n])))

    def save_gutzwiller_coefficients_at_one_site(self, int i_site, filename):
        with open(filename, 'w') as out:
            out.write('# site=%i\n' % i_site)
            out.write('# n, Re(f_n), Im(f_n), |f_n| \n')
            for n in range(self.nmax + 1):
                out.write('%3i %+.10e %+.10e %.10e\n' % (n, c_real(self.f[i_site, n]), c_imag(self.f[i_site, n]), c_abs(self.f[i_site, n])))

    def load_config(self, datafile):
        ''' Load full configuration from a file. '''
        new_f = np.loadtxt(datafile).view(complex)
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
        ''' Store full configuration in a file. '''
        np.savetxt(datafile, np.array(self.f).view(float))

    def save_densities(self, datafile):
        ''' Store local total/condensed densities on a file. '''
        np.savetxt(datafile, np.array(self.f).view(float))
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

    cpdef double [:] compute_center_of_mass(self):
        self.x_com[:] = 0.0
        cdef int i_site, i_dim
        for i_dim in range(self.D):
            for i_site in range(self.N_sites):
                self.x_com[i_dim] += self.site_coords[i_site, i_dim] * self.density[i_site]
            self.x_com[i_dim] /= self.N
        return self.x_com[:]

    @cython.cdivision(True)
    cpdef double compute_rsq(self):
        cdef int i_site, i_dim
        cdef double r_sq
        r_sq = 0.0
        for i_site in range(self.N_sites):
            for i_dim in range(self.D):
                r_sq += self.density[i_site] * c_pow(self.site_coords[i_site, i_dim] - self.trap_center, 2)
        r_sq /= self.N
        return r_sq

    cpdef int count_MI_particles(self, double tol_MI=0.04):
        cdef int i_site, n, N_MI
        N_MI = 0
        for i_site in range(self.N_sites):
            for n in range(1, self.nmax + 1):
                if c_pow(c_abs(self.f[i_site, n]), 2) > 1.0 - tol_MI:
                    N_MI += n
        return N_MI

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
        ''' Re-computes self.bmean, self.sum_bmeans and self.N_cond. '''
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
        ''' Re-computes self.density and self.N. '''
        cdef int i_site, n
        self.N = 0.0
        self.density[:] = 0.0
        for i_site in range(self.N_sites):
            for n in range(0, self.nmax + 1):
                self.density[i_site] += c_pow(c_abs(self.f[i_site, n]), 2) * n
            self.N += self.density[i_site]

    cpdef void update_energy(self):
        ''' Re-computes self.E (NOTE: This requires up-to-date self.bmean and self.density). '''
        cdef int i_site, j_site, j_nbr, n, m
        cdef double f_n_square, sum_density_on_neighbors
        self.E = 0.0
        for i_site in range(self.N_sites):
            # Nearest-neighbor hopping
            for j_nbr in range(self.N_nbr[i_site]):
                j_site = self.nbr[i_site, j_nbr]
                self.E -= self.J * c_real(c_conj(self.bmean[i_site]) * self.bmean[j_site])
            # On-site repulsion and local chemical potential
            for n in range(self.nmax + 1):
                f_n_square = c_pow(c_abs(self.f[i_site, n]), 2)
                self.E += 0.5 * self.U * f_n_square * n * (n - 1)
                self.E -= self.mu_local[i_site] * f_n_square * n
            # Nearest-neighbor interaction
            sum_density_on_neighbors = 0.0
            for j_nbr in range(self.N_nbr[i_site]):
                sum_density_on_neighbors += self.density[self.nbr[i_site, j_nbr]]
            self.E += 0.5 * self.Vnn * self.density[i_site] * sum_density_on_neighbors
 
    cdef void one_sequential_time_step(self, double complex dtau, int normalize_at_each_step=1, int update_variables=1):
        cdef int i_site, n, m, j_nbr
        cdef double complex old_bmean, diff_bmean
        cdef double sum_density_on_neighbors

        cdef double N_new = 0.0

        for i_site in range(self.N_sites):

            # Compute sum of densities over neighbors
            sum_density_on_neighbors = 0.0
            for j_nbr in range(self.N_nbr[i_site]):
                sum_density_on_neighbors += self.density[self.nbr[i_site, j_nbr]]

            # Build matrix
            self.M[:, :] = 0.0
            for m in range(self.nmax + 1):
                self.M[m, m] += 0.5 * self.U * m * (m - 1.0) - self.mu_local[i_site] * m + 0.5 * self.Vnn * sum_density_on_neighbors * m
                if m < self.nmax:
                    self.M[m + 1, m] -= self.J * self.sum_bmeans[i_site] * c_sqrt(m + 1)
                    self.M[m, m + 1] = c_conj(self.M[m + 1, m])

            # Update on-site coefficients
            self.exp_M = scipy.linalg.expm(1.0j * dtau * np.asarray(self.M))
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

            # Update on-site density
            self.density[i_site] = 0.0
            for n in range(0, self.nmax + 1):
                self.density[i_site] += c_pow(c_abs(self.f[i_site, n]), 2) * n
            N_new += self.density[i_site]

        # Update total number of particles
        self.N = N_new

        # Update energy
        if update_variables == 1:
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
        if trap_center > 0:
            self.trap_center = trap_center
        elif trap_center == -1:
            if self.L % 2 == 0:
                self.trap_center = (self.L - 1) // 2 + 0.5
            else:
                self.trap_center = (self.L - 1) // 2
        else:
            sys.exit('[update_trap_center] ERROR: Unrecognized value of trap_center=%f. Exit.' % trap_center)
        self.initialize_trap()

    cpdef int set_mu_via_bisection(self,
                                    double Ntarget=0.0, double mu_min=-3.0, double mu_max=3.0, double tol_N=0.1,
                                    double complex dtau_times_J=1.0, int nsteps=100, int Verbose=1):
        cdef double f_VTmin, f_VTmid, f_VTmax, df
        cdef double mu_mid = 0.5 * (mu_min + mu_max)   # This is only to avoid a compile-time warning ("mu_mid’ may be used uninitialized..")
        cdef double complex dtau = dtau_times_J / self.J * 1.0j
        cdef int bisection_iterations = 0

        self.update_mu(mu_min)
        self.initialize_gutzwiller_coefficients_random()
        self.many_time_steps(dtau, nsteps=nsteps)
        f_min = self.N - Ntarget
        if f_min > 0.0:
            sys.exit('[bisection] ERROR: mu_min=%.8f, Ntarget=%.8f, N_min=%.8f' % (mu_min, Ntarget, self.N))
        if Verbose == 1:
            print('[bisection] mu_min=%.8f, N_min=%.8f, Ntarget=%.8f' % (mu_min, self.N, Ntarget))

        self.update_mu(mu_max)
        self.initialize_gutzwiller_coefficients_random()
        self.many_time_steps(dtau, nsteps=nsteps)
        f_max = self.N - Ntarget
        if f_max < 0.0:
            sys.exit('[bisection] ERROR: mu_max=%.8f, Ntarget=%.8f, N_max=%.8f' % (mu_max, Ntarget, self.N))
        if Verbose == 1:
            print('[bisection] mu_max=%.8f, N_max=%.8f, Ntarget=%.8f' % (mu_max, self.N, Ntarget))

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
            if Verbose == 1:
                print('[bisection] (%02i) mu=(%.8f, %.8f), N=(%.8f, %.8f), Ntarget=%.8f' % (bisection_iterations, mu_min, mu_max, f_min + Ntarget, f_max + Ntarget, Ntarget))

        self.update_mu(mu_mid)
        self.initialize_gutzwiller_coefficients_random()
        self.many_time_steps(dtau, nsteps=nsteps)
        if abs(self.N - Ntarget) > tol_N:
            sys.exit('[bisection] ERROR: mu=%.8f, Ntarget=%.8f, N=%.8f' % (self.mu, Ntarget, self.N))

        return bisection_iterations

    cpdef int set_VT_via_bisection(self,
                                    double Ntarget=0.0, double VT_min=0.001, double VT_max=1.0, double tol_N=0.1,
                                    double complex dtau_times_J=1.0, int nsteps=100, int Verbose=1):
        cdef double f_VTmin, f_VTmid, f_VTmax, df
        cdef double VT_mid = 0.5 * (VT_min + VT_max)   # This is only to avoid a compile-time warning ("VT_mid’ may be used uninitialized..")
        cdef double complex dtau = dtau_times_J / self.J * 1.0j
        cdef int bisection_iterations = 0

        self.update_VT(VT_min)
        self.initialize_gutzwiller_coefficients_random()
        self.many_time_steps(dtau, nsteps=nsteps)
        f_VTmin = self.N - Ntarget
        if f_VTmin < 0.0:
            sys.exit('[bisection] ERROR: VT_min=%f, Ntarget=%f, N_VTmin=%f' % (VT_min, Ntarget, self.N))
        if Verbose == 1:
            print('[bisection] VT_min=%f, N_VTmin=%f, Ntarget=%f' % (VT_min, self.N, Ntarget))

        self.update_VT(VT_max)
        self.initialize_gutzwiller_coefficients_random()
        self.many_time_steps(dtau, nsteps=nsteps)
        f_VTmax = self.N - Ntarget
        if f_VTmax > 0.0:
            sys.exit('[bisection] ERROR: VT_max=%f, Ntarget=%f, N_VTmax=%f' % (VT_max, Ntarget, self.N))
        if Verbose == 1:
            print('[bisection] VT_max=%f, N_VTmax=%f, Ntarget=%f' % (VT_max, self.N, Ntarget))

        df = 10.0 * tol_N
        while abs(df) > tol_N:
            bisection_iterations += 1
            VT_mid = 0.5 * (VT_min + VT_max)
            self.update_VT(VT_mid)
            self.initialize_gutzwiller_coefficients_random()
            self.many_time_steps(dtau, nsteps=nsteps)
            f_VTmid = self.N - Ntarget

            if f_VTmid > 0.0:
                VT_min = VT_mid
                f_VTmin = f_VTmid
            else:
                VT_max = VT_mid
                f_VTmax = f_VTmid
            df = f_VTmin - f_VTmax
            if Verbose == 1:
                print('[bisection] (%02i) VT=(%f, %f), N=(%f, %f), Ntarget=%f' % (bisection_iterations, VT_min, VT_max, f_VTmin + Ntarget, f_VTmax + Ntarget, Ntarget))
            assert df > 0.0

        self.update_VT(VT_mid)
        self.initialize_gutzwiller_coefficients_random()
        self.many_time_steps(dtau, nsteps=nsteps)
        if abs(self.N - Ntarget) > tol_N:
            sys.exit('[bisection] ERROR: VT=%f, Ntarget=%f, N=%f' % (self.VT, Ntarget, self.N))

        return bisection_iterations
