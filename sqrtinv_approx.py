import numpy as np
import time
from scipy import special, linalg, sparse
from kernels import *
from sqrtinv_multi_shift_solver import *


class SquareRootApproximation(object):
    """ Rational approximations of A^1/2x = b and A^-1/2x = b. """

    def __init__(self, n_shift):
        self.n_shift = n_shift
        self.shifts, self.weights = self.compute_rational_approximation()
        self.s0 = self.shifts[0]
        self.shifts = (self.shifts-self.s0)*-1
        self.cg_solver = ShiftConjugateGradient(self.shifts)


    def compute_rational_approximation(self):
        # Compute shifts and quadrature weights
        m, M = 10e-6, 10e6
        k2 = m / M
        kp = special.ellipk(1.0 - k2)
        t = 1j * np.arange(0.5, self.n_shift) * kp / self.n_shift
        sn, cn, dn, ph = special.ellipj(t.imag, 1-k2)
        cn = 1./cn
        cn *= cn
        shifts = -m*sn*sn*cn
        weights = 2.*cn*dn*kp*np.sqrt(m) / (np.pi * self.n_shift)
        return (shifts, weights)


    def plus_diag(self, A, b, inplace=False):
        """Add vector or scalar b to diagonal of square array A

        By default returns a new matrix. Set inplace=True to do inplace"""
        if not inplace:
            A = A.copy()
        A[np.diag_indices_from(A)] += b
        return A


    def sqrt_product(self, A, b):
        return A.dot(self.sqrt_inverse_product(A, b))


    def sqrt_inverse_product(self, A, b):
        A = self.plus_diag(A, -self.s0)
        xs=self.cg_solver.solve(A, b, len(b), tol=1e-5)
        ret = np.matmul(self.weights, xs)
        return ret

	
def main():
    """ Test function - error should be close to zero """

    #Construct (dense) sparse matrix - must be able to handle sparse matrices
    n=100
    theta = np.log(.1)
    x = np.linspace(0, 1, n)
    b = np.random.normal(0, 1, n)
    A = kernel3(x, np.exp(theta))
    A = sparse.csc_matrix(A)
   
    # Time and compute sqrt inverse product
    t_start = time.time()
    ra = SquareRootApproximation(20)
    S = ra.sqrt_inverse_product(A, b)
    t_end = time.time()

    # Convert to dense matrix for the test case
    A = A.todense()
    Asqrt = linalg.sqrtm(A)
    S_true = np.linalg.solve(Asqrt, b)

    # Check norms and output
    print("Relative error in sup-norm " + \
        str(np.max(np.abs((S_true-S)/S_true))))
    print("Time taken is " + str(np.round(t_end-t_start, 5)) + " seconds.")
    return


if __name__ == "__main__":
    main()