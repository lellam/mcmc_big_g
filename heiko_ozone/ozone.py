import os
from scipy.io.matlab.mio import loadmat
from scipy.sparse.construct import eye
from scipy.sparse.csr import csr_matrix
from scikits.sparse.cholmod import cholesky
import numpy as np

import matplotlib.pyplot as plt

def load_ozone_data(folder):
    # actual observations
    y = loadmat(folder + "y.mat")["y"][:, 0]
    assert(y.ndim == 1)
    
    # triangulation of globe
    A = loadmat(folder + "A.mat")["A"]
    
    return y, A

def create_Q_matrix(kappa, folder, ridge=1e-6):
    GiCG = loadmat(folder + "GiCG.mat")["GiCG"]
    G = loadmat(folder + "G.mat")["G"]
    C0 = loadmat(folder + "C0.mat")["C0"]
    
    Q = GiCG + 2 * (kappa ** 2) * G + (kappa ** 4) * C0
    return Q + eye(Q.shape[0], Q.shape[1]) * ridge

if __name__ == "__main__":
    # posterior mode of marginals in RR paper
    kappa = 2.**(-3)
    tau = 2.**(-11.59)
    
    # folder with data files
    folder = os.sep.join([os.path.expanduser("~"), "data", "ozone"]) + os.sep
    
    # regulariser on diagonal
    ridge=1e-6
    
    Q = create_Q_matrix(kappa, folder, ridge)
    y,A = load_ozone_data(folder)
    print "Size of y", y.shape
    print "Size of A", A.shape
    
#     # can have a look at the sparsity pattern
#     plt.spy(csr_matrix(Q))
#     plt.title("Q")
#     plt.show()
    
    # example of an exact solve using scipy
    # this computes a factor of the last term of eq 7.4
    # (Q + tau*A^T *A)^-1 * A^T * y
    AtA = A.T.dot(A)
    M = Q + tau * AtA
    b = A.T.dot(y)
    print "Computing Cholesky"
    factor = cholesky(M)
    print "Backwards solving"
    result = factor.solve_A(b)
    print result
    
    # sample a Gaussian with covariance M and zero mean, need to respect permutation
    # see https://pythonhosted.org/scikits.sparse/cholmod.html#scikits.sparse.cholmod.Factor.L
    u = np.random.randn(Q.shape[0])
    L = factor.L()
    P = factor.P()
    z = L.dot(u)[P]
    print z