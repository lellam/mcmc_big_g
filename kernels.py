import numpy as np
import scipy.linalg
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, csc_matrix, hstack, eye

def kernel(space, alpha=1.0):
    return wendland_kernel(space, alpha)

def kernel3(x, l):
	"""Returns covariance matrix with exponential squared covariance function."""
	pairwise_sq_dists = squareform(pdist(np.atleast_2d(x/l).T, 'sqeuclidean'))
	k = np.exp(-pairwise_sq_dists)
	k[np.diag_indices_from(k)] += 10e-4
	return k


def kernel2(x1, hh=1.0):
    """Compute Gaussian kernel matrix with bandwidth hh

    x1 is NxD, x2 is MxD, output is NxM
    
    This is the version that's fast for big D. Although data should be
    centred."""
    x1 /= np.sqrt(hh) 
    psum =  np.sum(x1*x1, 1)
    Knm = np.exp(np.dot(x1, (2*x1.T)) - psum[:,np.newaxis] - psum[:,np.newaxis].T)
    Knm[np.diag_indices_from(Knm)] += 10e-4
    return Knm


def wendland_kernel(space, alpha=1.0):
    """ Returns compressed sparse column matrix with Wendland kernel """
    #Find pairs within a ball of radius alpha
    pts = np.tile(space, (1, 1)).T
    tree=cKDTree(pts)
    A = cKDTree.sparse_distance_matrix(tree,tree,alpha, output_type='coo_matrix')
    A.data = A.data/alpha

    #Construct Wendland kernel
    q = 2
    j = q+1
    A.data = (1.-A.data)**(j+2)*((j*j + 4.*j + 3.)*A.data**2 + (3*j + 6.)*A.data + 3.)/3.
    A = csc_matrix(A)
    A[np.diag_indices_from(A)] += 10e-4
    return A
    
    
def gauss_Knm(x1, x2=None, hh=1.0):
    """Compute Gaussian kernel matrix with bandwidth hh

    x1 is NxD, x2 is MxD, output is NxM
    
    This is the version that's fast for big D. Although data should be
    centred."""

    if x2 is None:
        x2 = x1 # Can't be bothered to caching sum(x1.*x1,1)

    if hh != 1.0:
        # Scaling data first is O(N), scaling square dists is O(N^2)
        x1 /= np.sqrt(hh) 
        x2 /= np.sqrt(hh) 

    return np.exp(np.dot(x1, (2*x2.T)) - np.sum(x1.T*x1.T, 0)[:,np.newaxis] \
            - np.sum(x2.T*x2.T, 0)[:,np.newaxis].T)

def main():
    N = 5
    M = 5
    D = 1
    hh = 0.7
    x1 = np.random.randn(N, D)
    x2 = np.random.randn(M, D)

    x1 = np.atleast_2d(np.linspace(0, 1, N)).T
    x2 = x1.copy()

    # Dumb computation for comparison
    K1 = np.zeros((N, M))
    for nn in range(N):
        for mm in range(M):
            rr = x1[nn,:] - x2[mm,:]
            K1[nn, mm] = np.exp(-np.dot(rr, rr) / hh)

    K2 = gauss_Knm(x1, x2, hh)

    print(np.round(K2, 2))

    print(np.all(np.isclose(K1, K2)))
    return
    
if __name__ == "__main__":

    
    main()


