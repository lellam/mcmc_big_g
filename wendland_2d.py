import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, csc_matrix
import matplotlib.pyplot as plt
from sqrtinv_approx import *
import time

def wendland_kernel_d(x1, x2, alpha=1.0):
    """Compute wendland_kernel matrix with width alpha

    x1 is NxD, x2 is MxD, output is NxM
    """
    #Find scaled r within a ball of radius alpha
    tree1=cKDTree(x1)
    tree2=cKDTree(x2)
    A = cKDTree.sparse_distance_matrix(tree1,tree2,alpha, output_type='coo_matrix')
    A.data = A.data/alpha

    #Construct Wendland kernel
    d=2
    j = 3 + np.floor(d*0.5)
    A.data = (1.-A.data)**(j+2)*((j*j + 4.*j + 3.)*A.data**2 + (3*j + 6.)*A.data + 3.)/3.
    A = csc_matrix(A)
    return A
    
def pts_of_grid1d(x):
    """ Returns N points on 1d line """
    return x[:, np.newaxis].T
    
def pts_of_grid2d(x):
    """ Returns N^2 points on 2d grid """
    N = len(x)
    pts1 = []
    for xx in x:
        for yy in x:
            pts1.append([xx, yy])
    return np.reshape(pts1, (N*N, 2))

def main():

    nN = 50
    nM = 50

    ptsN = pts_of_grid2d(np.linspace(0, 1, nN))
    ptsM = pts_of_grid2d(np.linspace(0, 1, nM))

    A = wendland_kernel_d(ptsN, ptsM, 2.)
    
    print("Kernel built")
    t0 = time.time()
    b = np.random.normal(0, 1, A.shape[0])
    ra = SquareRootApproximation(20)
    S1 = np.reshape(ra.sqrt_product(A, b), (nN, nN))
    t1 = time.time()
    print(t1-t0)
    S2 = np.reshape(np.random.multivariate_normal(np.zeros(nN*nM), A.todense()), (nN, nM))
    t2 = time.time()
    print(t2-t1)
    plt.subplot(1, 2, 1)
    plt.imshow(S1, interpolation='nearest')
    plt.subplot(1, 2, 2)
    plt.imshow(S2, interpolation='nearest') 
    plt.show()

    return
    
if __name__ == "__main__":
    main()