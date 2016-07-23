from kernels import *

def generate_data(x, theta):
	"""Generate observation data."""
	S = kernel(x, theta)
	y = np.random.multivariate_normal(np.zeros(np.size(x)), S)
	return y
	
def read_data():
    """ Read observation data """
    x = np.loadtxt("data/x.dat")
    y = np.loadtxt("data/y.dat")
    return(x,y)