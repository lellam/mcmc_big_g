import numpy as np
from kernels import *
from scipy.sparse import issparse
import scipy.sparse.linalg as splinalg

class small_gaussian_density(object):
	"""A model for a small to medium sized realization of a Gaussian process.  Methods assume that
	the Cholesky decomposition can be computed and stored.
	
	n:	Dimension of the Gaussian
	x:	An array of dimension n of the spatial locations
	y:	An array of dimension n of the observed values
	"""

	def __init__(self, x, y, sigma):
		"""Initialize spatial locations *x* and observed values *y*."""
		self.x = np.array(x)
		self.y = np.array(y)
		self.n = np.size(x)
		self.sigma = sigma
		assert(self.n == np.size(y))
		return

	def ln_likelihood(self, theta):
		"""Evaluates and returns p(y | theta)."""
		Knm=kernel(self.x, np.exp(theta))
		if(issparse(Knm)):
		    Knm = Knm.todense()
		L = scipy.linalg.cho_factor(Knm, lower=True)
		a = scipy.linalg.cho_solve(L, self.y)
		ret = -0.5*np.dot(self.y, a)
		ret += -np.log(np.diag(L[0])).sum()
		ret += -0.5*self.n*np.log(2.*np.pi)
		return ret
		
	def ln_prior(self, theta):
		"""Evaluates and returns p(theta)."""
		ret = -0.5*theta*theta/(self.sigma*self.sigma)
		ret += -np.log(self.sigma)
		ret += -0.5*np.log(2.*np.pi)
		return ret
		
	def ln_posterior(self, theta):
		"""Evalates and returns p(theta | y)."""
		return self.ln_likelihood(theta) + self.ln_prior(theta)
		
	def __call__(self, theta):
		return self.ln_posterior(theta)


class augmented_gaussian_density(object):

	def __init__(self, x, y, sigma):
		"""Initialize spatial locations *x* and observed values *y*."""
		self.x = np.array(x)
		self.y = np.array(y)
		self.n = np.size(x)
		self.sigma = sigma
		assert(self.n == np.size(y))
		return

	def ln_likelihood(self, theta, z):
		"""Evaluates and returns p(y | theta)."""
		S = kernel(self.x, np.exp(theta))
		a = splinalg.cg(S, self.y, tol=1e-5)[0]
		ret = -0.5*(np.dot(self.y, a)+np.dot(z, S.dot(z)))
		return ret
		
	def ln_prior(self, theta):
		"""Evaluates and returns p(theta)."""
		ret = -0.5*theta*theta/(self.sigma*self.sigma)
		return ret
		
	def ln_posterior(self, theta, z):
		"""Evalates and returns p(theta | y)."""
		return self.ln_likelihood(theta, z) + self.ln_prior(theta)
		
	def __call__(self, theta, z):
		return self.ln_posterior(theta, z)