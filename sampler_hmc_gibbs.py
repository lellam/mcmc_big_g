import matplotlib.pyplot as plt
from scipy import linalg
from random import randint
from kernels import *
from densities import *
from observation_data import *
import time

l = 0.1
sigma0 = 100.
data_n = 100
x, y = read_data()
#y = generate_data(x, l)
mcmc_step = .1
prop_n_theta = 0
accept_n_theta = 0
prop_n_z = 0
accept_n_z = 0
eps_0 = .1
eps_n = 50


target = augmented_gaussian_density(x, y, sigma0)


def hamiltonian(z, r, S):
	ret = z.dot(S.dot(z))
	ret += np.dot(r, r)
	ret *= .5
	return ret

def grad_E(z, S):
	return S.dot(z)

def theta_update(theta, z, mhk_n):

	global mcmc_step
	global prop_n_theta
	global accept_n_theta
	
	prev_ln_prob = target(theta, z)
	
	for i in range(mhk_n):
		prop = theta + np.random.normal(0, mcmc_step)
		prop_ln_prob = target(prop, z)
		ln_ap = prop_ln_prob - prev_ln_prob
		ln_u = np.log(np.random.uniform(0, 1))
		
		prop_n_theta+=1
		if(ln_u < ln_ap):
			theta = prop
			prev_ln_prob = prop_ln_prob
			accept_n_theta+=1

	return theta
	


def z_update(theta, z, mhk_n):
	global prop_n_z
	global accept_n_z
	
	""" Compute covariance matrix """
	S = kernel(x, np.exp(theta))
	
	for m in range(mhk_n):
		""" Determine leap direction """
		eps = eps_0
		if(np.random.uniform(0, 1)>.5):
			eps*=-1
	
		""" Update momentum variables """
		r = np.random.normal(0, 1, data_n)
		
		""" Leapfrog scheme """
		r_new = r -0.5*eps*grad_E(z, S)
		z_new = z +eps*r_new
		eps_n
		for i in range(eps_n-1):
			r_new += -eps*grad_E(z_new, S)
			z_new += +eps*r_new
		r_new += -0.5*eps*grad_E(z_new, S)
	
		""" Compute acceptance probability and accept/reject"""
		ap = np.exp(hamiltonian(z, r, S)-hamiltonian(z_new, r_new, S))
	
		u = np.random.uniform(0, 1)
		prop_n_z+=1
		if(u < ap):
			z = z_new
			accept_n_z+=1
			
	return z
	
	

def main():

	""" Initialize Gibbs sampler """
	gibbs_n = 10000
	gibbs_burn = 1000
	gibbs_thin = 1
	
	z = np.zeros(data_n)
	theta = 0.
	samples = []
	z_samples = []
	
	t_start = time.time()
	
	""" Iterations of Gibbs sampler """
	for i in range(gibbs_n+gibbs_burn):
	
		z = z_update(theta, z, 10)
		theta = theta_update(theta, z, 1)
	
		""" Store samples """
		if(i%gibbs_thin==0 and i > gibbs_burn):
				samples.append(theta)
				z_samples.append(z[0])
				
		""" Print """	
		if(i == 0):
			print("Burning...")
			
		if(i >= gibbs_burn and i%100 == 0):
			ar1 = accept_n_theta/float(prop_n_theta)
			ar2 = accept_n_z/float(prop_n_z)
			progress = int(float(i-gibbs_burn)/float(gibbs_n)*100)
			print(str(progress) + "% complete with ar " + str(ar1) + ", " + str(ar2))
			
				
	
	t_end = time.time()
	print("Time taken is " +  str(t_end - t_start))
	
	""" Summary statistics """
	samples = np.array(samples)
	expectation = np.exp(samples).mean()
	cum_mean = np.cumsum(np.exp(samples))/np.arange(1, len(samples)+1)

	print(expectation)
	
	plt.subplot(2, 2, 1)
	plt.title("GP")
	plt.plot(x, y)
	
	plt.subplot(2, 2, 2)
	plt.title("$l$")
	plt.plot(samples)
	
	plt.subplot(2, 2, 3)
	plt.title("$l$")
	plt.plot(cum_mean)
	
	plt.subplot(2, 2, 4)
	plt.title("$z_0$")
	plt.plot(z_samples)
	
	plt.show()
	
	return
	
if(__name__=="__main__"):
	main()