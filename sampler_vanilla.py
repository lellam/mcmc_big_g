import numpy as np
import matplotlib.pyplot as plt
from densities import *
from observation_data import *
import time



def main():
	
	"""Set true value"""
	l = 0.1
	
	"""MCMC parameters"""
	mcmc_n = 10000
	mcmc_burn = 1000
	mcmc_thin = 1
	mcmc_step = 0.1
	
	"""Generate observation data."""
	x, y = read_data()
	
	"""Build posterior measure."""
	sigma0 = 100.
	target = small_gaussian_density(x, y, sigma0)
	
	"""Run MCMC"""
	samples = []
	theta = 0.		#initialize from prior
	prev_ln_prob = target(theta)
	accept_i = 0
	
	t_start = time.time()
	
	for i in range(mcmc_n+mcmc_burn):
	
		prop = theta + np.random.normal(0, mcmc_step)
		prop_ln_prob = target(prop)
		ln_ap = prop_ln_prob - prev_ln_prob
		ln_u = np.log(np.random.uniform(0, 1))
		
		if(ln_u < ln_ap):
			theta = prop
			accept_i+=1
			prev_ln_prob = prop_ln_prob
		
		if(i > mcmc_burn and i%mcmc_thin==0):
			samples.append(theta)
		
		""" Print """	
		if(i == 0):
			print("Burning...")
			
		if(i >= mcmc_burn and i%100 == 0):
			ar = accept_i/float(i+1)
			progress = int(float(i-mcmc_burn)/float(mcmc_n)*100)
			print(str(progress) + "% complete with ar " + str(ar))
	
	t_end = time.time()
	print("Time taken is " +  str(t_end - t_start))
	
	samples = np.array(samples)
	accept_rate = float(accept_i)/float(mcmc_n)
	expectation = np.exp(samples).mean()
	cum_mean = np.cumsum(np.exp(samples))/np.arange(1, len(samples)+1)

	print(expectation)
	
	plt.subplot(2, 2, 1)
	plt.plot(x, y)
	
	plt.subplot(2, 2, 2)
	plt.plot(samples)
	
	plt.subplot(2, 2, 3)
	plt.plot(cum_mean)
	
	#plt.show()
	
	return
	
if(__name__=="__main__"):
	main()