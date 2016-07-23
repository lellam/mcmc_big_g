import numpy as np
import matplotlib.pyplot as plt
import time
from sqrtinv_approx import *
from kernels import *
from densities import *
from observation_data import *

l = 0.1
sigma0 = 100.
data_n = 100
x, y = read_data()

mcmc_step = .05
prop_n = 0
accept_n = 0
ra = SquareRootApproximation(20)

target = augmented_gaussian_density(x, y, sigma0)

def theta_update(theta, z, mhk_n):

    global mcmc_step
    global prop_n
    global accept_n
    
    prev_ln_prob = target(theta, z)
    
    for i in range(mhk_n):
        prop = theta + np.random.normal(0, mcmc_step)
        prop_ln_prob = target(prop, z)
        ln_ap = prop_ln_prob - prev_ln_prob
        ln_u = np.log(np.random.uniform(0, 1))
        prop_n+=1
        if(ln_u < ln_ap):
            theta = prop
            accept_n+=1
            prev_ln_prob = prop_ln_prob

    return theta


def z_update(theta, z):
    Knm = kernel(x, np.exp(theta))
    rhs = np.random.normal(0, 1, data_n)
    ret =  ra.sqrt_inverse_product(Knm, rhs)
    return ret


def main():

    gibbs_n = 10000
    gibbs_burn = 1000
    gibbs_thin = 1
    
    z = np.zeros(data_n)
    theta = np.log(.1)
    samples = []
    z_samples = []
    
    t_start = time.time()
    
    for i in range(gibbs_n+gibbs_burn):
    
        z = z_update(theta, z)
        theta = theta_update(theta, z, 1)
        
        if(i%gibbs_thin==0 and i > gibbs_burn):
            samples.append(theta)
            z_samples.append(z[0])
    
        """ Print """    
        if(i == 0):
            print("Burning...")
            
        if(i >= gibbs_burn and i%100 == 0):
            ar = accept_n/float(prop_n)
            progress = int(float(i-gibbs_burn)/float(gibbs_n)*100)
            print(str(progress) + "% complete with ar " + str(ar))
    
    t_end = time.time()
    print("Time taken is " +  str(t_end - t_start))
    
    """ Summary statistics """
    samples = np.array(samples)
    accept_rate = float(accept_n)/float(gibbs_n)
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