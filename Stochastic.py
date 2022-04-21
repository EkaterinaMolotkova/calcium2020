# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 16:20:46 2022

@author: XPS
"""
import pandas as pd 
import numpy as np
import scipy.stats as st
# import biocircuits

import time
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib


v1 = 6.0
v2 = 0.11
v3 = 0.9
C0 = 2.5
c1 = 0.185
K3 = 0.04
d1 = 0.13
d2 = 1.049
d3 = 0.9434
d5 = 0.08234
a2 = 0.2

I = 0.6

ca_norm = 100000
h_norm  = 100000


def Jleak(u): return v2*(C0-(1.0+c1)*u[0])
def Jpump(u): return v3*u[0]*u[0]/(K3*K3+u[0]*u[0])
def minf(u) : return (I*u[0])/(I+d1)/(u[0]+d5)
def Q2(u)   : return (I+d1)*d2/(I+d3)
def hinf(u) : return Q2(u)/(Q2(u)+u[0])
def tauh(u) : return 1.0/a2/(Q2(u)+u[0])
def Jchan(u): return v1*(minf(u)**3)*(u[1]**3)*(C0-(1.0+c1)*u[0])

def Jchan_plus(u) : return v1*(minf(u)**3)*(u[1]**3)*C0
def Jleak_plus(u) : return v2*C0
def Jchan_min(u)  : return v1*((minf(u)*u[1])**3)*(1.0+c1)*u[0]
def Jleak_min(u)  : return v2*(1.0+c1)*u[0]

def dotu(x,t):
    dx = np.zeros(2)
    dx[0] = Jchan(x)+Jleak(x)-Jpump(x)
    dx[1] = (hinf(x)-x[1])/tauh(x)
    return dx

def simple_propensity(propensities, population, t, *args):
    """Updates an array of propensities given a set of parameters
    and an array of populations.
    """
    # Unpack population
    c, h = population
    
    c = c/ca_norm
    h = h/h_norm
    x = np.array([c,h])
    
    # Update propensities
    propensities[0] = (Jleak_plus(x) + Jchan_plus(x))         *ca_norm
    propensities[1] = (Jleak_min(x) + Jchan_min(x) + Jpump(x))*ca_norm 
    propensities[2] = (hinf(x)/tauh(x))                       *h_norm
    propensities[3] = (x[1]/tauh(x))                          *h_norm

    
def sample_discrete_scipy(probs):
    """Randomly sample an index with probability given by probs."""
    return st.rv_discrete(values=(range(len(probs)), probs)).rvs()

def sample_discrete(probs):
    """Randomly sample an index with probability given by probs."""
    # Generate random number
    q = np.random.rand()
    
    # Find index
    i = 0
    p_sum = 0.0
    while p_sum < q:
        p_sum += probs[i]
        i += 1
    return i - 1

def gillespie_draw(propensity_func, propensities, population, t, args=()):
    """
    Draws a reaction and the time it took to do that reaction.

    Returns
    -------
    rxn : int
        Index of reaction that occured.
    time : float
        Time it took for the reaction to occur.
    """
    # Compute propensities
    propensity_func(propensities, population, t, *args)
    
    # Sum of propensities
    props_sum = propensities.sum()
#     print('prop sum', props_sum)
    
    # Compute next time
    time_ = np.random.exponential(1.0 / props_sum)
    
    # Compute discrete probabilities of each reaction
    rxn_probs = propensities / props_sum
    
    # Draw reaction from this distribution
    rxn = sample_discrete(rxn_probs)
    
    return rxn, time_

def gillespie_ssa(propensity_func, update, population_0, time_points, args=()):
    """
    Uses the Gillespie stochastic simulation algorithm to sample
    from probability distribution of particle counts over time.   

    Returns
    -------
    sample : ndarray, shape (num_time_points, num_chemical_species)
        Entry i, j is the count of chemical species j at time
        time_points[i].
    """

    # Initialize output
    pop_out = np.empty((len(time_points), update.shape[1]), dtype=np.int)

    # Initialize and perform simulation
    i_time = 1
    i = 0
    dt = 0.0001
    t = time_points[0]
    population = population_0.copy()
    pop_out[0,:] = population
    propensities = np.zeros(update.shape[0])
    while i < len(time_points):
        while t < time_points[i_time]:
            # draw the event and time step
            event, dt = gillespie_draw(propensity_func, propensities, population, t, args)

            # Update the population
            population_previous = population.copy()
            population += update[event,:]
            # Increment time
            t += dt

        # Update the index
        i = np.searchsorted(time_points > t, True)
        
        # Update the population
        pop_out[i_time:min(i,len(time_points))] = population_previous
        
        # Increment index
        i_time = i
                           
    return pop_out

def plotting(samples):
    plt.figure(figsize=(10,5), dpi=200)
    ax1 = plt.axes()
    matplotlib.rcParams.update({'font.size': 10})
    ax1.set_xlabel("Время [сек]", fontsize=10)
    ax1.set_ylabel("Кальций [мкМ]", fontsize=10)
    ax1.tick_params(labelsize=10)
    # ax1.set_ylim([-0.08,1.66])
    # ax1.set_xlim([0,120])
    max_tlim = len(samples)
    min_tlim = 0
    # t = np.linspace(min_tlim, int(max_tlim), int(max_tlim-min_tlim))
    t = np.linspace(min_tlim, 500, int(max_tlim-min_tlim))
    ax1.plot(t, samples[min_tlim:max_tlim,0]/ca_norm, lw=1, color="black")
    ax1.set_xticks(np.linspace(min_tlim, 500, int(21)))
    # ax1.set_xticks(np.linspace(0,1000,21))
    
    ax2 = ax1.twinx()
    ax2.plot(t, samples[min_tlim:max_tlim,1]/h_norm, lw=1, color="red")
    ax2.set_ylabel("Инактивация, h", fontsize=10, color='r')
    ax2.tick_params('y', colors='r')
    ax2.set_xticks(np.linspace(0,500,21))
    ax2.set_ylim([0,2])
    ax1.set_ylim([0,2])
    # ax2.set_xlim([0,120])
    
    # plt.subplots_adjust(left=0.21, bottom=0.22, right=0.80, top=0.95)
    # plt.show()
    pl_path = r"C:/Users/XPS/Desktop/фф/6 сем/научка/Stochastic/St1,c={}, h={}, I={}, K3={}.png".format(ca_norm, h_norm,I, K3)
    plt.savefig(pl_path)
    
def main():
    # Column 0 is change in c, column 1 is change in h
    simple_update = np.array([[1, 0],   
                              [-1, 0],  
                              [0, 1],   
                              [0, -1]], 
                             dtype=np.int)
    # Specify parameters
    args = (d1,d2,d3,d5,v1,v2,v3,C0,a2,c1,K3,I)
    time_points = np.linspace(0, int(500), int(10000))
    population_0 = np.array([ca_norm/2, h_norm/2], dtype=int)
    
    # Seed random number generator for reproducibility
    np.random.seed(42)
    
    # Initialize output array
    samples = np.empty((len(time_points), 2), dtype=int)
    
    # Run the calculations
    
    #
    start = time.time()
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start Time is :", current_time)
    
    samples[:,:] = gillespie_ssa(simple_propensity, simple_update, population_0, time_points, args=args)
    
    end = time.time()
    print(end - start, 'seconds')
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("End Time is :", current_time)
    
    file_path = r"C:/Users/XPS/Desktop/фф/6 сем/научка/Stochastic/St1,c={}, h={}, I={}, K3={}.csv".format(ca_norm, h_norm,I, K3)
    pd.DataFrame(samples[:,:]).to_csv(file_path)
    return samples

file_path = r"C:/Users/XPS/Desktop/фф/6 сем/научка/Stochastic/St1,c={}, h={}, I={}, K3={}.csv".format(ca_norm, h_norm,I, K3)
samples = np.loadtxt(file_path, dtype=float, delimiter=',', skiprows=1, usecols = (1,2))

# samples = main()

plotting(samples)
