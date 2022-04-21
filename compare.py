# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 22:48:37 2022

@author: XPS
"""
import matplotlib.pyplot as plt
import matplotlib
# import pandas as pd 
import numpy as np



def plotting(x, samples):
    plt.figure(figsize=(25,5), dpi=200)
    ax1 = plt.axes()
    matplotlib.rcParams.update({'font.size': 10})
    ax1.set_xlabel("Время [сек]", fontsize=10)
    ax1.set_ylabel("Детерменистика, c [мкМ]", fontsize=10)
    ax1.tick_params(labelsize=10)
    ax1.set_ylim([-0.08,1.66])
    ax1.set_xlim([0,100])
    ax1.set_xticks(np.linspace(0,100,21))

    shift = 0
    t = np.linspace(0.0, 5000.0, 5000)
    ax1.plot(t, x[:5000,0], lw=1, color="black")
    ax2 = ax1.twinx()
    ax2.plot(t, samples[shift:5000+shift,0]/ca_norm, lw=1, color="blue")
    ax2.set_ylabel("Стохастика, c [мкМ]", fontsize=10, color='b')
    ax2.tick_params('y', colors='b')
    ax2.set_ylim([-0.08,1.66])
    ax2.set_xlim([0,100])


    plt.subplots_adjust(left=0.21, bottom=0.22, right=0.80, top=0.95)
    plt.show()
    pl_path = r"C:/Users/XPS/Desktop/фф/6 сем/научка/Determenistic/Compare1,c={}, h={},I={},K3={}.png".format(ca_norm, h_norm,I, K3)
    plt.savefig(pl_path)
    
I = 0.6
K3 = 0.04

ca_norm = 100000
h_norm  = 100000

st_path = r"C:/Users/XPS/Desktop/фф/6 сем/научка/Stochastic/St,c={}, h={}, I={}, K3={}.csv".format(ca_norm, h_norm,I, K3)
det_path = r"C:/Users/XPS/Desktop/фф/6 сем/научка/Determenistic/Det, I={}, K3={}.csv".format(I, K3)

x    = np.loadtxt(det_path, dtype=float, delimiter=',', skiprows=1, usecols = (1,2))
samp = np.loadtxt(st_path,  dtype=float, delimiter=',', skiprows=1, usecols = (1,2))
print('x shape ',    x.shape)
print('samp shape ', samp.shape)

plotting(x,samp)