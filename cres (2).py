#! /usr/bin/env python
# -*- coding: utf-8 -*-

from scipy.integrate import odeint
from scipy.optimize import fsolve
from scipy.misc import derivative
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import time
from datetime import datetime
import pandas as pd 

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
def Jleak(u): return v2*(C0-(1.0+c1)*u[0])
def Jpump(u): return v3*u[0]*u[0]/(K3*K3+u[0]*u[0])
def minf(u): return (I*u[0])/(I+d1)/(u[0]+d5)
def Q2(u): return (I+d1)*d2/(I+d3)
def hinf(u): return Q2(u)/(Q2(u)+u[0])
def tauh(u): return 1.0/a2/(Q2(u)+u[0])
def Jchan(u): return v1*(minf(u)**3)*(u[1]**3)*(C0-(1.0+c1)*u[0])

def dotu(x,t):
    dx = np.zeros(2)
    dx[0] = Jchan(x)+Jleak(x)-Jpump(x)
    dx[1] = (hinf(x)-x[1])/tauh(x)
    return dx

def isocline2(x,y):
    u = [y,x]
    du = (hinf(u)-u[1])/tauh(u)
    return du
    
def isocline1(x,y):
    u = [y,x]
    du = Jchan(u)+Jleak(u)-Jpump(u)
    return du
    
def f(u):
    return Jchan(u)+Jleak(u)
    
def lol(x,y):
    u = [y,x]
    return (hinf(u)-u[1])/tauh(u)
    
def fun1():
    plt.figure(figsize=(30,10), dpi=200)
    ax1 = plt.axes()
    matplotlib.rcParams.update({'font.size': 10})
    ax1.set_xlabel("Время [сек]", fontsize=10)
    ax1.set_ylabel("Кальций [мкМ]", fontsize=10)
    ax1.tick_params(labelsize=10)
    ax1.set_ylim([-0.08,1.66])
    ax1.set_xlim([0,500])

    global K3, I
    K3 = 0.04
    I = 0.6
    t = np.linspace(0.0, 500.0, 10000)
    x = np.array(odeint(dotu, [0.0,1.0], t)).transpose()
    print('X shape', x.shape)
    ax1.plot(t, x[0], lw=1, color="black")
    ax2 = ax1.twinx()
    ax2.plot(t, x[1], lw=1, color="red")
    ax2.set_ylabel("Инактивация, h", fontsize=10, color='r')
    ax2.tick_params('y', colors='r')
    ax2.set_ylim([-0.08,1.66])
    ax2.set_xlim([0,500])

    plt.subplots_adjust(left=0.21, bottom=0.22, right=0.80, top=0.95)
    # plt.show()
    pl_path = r"C:/Users/XPS/Desktop/фф/6 сем/научка/Determenistic/Det, I={}, K3={}.png".format(I, K3)
    plt.savefig(pl_path)
    return x
    
    
def fun2():
    plt.figure(figsize=(4,3), dpi=80)
    matplotlib.rcParams.update({'font.size': 18})
    plt.xlabel("Кальций [мкМ]", fontsize=18)
    plt.ylabel("Поток [мкМ/сек]", fontsize=18)
    plt.tick_params(labelsize=18)
    
    global K3, I
    K3 = 0.04
    I = 0.4
    plt.xscale("log")
    plt.ylim([0.0,1.0])
    plt.xlim([0.01,1.0])
    x = np.logspace(-2,1,100)
    y = Jpump([x,0])
    plt.plot(x,y,lw=5,color='black')
    
    w = []
    s = 10.0
    for i in x:
        r = fsolve(lol, s, args=i)
        s = r[0]
        w.append(f([i,s]))
           
    plt.plot(x,w,lw=5,color='red')

    w = []
    s = 10.0
    I = 0.0
    for i in x:
        r = fsolve(lol, s, args=i)
        s = r[0]
        w.append(f([i,s]))
           
    plt.plot(x,w,lw=5,color='red',ls='--')

    plt.subplots_adjust(left=0.25, bottom=0.23, right=0.94, top=0.95)
    plt.show()
    
def fun3():
    plt.figure(figsize=(4,3), dpi=80)
    matplotlib.rcParams.update({'font.size': 18})
    plt.xlabel("Кальций [мкМ]", fontsize=18)
    plt.ylabel("Инактивация, h", fontsize=18)
    plt.tick_params(labelsize=18)
    plt.ylim([0.45,1.0])
    plt.xlim([0.02,1.7])
    plt.xscale("log")
    
    global K3, I
    K3 = 0.04
    I = 0.4

    x = np.linspace(0.02,1.7,1000)
    w = []
    s = 0.5
    for i in x:
        r = fsolve(isocline1, s, args=i)
        s = r[0]
        w.append(r[0])       
    plt.plot(x,w,lw=3,color="red",ls="--")
    
    w = []
    s = 1.0
    x = np.linspace(0.0,1.7,1000)
    for i in x:
        r = fsolve(isocline2, s, args=i)
        s = r[0]
        w.append(r[0])       
    plt.plot(x,w,lw=3,color="red")

    r1 = fsolve(dotu, [0.04,0.9],args=0)
    plt.scatter(r1[0],r1[1],s=120,marker='o',c='black',zorder=10)

    r2 = fsolve(dotu, [0.09,0.8],args=0)
    plt.scatter(r2[0],r2[1], s=120, marker='v', facecolors='none', edgecolors='black',zorder=10)

    r3 = fsolve(dotu, [0.21,0.65],args=0)
    plt.scatter(r3[0],r3[1],s=120, marker='s', facecolors='none', edgecolors='black',zorder=10)

    t = np.linspace(0.0, 200.0, 1e4)
    x = np.array(odeint(dotu, [r2[0]-0.001,r2[1]], t)).transpose()
    plt.plot(x[0],x[1],lw=2,color="black")

    x = np.array(odeint(dotu, [r2[0]+0.001,r2[1]], t)).transpose()
    plt.plot(x[0],x[1],lw=2,color="black")

    x = np.array(odeint(dotu, [r3[0]-0.001,r3[1]], t)).transpose()
    plt.plot(x[0],x[1],lw=2,color="black") 
    x = np.array(odeint(dotu, [r3[0]+0.001,r3[1]], t)).transpose()
    plt.plot(x[0],x[1],lw=2,color="black")

    t = np.linspace(0.0, -200.0, 1e4)
    x = np.array(odeint(dotu, [r2[0],r2[1]-0.001], t)).transpose()
    plt.plot(x[0],x[1],lw=4,color="blue",zorder=5,alpha=0.7)
    t = np.linspace(0.0, -140.0, 1e4)
    x = np.array(odeint(dotu, [r2[0],r2[1]+0.001], t)).transpose()
    plt.plot(x[0],x[1],lw=4,color="blue",zorder=5,alpha=0.7)

    t = np.linspace(0.0, -20.0, 1e4)
    x = np.array(odeint(dotu, [r1[0]-0.001,r1[1]], t)).transpose()
    plt.plot(x[0],x[1],color="black",lw=2) 
    x = np.array(odeint(dotu, [r1[0]+0.001,r1[1]], t)).transpose()
    plt.plot(x[0],x[1],color="black",lw=2)

    t = np.linspace(0.0, 20.0, 1e4)
    x = np.array(odeint(dotu, [0.02,0.975], t)).transpose()
    plt.plot(x[0],x[1],color="black",lw=2)
    x = np.array(odeint(dotu, [0.02,0.823], t)).transpose()
    plt.plot(x[0],x[1],color="black",lw=2)
    x = np.array(odeint(dotu, [0.02,0.673], t)).transpose()
    plt.plot(x[0],x[1],color="black",lw=2)
    x = np.array(odeint(dotu, [0.02,0.512], t)).transpose()
    plt.plot(x[0],x[1],color="black",lw=2)

    plt.subplots_adjust(left=0.22, bottom=0.23, right=0.95, top=0.95)
    plt.show()


def fun4():
    plt.figure(figsize=(4,3), dpi=80)
    matplotlib.rcParams.update({'font.size': 18})
    plt.xlabel("Кальций [мкМ]", fontsize=18)
    plt.ylabel("Инактивация, h", fontsize=18)
    plt.tick_params(labelsize=18)
    plt.ylim([0.45,1.0])
    plt.xlim([0.02,1.7])
    plt.xscale("log")
    
    global K3, I
    K3 = 0.04
    I = 0.5

    x = np.linspace(0.02,1.7,1000)
    w = []
    s = 0.5
    for i in x:
        r = fsolve(isocline1, s, args=i)
        s = r[0]
        w.append(r[0])       
    plt.plot(x,w,lw=3,color="red",ls="--")
    
    w = []
    s = 1.0
    x = np.linspace(0.0,1.7,1000)
    for i in x:
        r = fsolve(isocline2, s, args=i)
        s = r[0]
        w.append(r[0])       
    plt.plot(x,w,lw=3,color="red")

    r3 = fsolve(dotu, [0.32,0.59],args=0)
    plt.scatter(r3[0],r3[1],s=120, marker='s', facecolors='none', edgecolors='black',zorder=10)

    t = np.linspace(0.0, 200.0, 1e4)
    x = np.array(odeint(dotu, [r3[0]+0.001,r3[1]], t)).transpose()
    plt.plot(x[0],x[1],lw=2,color="black")

    x = np.array(odeint(dotu, [0.02,0.95], t)).transpose()
    plt.plot(x[0],x[1],color="black",lw=2)
    x = np.array(odeint(dotu, [0.02,0.75], t)).transpose()
    plt.plot(x[0],x[1],color="black",lw=2)
    x = np.array(odeint(dotu, [0.02,0.605], t)).transpose()
    plt.plot(x[0],x[1],color="black",lw=2)
    x = np.array(odeint(dotu, [0.02,0.512], t)).transpose()
    plt.plot(x[0],x[1],color="black",lw=2)

    plt.subplots_adjust(left=0.22, bottom=0.23, right=0.95, top=0.95)
    plt.show()


def fun5():
    plt.figure(figsize=(4,3), dpi=80)
    matplotlib.rcParams.update({'font.size': 18})
    plt.xlabel("Кальций [мкМ]", fontsize=18)
    plt.ylabel("Инактивация, h", fontsize=18)
    plt.tick_params(labelsize=18)
    plt.ylim([0.45,1.0])
    plt.xlim([0.02,1.7])
    plt.xscale("log")
    
    global K3, I
    K3 = 0.04
    I = 0.9

    x = np.linspace(0.02,1.7,1000)
    w = []
    s = 0.5
    for i in x:
        r = fsolve(isocline1, s, args=i)
        s = r[0]
        w.append(r[0])       
    plt.plot(x,w,lw=3,color="red",ls="--")
    
    w = []
    s = 1.0
    x = np.linspace(0.0,1.7,1000)
    for i in x:
        r = fsolve(isocline2, s, args=i)
        s = r[0]
        w.append(r[0])       
    plt.plot(x,w,lw=3,color="red")

    r3 = fsolve(dotu, [0.54,0.52],args=0)
    plt.scatter(r3[0],r3[1],s=120, marker='o',c='black',zorder=10)

    t = np.linspace(0.0, -200.0, 1e4)
    x = np.array(odeint(dotu, [r3[0]+0.001,r3[1]], t)).transpose()
    plt.plot(x[0],x[1],lw=2,color="black")
    x = np.array(odeint(dotu, [r3[0]-0.001,r3[1]], t)).transpose()
    plt.plot(x[0],x[1],lw=2,color="black")

    t = np.linspace(0.0, 200.0, 1e4)
    x = np.array(odeint(dotu, [0.02,0.95], t)).transpose()
    plt.plot(x[0],x[1],lw=2,color="black")

    t = np.linspace(0.0, 200.0, 1e4)
    x = np.array(odeint(dotu, [0.05,0.7], t)).transpose()
    plt.plot(x[0],x[1],lw=2,color="black")

    t = np.linspace(0.0, -200.0, 1e4)
    x = np.array(odeint(dotu, [0.05,0.7], t)).transpose()
    plt.plot(x[0],x[1],lw=2,color="black")

    t = np.linspace(0.0, 200.0, 1e4)
    x = np.array(odeint(dotu, [0.20,0.80], t)).transpose()
    plt.plot(x[0],x[1],lw=2,color="black")

    t = np.linspace(0.0, 200.0, 1e4)
    x = np.array(odeint(dotu, [0.02,0.6], t)).transpose()
    plt.plot(x[0],x[1],lw=2,color="black")

    t = np.linspace(0.0, -200.0, 1e4)
    x = np.array(odeint(dotu, [0.20,0.80], t)).transpose()
    plt.plot(x[0],x[1],lw=2,color="black")

    x = np.array(odeint(dotu, [r3[0]+0.001,r3[1]], t)).transpose()
    plt.plot(x[0][5000:],x[1][5000:],lw=4,color="blue",zorder=5,alpha=0.7)

    plt.subplots_adjust(left=0.22, bottom=0.23, right=0.95, top=0.95)
    plt.show()


def fun6():
    plt.figure(figsize=(4,3), dpi=80)
    matplotlib.rcParams.update({'font.size': 18})
    plt.xlabel("Кальций [мкМ]", fontsize=18)
    plt.ylabel("Инактивация, h", fontsize=18)
    plt.tick_params(labelsize=18)
    plt.ylim([0.45,1.0])
    plt.xlim([0.02,1.7])
    plt.xscale("log")
    
    global K3, I
    K3 = 0.04
    I = 1.0

    x = np.linspace(0.02,1.7,1000)
    w = []
    s = 0.5
    for i in x:
        r = fsolve(isocline1, s, args=i)
        s = r[0]
        w.append(r[0])       
    plt.plot(x,w,lw=3,color="red",ls="--")
    
    w = []
    s = 1.0
    x = np.linspace(0.0,1.7,1000)
    for i in x:
        r = fsolve(isocline2, s, args=i)
        s = r[0]
        w.append(r[0])       
    plt.plot(x,w,lw=3,color="red")

    r3 = fsolve(dotu, [0.54,0.52],args=0)
    plt.scatter(r3[0],r3[1],s=120, marker='o', c='black',zorder=10)


    t = np.linspace(0.0, 200.0, 1e4)
    x = np.array(odeint(dotu, [0.02,0.95], t)).transpose()
    plt.plot(x[0],x[1],lw=2,color="black")

    t = np.linspace(0.0, 200.0, 1e4)
    x = np.array(odeint(dotu, [0.02,0.605], t)).transpose()
    plt.plot(x[0],x[1],lw=2,color="black")

    x = np.array(odeint(dotu, [0.06,0.6], t)).transpose()
    plt.plot(x[0],x[1],lw=2,color="black")

    t = np.linspace(0.0, -10.0, 1e4)
    x = np.array(odeint(dotu, [0.06,0.6], t)).transpose()
    plt.plot(x[0],x[1],lw=2,color="black")

    plt.subplots_adjust(left=0.22, bottom=0.23, right=0.95, top=0.95)
    plt.show()


def fun7():
    plt.figure(figsize=(4,3), dpi=80)
    matplotlib.rcParams.update({'font.size': 18})
    plt.xlabel("Кальций [мкМ]", fontsize=18)
    plt.ylabel("Инактивация, h", fontsize=18)
    plt.tick_params(labelsize=18)
    plt.ylim([0.45,1.0])
    plt.xlim([0.02,1.7])
    plt.xscale("log")
    # ax = plt.axes
    
    global K3, I
    K3 = 0.04
    I = 0.35

    x = np.linspace(0.02,1.7,1000)
    w = []
    s = 0.5
    for i in x:
        r = fsolve(isocline1, s, args=i)
        s = r[0]
        w.append(r[0])       
    plt.plot(x,w,lw=3,color="red",ls="--")
    
    w = []
    s = 1.0
    x = np.linspace(0.0,1.7,1000)
    for i in x:
        r = fsolve(isocline2, s, args=i)
        s = r[0]
        w.append(r[0])       
    plt.plot(x,w,lw=3,color="red")

    r1 = fsolve(dotu,[0.04,0.9],args=0)
    plt.scatter(r1[0],r1[1],s=120,marker='o',c='black',zorder=10)

    t = np.linspace(0.0, 200.0, 1e4)
    x = np.array(odeint(dotu, [0.02,0.999], t)).transpose()
    plt.plot(x[0],x[1],lw=2,color="black")

    x = np.array(odeint(dotu, [0.02,0.96], t)).transpose()
    plt.plot(x[0],x[1],lw=2,color="black")

    x = np.array(odeint(dotu, [0.02,0.547], t)).transpose()
    plt.plot(x[0],x[1],lw=2,color="black")

    x = np.array(odeint(dotu, [0.02,0.485], t)).transpose()
    plt.plot(x[0],x[1],lw=2,color="black")

    x = np.array(odeint(dotu, [0.25,0.80], t)).transpose()
    plt.plot(x[0],x[1],lw=2,color="black")

    x = np.array(odeint(dotu, [0.16,0.47], t)).transpose()
    plt.plot(x[0],x[1],lw=2,color="black")

    t = np.linspace(0.0, -60.0, 1e4)
    x = np.array(odeint(dotu, [0.25,0.80], t)).transpose()
    plt.plot(x[0],x[1],lw=2,color="black")

    t = np.linspace(0.0, -4.0, 1e4)
    x = np.array(odeint(dotu, [0.16,0.47], t)).transpose()
    plt.plot(x[0],x[1],lw=2,color="black")

    plt.subplots_adjust(left=0.22, bottom=0.23, right=0.95, top=0.95)
    plt.show()
    
def main():
    start = time.time()
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start Time is :", current_time)
    
    # samples = fun1()
    t = np.linspace(0.0, 6000.0, 6000)
    x = np.array(odeint(dotu, [0.5,0.5], t))
    print('shape ',x.shape)
    
    end = time.time()
    print(end - start, 'seconds')
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("End Time is :", current_time)
    
    file_path = r"C:/Users/XPS/Desktop/фф/6 сем/научка/Determenistic/Det1, I={}, K3={}.csv".format(I, K3)
    pd.DataFrame(x).to_csv(file_path)
    

main()
print("I = ",I)
