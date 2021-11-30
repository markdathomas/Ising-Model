# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 21:37:16 2021

@author: kramm
"""
import numpy as np
from Energy_function import energy
from all_vectors import all_vectors_ising
from math import isclose
from tqdm import trange


#v = np.array([1,0,1])
#h = np.array([1,0])


#theta = (w,b,c)


def generate_random_theta(m_visible, n_hidden):
    """Generate a random set of parameters theta given network structure"""
    w = np.asarray([np.zeros(m_visible) for i in range(n_hidden)])
    b = np.asarray(np.zeros(m_visible))
    c = np.asarray(np.zeros(n_hidden))
    
    for i in range(n_hidden):
        c[i] = np.random.uniform(-1,1)
        for j in range(m_visible):
            w[i][j] = np.random.uniform(-1,1)
    for j in range(m_visible):
        b[j] = np.random.uniform(-1,1)
    theta = (w,b,c)
    return theta





def partition_function(All_v, All_h, theta):
    no_v = len(All_v)
    no_h = len(All_h)
    Z = 0
    for i in range(no_v):
        for j in range(no_h):
            Z+=np.exp(-energy(All_v[i], All_h[j], theta))
    return Z
            
    
    return

def prob_v_given_theta(all_visible, all_hidden, v, theta,Z):
    """Calculates p(v|theta) eqn 5"""
    m = len(v)
    n_hidden = len(theta[2])
    
    w,b,c = theta
    
    prod_1 = 1
    for j in range(m):
        prod_1 *= np.exp(b[j]*v[j])
    prod_2 = 1
    for i in range(n_hidden):
        sum_1 = 0
        for j in range(m):
            sum_1+=w[i][j]*v[j]
        prod_2 *= (1+np.exp(c[i]+sum_1))
    p = prod_1*prod_2/Z
    return p

def data_average(S, function, theta, arg):
    """Calculate the average of the function over the batch S of vectors v
    eqn 14"""
    average = 0
    for v in S:
        average+=function(v,theta, arg)
    if len(S)!=0:
        average = average/len(S)
    return average

def model_average(All_v, All_h, theta, function, arg, Z):
    "Works out the model average of the function f, eqn 15"

    average = 0
    for v in All_v:
        pv = prob_v_given_theta(All_v, All_h, v, theta, Z)
        average+=pv*function(v,theta, arg)
    return average


def check_normalisation(theta,Z):
    m_visible = len(theta[1])
    n_hidden = len(theta[2])
    all_visible = all_vectors_ising(m_visible)
    all_hidden = all_vectors_ising(n_hidden)
    cumulative_prob = 0
    for i in range(len(all_visible)):
        v = all_visible[i]
        pi = prob_v_given_theta(all_visible,all_hidden, v, theta,Z)
        cumulative_prob +=pi
    return cumulative_prob
    
def check_normalisation_many_graphs(end_m, end_n, checks_per_config):
    """Check every graphs probability normalisation for every
    all possible number of visible nodes m and hidden nodes n up
    to end_m and end_n"""
    """Checks_per_configu is the number of theta sets generated
    per set of (m,n) values."""
    all_normalised = True
    for j in trange(checks_per_config):
        for m in trange(end_m):
            for n in trange(end_n):
                theta = generate_random_theta(end_m-m,end_n) #Check the slow ones first
                all_visible = all_vectors_ising(end_m-m)#SLOW STEP
                all_hidden = all_vectors_ising(end_n) # SLOW STEP
                Z = partition_function(all_visible, all_hidden, theta)
                checked_norm = check_normalisation(theta,Z)
                truth = isclose(checked_norm, 1, abs_tol=10**(-17))
                if not truth:
                    print("Issue! Norm is ", checked_norm, "for (m,n) = ", (end_m-m,end_n-n))
                    all_normalised = False
    if all_normalised:
        print()
        print("All configurations checked normalised to machine accuracy")
    return 
    

#m_visible = 3
#n_hidden = 2
#checks_per_config = 10
#check_normalisation_many_graphs(4,4, checks_per_config)