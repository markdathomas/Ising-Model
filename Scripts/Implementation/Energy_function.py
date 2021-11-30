# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 18:17:47 2021

@author: kramm
"""
import numpy as np

def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))

"""
v = np.array([1,0,1])
h = np.array([1,0])

w = np.array([[2,3,2],[1,2,3]])
b = np.array([4,1,5])
c = np.array([5,-1])

theta = (w,b,c)
"""


def energy(v_vec, h_vec,theta):
    "Calculate the energy associated with a configuration given the weights"
    #print("Theta is: ", theta)
    w_matrix,b_vec,c_vec = theta
    empty = np.array([])
    assert type(w_matrix) == type(empty)
    n = len(h_vec) # Number of hidden nodes
    m = len(v_vec) # Number of visible nodes
    assert np.shape(w_matrix) == (n,m) #Ensure the right amount of weights
    assert len(b_vec) == m
    assert len(c_vec) == n
    
    E = 0
    for j in range(m):
        E+=b_vec[j]*v_vec[j]
    for i in range(n):
        E+=c_vec[i]*h_vec[i]
        for j in range(m):
            E+=h_vec[i]*w_matrix[i][j]*v_vec[j]
    E = -E
    return E 

"""
e = energy(v,h,theta)
"""