# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 19:34:26 2021

@author: kramm
"""

import numpy as np
import random


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))




def prob_hi_is_1(i, v_vec,theta):
    w_matrix,b_vec,c_vec = theta
    if type(v_vec)==int:
        return 0
    
    else: 
        m = len(v_vec) # Number of visible nodes
        assert i <= m #Requested index not in list!
        marg_number = 0
        for j in range(m):
            marg_number+= w_matrix[i][j]*v_vec[j]
        marg_number += c_vec[i]
        prob = sigmoid(marg_number)
        return prob


def prob_vj_is_1(j, h_vec,theta):
    w_matrix,b_vec,c_vec = theta
    n= len(h_vec) # Number of visible nodes
    assert j <= n #Requested index not in list!
    marg_number = 0
    for i in range(n):
        marg_number+= w_matrix[i][j]*h_vec[i]
    marg_number += b_vec[j]
    prob = sigmoid(marg_number)
    return prob




def cd_hi_step(i,vn,theta):
    """For this step of the cd, determine hi value given training vector vn"""
    u = random.uniform(0.0, 1.0)
    p_hi = prob_hi_is_1(i, vn, theta)
    if p_hi>u:
        return 1
    else:
        return 0


def cd_h_step(number_of_hidden_nodes, vn, theta):
    hn = np.zeros(number_of_hidden_nodes)
    for i in range(number_of_hidden_nodes):
        hn[i] = cd_hi_step(i, vn, theta)
    return hn



def cd_vj_step(j, hn, theta):
    """For this step of the cd, determine hi value given hn"""
    u = random.uniform(0.0, 1.0)
    p_vj = prob_vj_is_1(j, hn, theta)
    if p_vj>u:
        return 1
    else:
        return 0


def cd_v_step(number_of_visible_nodes, hn, theta):
    vn = np.zeros(number_of_visible_nodes)
    for j in range(number_of_visible_nodes):
        vn[j] = cd_vj_step(j, hn, theta)
    return vn




def cdk(k, n_hidden, training_v, theta):
    n_visible = len(training_v)
    vn = training_v
    if k==0:
        return 0,0
    else:
        
        for n in range(k):
            hn = cd_h_step(n_hidden, vn, theta)
            vn = cd_v_step(n_visible, hn, theta)
            return vn, hn







def batch_generation(k, n_hidden, theta, batch_size, initial_training_v):
    """Generate a batch of cdk vectors"""
    #S = np.asarray(np.zeros(batch_size), dtype=object)
    S = []
    
    for i in range(batch_size):
        vk,hk = cdk(k,n_hidden, initial_training_v, theta)
        entry = vk
        S.append(entry)
        
    return S
    
    
    

"""
v = np.array([1,0,1])
h = np.array([1,0])

w = np.array([[0.1,0.1,0.2],[0.1,0.2,-0.3]])
b = np.array([0.1,0.2,0.5])
c = np.array([0.5,-0.1])

theta = (w,b,c)


training_v = v
k= 5
n_hidden = 2
n_visible = 3

vk,hk = cdk(k,n_hidden,training_v, theta)
a = batch_generation(k, n_hidden, theta, 1000, vk)
print(a)
"""