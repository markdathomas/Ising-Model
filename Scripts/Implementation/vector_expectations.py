# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 18:39:39 2021

@author: kramm
"""


from model_and_data_averages import data_average, model_average
from contrastive_divergence import prob_hi_is_1



# Expectations over data
def data_exp_wij(v, theta, arg):
    i,j = arg
    prob_term = prob_hi_is_1(i,v,theta)
    if type(v)==int:
        return 0
    else:
        vj = v[j]
        output = prob_term*vj
        return output


def data_exp_bj(v, theta, arg):
    j = arg
    if type(v)==int:
        return 0
    else:
        vj_term = v[j]
        return vj_term

def data_exp_ci(v, theta, arg):
    i = arg
    p_hi_term = prob_hi_is_1(i, v, theta)
    return p_hi_term


def model_exp_wij(v,theta, arg):
    i,j=arg
    prob_term = prob_hi_is_1(i, v, theta)
    vj_term = v[j]
    output = prob_term*vj_term
    return output

def model_exp_bj(v, theta, arg):
    j = arg
    vj_term=v[j]
    return vj_term

def model_exp_ci(v, theta, arg):
    i = arg
    prob_term = prob_hi_is_1(i, v, theta)
    return prob_term

def e_data_wij(S,i, j, theta,Z):
    """Expected value for the data term in equation 13"""
    arg = i,j
    function = data_exp_wij
    average = data_average(S, function, theta, arg)
    return average

def e_data_bj(S, j, theta):
    """Expected value for the data term in equation 17"""
    arg = j
    function = data_exp_bj
    average = data_average(S, function,theta, arg)
    return average

def e_data_ci(S, i, theta):
    """Expected value for the data term in equation 19"""
    arg = i
    function = data_exp_ci
    average = data_average(S,function,theta, arg)
    return average


def e_model_wij(i, j, All_v, All_h, theta,Z):
    """Expected value for the model term in equation 13"""
    function = model_exp_wij
    arg = i,j
    average = model_average(All_v, All_h, theta,function, arg,Z)
    return average
    
def e_model_bj(j, All_v, All_h, theta,Z):
    """Expected value for the model term in equation 17"""
    function = model_exp_bj
    arg = j
    average = model_average(All_v, All_h, theta, function, arg,Z)
    return average

def e_model_ci(i, All_v, All_h, theta,Z):
    """Expected value for the model term in equation 19"""
    function = model_exp_ci
    arg = i
    average = model_average(All_v, All_h, theta, function, arg, Z)
    return average
"""
v1 = np.array([0,0,1])
v2 = np.array([0,1,1])

h = np.array([1,0])

w = np.array([[0.1,0.1,0.2],[0.1,0.2,-0.3]])
b = np.array([0.1,0.2,0.5])
c = np.array([0.5,-0.1])

theta = (w,b,c)



S = np.array([v1,v2], dtype = object)




allv = all_vectors_ising(len(v1))
allh = all_vectors_ising(len(h))

m = len(w[0])
n = len(w)
"""
"""

wij_step = w
ci_step = c
bj_step = b
for i in range(n): #j index
    edci = e_data_ci(S, i, theta)
    emci = e_model_ci(i, allv, allh, theta)
    diff_ci = edci-emci
    ci_step[i] = diff_ci
    for j in range(m): #i index
        edwij = e_data_wij(S, i, j, theta)
        emwij = e_model_wij(i,j, allv, allh, theta)
        diff_wij = edwij-emwij
        wij_step[i][j]=diff_wij

for j in range(m):
    wdbj = e_data_bj(S, j, theta)
    embj = e_model_bj(j, allv, allh, theta)
    diff_bj = wdbj-embj
    bj_step[j]=diff_bj
    

print(wij_step)
print()
print(ci_step)
print()
print(bj_step)    
"""