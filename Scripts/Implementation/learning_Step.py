# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 13:22:18 2021

@author: kramm
"""

from vector_expectations import e_data_ci, e_model_ci, e_data_wij, e_model_wij, e_data_bj, e_model_bj
from contrastive_divergence import batch_generation
import numpy as np
from model_and_data_averages import partition_function


def theta_step(S, allv, allh, theta,Z):
    w,b,c = theta
    wij_step = w.copy()
    ci_step = c.copy()
    bj_step = b.copy()
    m = len(w[0])
    n = len(w)
    
    for i in range(n): #j index
        edci = e_data_ci(S, i, theta)
        emci = e_model_ci(i, allv, allh, theta, Z)
        diff_ci = edci-emci
        ci_step[i] = diff_ci
        for j in range(m): #i index
            edwij = e_data_wij(S, i, j, theta,Z)
            emwij = e_model_wij(i,j, allv, allh, theta,Z)
            diff_wij = edwij-emwij
            wij_step[i][j]=diff_wij
    for j in range(m):
        wdbj = e_data_bj(S, j, theta)
        embj = e_model_bj(j, allv, allh, theta,Z)
        diff_bj = wdbj-embj
        bj_step[j]=diff_bj
    
    theta_step = wij_step, bj_step, ci_step
    return theta_step
    


def increment_theta(S, theta, allv, allh, alpha,Z):
    
    thetastep= theta_step(S, allv, allh, theta,Z)
    
    wstep, bstep, cstep = thetastep
    #new_theta = theta + alpha*thetastep
    w,b,c = theta
    
    wnew = w.copy()#Has to be a copy else get overwritten data later
    bnew = b.copy()
    cnew = c.copy()
    
    m = len(bnew)#theta[1])
    n = len(cnew)#theta[2])
    
    count = 1
    
      
    for j in range(m):
        
        """BUG CHECK HERE
        if count ==1:
            print("Before:", this_step_theta_history)
            print("m is", m)
            count+=1  
        """ 
        bnew[j]=b[j]+alpha*bstep[j]
        """
        if count == 2:
            print("After:", this_step_theta_history)
            count+=1
        """   
        for i in range(n):
            wnew[i][j] = w[i][j]+alpha*wstep[i][j]
    
    for i in range(n):
        cnew[i] = c[i]+alpha*cstep[i]
    
    
    
    new_theta = wnew, bnew, cnew
    
    return new_theta
    



def take_N_cdk_steps(allv, allh, theta0,alpha, N, k, n_hidden, batch_size, init_v):
    """Increment theta N times according to the algorithm"""
    
    theta_record = []#np.zeros(N, dtype = object)
    theta1 = theta0
    batch_record = []
    

    #print(this_step_theta_history)

    if N==0:
        print("No steps taken!")
        return theta0
    else:
        
        for step in range(N):
                
            S_current = batch_generation(k,n_hidden, theta1,batch_size, init_v)
            
            Z = partition_function(allv, allh, theta1)
            #print()
            #print(theta1)
                
            theta2 = increment_theta(S_current,theta1, allv, allh, alpha,Z)
            #print(theta1)
            
            
            theta_record.append(theta2)#[step] = theta2
            batch_record.append(S_current)
            
            theta1 = theta2
            
        
    return theta_record, batch_record

