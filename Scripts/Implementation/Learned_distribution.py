# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 12:52:06 2021

@author: kramm
"""
import matplotlib.pyplot as plt
import numpy as np
import random

from learning_Step import take_N_cdk_steps
from model_and_data_averages import prob_v_given_theta, partition_function
from all_vectors import all_vectors_ising

from tqdm import trange, tqdm
import time

import sys
sys.path.insert(0, "../Analysis")
from log_likelihood import log_likelihood

initial_time= time.time()



def learned_distribution(theta, m_visible, n_hidden):
    
    all_visible = all_vectors_ising(m_visible)
    all_hidden = all_vectors_ising(n_hidden)
    Z = partition_function(all_visible, all_hidden, theta) #SLOW STEP HERE
    distribution = np.zeros(2**m_visible)
    for i in range(len(all_visible)):
        v = all_visible[i]
        pv = prob_v_given_theta(all_visible, all_hidden, v, theta,Z)
        distribution[i] = pv
    return distribution


def get_learned_distribution(theta, m_visible, n_hidden, N_curves, alpha_list, k_list, batch_size, init_v, steps = [100,100,100],  plot=False):
    
    assert N_curves == len(steps)
    
    allv = all_vectors_ising(m_visible)
    allh = all_vectors_ising(n_hidden)
    
    if plot:
        plt.figure()
        plt.title("Learned distribution", fontsize = 20)
    
    
    total_steps = sum(steps)
    theta_history_list = np.zeros(sum(steps), dtype = object)
    
    
    
    batch_history = np.zeros(total_steps, dtype = object)
    cdk_history = np.zeros(total_steps, dtype = object)
    
    labels = [i for i in range(2**m_visible)]
    init_dist = learned_distribution(theta, m_visible, n_hidden)

    if plot:
        plt.plot(labels, init_dist, label = "Initial distribution", color = 'k')
    
    
    for step in tqdm(range(N_curves),  position=0, leave=True):
        N_steps = steps[step]
        alpha = alpha_list[step]
        k = k_list[step]
        
        batches_current_run = np.zeros(N_steps, dtype = object)
        
        this_step_theta_history = []
        for this_step in tqdm(range(N_steps),  position=0, leave=True):
            
            theta_new, batch = take_N_cdk_steps(allv, allh, theta,alpha, 1, k, n_hidden, batch_size, init_v)
            theta = theta_new[-1] #Take the kth cdk theta output
            this_step_theta_history.append(theta)
           
            i = sum(steps[:step]) + this_step
            theta_history_list[i] = theta

            batches_current_run[this_step] = batch[-1] #Take the kth batch 
        
            batch_history[step] = batches_current_run
            
      
        
        
        
        if plot:
            learned_dist = learned_distribution(theta, m_visible, n_hidden)
            plt.plot(labels,learned_dist, label = "Epoch number = "+str(step+1))

   
    init_rep = 0
    for ii in range(len(init_v)):
        jj=len(init_v)-ii-1
        init_rep+=init_v[jj]*(2**ii)
    #print("Used v configu number is", init_rep)
    
    Z = partition_function(allv, allh,theta)
    pvgt = prob_v_given_theta(allv, allh, init_v,theta,Z)
    #print("Prob of used v in final parameter choice is", pvgt)
    if plot:    
        plt.scatter(init_rep, pvgt, marker = 'x', color = 'r', label = "Initial vector probability")

        
        
    if plot:
        plt.ylabel("Learned configuration probability", fontsize = 20)
        plt.xlabel("Ising configuration number", fontsize = 20)
        plt.yscale('log')
        plt.legend()
        plt.show()
    
  
    
    return theta_history_list, batch_history, cdk_history, init_dist, init_rep, pvgt



def learn_distribution(run_parameters):#, init_v,init_theta):
    """Generate random initial theta and init v data, go with it and then plot the learned distribution"""
    
    if len(run_parameters)==6: 
        plot_distribution = False
        step_size_list,m_visible,n_hidden, alpha_list, k_steps_list, batch_size = run_parameters 
    else: 
        step_size_list,m_visible,n_hidden, alpha_list, k_steps_list, batch_size,plot_distribution = run_parameters 
    N_curves = len(step_size_list)
    
    init_b = np.asarray([random.uniform(-1,1) for j in range(m_visible)])
    init_c = np.asarray([random.uniform(-1,1) for i in range(n_hidden)])
    
    init_w = np.asarray([[random.uniform(-1,1) for j in range(m_visible)] for i in range(n_hidden)])
    
    init_v = np.asarray(np.random.randint(2, size=m_visible))
    
    init_theta = init_w, init_b, init_c
    

    theta_history_list, batch_history, cdk_history, init_dist, init_rep, pvgt = get_learned_distribution(init_theta, m_visible, n_hidden,
                              N_curves, alpha_list, k_steps_list,
                              batch_size, init_v, step_size_list, plot_distribution)
    
    
    distribution_data = init_v, init_theta, theta_history_list, batch_history, cdk_history
    return distribution_data


    




#print("Time taken for everything was ", time.time() - initial_time, " seconds.")

#print("Theta history length is", len(theta_history_list))


#print("Batch history length is", len(batch_history))

#batch history is a list of all of the bathches used, divided into the step sets
#batch_history[0] is the set of batches used for each step, for the first set
#of steps
#batch_history[0][0] is batch used on the first step of the first set of steps
#batch_history[0][0][0] is the first vector used in batch_histroy[0][0]
