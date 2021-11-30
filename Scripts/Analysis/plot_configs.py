# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 23:06:49 2021

@author: kramm
"""
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import sys
import os.path
from file_restructure import load_numpy_array, make_directory

sys.path.insert(0, "../Data_generation")
try:
    from get_loglik_data import generate_loglik_data
    from file_restructure import save_numpy_array

except ImportError:
    print('No Import')
    
sys.path.insert(0, "../Implementation")
try:
    from all_vectors import all_vectors_ising
    from model_and_data_averages import prob_v_given_theta, partition_function
except ImportError:
    print('No Import')    




def generate_config_plot(topdir, data_params, data_date, folder_date, data_name):
    
    raw_data_filepath = topdir + "Data/"+folder_date+" "+data_params+"/"+data_date+" "+data_name+".npy"
        
    
    
    
    raw_data_filepath = topdir + "Data/"+folder_date+" "+data_params+"/"+data_date+" "+data_name+".npy"
    dist_filepath = topdir + "Data/"+folder_date+" "+data_params+"/"+data_date+" "+"dist.npy"
    
    run_parameters, distribution_data = load_numpy_array(raw_data_filepath)
    step_size_list = run_parameters[0]
    
    theta_history_list = distribution_data[2]
    m_visible = run_parameters[1]
    n_hidden = run_parameters[2]
   
    alpha_list = run_parameters[3]
    allv = all_vectors_ising(m_visible)
    if not os.path.isfile(dist_filepath): #If the loglik file hasn't been generated yet
        print ("Generating the distribution data")
       
        
        allh = all_vectors_ising(n_hidden)
        
        
        
        
        current_step = 0
        relevant_theta_list = []
        pvgt_list = []
        
        loglik_name = "loglik"# Putting "raw_data" here overwrites old file
        file_to_save =   data_date + " " + loglik_name
        folder_path = "../../Data/"+folder_date+" "+data_params+"/"
        
        loglik_name = file_to_save#date_file(file_to_save)
        
        
        for epoch_number in range(len(step_size_list)):
            current_step += step_size_list[epoch_number]
            current_theta = theta_history_list[current_step-1]
            relevant_theta_list.append(current_theta)
            
            Z = partition_function(allv, allh, current_theta)
            pvgt = []
            for v in allv:
                pvgt.append(prob_v_given_theta(allv, allh, v,current_theta, Z))
            pvgt_list.append(pvgt)
    
        dist_name = "dist"# Putting "raw_data" here overwrites old file
        file_to_save =   data_date + " " + dist_name
        folder_path = "../../Data/"+folder_date+" "+data_params+"/"
        
        dist_name = file_to_save#date_file(file_to_save)
        
        data_to_save = [pvgt_list, relevant_theta_list]
        #print(folder_path)
        save_numpy_array(dist_name, data_to_save, folder_path)
    
    
    
    
    pvgt_list, relevant_theta_list = load_numpy_array(dist_filepath)
    config_number_list = [i for i in range(len(allv))]
        
        
    plots_folder_path = topdir+"Plots/"+folder_date+" "+data_params
    plot_file_path = plots_folder_path+"/"+"Learned distribution"
    #If a folder for the plots doesn't exist yet, create one:
    os.makedirs(plots_folder_path, exist_ok=True)  
    
    

    
    plt.figure()
    plt.title("Learned distribution", fontsize = 30)
    plt.xlabel("Config number", fontsize = 20)
    plt.ylabel(r'$p\left(v|\theta \right)$', fontsize = 20)
    for i in range(len(pvgt_list)):
        plt.plot(config_number_list, pvgt_list[i], label = r'$\alpha$ = '+str(alpha_list[i]))
    
    plt.legend()
    plt.savefig(plot_file_path)
    plt.show()
    
    """
    for item in relevant_theta_list:
        print(item)
        print()
    """
    return relevant_theta_list

topdir = "../../" 
data_params = "m 6 n 6b 200 [1000, 1000, 1000, 1000][0.1, 0.01, 0.001, 0.0001]"
data_date = "2021-11-29"
folder_date = "2021-11-29"
data_name = "raw_data"

t = generate_config_plot(topdir, data_params, data_date, folder_date, data_name)