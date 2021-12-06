# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 00:14:44 2021

@author: Mark
"""


import matplotlib.pyplot as plt
import sys
import os.path


sys.path.insert(0, "../Data_generation")
try:
    from get_loglik_data import generate_loglik_data
except ImportError:
    print('No Import')
    
sys.path.insert(0, "../Analysis")
try:
    from file_restructure import load_numpy_array
except ImportError:
    print('No Import')    




def generate_loglik_plot(topdir, data_params, data_date, folder_date, data_name):
    
    raw_data_filepath = topdir + "Data/"+folder_date+" "+data_params+"/"+data_name+".npy"
    loglik_filepath = topdir + "Data/"+folder_date+" "+data_params+"/"+"loglik.npy"
    
    if not os.path.isfile(loglik_filepath): #If the loglik file hasn't been generated yet
        print ("Generating the loglik data")
        generate_loglik_data(folder_date, data_params, data_date, data_name)
    else:
        print("Using older data")
    loglik_list = load_numpy_array(loglik_filepath)
    
    run_parameters, distribution_data = load_numpy_array(raw_data_filepath)
    step_size_list = run_parameters[0]
    
    
    step_number_list = []
    for epoch_number in range(len(step_size_list)):
        for step in range(step_size_list[epoch_number]):
            i = sum(step_size_list[:epoch_number]) + step
            step_number_list.append(i)
    
    
    plots_folder_path = topdir+"Plots/"+folder_date+" "+data_params
    plot_file_path = plots_folder_path+"/"+"Loglik plot"
    #If a folder for the plots doesn't exist yet, create one:
    os.makedirs(plots_folder_path, exist_ok=True)  
        
    plt.figure()
    plt.plot(step_number_list, loglik_list)
    plt.title("Log likelihood vs epoch number", fontsize = 30)
    plt.xlabel("Epoch number", fontsize = 20)
    plt.ylabel(r'$\mathcal{L}\left(\theta |S \right)$', fontsize = 20)
    plt.savefig(plot_file_path)
    plt.show()
    return

topdir = "../../" 
data_params = "m 3 n 3b 200 [1000, 100, 100, 100][-0.1, -0.01, -0.001, -0.0001]"
data_date = "2021-12-06"
folder_date = "2021-12-06"
data_name = "raw_data"


generate_loglik_plot(topdir, data_params, data_date, folder_date, data_name)