# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 19:26:41 2021

@author: kramm
"""
import numpy as np

import sys
import os


from file_restructure import date_file, make_directory, save_numpy_array
#from Learned_distribution import learn_distribution
sys.path.insert(0, "../Implementation")


from Learned_distribution import learn_distribution

    
    


def generate_data_folder(run_parameters, new_folder_absolute_path, folder_name,array_name, topdir = "../../"):
    #Specify the directory from this file
   
    
    
    #Define the new folder location and name
    new_folder_name = date_file(folder_name)
    
    path = topdir+new_folder_absolute_path+date_file(folder_name)
    print("Requested path: ", path)
    truth = os.path.exists(path)
    
    if not truth: #If the folder isn't yet made, create it
        make_directory(new_folder_name,new_folder_absolute_path, topdir)
    folder_path = topdir+new_folder_absolute_path+new_folder_name+"/"
    
    
    distribution_data = learn_distribution(run_parameters)
    #init_v, init_theta, theta_history_list, batch_history, cdk_history = distribution_data
    
    #Save data to the new folder
    saving_array = np.array([run_parameters, distribution_data], dtype = object) #Array to be saved

    #array_name =  folder_name #Giving the array a name the same as the folder
    dated_array_name = date_file(array_name) #Dating the array name for the file

    #Save the array
    saved_array_location = save_numpy_array(dated_array_name, saving_array, folder_path)
    
    return saved_array_location
