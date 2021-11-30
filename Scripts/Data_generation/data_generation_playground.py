# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 23:58:48 2021

@author: kramm
"""

import numpy as np
from generate_data import generate_data_folder

step_size_list = [10**4, 10**4, 10**4, 10**4]
alpha_list = [0.1, 0.01, 0.001, 0.0001]
m_visible = 6
n_hidden = 6
k_steps_list = [1 for i in range(len(step_size_list))]
batch_size = 200

run_parameters = np.array([step_size_list, m_visible, n_hidden, alpha_list, k_steps_list, batch_size], dtype = object)
save = True


new_folder_absolute_path = "Data/"
folder_name = "m " +str(m_visible)+" " +"n "+str(n_hidden) + "b "+ str(batch_size) + " "  + str(step_size_list) + str(alpha_list)
array_name = "raw_data"

topdir = "../../"
output_location = generate_data_folder(run_parameters, new_folder_absolute_path, folder_name,array_name, topdir)
print("Location of output saved data is: ", output_location)