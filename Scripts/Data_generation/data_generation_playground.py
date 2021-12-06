# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 23:58:48 2021

@author: Mark Thomas
"""

from generate_data import run


step_size_list = [10**3, 10**2, 10**2, 10**2]
alpha_list = [0.1, 0.01, 0.001, 0.0001]
m_visible = 3
n_hidden = 3
k_steps_list = [1 for i in range(len(step_size_list))]
batch_size = 200
save = True


output_location = run(step_size_list, m_visible, n_hidden, alpha_list, k_steps_list, batch_size)
print("Location of output saved data is: ", output_location)