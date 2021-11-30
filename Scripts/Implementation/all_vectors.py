# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 22:06:04 2021

@author: kramm
"""
import numpy as np


def dec_to_bin(x):
    return int(bin(x)[2:])



def bin_to_rep(x, number_of_nodes):
    a = str(x)
    c = str()
    b = number_of_nodes - len(a)
    for i in range(b):
        c+=str(0)
    c+=a
    return  c


def all_vectors_ising(number_of_nodes):
    vector_collection = []
    for i in range(2**number_of_nodes):
        bin_i = dec_to_bin(i)
        rep_i = bin_to_rep(bin_i, number_of_nodes)    
        vi = np.zeros(number_of_nodes)
        for j in range(number_of_nodes):
            vi[j] = int(str(rep_i)[j])
        vector_collection.append(vi)
    vector_collection = np.asarray(vector_collection)
    return vector_collection



#all_vectors = all_vectors_ising(5)
#print(all_vectors)