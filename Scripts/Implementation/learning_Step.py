# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 13:22:18 2021

@author: Mark Thomas
"""

#Imports from current directory:
from vector_expectations import e_data_ci, e_model_ci, e_data_wij, e_model_wij, e_data_bj, e_model_bj
from contrastive_divergence import batch_generation
from model_and_data_averages import partition_function


def theta_step(S, allv, allh, theta,Z):
    """Given an input batch of training vectors (S) and the current 
    theta, computes the derivative of the log likelihood function
    in order to maximise log(L).
    
    This is done by averaging over the batch S, as in https://arxiv.org/pdf/1810.11503.pdf
    equations 13-19.
    
    Returns these derivatives in the same form/structure as theta is specified."""
    w,b,c = theta
    #Make a copy of each parameter of the right dimensions
    #(Done for ease, the values are all edited later):
    wij_step = w.copy()
    ci_step = c.copy()
    bj_step = b.copy()
    m = len(w[0])
    n = len(w)
    
    for i in range(n): #i index
        #Calculate the change in each ci:
        #Calculate eqn19 E_{data}:
        edci = e_data_ci(S, i, theta)
        #Calculate eqn19 E_{model}:
        emci = e_model_ci(i, allv, allh, theta, Z)
        #Difference of these (as in eqn 19):
        diff_ci = edci-emci
        #Set the ith entry of the derivative calculator equal to the calculated difference.
        ci_step[i] = diff_ci
        for j in range(m): #j index
            #Calculate the change in each wij:
            #Calculate eqn13 E_{data}:
            edwij = e_data_wij(S, i, j, theta,Z)
            #Calculate eqn13 E_{model}:
            emwij = e_model_wij(i,j, allv, allh, theta,Z)
            #Difference of these (as in eqn 13):
            diff_wij = edwij-emwij
            #Set the entry ij of the derivative calculator equal to the calculated difference.
            wij_step[i][j]=diff_wij
  
    for j in range(m): #j index
        #Calculate the change in each bj:
        #Calculate eqn17 E_{data}:
        wdbj = e_data_bj(S, j, theta)
        #Calculate eqn17 E_{model}:
        embj = e_model_bj(j, allv, allh, theta,Z)
        #Difference of these as in eqn 17:
        diff_bj = wdbj-embj
        #Set entry j of the derivative calculator equal to this calculated difference.
        bj_step[j]=diff_bj
    
    #Return these differences as a "delta theta" term:
    theta_step = wij_step, bj_step, ci_step
    
    return theta_step
    


def increment_theta(S, theta, allv, allh, alpha,Z):
    """Given a batch of vectors to use, the current theta 
    value and a learning rate (all other arguments can be generated
    from those, but are asked as an argument for computational efficiency),
    calculates the step in each parameter needed to be taken, and using the
    learning rate (alpha), returns the new value of theta."""
    #Use the contrastive divergence algorithm to find estimates
    #for the variation needed in the parameters theta needed 
    #to take a step in the direction of fastest increasing log likelihood:
    thetastep= theta_step(S, allv, allh, theta,Z)
    
    #Extract the individual components of delta theta:
    wstep, bstep, cstep = thetastep
    #(will have: new_theta = theta + alpha*thetastep)
    
    #Extract the old parameters:
    w,b,c = theta
    
    #Make a copy of the old parameters with specified dimensionality
    #(Done for convenience)
    wnew = w.copy()
    bnew = b.copy()
    cnew = c.copy()
    
    #Lengths of the loops over each parameter needed:
    m = len(bnew)
    n = len(cnew)
    
    for j in range(m):
        #Increment bj at the current learning rate:
        bnew[j]=b[j]+alpha*bstep[j]
        
        for i in range(n):
            #Increment wij at the current learning rate:
            wnew[i][j] = w[i][j]+alpha*wstep[i][j]
    
    for i in range(n):
        #Increment ci at the current learning rate:
        cnew[i] = c[i]+alpha*cstep[i]
    
    
    #Combine the new parameters into one object theta again:
    new_theta = wnew, bnew, cnew
    
    return new_theta
    



def take_N_cdk_steps(allv, allh, theta0,alpha, N, k, batch_size):
    """Increment theta N times according to the contrastive divergence algorithm.
    Returns the pair (theta_record, batch_record), which are the theta values
    and batches of vectors used.
    
    Currently assumes the Ising model is being used, and as such, functions called
    from here require the set of all hidden and all visible nodes (allh, allv),
    and uses these quantities accordingly"""
    
    #Empty arrays to be recorded into:
    theta_record = []
    theta1 = theta0
    batch_record = []

    if N==0:
        print("No steps taken!")
        return theta0
    else:
        #For each step requested:
        for step in range(N):
            #Generate a batch using the "current theta":
            S_current = batch_generation(k, theta1, batch_size)
            
            #Find the partition function (ISING MODEL SLOW METHOD):
            Z = partition_function(allv, allh, theta1)
            
            #Increment theta using averages over the generated batch.
            theta2 = increment_theta(S_current,theta1, allv, allh, alpha,Z)
            
            #Record the batch used and the new theta value
            batch_record.append(S_current)
            theta_record.append(theta2)
            
            #Update the "current theta" to be the new theta for repeat of above algorithm.
            theta1 = theta2

    return theta_record, batch_record

