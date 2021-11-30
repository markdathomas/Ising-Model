import sys
from tqdm import trange, tqdm




from file_restructure import save_numpy_array


sys.path.insert(0, "../Implementation")
try:
    from all_vectors import all_vectors_ising
except ImportError:
    print('No Import')
    
sys.path.insert(0, "../Analysis")
try:
    from Data_load import load_data
    from log_likelihood import log_likelihood
except ImportError:
    print('No Import')



def generate_loglik_data(folder_date, data_params, data_date, data_name = "raw_data"):
    """Load the data for the specified file, find loglik data and save"""

    # Location of output saved data is:  ../../Data/2021-11-24 m 6 n 6b 200 [20000, 10000, 10000][0.01, 0.001, 0.0001]/2021-11-25 m 6 n 6b 200 [20000, 10000, 10000][0.01, 0.001, 0.0001].npy
    # Location of output saved data is:  ../../Data/2021-11-24 m 6 n 6b 200 [2000, 1000, 1000][0.01, 0.001, 0.0001]/2021-11-25 m 6 n 6b 200 [2000, 1000, 1000][0.01, 0.001, 0.0001].npy
    
    

    #2021-11-24 m 6 n 6b 200 [2000, 1000, 1000][0.01, 0.001, 0.0001]/2021-11-25 m 6 n 6b 200 [2000, 1000, 1000][0.01, 0.001, 0.0001].npy'
    
    #./../Data/2021-11-25 m 6 n 6b 200 [200, 100, 100, 100][0.1, 0.01, 0.001, 0.0001]/2021-11-25 m 6 n 6b 200 [200, 100, 100, 100][0.1, 0.01, 0.001, 0.0001].npy 
    relevant_data = load_data(folder_date, data_date,data_params, data_name)
    
    run_parameters, distribution_data = relevant_data
    step_size_list, m_visible, n_hidden, alpha_list, k_steps_list, batch_size = run_parameters
    init_v, init_theta, theta_history_list, batch_history, cdk_histor = distribution_data
    

    
    allv = all_vectors_ising(m_visible)
    allh = all_vectors_ising(n_hidden)
    #print(theta_history)
    step_number_list = []
    loglik_list = []

    
    
    for epoch_number in trange(len(step_size_list)):
        
        for step in tqdm(range(step_size_list[epoch_number]),  position=0, leave=True):
            i = sum(step_size_list[:epoch_number]) + step
            step_number_list.append(i)
            
            current_theta = theta_history_list[i]
            current_batch = batch_history[epoch_number][step] 
            ll = log_likelihood(current_theta, current_batch, allv, allh)
            
            loglik_list.append(ll)
    
    
    loglik_name = "loglik"# Putting "raw_data" here overwrites old file
    file_to_save =   data_date + " " + loglik_name
    folder_path = "../../Data/"+folder_date+" "+data_params+"/"
    
    loglik_name = file_to_save#date_file(file_to_save)
    
    
    #print(folder_path)
    save_numpy_array(loglik_name, loglik_list, folder_path)
    return 
    
    """
data_params = "m 6 n 6b 200 [20, 10, 10, 10][0.1, 0.01, 0.001, 0.0001]"
data_date = "2021-11-25"
folder_date = "2021-11-25"
data_name = "raw_data"


generate_loglik_data(folder_date, data_params, data_date, data_name)
"""