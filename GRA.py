import numpy as np
import random

def create_GRA_sol(F, seed):
    random.seed(seed)
    n = F.shape[1]
    m = F.shape[0]
    B = np.eye(n)
    dist_thr = 0.05
    # Select a sample randomly
    sample_idx = random.choice(np.arange(m))
    VAF_values = np.copy(F[sample_idx, :])
    # Build the solution iteratively
    for i in np.arange(n):
        candidate_top_VAF = max(VAF_values)
        candidate_list = np.where(np.array(VAF_values) >= candidate_top_VAF - dist_thr)[0]
        selected_mutation = random.choice(candidate_list)
        B[VAF_values != -1000, selected_mutation] = 1
        VAF_values[selected_mutation] = -1000
    return B



def create_GRA_sol_fixed_c(F, seed, c_length):
    random.seed(seed)
    n = F.shape[1]
    m = F.shape[0]
    B = np.eye(n)
    # Root of tree
    max_vaf = np.max(F)
    selected_mutation = np.argmax(np.sum(F == max_vaf, axis=0)) #In case of multiple occurrences of the maximum values, the indices corresponding to the first occurrence are returned.
    sample_idx = random.choice(np.where(F[:, selected_mutation] == max_vaf)[0])
    VAF_values = np.copy(F[sample_idx, :])
    B[:, selected_mutation] = 1
    VAF_values[selected_mutation] = np.nan
    # Place rest of nodes iteratively
    for i in np.arange(n-1):
        nan_pos = np.where(np.isnan(VAF_values))[0]
        candidate_list = np.argsort(VAF_values)[::-1]
        candidate_list = np.array([x for x in candidate_list if x not in nan_pos])[:c_length]
        selected_mutation = random.choice(candidate_list)
        B[~np.isnan(VAF_values), selected_mutation] = 1
        VAF_values[selected_mutation] = np.nan
    return B
