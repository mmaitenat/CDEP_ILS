import numpy as np
import random
from helpers import *
from calc_U import *
from calc_F import *
from calc_error import *
from fill_log import *
from tune_U import *

def move_randomly(B, local_change):
    # Define a dictionary to map movement names to their calculation functions
    move_functions = {
        "spr": move_randomly_spr,
        "leaf": move_randomly_leaf,
        "sumoted": move_randomly_sumoted
    }
    if local_change in move_functions:
        # Call the corresponding local change calculation function
        return move_functions[local_change](B)
    else:
        raise ValueError("Local change calculation method not recognized.")



def move_randomly_spr(B):
    n = B.shape[0]
    parents = [get_parent(B, idx) for idx in np.arange(n)]
    root = np.where(np.isnan(parents))[0]
    # Find new parent options until there are valid options
    while True:
        node_change_parent = random.choice(np.setdiff1d(np.arange(n), root))
        new_parent_opts = get_new_parent_opts(node_change_parent, parents, B)
        if new_parent_opts:
            break
    new_parent = random.choice(new_parent_opts)
    # Create a new B matrix
    new_B = np.copy(B)
    # The rows that change will be the one where we changed the parent and its descendants
    node_change_parent_descendants = get_descendants(B, node_change_parent)
    # Change itself
    mut_idx = get_mutation_idx(B, node_change_parent)
    new_B[node_change_parent, :] = get_new_B_row(mut_idx, B, new_parent)
    # Change descendants
    for idx in node_change_parent_descendants:
        child_mut_idx = get_mutation_idx(B, idx)
        new_B[idx, :] = get_new_B_row(child_mut_idx, new_B, get_parent(B, idx))
    return new_B



def move_randomly_leaf(B):
    n = B.shape[0]
    parents = [get_parent(B, idx) for idx in np.arange(n)]
    root = np.where(np.isnan(parents))[0]
    leaf_mutations = get_leaf_mutations(B)
    while True:    
        node_change_parent = random.choice(leaf_mutations)
        new_parent_opts = get_new_parent_opts(node_change_parent, parents, B)
        if new_parent_opts:
            break
    new_parent = random.choice(new_parent_opts)
    # Create a new B matrix
    new_B = np.copy(B)
    # Rows that change are the one we modified the parent and its descendants
    node_change_parent_descendants = get_descendants(B, node_change_parent)
    # Change itself
    mut_idx = get_mutation_idx(B, node_change_parent)
    new_B[node_change_parent, :] = get_new_B_row(mut_idx, B, new_parent)
    # Change descendants
    for idx in node_change_parent_descendants:
        child_mut_idx = get_mutation_idx(B, idx)
        new_B[idx, :] = get_new_B_row(child_mut_idx, new_B, get_parent(B, idx))
    return new_B



def move_randomly_sumoted(B):
    n = B.shape[0]
    parents = [get_parent(B, idx) for idx in np.arange(n)]
    root = np.where(np.isnan(parents))[0]
    while True:
        node = random.choice(np.setdiff1d(np.arange(n), root))
        grandparent = get_parent(B, get_parent(B, node))
        siblings = get_siblings(B, node)
        new_parent_opts = [grandparent] + siblings
        new_parent_opts = [x for x in new_parent_opts if not np.isnan(x)] # cuando no hay abuelo devuelve NA
        if new_parent_opts:
            break
    new_parent = random.choice(new_parent_opts)
    # Create a new B matrix
    new_B = np.copy(B)
    # Rows that change are the one we modified the parent and its descendants
    node_change_parent_descendants = get_descendants(B, node)
    # Change itself
    mut_idx = get_mutation_idx(B, node)
    new_B[node, :] = get_new_B_row(mut_idx, B, new_parent)
    # Change descendants
    for idx in node_change_parent_descendants:
        child_mut_idx = get_mutation_idx(B, idx)
        new_B[idx, :] = get_new_B_row(child_mut_idx, new_B, get_parent(B, idx))
    return new_B



def escape_from_local_optimum(log, last_iter, F_true, escape_method, perturb_param, sol_to_perturbate, local_change):
    if sol_to_perturbate == "last":
        B = log["B_matrix"][-1]
    elif sol_to_perturbate == "best":
        perturb_iter = np.where(log["error"] == min(log["error"]))[0]
        perturb_iter = random.choice(perturb_iter)
        B = log["B_matrix"][perturb_iter]
    else:
        raise ValueError("Unrecognized option for sol_to_perturbate. Please choose one between last or best.")

    n = F_true.shape[1]
    if escape_method == "ils":
        n_jumps = np.ceil(perturb_param * n**2).astype(int)
        for jump in range(n_jumps):
            B = move_randomly(B, local_change)
    elif escape_method == "multistart":
        B = create_GRA_sol_fixed_c(F=F_true, seed=100, c_length=round(0.2*n))
    else:
        raise ValueError("Unrecognized option for escape_method. Please choose one between ils or multistart.")

    U = calc_U(F = F_true, B = B, heterozygous = False)
    U = tune_U(U)
    F = calc_F(U = U, B = B, heterozygous = False)
    error = calc_error(F_old = F_true, F_new = F)
    log0 = fill_log(log={}, iter = 1, B_matrix = B, U_matrix = U, F_matrix = F, error = error, node_change_parent=np.nan, new_parent=np.nan, new_child=np.nan, neighbourhood_size=np.nan)
    return {key: value[0] for key, value in log0.items()}