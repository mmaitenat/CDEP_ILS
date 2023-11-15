import numpy as np
import random
from helpers import *
from calc_U import *
from calc_F import *
from calc_error import *
from fill_log import *
from tune_U import *


def get_neighbourhood(B, F_true, iter, max_iter, algorithm, neighbourhood, last_error):
    # Define a dictionary to map neighborhood names to their calculation functions
    neighbourhood_functions = {
        "spr": calc_spr_neighbourhood,
        "leaf": calc_leaf_neighbourhood,
        "sumoted": calc_sumoted_neighbourhood
    }
    if neighbourhood in neighbourhood_functions:
        # Call the corresponding neighborhood calculation function
        return neighbourhood_functions[neighbourhood](B, F_true, iter, max_iter, algorithm, last_error)
    else:
        raise ValueError("Neighbourhood calculation method not recognized.")
    

def calc_spr_neighbourhood(B, F_true, iter, max_iter, algorithm, last_error):
    stop = False
    log0 = {}
    n = B.shape[0]
    parents = [get_parent(B, idx) for idx in np.arange(n)]
    root = np.where(np.isnan(parents))[0]
    nodes_move = np.setdiff1d(np.arange(n), root)
    random.shuffle(nodes_move)  # Randomize to avoid biases in the last iteration
    start_iter = iter # scalars are passed by value in Python, so this is a copy of the original value
    for node in nodes_move:
        new_parent_opts = get_new_parent_opts(node = node, parents = parents, B = B)
        random.shuffle(new_parent_opts)  # Randomize to avoid biases in the last iteration
        node_change_parent_descendants = get_descendants(B, node)
        mut_idx = get_mutation_idx(B, node)
        for new_parent in new_parent_opts:
            if iter < max_iter:
                iter += 1
                # Change itself
                B_new = np.copy(B)
                B_new[node, :] = get_new_B_row(mut_idx, B, new_parent)
                # Change descendants
                for idx in node_change_parent_descendants:
                    child_mut_idx = get_mutation_idx(B, idx)
                    B_new[idx, :] = get_new_B_row(child_mut_idx, B_new, get_parent(B, idx))
                # Solve the equation
                U = calc_U(F=F_true, B=B_new, heterozygous=False)
                U = tune_U(U)
                F = calc_F(U=U, B=B_new, heterozygous=False)
                error = calc_error(F_old=F_true, F_new=F)
                # Save log
                log0 = fill_log(log=log0, iter=iter - start_iter, B_matrix=B_new, U_matrix=U, F_matrix=F,
                                error=error, node_change_parent=node, new_parent=new_parent, new_child=np.nan)
                # If algorithm is "first-improvement," check if the current solution is already better than the previous one
                if algorithm == "first-improvement":
                    if error < last_error:
                        print("First-improvement: improving solution found before examining the whole neighbourhood.")
                        stop = True
                        break
            else:
                print("Incomplete neighbourhood due to the maximum number of iterations reached.")
                stop = True
                break
        if stop:
            break
    return {"log": log0, "iters": iter}


def calc_sumoted_neighbourhood(B, F_true, iter, max_iter, algorithm, last_error):
    stop = False
    log0 = {}
    n = B.shape[0]
    parents = [get_parent(B, idx) for idx in np.arange(n)]
    root = np.where(np.isnan(parents))[0]
    nodes_move = np.setdiff1d(np.arange(n), root)
    random.shuffle(nodes_move)  # Randomize to avoid biases in the last iteration
    start_iter = iter
    for node in nodes_move:
         # In this case, the new parent options are only the grandparent or a sibling
        grandparent = get_parent(B, get_parent(B, node))
        siblings = get_siblings(B, node)
        new_parent_opts = [grandparent] + siblings
        new_parent_opts = [x for x in new_parent_opts if not np.isnan(x)] # cuando no hay abuelo devuelve NA
        random.shuffle(new_parent_opts)  # Randomize to avoid biases in the last iteration
        node_change_parent_descendants = get_descendants(B, node)
        mut_idx = get_mutation_idx(B, node)
        for new_parent in new_parent_opts:
            if iter < max_iter:
                iter += 1
                # Change itself
                B_new = np.copy(B)
                B_new[node, :] = get_new_B_row(mut_idx, B, new_parent)
                # Change descendants
                for idx in node_change_parent_descendants:
                    child_mut_idx = get_mutation_idx(B, idx)
                    B_new[idx, :] = get_new_B_row(child_mut_idx, B_new, get_parent(B, idx))
                # Solve the equation
                U = calc_U(F=F_true, B=B_new, heterozygous=False)
                U = tune_U(U)
                F = calc_F(U=U, B=B_new, heterozygous=False)
                error = calc_error(F_old=F_true, F_new=F)
                # Save log
                log0 = fill_log(log=log0, iter=iter - start_iter, B_matrix=B_new, U_matrix=U, F_matrix=F,
                                error=error, node_change_parent=node, new_parent=new_parent, new_child=np.nan)
                # If algorithm is "first-improvement," check if the current solution is already better than the previous one
                if algorithm == "first-improvement":
                    if error < last_error:
                        print("First-improvement: improving solution found before examining the whole neighbourhood.")
                        stop = True
                        break
            else:
                print("Incomplete neighbourhood due to the maximum number of iterations reached.")
                stop = True
                break
        if stop:
            break
    return {"log": log0, "iters": iter}




def calc_leaf_neighbourhood(B, F_true, iter, max_iter, algorithm, last_error):
    stop = False
    log0 = {}
    n = B.shape[0]
    parents = [get_parent(B, idx) for idx in np.arange(n)]
    root = np.where(np.isnan(parents))[0]
    nodes_move = get_leaf_mutations(B)
    random.shuffle(nodes_move)  # Randomize to avoid biases in the last iteration
    start_iter = iter
    for node in nodes_move:
        new_parent_opts = get_new_parent_opts(node=node, parents=parents, B=B)
        random.shuffle(new_parent_opts)  # Randomize to avoid biases in the last iteration
        node_change_parent_descendants = get_descendants(B, node)
        mut_idx = get_mutation_idx(B, node)
        for new_parent in new_parent_opts:
            if iter < max_iter:
                iter += 1
                # Change itself
                B_new = np.copy(B)
                B_new[node, :] = get_new_B_row(mut_idx, B, new_parent)
                # Change descendants
                for idx in node_change_parent_descendants:
                    child_mut_idx = get_mutation_idx(B, idx)
                    B_new[idx, :] = get_new_B_row(child_mut_idx, B_new, get_parent(B, idx))
                # Solve the equation
                U = calc_U(F=F_true, B=B_new, heterozygous=False)
                U = tune_U(U)
                F = calc_F(U=U, B=B_new, heterozygous=False)
                error = calc_error(F_old=F_true, F_new=F)
                # Save log
                log0 = fill_log(log=log0, iter=iter - start_iter, B_matrix=B_new, U_matrix=U, F_matrix=F,
                                error=error, node_change_parent=node, new_parent=new_parent, new_child=np.nan)
                # If algorithm is "first-improvement," check if the current solution is already better than the previous one
                if algorithm == "first-improvement":
                    if error < last_error:
                        print("First-improvement: improving solution found before examining the whole neighbourhood.")
                        stop = True
                        break
            else:
                print("Incomplete neighbourhood due to the maximum number of iterations reached.")
                stop = True
                break
        if stop:
            break
    return {"log": log0, "iters": iter}


def get_best_move(B, F_true, iter, max_iter, last_error, neighbourhood, algorithm):
    # Calculate neighbourhood
    neighbourhood = get_neighbourhood(B=B, F_true=F_true, iter=iter, max_iter=max_iter, algorithm=algorithm,
                                       neighbourhood=neighbourhood, last_error=last_error)
    neighbourhood_log = neighbourhood["log"]
    iter = neighbourhood["iters"]
    min_error = min(neighbourhood_log["error"])
    # Catch the situation in which the solution cannot be improved
    if min_error >= last_error:
        return {"log": None, "iter": iter}
    # get the positions where neighbourhood_log["error"] is the minimum
    which_min_error = np.where(neighbourhood_log["error"] == min_error)[0]
    # In case multiple solutions lead to the same min error solution, we'll just keep one, randomly
    which_min_error = random.choice(which_min_error)
    # Calculate neighbourhood size
    neighbourhood_size = len(neighbourhood_log["error"])
    # Return the best neighbor solution
    return {"log": {**{key: value[which_min_error] for key, value in neighbourhood_log.items()}, "neighbourhood_size" : neighbourhood_size},
            "iter": iter}