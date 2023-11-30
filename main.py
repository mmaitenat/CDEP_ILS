import argparse
import time
import random
import numpy as np
import os
import pickle
from calc_U import *
from tune_U import *
from calc_F import calc_F
from calc_error import *
from apply_change import *
from fill_log import fill_log
from escape import *
from helpers import *
from return_ils import return_ils
from GRA import *

def ils(F_mat, outdir, algorithm = "first-improvement", seed = 1, jump_param = 0.3, neighbourhood = "leaf", verbosity = 2):
    F_true = np.copy(F_mat)
    error_thr = 10**(-17)
    n = F_mat.shape[1]
    samples = F_mat.shape[0]
    max_iter = 50 * n**2
    local_opt = False
    escape = True
    escape_method = 'ils'
    sol_to_perturbate = 'best'
    escapes = 0
    c_length = round(0.2*n)
    # Initial solution
    iter = 0
    start = "GRA-fixed_c"
    if start == "GRA-fixed_c":
        print("Greedy random adaptive starting solution. Fixed length candidate list.")
        B_initial = create_GRA_sol_fixed_c(F = F_mat, seed = seed, c_length = c_length)
    elif start == "GRA-thr":
        print("Greedy random adaptive starting solution. Candidate list based on thr.")
        B_initial = create_GRA_sol(F = F_mat, seed = seed)
    else:
        raise ValueError("Method for calculating starting solution not recognized")

    # Initialize log
    error_log = []
    iter_log = []
    node_change_log =[]
    new_parent_log = []
    new_child_log = [] # Only used is tree-edit distance
    neighbourhood_size_log = []
    B_matrix_log = []
    U_matrix_log = []
    F_matrix_log = []

    log = {'error': error_log, 'iter': iter_log, 'node_change_parent': node_change_log, 'new_parent': new_parent_log, 'new_child': new_child_log, 'B_matrix': B_matrix_log, 'U_matrix': U_matrix_log, 'F_matrix': F_matrix_log, 'neighbourhood_size': neighbourhood_size_log}
    # Start timing
    start_time = time.time()
    # Initial solution
    random.seed(seed)
    B = np.copy(B_initial)
    U = calc_U(F = F_true, B = B, heterozygous = False)
    U = tune_U(U)
    F = calc_F(U = U, B = B, heterozygous = False)
    error = calc_error(F_old = F_true, F_new = F)
    log = fill_log(log = log, iter = iter, node_change_parent = np.nan, new_parent = np.nan, new_child = np.nan, B_matrix = B, U_matrix = U, F_matrix = F, neighbourhood_size = np.nan, error = error)
    # Start local search if not optimal solution
    if error:
        while round(error, 10) > error_thr and iter < max_iter:
            # Get move by greedy or first-improvement
            move = get_best_move(F_true=F_true, B=B, algorithm=algorithm, iter=iter, max_iter=max_iter, last_error=error, neighbourhood=neighbourhood)
            if move["log"] is None:  # We will get a None when stuck in a local optimum
                print(f"Stuck in a local optimum in iteration {move['iter']}")
                if escape:
                    print(f"Trying to escape by {escape_method}.")
                    iter = move["iter"]
                    if iter < max_iter:
                        escapes += 1  # Sums to the number of times we try to escape from a local optimum
                        move = {"log": escape_from_local_optimum(log=log, last_iter=iter, F_true=F_true, escape_method=escape_method, perturb_param=jump_param, sol_to_perturbate=sol_to_perturbate, local_change=neighbourhood), "iter": iter + 1}
                    else:
                        print("Maximum number of iterations reached.")
                        break
                else:
                    local_opt = True
                    break
            # Save values of iteration to log
            move_log = move["log"]
            iter = move["iter"]
            error = move_log["error"]
            B = move_log["B_matrix"]
            # Fill the log
            log = fill_log(log=log, iter=iter, B_matrix=B, U_matrix=move_log["U_matrix"], F_matrix=move_log["F_matrix"], error=error, node_change_parent=move_log["node_change_parent"], new_parent=move_log["new_parent"], neighbourhood_size=move_log["neighbourhood_size"], new_child=move_log["new_child"])
        print("Maximum number of iterations or converge criteria reached.")
    else:
        print("Initial solution is already optimal.")

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Output log and solution
    sol = return_ils(log=log, iter=iter, time=elapsed_time, verbosity=verbosity)
    return sol

def parse_arguments():
    parser = argparse.ArgumentParser(description='Iterated local search for the clonal deconvolution and evolution problem.')
    parser.add_argument('filedir', type=str, help='Directory where the input F file is located.')  
    parser.add_argument('outdir', type=str, help='Directory where the output files will be stored.')
    parser.add_argument('--algorithm', type=str, default="first-improvement", help='Algorithm to use for the local search. Options: first-improvement, greedy. Default: first-improvement.')
    parser.add_argument('--seed', type=int, default=1, help='Seed for steps involving random number generation. Default: 1.')
    parser.add_argument('--jump_param', type=float, default=0.3, help='Parameter that determines the number of random changes to make when escaping from a local optimum. The number of changes is jump_param x n. Default: 0.3.')
    parser.add_argument('--neighbourhood', type=str, default="leaf", help='Neighbourhood or local move to use for the local search. Options: spr, leaf, sumoted. Default: leaf.')
    parser.add_argument('--verbosity', type=int, default=2, help='Verbosity level. Options: 1, 2, 3. Default: 2.')
    return parser.parse_args()


if __name__ == '__main__':
    # filedir = "/Users/maitena/Research/clonalDeconvolution/my_algorithm/data/second_experimentation/evaluation//n-10_m-6_k-5_sel-neutral_noisy-FALSE_depth-NA_rep-1_seed-461"
    args = parse_arguments()
    F_mat = np.loadtxt(os.path.join(args.filedir, "F_normal.txt"))
    solution = ils(F_mat, args.outdir, algorithm = args.algorithm, seed = args.seed, jump_param = args.jump_param, neighbourhood = args.neighbourhood, verbosity = args.verbosity)

    with open(os.path.join(args.outdir, 'log_error.pkl'), 'wb') as f:
        pickle.dump(solution, f)