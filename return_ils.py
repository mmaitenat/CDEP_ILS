import numpy as np
import random

def return_ils(log, time, verbosity):
    # Add time to solution
    log['time'] = time

    # Output different things, values, and formats depending on verbosity
    if verbosity == 1:
        min_error = min(log['error'])
        min_error_iter = np.where(log["error"] == min(log["error"]))[0]
        min_error_iter = min_error_iter[0]
        opt_sol = log['B_matrix'][min_error_iter]
        with open("log.txt", "wb") as f:
            np.savetxt(f, opt_sol, delimiter="\t", fmt='%d')
        with open("log.txt", "a") as f:
            f.write(str(min_error))
            f.write("\n")
            f.write(str(time))
    elif verbosity == 2:
        min_error = min(log['error'])
        min_error_iter = np.where(log["error"] == min(log["error"]))[0]
        min_error_iter = min_error_iter[0]
        final_log = {'opt': {key: value[min_error_iter] for key, value in log.items() if key != 'time'}, 'log': {'error': log['error'], 'neighbourhood_size': log['neighbourhood_size'], 'iter': log['iter'],'time': time}}
        return final_log
    elif verbosity == 3:
        min_error = min(log['error'])
        min_error_iter = np.where(log["error"] == min(log["error"]))[0]
        min_error_iter = min_error_iter[0]
        final_log = {'opt': {key: value[min_error_iter] for key, value in log.items() if key != 'time'}, 'log': {'error': log['error'], 'B_matrix': log['B_matrix'], 'neighbourhood_size': log['neighbourhood_size'], 'iter': log['iter'], 'time': time}}
        return final_log
    else:
        return log
