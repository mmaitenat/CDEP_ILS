import numpy as np

def force_sum_one(U):
    return np.apply_along_axis(lambda x: x / np.sum(x), axis=1, arr=U)


def tune_U(U):
    # Non-negativity
    new_U = np.copy(U)
    new_U[new_U < 0] = 0
    # Rows sum to 1
    new_U = force_sum_one(new_U)
    return new_U