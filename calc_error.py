def calc_error(F_old, F_new):
    m, n = F_old.shape  # F_old is same size as F_new
    error = (abs(F_new - F_old).sum()) / (m * n)
    return error