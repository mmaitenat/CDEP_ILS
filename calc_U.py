import numpy as np

def calc_U(F, B, heterozygous=True):
    B_inv = np.linalg.inv(B)
    U_homozygous = np.dot(F, B_inv)
    if heterozygous:
        U = 2 * U_homozygous
    else:
        U = U_homozygous
    return U