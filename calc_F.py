import numpy as np

def calc_F(U, B, heterozygous=True):
    F_homozygous = np.dot(U, B)
    if heterozygous:
        F = 0.5 * F_homozygous
    else:
        F = F_homozygous
    return F