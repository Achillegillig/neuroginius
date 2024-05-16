import matplotlib.pyplot as plt
import numpy as np

from itertools import combinations
import math


def reshape_pvalues(pvalues):
    l = len(pvalues)
    
    # Mat size is the positive root of :
    # n**2 - n - 2l = 0 
    # Where l is the length of pvalues array
    # and n is the square matrix size
    n = (1 + math.sqrt(1 + 8 * l)) / 2
    if n != int(n):
        raise ValueError(f"Array of lenght {l} cannot be reshaped as a square matrix")
    n = int(n)
    
    arr = np.zeros((n, n))
    pointer = 0
    for i in range(n):
        if i + pointer > pointer:
            arr[i, :i] = pvalues[pointer:pointer+i]
        pointer += i

    return arr + arr.T

def fast_hist(matrix:np.ndarray):
    """Plot values of arrays containing
    individuals correlations

    Args:
        matrix (np.ndarray): (n_subjects, n_regions, r_regions)

    """
    n_regions = matrix.shape[1]
    n_subjects = matrix.shape[0]
    fig, ax = plt.subplots(1, 1)

    # Passing the array is slower
    for i in range(n_subjects):
        tst = matrix[i, :, :].reshape((n_regions ** 2))
        ax.hist(tst, histtype="step")
    ax.set_xlim(-1, 1)
    return fig, ax


