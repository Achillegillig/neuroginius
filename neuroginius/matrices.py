import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

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

def to_matrix(pairwise_results, n_parcels, k=0):
    new_mat = np.zeros((n_parcels, n_parcels))
    new_mat[np.triu_indices_from(new_mat, k=k)] = pairwise_results

    new_mat = new_mat + new_mat.T
    if k==0:
        new_mat[np.diag_indices_from(new_mat)] = new_mat[np.diag_indices_from(new_mat)] / 2
    return new_mat

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

# IMPLEMENT THIS 
from neuroginius.networks import group_by_networks
from neuroginius.atlas import Atlas
from neuroginius.iterables import unique
import itertools as it

def default_agg_func(block):
    return (block.mean(),)

class MatrixResult:
    def __init__(self, matrix, atlas) -> None:
        self.atlas = atlas
        self.matrix = matrix
        self._set_sorted_matrix()
    
    def _set_sorted_matrix(self):
        """Reorganize the matrix by macro labels, store
        the sorted matrix and a mapping from networks name
        to indexes in the sorted matrix
        """
        ticks, sort_index = group_by_networks(self.atlas.macro_labels)
        matrix_sort = np.ix_(sort_index, sort_index)

        self.sorted_matrix = self.matrix[matrix_sort]
        new_labels = sorted(tuple(unique(self.atlas.macro_labels)))
        self.network_to_idx = pd.Series(dict(zip(
            new_labels,
            it.pairwise(ticks)
        )))
        self.labels = new_labels

    def get_macro_matrix(self, agg_func=default_agg_func):
        """Get a matrix reorganized by networks
        Args:
            agg_func (function, optional): function to compute
            the aggregated of each cell, from the block of original
            values. Defaults to default_agg_func, which performs a simple
            mean

        Returns:
            DataFrame: summary per network of the original matrix.
        """
        gen = self._gen_macro_values(
            agg_func=agg_func
        )
        comparisons = pd.DataFrame(gen, columns=["node_a", "node_b", "connectivity"])
        pivoted = comparisons.pivot(index="node_a", columns="node_b")
        return pivoted.loc[:, "connectivity"]

    # This could be a function on its own
    def _gen_macro_values(self, agg_func):
        for network_a, network_b in it.product(self.network_to_idx.index, self.network_to_idx.index):
            loc_a, loc_b = self.network_to_idx[network_a], self.network_to_idx[network_b]
            block = self.sorted_matrix[loc_a[0]:loc_a[1], loc_b[0]:loc_b[1]]

            yield network_a, network_b, *agg_func(block)


