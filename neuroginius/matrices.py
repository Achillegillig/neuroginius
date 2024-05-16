import matplotlib.pyplot as plt
import numpy as np

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import seaborn as sns
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

# TODO Return axes instead of figure
def plot_matrix(
    mat, atlas, macro_labels=True, bounds=None, cmap="seismic", axes=None
):
    """Simplified version of the plot_matrices function. Only displays
    a single matrix.

    Args:
        mat (_type_): _description_
        atlas (Bunch): sklearn bunch containing labels and
        macro labels id macro_labels is True
        macro_labels (bool, optional): _description_. Defaults to True.
        bounds (_type_, optional): _description_. Defaults to None.
        cmap (str, optional): _description_. Defaults to "seismic".

    """
    mat = mat.copy()
    n_regions = mat.shape[0]
    mat[list(range(n_regions)), list(range(n_regions))] = 0

    # In general we want a colormap that is symmetric around 0
    span = max(abs(mat.min()), abs(mat.max()))
    if bounds is None:
        bounds = (-span, span)

    if axes is None:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))

    if macro_labels:
        networks = np.array(atlas.macro_labels)

        sort_index = np.argsort(networks)
        ticks = []
        lbls = []
        prev_label = None
        for i, label in enumerate(networks[sort_index]):
            if label != prev_label:
                ticks.append(i)
                lbls.append(label)
                prev_label = label
                axes.hlines(i, 0, n_regions, colors="black", linestyles="dotted")
                axes.vlines(i, 0, n_regions, colors="black", linestyles="dotted")

        ticks.append(i + 1)

    else:
        sort_index = np.arange(n_regions)

    sns.heatmap(
        mat[np.ix_(sort_index, sort_index)],
        ax=axes,
        vmin=bounds[0],
        vmax=bounds[1],
        cmap=cmap
    )

    if macro_labels:
        axes.yaxis.set_minor_locator(FixedLocator(ticks))
        axes.yaxis.set_major_locator(FixedLocator([(t0 + t1) / 2 for t0, t1 in zip(ticks[:-1], ticks[1:])]))
        axes.xaxis.set_major_locator(FixedLocator([(t0 + t1) / 2 for t0, t1 in zip(ticks[:-1], ticks[1:])]))
        axes.set_yticklabels(lbls, rotation=0)
        axes.set_xticklabels(lbls, rotation=30)

    return axes
