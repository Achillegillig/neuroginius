
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator
import numpy as np
import seaborn as sns


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