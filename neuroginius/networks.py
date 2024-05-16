from itertools import product, combinations
import numpy as np
import itertools
from typing import Iterable, Tuple

import pandas as pd

from neuroginius.atlas import Atlas
from neuroginius.iterables import unique, all_equal, join, all_connectivities

def edge_format(block, labels):
    unstacked = block.unstack().reset_index().rename(columns={0: "FC"})
    unstacked["edge"] = unstacked["node_a"] + "_" + unstacked["node_b"]

    target_cols = tuple(join(all_connectivities(labels)))
    triangle_mask = unstacked.edge.isin(target_cols)
    c = unstacked[triangle_mask].loc[:, ["edge", "FC"]]
    return c.set_index("edge").T

def network_to_network_connectivity(matrix, network_to_idx, pairing_func=combinations):
    """

    Args:
        matrix (_type_): Matrix should have the block structure 
        described in network_to_idx!
        network_to_idx (_type_): _description_

    Yields:
        _type_: _description_
    """
    for network_a, network_b in pairing_func(network_to_idx.index, 2):
        loc_a, loc_b = network_to_idx[network_a], network_to_idx[network_b]
        connectivity = matrix[loc_a[0]:loc_a[1], loc_b[0]:loc_b[1]].mean()
        yield network_a, network_b, connectivity

def edge_counts(block):
    n_positive_edges = np.count_nonzero(block > 0)
    n_negative_edges = np.count_nonzero(block < 0)
    block_activation = (n_negative_edges > 0) or (n_positive_edges > 0)
    return n_positive_edges, n_negative_edges, block_activation
    

def block_block(matrix, network_to_idx, aggregating_func=edge_counts):
    for network_a, network_b in product(network_to_idx.index, network_to_idx.index):
        loc_a, loc_b = network_to_idx[network_a], network_to_idx[network_b]
        block = matrix[loc_a[0]:loc_a[1], loc_b[0]:loc_b[1]]

        yield network_a, network_b, *aggregating_func(block)


def macro_matrix(matrix, network_to_idx):
    gen = block_block(matrix, network_to_idx, aggregating_func=lambda block : (block.mean(),))
    comparisons = pd.DataFrame(gen, columns=["node_a", "node_b", "connectivity"])
    pivoted = comparisons.pivot(index="node_a", columns="node_b")
    return pivoted.loc[:, "connectivity"]

def groupby_blocks(matrix, atlas):
    ticks, sort_index = group_by_networks(atlas.macro_labels)
    matrix_sort = np.ix_(sort_index, sort_index)
    sorted_matrix = matrix[matrix_sort]
    new_labels = sorted(tuple(unique(atlas.macro_labels)))

    network_to_idx = pd.Series(dict(zip(
        new_labels,
        itertools.pairwise(ticks)
    )))
    return macro_matrix(sorted_matrix, network_to_idx), new_labels

def group_by_networks(macro_labels):
    networks = np.array(macro_labels)
    sort_index = np.argsort(networks)

    ticks = []
    lbls = []
    prev_label = None
    for i, label in enumerate(networks[sort_index]):
        if label != prev_label:
            ticks.append(i)
            lbls.append(label)
            prev_label = label

    ticks.append(i+1)
    return ticks, sort_index

def group_groupby(matrices: Iterable, atlas: Atlas) -> Tuple[np.ndarray, list[str]]:
    """Apply grouping by networks to an iterable of connectivity matrices

    Returns:
        Tuple[np.ndarray, list[str]]: Array of new connectivity matrices with
        the corresponding macro labels
    """
    res = (groupby_blocks(mat, atlas) for mat in matrices)
    blocks, labels = zip(*res)
    
    assert all_equal(labels)
    labels = labels[0]

    return blocks, labels