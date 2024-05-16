import numpy as np
from scipy.spatial.distance import squareform
from scipy.stats import zscore
from itertools import product
import pandas as pd

def compute_edge_timeseries(rsData, matrix_form=False):
    
    # zscore
    rsData = zscore(rsData, axis=0)

    pairwise_products = np.array([np.outer(rsData[:, i], rsData[:, i]) for i in range(rsData.shape[1])])

    if matrix_form:
        edge_ts = pairwise_products
        # np.fill_diagonal(edge_ts, 0)
    else:
        indices_uppertriangle = [np.triu_indices(pairwise_products[0].shape[0], 1) for i in range(pairwise_products.shape[0])]
        edge_ts = np.array([prod[ind] for prod, ind in zip(pairwise_products, indices_uppertriangle)])
    return edge_ts

def label_ets(labels):
    """
    labels: name of ROIs, size n_ROIs
    return: ets_labels
    """
    indices = np.triu_indices(len(labels), 1)

    ets_labels = []
    for i, (x,y) in enumerate(zip(*indices)):
        ets_labels.append((labels[x], labels[y]))

    return ets_labels

def ets_labels_to_int(ets_labels):
    # Normalize each label by sorting the tuples
    normalized_labels = [tuple(sorted(label)) for label in ets_labels]
    
    # Create a list of unique labels using a set to remove duplicates, then convert it back to a list and sort it
    unique_labels = sorted(list(set(normalized_labels)))
    
    # Create an array of integer labels where each label corresponds to the index of the tuple in unique_labels plus one
    int_labels = np.array([unique_labels.index(label) + 1 for label in normalized_labels], dtype=int)
    
    return int_labels
        
