import numpy as np
from scipy.spatial.distance import squareform
from scipy.stats import zscore

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