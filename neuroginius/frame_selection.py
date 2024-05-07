import numpy as np
import sys
sys.path.insert(1, '/homes_unix/agillig/Projects/DynaPred/code')

import edge_ts as ets

def framewise_selection(X, lower_q, upper_q, criterion='RSS'):
    """
    Select frames based on the given criterion.
    
    Parameters
    ----------
    X : array-like, rs timeseries; shape (n_features, n_TR)
        The input data.

    lower_q : float
        The lower quantile.
    upper_q : float
        The upper quantile.
    criterion : str, default='RSS'
        The criterion to use for selecting frames.
        Possible values are 'RSS' and 'BIC'.
    
    Returns
    -------
    X_selected : array-like, shape (n_edges, n_selected_TR)
    """
    n_edges, n_TR = X.shape

    edge_ts = ets.compute_edge_ts(X)

    if criterion == 'RSS':
        rss = np.sqrt(np.sum(edge_ts**2, axis=1)).reshape(1,-1)
        selector = rss
        # print(f'shape of RSS: {rss.shape}')
    else:
        raise ValueError("Invalid criterion: %s, valid choices are: RSS (yes that's it)" % criterion)
    #identify the lower and upper quantile of the RSS
    lower = np.percentile(selector, lower_q)
    upper = np.percentile(selector, upper_q)

    lower_mask = selector > lower
    upper_mask = selector < upper
    selection_mask = np.logical_and(lower_mask, upper_mask).squeeze()
    X_selected = X[:, selection_mask]

    return X_selectedimport numpy as np
import sys
sys.path.insert(1, '/homes_unix/agillig/Projects/DynaPred/code')

import edge_ts as ets

def framewise_selection(X, lower_q, upper_q, criterion='RSS'):
    """
    Select frames based on the given criterion.
    
    Parameters
    ----------
    X : array-like, rs timeseries; shape (n_features, n_TR)
        The input data.

    lower_q : float
        The lower quantile.
    upper_q : float
        The upper quantile.
    criterion : str, default='RSS'
        The criterion to use for selecting frames.
        Possible values are 'RSS' and 'BIC'.
    
    Returns
    -------
    X_selected : array-like, shape (n_edges, n_selected_TR)
    """
    n_edges, n_TR = X.shape

    edge_ts = ets.compute_edge_ts(X)

    if criterion == 'RSS':
        rss = np.sqrt(np.sum(edge_ts**2, axis=1)).reshape(1,-1)
        selector = rss
        # print(f'shape of RSS: {rss.shape}')
    else:
        raise ValueError("Invalid criterion: %s, valid choices are RSS (yes that's it)" % criterion)
    #identify the lower and upper quantile of the RSS
    lower = np.percentile(selector, lower_q)
    upper = np.percentile(selector, upper_q)

    lower_mask = selector > lower
    upper_mask = selector < upper
    selection_mask = np.logical_and(lower_mask, upper_mask).squeeze()
    X_selected = X[:, selection_mask]

    return X_selected