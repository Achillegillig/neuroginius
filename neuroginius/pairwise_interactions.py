#computation of functional connectivity matrices
import itertools
import numpy as np 
import pandas as pd
from scipy.stats import t
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore, spearmanr
from statsmodels.tsa.api import VAR
from numpy.matlib import repmat

def compute_connectivity(data, method='pearson', matrix_form=True, to_numpy=False,
                         fisher_transform=True,
                         **kwargs):
    """
    Compute the correlation matrix of a given dataset
    :param data: 2D numpy array
    :param method: string, 'pearson'
    :return: 2D numpy array, the correlation matrix
    :or a pandas DataFrame if nodenames are provided
    """
    nodenames = kwargs.get('nodenames', None)
    #z score data
    # data = zscore(data, axis=1)

    if method == 'pearson':
        # corr_matrix = np.corrcoef(data, rowvar=rowvar)
        corr_matrix = np.corrcoef(data, rowvar=True)
    else:
        raise ValueError('method should be pearson. The developpers were too lazy to implement another method yet.')
    if fisher_transform:
        print('Fisher transforming the correlation matrix')
        corr_matrix = np.arctanh(corr_matrix)
        
    if nodenames is not None:
        if len(nodenames) != corr_matrix.shape[0]:
            raise ValueError('Number of nodenames should be equal to the number of nodes in the data')
        return pd.DataFrame(corr_matrix, index=nodenames, columns=nodenames)
    # elif to_numpy:
    #     return corr_matrix.reshape(1,-1)
    else:
        if matrix_form == False:
            # get upper triangle
            corr_matrix = corr_matrix[np.triu_indices(corr_matrix.shape[1], k=1)]
            corr_matrix = corr_matrix.reshape(1,-1).squeeze()
        if to_numpy:
            return corr_matrix
        return pd.DataFrame(corr_matrix, columns=[i for i in range(corr_matrix.shape[1])])
    



def multivariate_distance_correlation(X, centering='U-centering', standardize=True):
    '''
    Translated from https://github.com/rayksyoo/dCorMV/blob/master/calc_distCorrMV.m
    All credits to the original authors
    MIT License
    Copyright 2019. Kwangsun Ray Yoo (K Yoo), PhD
    E-mail: kwangsun.yoo@yale.edu / rayksyoo@gmail.com
    Dept. of Psychology
    Yale University

    ----------------
    parameters:
    X: list of numpy arrays: list of time series of n nodes / brain regions. Each numpy array has shape (n_timepoints, n_voxels)

    centering: str: 'U-centering' or 'D-centering'
    whether to use U-centering or D-centering for distance matrix centering

    ----------------
    returns:
    dCor: numpy array: distance correlation matrix (n x n)
    '''
    m = len(X)
    n = X[0].shape[0]

    if standardize:
        X = [zscore(region_ts, axis=0) for region_ts in X]

    Edist_centered = []

    for x, region_ts in enumerate(X):
        # Removing voxels of which values are zero all the time
        check_nonzero = np.where(np.sum(region_ts, axis=0) != 0, True, False)
        region_ts = region_ts[:,check_nonzero]

        # Euclidean distance among time points of nodes in X
        temp_Edist1d = pdist(region_ts, metric='euclidean')

        # 1D distance array to 2D matrix
        temp_Edist = squareform(temp_Edist1d)

        # Centering
        if centering == 'U-centering':
            # U-centering
            # temp_EdistCent = temp_Edist - np.sum(temp_Edist, axis=1, keepdims=False) / (n-2) - np.sum(temp_Edist, axis=0, keepdims=False) / (n-2) + np.sum(temp_Edist, keepdims=False) / ((n-1)*(n-2))
            temp_EdistCent = temp_Edist - repmat(np.sum(temp_Edist, axis=1, keepdims=True) / (n-2), 1, n) \
                                 - repmat(np.sum(temp_Edist, axis=0, keepdims=True) / (n-2), n, 1) \
                                 + repmat(np.sum(temp_Edist, keepdims=True) / ((n-1)*(n-2)), n, n)
            np.fill_diagonal(temp_EdistCent, 0)
        elif centering == 'D-centering':
            # Double-centering
            temp_EdistCent = temp_Edist - np.mean(temp_Edist, axis=1) - np.mean(temp_Edist, axis=0) + np.mean(temp_Edist)
        else: 
            raise ValueError('centering should be either U-centering or D-centering')
        Edist_centered.append(temp_EdistCent)

    Edist_centered = np.stack(Edist_centered, axis=2)

    if centering == 'U-centering':
        K = n * (n - 3)
    else:
        K = n ** 2

    # Distance variance
    # dVar = np.sum(np.sum(Edist_centered ** 2, axis=0), axis=0) / K
    dVar = (Edist_centered ** 2).sum(axis=0).sum(axis=0) / K

    # Distance covariance
    dCov = np.zeros((m, m))
    for nd1 in range(m-1):
        for nd2 in range(nd1 + 1, m):
            dCov[nd1, nd2] = np.sum(Edist_centered[:, :, nd1] * Edist_centered[:, :, nd2]) / K
    dCov = dCov + dCov.T
    np.fill_diagonal(dCov, dVar)
    # Distance correlation
    dVarSqrt = np.sqrt(np.outer(dVar, dVar))
    dCor = np.sqrt(dCov / dVarSqrt)
    dCor[dCov <= 0] = 0
    np.fill_diagonal(dCor, 0)

    # Stats
    stats = {}
    stats['df'] = K / 2 - 1
    stats['T'] = np.sqrt(stats['df']) * (dCor ** 2 / np.sqrt(1 - dCor ** 4))
    stats['p'] = t.cdf(stats['T'], stats['df'])

    # return dCor, dCov, dVar, stats
    return dCor


def ar_connectivity(X, lag=1, time_first=True):
    '''
    X: shape (n_timepoints, n_nodes) if time_first or (n_nodes, n_timepoints)
    '''
    if time_first == False:
        X = X.T

    model = VAR(X)
    results = model.fit(maxlags=lag)

    # Extract the weight matrices (coefficient matrices for lags)
    coef_matrices = results.coefs[lag-1]  # Shape: (lags, n_vars, n_vars)
    coef_matrices = coef_matrices[np.triu_indices(coef_matrices.shape[1], k=1)].reshape(1,-1).squeeze()
    return coef_matrices


def multivariate_integration(X_ts, standardize=True, time_first=False):
    '''
    X: list with each element of shape (n_nodes, n_timepoints)
    '''
    if time_first:
        X_ts = [X.T for X in X_ts]

    if standardize:
        X_ts = [zscore(X, axis=0) for X in X_ts]

    blocks = np.arange(len(X_ts))
    print(blocks)
    block_pairs = list(itertools.combinations(range(len(X_ts)), 2))
    seen = {}
    initial_data = np.zeros((len(X_ts), len(X_ts)))
    np.fill_diagonal(initial_data, 1)
    results = pd.DataFrame(initial_data, index=blocks, columns=blocks)

    for block_a, block_b in block_pairs:
        if (block_a, block_b) in seen:
            continue
        if (block_b, block_a) in seen:
            continue
        if block_a == block_b:
            continue

        X_dist = pdist(X_ts[block_a].T, metric='cosine')
        X_sim_a = 1 / (1 + X_dist)   

    
        X_dist_b = pdist(X_ts[block_b].T, metric='cosine')
        X_sim_b = 1 / (1 + X_dist_b)

        sim = spearmanr(X_sim_a, X_sim_b).statistic
        results.loc[block_a, block_b] = sim
        results.loc[block_b, block_a] = sim

        seen[(block_a, block_b)] = True

    return results.values[np.triu_indices_from(results, k=1)].reshape(1,-1).squeeze()

def regionavg_multivariate_integration(X_ts, network_labels, standardize=True):
    '''
    X: shape (n_nodes, n_timepoints)
    '''

    if standardize:
        X_ts = zscore(X_ts, axis=0)

    blocks = np.unique(network_labels)
    seen = {}
    initial_data = np.zeros((len(blocks), len(blocks)))
    np.fill_diagonal(initial_data, 1)
    results = pd.DataFrame(initial_data, index=blocks, columns=blocks)

    for block_a in blocks:
        for block_b in blocks:
            if (block_a, block_b) in seen:
                continue
            if (block_b, block_a) in seen:
                continue
            if block_a == block_b:
                continue
            mask = np.array([1 if lbl == block_a else 0 for lbl in network_labels], dtype=bool)
            X_masked = X_ts[mask,:]
            X_dist = pdist(X_masked.T, metric='cosine')
            X_sim_a = 1 / (1 + X_dist)

            mask = np.array([1 if lbl == block_b else 0 for lbl in network_labels], dtype=bool)
            X_masked = X_ts[mask,:]
            X_dist_b = pdist(X_masked.T, metric='cosine')
            X_sim_b = 1 / (1 + X_dist_b)

            sim = spearmanr(X_sim_a, X_sim_b).statistic
            results.loc[block_a, block_b] = sim
            results.loc[block_b, block_a] = sim

            seen[(block_a, block_b)] = True

    return results.values[np.triu_indices_from(results, k=1)].reshape(1,-1).squeeze()