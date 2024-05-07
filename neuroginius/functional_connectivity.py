#computation of functional connectivity matrices
import numpy as np 
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore

def compute_connectivity(data, method='pearson', matrix_form=True, to_numpy=False, **kwargs):
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