#computation of functional connectivity matrices
import numpy as np 
import pandas as pd
from scipy.spatial.distance import pdist, squareform

def compute_correlation_matrix(data, method='pearson', matrix_form=True, **kwargs):
    """
    Compute the correlation matrix of a given dataset
    :param data: 2D numpy array
    :param method: string, 'pearson'
    :return: 2D numpy array, the correlation matrix
    :or a pandas DataFrame if nodenames are provided
    """
    nodenames = kwargs.get('nodenames', None)

    if method == 'pearson':
        # corr_matrix = np.corrcoef(data, rowvar=rowvar)
        if matrix_form:
            corr_matrix = squareform(pdist(data, metric='correlation'))
        else:
            corr_matrix = pdist(data, metric='correlation')
    else:
        raise ValueError('Method should be pearson')
     
    if nodenames is not None:
        if len(nodenames) != corr_matrix.shape[0]:
            raise ValueError('Number of nodenames should be equal to the number of nodes in the data')
        return pd.DataFrame(corr_matrix, index=nodenames, columns=nodenames)
    else: 
        return corr_matrix
    