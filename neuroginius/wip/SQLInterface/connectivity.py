from joblib import Parallel, delayed
import sqlite3
from SQLInterface import RetrieveData
import functional_connectivity as FC
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import warnings


 

def ConnectivityToSQL(FCData_db, SubList, if_exists='continue', n_jobs=None):
    """
    Compute the functional connectivity matrices for a list of subjects and save them to a SQL database
    :param FCData_db: string, path to the database where the functional connectivity matrices will be saved
    :param SubList: list of strings, the list of subjects
    :param if_exists: string, 'continue' or 'replace'
    """
    pass

def ConnectivityFromParcellationSQL(ParcellationDb, SubList=None, method='pearson', n_jobs=1, ConnectivityDb=None, in_memory=False):
    """
    Compute the functional connectivity matrices from a parcellation database
    :param ParcellationDb: string, path to the database where the parcellation data is stored
    :param SubList: list of strings, the list of subjects
    :return: 3D numpy array, the functional connectivity matrices
    """
    if type(SubList) == 'str':
        SubList = [SubList]

    print('loading parcellated data')
    DataList, SubIDs = RetrieveData.RetrieveTablesFromSQL(ParcellationDb, SubList, index_col='ROI', n_jobs=n_jobs)
    #!!!!! fix the orientation of parcellated data directly in df
    warnings.warn('Incorrect orientation of parcellated data, fix the saving parcellation step')
    DataList = [data.T for data in DataList]

    print(DataList[0].shape)
    print('computing functional connectivity')
    FCData = Parallel(n_jobs=n_jobs)(delayed(FC.compute_connectivity)(data, method=method, matrix_form=False) for data in tqdm(DataList))

    FCData = pd.concat(FCData, axis=0).T
    FCData.columns = [i for i in range(FCData.shape[1])]
    print(FCData.columns)
    print(FCData.shape)
    if ConnectivityDb is not None:
        print('saving to database')
        # [RetrieveData.SaveToSQLTable(data, ConnectivityDb, index_label='ID') for data, ID in tqdm(zip(FCData, SubIDs))]
        
        if not in_memory:
            # Parallel(n_jobs=1)(delayed(RetrieveData.SaveToSQLTable)(data, ConnectivityDb, index_label='ID') for data, ID in tqdm(zip(FCData, SubIDs)))
            RetrieveData.SaveToSQLTable(FCData, ConnectivityDb, index_label='ID')
        else:
            conn = sqlite3.connect(':memory:', check_same_thread=False)
            # Parallel(n_jobs=n_jobs)(delayed(RetrieveData.save_to_sql_in_memory)(data, conn, index_label='ID') for data, ID in tqdm(zip(FCData, SubIDs)))
            # Use the function in your parallel job
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                futures = [executor.submit(RetrieveData.save_to_sql_in_memory, data, conn, index_label='ID') for data, ID in tqdm(zip(FCData, SubIDs))]
                results = [future.result() for future in futures]
            #dump to file
            conn_disk = sqlite3.connect(ConnectivityDb)
            conn.backup(conn_disk)
        print(f'saved to {ConnectivityDb}')
    return FCData, SubIDs