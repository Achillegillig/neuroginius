
import sqlite3
import pandas as pd
import numpy as np
import parcellate as par
from pathlib import Path
import os
from abc import ABC

def get_data(SubID, database):
    #database: str referring to the fMRI database. Can be "ISHARE" or "MEMENTO".
    
    exists = check_data_existence(SubID, database)

    if exists:
        return retrieve_from_database(SubID, database)
    else:
        print(f'{SubID} not in the database, computing and saving parcellation')
        # return parcellate_and_retrieve(SubID, database)
        pass


class DataRetriever(ABC):
    # @abstractmethod
    def retrieve(self, SubID):
        pass

class ISHARERetriever(DataRetriever):
    def retrieve(self, SubID):
        pass


def retrieve_from_database(SubID, database):
    pass
    # if database == 'ISHARE':
    #     retriever = 

def retrieve_parcellated_data(SubID, db_file, DataFile, **kwargs):
    """
    Retrieve parcellated data from the database
    """
    atlas = kwargs.get('atlas', None)

    # get the data from the database
    query = f"SELECT * FROM {SubID}"

    # Execute a SQL query and load the result into a DataFrame
    conn = sqlite3.connect(db_file)
    #check if table for subject exists
    cur = conn.cursor()

    # Execute a SQL query to check if the table exists
    cur.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{SubID}';")

    # Fetch the result
    in_df = cur.fetchone()

    # Close the cursor
    cur.close()

    # Check if the table exists
    if in_df:
        data = pd.read_sql_query(query, conn) 
        return data

    print(f'{SubID} not in the database, computing and saving parcellation')

    # if the data is not in the database, parcellate the data
    if atlas is None:
        raise ValueError(f'an atlas is required to parcellate the data. subject {SubID} not found in the database and no atlas provided')
    #atlas = Path(".").absolute().parent / "atlases/"

    img = ''.join(DataFile)

    data = par.parcellate(img, atlas)
    data = pd.DataFrame(data)
    data.to_sql(f'{SubID}', conn, index=True, index_label='ROI')
    
    conn.commit()
    conn.close()

    return data

def retrieve_all_parcellated_data(db_file, **kwargs):
    import warnings
    warnings.warn('retrieve_all_parcellated_data not tested')

    n_sub = kwargs.get('n_sub', None)

    query = "SELECT * FROM parcellated_data"

    conn = sqlite3.connect(db_file)
    # Execute a SQL query and load the result into a DataFrame
    data = pd.read_sql_query(query, conn)

    if data.shape[0] != 0:
        return data
    else:
        raise ValueError('No data in the database')