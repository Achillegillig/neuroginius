
import sqlite3
import pandas as pd
import numpy as np
import parcellate as par
from pathlib import Path
import os
from abc import ABC
import warnings
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
import time
import FunctionalConnectivity as FC

def get_data(SubID, database):
    #database: str referring to the fMRI database. Can be "ISHARE" or "MEMENTO".
    
    exists = check_data_existence(SubID, database)

    if exists:
        return retrieve_from_database(SubID, database)
    else:
        print(f'{SubID} not in the database, computing and saving parcellation')
        # return parcellate_and_retrieve(SubID, database)
        pass

def get_fc_data(db_file, SubList, compute_missing=True, parcellation_db=None, matrix_form=False, atlas=None):
    #TODO: add support for parcellated data as pandas df / np array
    if type(SubList) == 'str':
        SubList = [SubList]
    if matrix_form:
        warnings.warn('Warning: matrix_form=True. will break the database if automatic saving')
    #check if all subjects are in the database
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    # Execute a SQL query to get all table names
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    # Fetch all results
    table_names = cur.fetchall()
    # Close the cursor and connection
    cur.close()
    conn.close()
    table_names = [table[0] for table in table_names]
    missing = [SubID for SubID in SubList if SubID not in table_names]
    if len(missing) == 0:
        return retrieve_tables_from_sql(db_file)
    else:
        print(f'subjects not in the database, computing and saving functional connectivity')
        # return parcellate_and_retrieve(SubID, database)
        if parcellation_db is None:
            raise ValueError('parcellation_db is required to compute functional connectivity')
        for SubID in tqdm(missing):
            if check_table_in_sql(SubID, parcellation_db) == False and compute_missing == False:
                continue
            data = retrieve_parcellated_data(SubID, parcellation_db, atlas=atlas, compute_missing=compute_missing)
            FC.compute_correlation_matrix(data, matrix_form=matrix_form)
            save_to_sql_table(SubID, data, db_file, SubID)

def check_missing_datatables(SubList, table_names):
    return [SubID for SubID in SubList if SubID not in table_names]

def check_data_in_sql_table(SubID, db_file, colname):
    #works
    conn = sqlite3.connect(db_file)

    cur = conn.cursor()
    # Execute a SQL query to get all table names
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    # Fetch all results
    table_names = cur.fetchall()
    print(table_names)
    table_name = table_names[0][0]
    print(table_name)
    # Close the cursor and connection
    cur.close()

    cur = conn.cursor()
    # Execute a SQL query to check if the table exists
    cur.execute(f"SELECT EXISTS(SELECT 1 FROM {table_name} WHERE {colname} = '{SubID}')")

    # Fetch the result
    in_df = cur.fetchone()[0]
    # Close the cursor
    cur.close()
    conn.close()
    print(in_df)
    # Check if the table exists
    if in_df:
        return True
    else:
        return False
    
def check_table_in_sql(table_name, db_file):
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    # Execute a SQL query to check if the table exists
    cur.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")

    # Fetch the result
    in_df = cur.fetchone()

    if in_df is not None:
        in_df = in_df[0]   

    # Close the cursor
    cur.close()

    # Check if the table exists
    if in_df:
        return True
    else:
        return False
    
def check_tables_in_sql(SubList, db_file):
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    # Execute a SQL query to check if the table exists
    cur.execute(f"SELECT name FROM sqlite_master WHERE type='table';")

    # Fetch the result
    in_df = cur.fetchall()
    in_df = [table[0] for table in in_df]
    cur.close()
    conn.close()

    missing = [SubID for SubID in SubList if SubID not in in_df]
    return missing

    

def check_missing_tables(SubList, db_file):
    #terribly inefficient
    return [SubID for SubID in SubList if not check_table_in_sql(SubID, db_file)]
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

def retrieve_parcellated_data(SubID, db_file, DataFile=None, compute_missing=True, **kwargs):
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

    if compute_missing == False:
        print('computing missing data is disabled, exciting')
        return

    # if the data is not in the database, parcellate the data
    if atlas is None:
        raise ValueError(f'an atlas is required to parcellate the data. subject {SubID} not found in the database {db_file} and no atlas provided')
    #atlas = Path(".").absolute().parent / "atlases/"
    if DataFile is None:
        raise ValueError('DataFile is required to parcellate the data')
    img = ''.join(DataFile)

    data = par.parcellate(img, atlas)
    data = pd.DataFrame(data)
    try:
        data.to_sql(f'{SubID}',
                    conn,
                    index=True,
                    index_label='ROI')
    except sqlite3.OperationalError:
        time.sleep(0.01)
        data.to_sql(f'{SubID}',
                    conn,
                    index=True,
                    index_label='ROI')

        
    
    
    conn.commit()
    conn.close()

    return data

def retrieve_all_parcellated_data(db_file, SubList, **kwargs):
    #TODO, add condition to check n_subjects vs len(database); compute the missing subjects
    compute_missing = kwargs.get('compute_missing', False)
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    tables = [table[0] for table in tables]
    # Retrieve all tables and store them in a list
    df_list = []

    missing = check_missing_datatables(SubList, tables)

    if len(missing) != 0:
        print('missing parcellations for some subjects')
        if compute_missing:
            parcellation_db = kwargs.get('parcellation_db', None)
            atlas = kwargs.get('atlas', None)
            if parcellation_db is None or atlas is None:
                raise ValueError('parcellation_db and atlas are required to compute missing subjects')
            for SubID in list(missing):
                data = retrieve_parcellated_data(SubID, parcellation_db, atlas=atlas)
                save_to_sql_table(SubID, data, db_file, SubID)
        else:
            print('missing parcellations will not be computed')


    with ThreadPoolExecutor() as executor:
        for table_name in tqdm(tables):
            query = f"SELECT * FROM {table_name}"
            # df = pd.read_sql_query(query, conn)
            df = pd.read_sql_query(query,
                                   conn
                                   )
            df_list.append(df)

    # df = pd.concat(df_list, axis=2)

    return df_list

def retrieve_from_sql_table(db_file, SubID=None):
    #NOT TESTED; TEMPLATE
    pass
    

def retrieve_tables_from_sql(db_file, table_names=None, index_col=None):

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()

    if table_names is not None:
        if type(table_names) != str:
            tables = table_names
        else:
            query = f"SELECT * FROM {table_names}"
            data = pd.read_sql_query(query, conn, index_col='ROI')
            return data

    

    df_list = []
    print(tables)
    for table_name in tqdm(tables):
        if type(table_name) == 'tuple':
            table_name = table_name[0]
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, conn, index_col=index_col)
        df_list.append(df)

    return df_list

def save_to_sql_table(SubID, data, db_file, table_name='data', index_label=None, **kwargs):
    if index_label is None:
        index = False
    else:
        index = True
    conn = sqlite3.connect(db_file)
    data.to_sql(f'{table_name}',
                    conn,
                    index=index,
                    index_label=index_label)
    conn.close()