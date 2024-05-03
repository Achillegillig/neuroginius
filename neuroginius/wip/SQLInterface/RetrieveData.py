
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
import functional_connectivity as FC
from joblib import Parallel, delayed
import warnings
from threading import Lock

def RetrieveTablesFromSQL(db_file, table_names=None, index_col=None, n_jobs=1):
    #TODO: implement joblib / mutltithread parallelization
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    tables = [table[0] for table in tables]
    cursor.close()
    
    if table_names is not None:
        tables = table_names
        if type(tables) == str:
            query = f"SELECT * FROM {tables}"
            data = pd.read_sql_query(query, conn, index_col=index_col)
            return data, tables
    conn.close()
    
    data = []
    # for table in tqdm(tables):
        # query = f"SELECT * FROM {table}"
        # data.append(pd.read_sql_query(query, conn, index_col=index_col))
        # conn.close()
    # args = [(db_file, table, index_col) for table in tables]
    timeout=9999
    # Ignore UserWarning: A worker stopped while some jobs were given to the executor
    warnings.filterwarnings("ignore", message="A worker stopped while some jobs were given to the executor")
    data = Parallel(n_jobs=n_jobs, timeout=timeout)(delayed(retrieve_table)(db_file, table_name, index_col) for table_name in tqdm(tables))
    return data, tables
    

    df_list = []
    for table_name in tqdm(tables):
        if type(table_name) == 'tuple':
            table_name = table_name[0]
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, conn, index_col=index_col)
        df_list.append(df)

    return df_list

def retrieve_table(db_file, table_name, index_col, conn=None):
    with sqlite3.connect(db_file) as conn:
        query = f"SELECT * FROM {table_name}"
        return pd.read_sql_query(query, conn, index_col=index_col)


def SaveToSQLTable(data, db_file, index_label=None, table_name='data', **kwargs):
    # if isinstance(data, pd.DataFrame) == False:
    #     if data.shape[0] != 1:
    # data = data.reshape(1, -1)
    
    if index_label is None:
        index = False
    else:
        index = True
    conn = sqlite3.connect(db_file)
    lock = Lock()
    with lock:
        while True:
            try:
                data.to_sql(f'{table_name}',
                                conn,
                                if_exists='append',
                                index=index,
                                index_label=index_label)
                conn.close()
                break
            except sqlite3.OperationalError:
                time.sleep(0.01)
                data.to_sql(f'{table_name}',
                                conn,
                                if_exists='append',
                                index=index,
                                index_label=index_label)
    conn.close()


def save_to_sql_in_memory(data, conn, index_label=None, table_name='data', **kwargs):
    # if isinstance(data, pd.DataFrame) == False:
    #     if data.shape[0] != 1:
    # data = data.reshape(1, -1)
    data = pd.DataFrame(data)
    
    if index_label is None:
        index = False
    else:
        index = True

    # lock = Lock()

    # with lock:
    while True:
        try:
            data.to_sql(f'{table_name}',
                            conn,
                            if_exists='append',
                            index=index,
                            index_label=index_label)
            break
        except sqlite3.OperationalError:
            time.sleep(0.01)
            data.to_sql(f'{table_name}',
                            conn,
                            if_exists='append',
                            index=index,
                            index_label=index_label)



def get_data(SubID, database):
    #database: str referring to the fMRI database. Can be "ISHARE" or "MEMENTO".
    
    exists = check_data_existence(SubID, database)

    if exists:
        return retrieve_from_database(SubID, database)
    else:
        print(f'{SubID} not in the database, computing and saving parcellation')
        # return parcellate_and_retrieve(SubID, database)
        pass

def get_fc_data(db_file, SubList=None, check_missing=True, compute_missing=True, parcellation_db=None, matrix_form=False, atlas=None, **kwargs):
    args = (db_file, SubList, check_missing, compute_missing,parcellation_db, matrix_form, atlas)
    #TODO: add support for parcellated data as pandas df / np array
    #TODO: make it abstract and applicable to different fetching types such as pickle, csv, etc.
    #TODO: take out automatic computing of missing subjects. this should be handled by another function
    if type(SubList) == 'str':
        SubList = [SubList]

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
    missing = check_missing_datatables(SubList, table_names)

    #TODO: ignore files for which parcellation is also missing

    if len(missing) == 0 or check_missing == False:
        return retrieve_from_sql_table(db_file, index_col=None)
    else:
        raise ValueError(f'{len(missing)} subjects not in the database, compute and save functional connectivity first')
        # return parcellate_and_retrieve(SubID, database)
        if parcellation_db is None:
            raise ValueError('parcellation_db is required to compute functional connectivity')
        # for SubID in tqdm(missing):
        #     if check_table_in_sql(SubID, parcellation_db) == False or compute_missing == False:
        #         continue
        #     data = retrieve_parcellated_data(SubID, parcellation_db, atlas=atlas, compute_missing=compute_missing)
        #     FC.compute_connectivity(data, matrix_form=matrix_form)
        #     SaveToSQLTable(SubID, data, db_file, SubID)

        Parallel(n_jobs=40)(delayed(process_fc_subid)(SubID, parcellation_db, atlas, compute_missing, matrix_form, db_file) for SubID in tqdm(missing))
        return get_fc_data(*args, **kwargs)

def process_fc_subid(SubID, parcellation_db, atlas, compute_missing, matrix_form, db_file):
    if check_table_in_sql(SubID, parcellation_db) == False or compute_missing == False:
        return
    data = retrieve_parcellated_data(SubID, parcellation_db, atlas=atlas, compute_missing=compute_missing)
    fc_data = FC.compute_connectivity(data, matrix_form=matrix_form)
    max_retry = 20
    c = 0
    while c < max_retry:
        try:
            return SaveToSQLTable(fc_data, db_file, SubID)
        except sqlite3.OperationalError:
            time.sleep(0.01)
            c += 1
            continue


def check_missing_datatables(SubList, table_names):
    return [SubID for SubID in SubList if SubID not in table_names]


def check_data_in_sql_table(SubID, db_file, colname='SHARE_ID'):
    #works
    conn = sqlite3.connect(db_file)

    cur = conn.cursor()
    # Execute a SQL query to get all table names
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    # Fetch all results
    table_names = cur.fetchall()
    table_name = table_names[0][0]
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
                retry(SaveToSQLTable, SubID, data, db_file, SubID)
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

def retrieve_from_sql_table(db_file, SubID=None, index_col="SHARE_ID"):
    #NOT TESTED; TEMPLATE
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    # Execute a SQL query to get all table names
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    # Fetch all results
    table_name = cur.fetchone()[0]

    # Get the table info
    cur.execute(f"PRAGMA table_info({table_name})")
    table_info = cur.fetchall()

    # Close the cursor and connection
    cur.close()
    
    if SubID is None:
        query = f"SELECT * FROM {table_name}"
    else:
        query = f"SELECT * FROM {table_name} WHERE {index_col} = '{SubID}'"

    data = pd.read_sql_query(query, conn, index_col=index_col)
    conn.close()
    return data

    

    


# def retry(func, *args, **kwargs):
#     max_retry_count = 10  # Maximum number of retries
#     retry_count = 0

#     while retry_count < max_retry_count:
#         try:
#             return func(*args, **kwargs)
#         except sqlite3.OperationalError as e:
#             if 'database is locked' in str(e):
#                 time.sleep(1)  # Wait for 1 second
#                 retry_count += 1
#                 if retry_count == max_retry_count:  # If this was the last retry
#                     raise Exception("Maximum number of retries exceeded")  # Raise a custom exception