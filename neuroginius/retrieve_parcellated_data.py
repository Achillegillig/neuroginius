
import sqlite3
import pandas as pd
import parcellate as par
from pathlib import Path


def retrieve_parcellated_data(SubID, db_file, **kwargs):
    """
    Retrieve parcellated data from the database
    """
    atlas = kwargs.get('atlas', None)

    # get the data from the database
    query = f"SELECT * FROM parcellated_data WHERE SubID = '{SubID}'"

    conn = sqlite3.connect(db_file)
    # Execute a SQL query and load the result into a DataFrame
    data = pd.read_sql_query(query, conn)

    if sequences_df.shape[0] != 0:
        return data

    print(f'{SubID} not in the database, computing and saving parcellation')
    # if the data is not in the database, parcellate the data
    if atlas is None:
        raise ValueError(f'an atlas is required to parcellate the data. subject {SubID} not found in the database and no atlas provided')
    atlas = Path(".").absolute().parent / "atlases/"

    data = par.parcellate(SubID, atlas)
    data = pd.DataFrame(data, index=[SubID])
    data.to_sql('parcellated_data', conn, if_exists='append', index=True, index_label='SubID')
    
    conn.commit()
    conn.close()

    return data