import pandas as pd
import numpy as np
from nilearn.maskers import NiftiLabelsMasker
from sklearn.metrics import adjusted_mutual_info_score, mutual_info_score
import os
import sqlite3
import itertools
import sys
sys.path.insert(1, '/homes_unix/agillig/neuroginius/neuroginius')
import retrieve_data as rtr
import functional_connectivity as FC
from tqdm.auto import tqdm


from SQLInterface import connectivity as SQLConn


def decoding_pipeline(job, **kwargs):
    debug = kwargs.get('debug', False)
    index = int(job) - 1

    atlas_name = 'schaefer200' 
    #params
    parallel = 'slurm' #whether to use slurm for massive parallelisation. If 'multiproc', will use multiprocessing?
    base_dir = '/homes_unix/agillig/Projects/DynaPred'
    ParcellatedData_db = base_dir + f'/processing/databases/SHARE_parcellated_{atlas_name}.db'
    FCData_db = base_dir + f'/processing/databases/SHARE_FC_{atlas_name}.db'

    rsDataList_file = base_dir + '/processing/MRiShare_gsreg_list.txt'
    rsDataList = np.loadtxt(rsDataList_file, dtype=str)
    SubList = [rsfile.split('subject_id_')[1].split('/')[0] for rsfile in rsDataList]
    SubList.sort()
    Atlases_dir = '/homes_unix/agillig/Atlases' ##TO BE UPDATED
    # atlas_file = '/homes_unix/agillig/Atlases/M5_clean2_ws/RSN_N41_atlas_M5_clean2_wscol.nii'
    atlas_file = Atlases_dir + '/Schaefer2018/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz'
    #INIT DIRS

    #INIT VARS 

    



    ## OPTIONAL: specify type of decoding to be performed via indexing array

    ##


    ###STEP 1: DATA PREPARATION
    #temporary; this is just for me
    # # find the rsDataList files that correspond to the SubID
    # ID_list = [rsfile.split('subject_id_')[1].split('/')[0] for rsfile in rsDataList]
    # #filter = np.isin(filter, SubID)
    # SubID = ID_list[index]
    # rsData_file = rsDataList[index]
    # rsData_file = ''.join(rsData_file)
	
    # if debug:
    #     rsData_file = '/homes_unix/agillig/Projects/DynaPred/data/tmp/SHARE001.nii.gz'

    # print(f'data file: {rsData_file}')



    #TODO: add interface that #1 check if there are missing subjects according to the provided list,
    #2 computes the missing ones, 3 retrieves everything

    # check if all parcellations are here
    missing = rtr.check_tables_in_sql(SubList, ParcellatedData_db)
    print(f'{len(missing)} missing parcellations')
    #temporary too, TODO: implement in neuroginius
    # will work only on pipeau, obviously. but should work
    # if len(missing) > 0:
    #     print(f'{len(missing)} missing parcellations')
    #     for subject in missing:
    #         file = rsDataList[SubList.index(subject)]
    #         print(subject, file)
    #         print(f'computing parcellation for {subject}')
    #         rtr.retrieve_parcellated_data(subject,
    #                                       ParcellatedData_db,
    #                                       DataFile=file,
    #                                       atlas=atlas_file)

    # rsData = rtr.retrieve_all_parcellated_data(ParcellatedData_db,
    #                                            SubList, 
    #                                            compute_missing=False)

    ## TO BUILD: frame filtering thanks to edge time series / RSS

    # print('computing functional connectivity')
    # for data in tqdm(rsData):
    #     tmp_fc_data = FC.compute_connectivity(data, method='pearson', matrix_form=False)

    # print('loading functional connectivity matrices')
    SubList = [sub for sub in SubList if sub not in missing]
    SubList = SubList[:100]
    # FCData = rtr.get_fc_data(FCData_db, SubList, check_missing=True, compute_missing=True, parcellation_db=ParcellatedData_db, matrix_form=False, atlas=atlas_file)

    FCData, SubList_new = SQLConn.ConnectivityFromParcellationSQL(ParcellatedData_db,
                                                                  SubList=SubList, 
                                                                  method='pearson', 
                                                                  n_jobs=16,
                                                                  ConnectivityDb=FCData_db
                                                                  )
    print(type(FCData))
    print(len(FCData))
    print(SubList_new[0], SubList[0], FCData[0])

    differences = [(i, a, b) for i, (a, b) in enumerate(zip(SubList, SubList_new)) if a != b]

    for i, a, b in differences:
        print(f"Element at index {i} is not the same: {a} in SubList, {b} in SubList_new")
    ## TO BUILD: decoding with nested cv & non parametric testing

    ## To build: permutation feature importance to interpret decoding drivers

    ## end goal: "multivariate" optimization to find the network that best predicts individual differences


    print('end of file, done')


if __name__ == '__main__':
    decoding_pipeline(1)