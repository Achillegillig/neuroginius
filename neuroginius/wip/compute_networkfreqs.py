
def compute_network_freqs(job, **kwargs):
    debug = kwargs.get('debug', False)
    index = int(job) - 1 
    import pandas as pd
    import numpy as np
    from nilearn.maskers import NiftiLabelsMasker
    from sklearn.metrics import adjusted_mutual_info_score
    import os
    import sqlite3

    base_dir = '/homes_unix/agillig/Projects/DynaPred'
    #SubList_file = base_dir + '/processing/SHARE_tmp_SubID.csv'
    #SubList = pd.read_csv(SubList_file, header=0, index_col=0).to_numpy().flatten()

    #SubID = SubList[index]

    rsDataList_file = base_dir + '/processing/MRiShare_gsreg_list.txt'
    rsDataList = np.loadtxt(rsDataList_file, dtype=str)

    # find the rsDataList files that correspond to the SubID
    ID_list = [rsfile.split('subject_id_')[1].split('/')[0] for rsfile in rsDataList]
    #filter = np.isin(filter, SubID)
    SubID = ID_list[index]
    rsData_file = rsDataList[index]
    rsData_file = ''.join(rsData_file)
	
    if debug:
        rsData_file = '/homes_unix/agillig/Projects/DynaPred/data/tmp/SHARE001.nii.gz'

    print(f'data file: {rsData_file}')

    # extract data from the file using NiftiLabelMasker (mean within parcels)
    atlas_file = '/homes_unix/agillig/Atlases/Schaefer2018/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz'

    LabelMasker = NiftiLabelsMasker(labels_img=atlas_file, strategy='mean',
                  standardize='zscore_sample')

    print(f'loading data for subject {SubID}')
    rsData = LabelMasker.fit_transform(rsData_file).T
    print('done')


    ## create network partitions ##
    print('creating network partitions')
    Atlases_dir = '/homes_unix/agillig/Atlases' ##TO BE UPDATED
    centroids_file = Atlases_dir + '/Schaefer2018/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv' 

    centroids = pd.read_csv(centroids_file)

    networks = centroids['ROI Name'].to_numpy()
    networks = [netstr.split('_')[2] for netstr in networks]
    networks = np.array(networks)

    networks_names = np.unique(networks)

    templates = []
    for network in networks_names:
        print(f'creating partition for network {network}')
        template_mask = np.array(networks == network).astype(int)

        pairwise_products = np.outer(template_mask, template_mask)

        indices = np.triu_indices(pairwise_products[0].shape[0], 1)
        template_edges = np.array(pairwise_products[indices])

        templates.append(template_edges)

    templates = pd.DataFrame(np.array(templates).T, columns=networks_names)


    # compute the edge time series & bipartition
    print('computing edge time series')
    ### step1 : pairwise products between the z scores of the time series - > edge time series
    pairwise_products = np.array([np.outer(rsData[:, i], rsData[:, i]) for i in range(rsData.shape[1])])

    indices_uppertriangle = [np.triu_indices(pairwise_products[0].shape[0], 1) for i in range(pairwise_products.shape[0])]
    edge_ts = np.array([prod[ind] for prod, ind in zip(pairwise_products, indices_uppertriangle)])

    bipartitions = (edge_ts > 0).astype(int)


    # compute the mutual information at each time point between the edge time series and the bipartition
    print(f'computing mutual information with network templates')
    mutual_info = []
    for i, network in enumerate(np.unique(networks)):
        template = templates[network]

        if debug:
            print(f'shape of template: {template.shape}, shape of bipartitions: {bipartitions.shape}')
        tmp_mutual_info = [adjusted_mutual_info_score(x, template) for x in bipartitions]
        tmp_mutual_info = np.array(tmp_mutual_info)
        mutual_info.append(tmp_mutual_info)

    # threshold of significance of the mutual information; computed using random permutations of the network labels
    value_significance_file = base_dir + '/processing/SHARE_tmp_value_significance.csv'
    value_significance = np.loadtxt(value_significance_file)

    print(f'computing frequency of networks occurences')
    freqs_occurences = []
    for mi in mutual_info:
        is_significant = (mi > value_significance).astype(int)
        n_occurences = len(mi[is_significant == 1])
        freq_occurence = n_occurences / len(is_significant)
        freqs_occurences.append(freq_occurence)

    freqs_occurences = np.array(freqs_occurences).reshape(1, -1)
    if debug:
        print(f'freqs_occurences shape: {freqs_occurences.shape}')
    freqs_occurences = pd.DataFrame(freqs_occurences, index=[SubID], columns=networks_names)


    print(freqs_occurences)
    # find a way to save: databasing? simple file?
    #<<use sqlite databasing

    db_file = base_dir + '/processing/network_freqs/database/SHARE_network_freqs.db'

    os.makedirs(os.path.dirname(db_file), exist_ok=True)

    conn = sqlite3.connect(db_file)

    # Append the DataFrame to the table in the SQLite database
    freqs_occurences.to_sql('network_freqs', conn, if_exists='append', index=True, index_label='SHARE_ID')

    conn.commit()
    conn.close()
