from nilearn.maskers import NiftiLabelsMasker
from nilearn.image import load_img
from neuroginius.atlas import Atlas
import numpy as np
import os

def parcellate(img, atlas):
    #img and atlas can either be a path to a nifti file or a nifti image object

    if isinstance(atlas, Atlas):
        atlas = atlas.maps
    masker = NiftiLabelsMasker(labels_img=atlas, strategy='mean')

    avg_data = masker.fit_transform(img).T
    
    return avg_data


def split_multivariate_timeseries(input, atlas):
    """
    Returns a list of timeseries for each region in the atlas.

    Parameters:
    input: nifti image compatible format (e.g. path to nifti file or 4D Nifti1Image object)
        The input data to be split into regions.
    """
    #todo: assert 4D image
    
    regions_ts = []
    if isinstance(input, np.ndarray):
        if len(input.shape) == 4:
            input = input.reshape(-1, input.shape[3])
    elif os.path.isfile(input):
        if input.endswith('.nii') or input.endswith('.nii.gz'):
            input = load_img(input).get_fdata()
            input = input.reshape(-1, input.shape[3])

    
    if isinstance(atlas, np.ndarray):
        mask_data = atlas
        if len(mask_data.shape) == 3:
            mask_data = mask_data.ravel()
    elif isinstance(atlas, Atlas):
        mask_data = load_img(atlas.maps).get_fdata().ravel()
    elif isinstance(atlas, str):
        if (atlas.endswith('.nii') or atlas.endswith('.nii.gz')) == False:
            raise ValueError('atlas must be a path to a nifti file')
        mask_data = load_img(atlas).get_fdata().ravel()
    else:
        raise ValueError('atlas must be an Atlas object, ndarray or a path to an atlas nifti file')
    
    regions_id = np.unique(mask_data)
    regions_id = regions_id[regions_id != 0] #background

    for id in regions_id:
        region_ts = input[mask_data == id, :]
        regions_ts.append(region_ts.T) # time x voxels

    return regions_ts

