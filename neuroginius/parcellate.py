from nilearn.maskers import NiftiLabelsMasker
from neuroginius.atlas import Atlas

def parcellate(img, atlas):
    #img and atlas can either be a path to a nifti file or a nifti image object

    if isinstance(atlas, Atlas):
        atlas = atlas.maps
    masker = NiftiLabelsMasker(labels_img=atlas, strategy='mean')

    avg_data = masker.fit_transform(img).T
    
    return avg_data