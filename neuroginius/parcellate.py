from nilearn.maskers import NiftiLabelsMasker

def parcellate(img, atlas):
    #img and atlas can either be a path to a nifti file or a nifti image object

    masker = NiftiLabelsMasker(labels_img=atlas, strategy='mean')

    avg_data = masker.fit_transform(img)
    
    return avg_data