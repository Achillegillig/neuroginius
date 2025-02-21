from abc import ABC, abstractmethod
import numpy as np
from nilearn.image import load_img
import os
import pandas as pd
import re
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm.auto import tqdm
from neuroginius.pairwise_interactions import multivariate_distance_correlation, ar_connectivity, multivariate_integration
from neuroginius.parcellate import split_multivariate_timeseries, parcellate
import warnings

class BaseDerivatives(TransformerMixin, BaseEstimator):
        """
        Initialize the derivatives object.

        """
        def load_individual(self, file=None, index=None, matrix_form=False):
            if file is None and index is None:
                raise ValueError("Specify either file or index.")
            if index is not None:
                return np.loadtxt(self.files[index], delimiter=",")
            if file.endswith(".csv"):
                data = np.loadtxt(file, delimiter=",")
            elif file.endswith(".nii"):
                data = load_img(file).get_fdata()
            if matrix_form:
                return data
            return data[np.triu_indices(data.shape[0], k=1)]
                
            

        def exists(self, n_subjects=None, index=None):
            # TODO: implement checking based on a subject list / filelist?
            if self.path is None:
                raise ValueError("Path is not specified.")
            if self.files is None:
                self.files = self._list_files()
            if len(self.files) == 0:
                return False
            if index is None:
                if n_subjects is None:
                    raise ValueError("Specify either n_subjects or index.")
                return len(self.files) == n_subjects
            else:
                if type(index) == str:
                    if self.subjects is None:
                        raise ValueError("Subjects are not specified. use subjects_from_prefix.")
                    return index in self.subjects               
                return self.files[index] is not None

        def set_derivatives_path(self, path, make_subdir=True):
            self.derivatives_path = Path(path)
            suffix = f'{self.extraction_method}'
            if isinstance(self, PairwiseInteraction):
                if self.extraction_method == 'mean':
                    suffix = f'{self.extraction_method}'
                elif self.dimensionality_reduction is None:
                    suffix = f'{self.extraction_method}/complete'
                else:
                    suffix = f'{self.extraction_method}/{self.dimensionality_reduction}'
                suffix = f'{suffix}/{self.metric}'
            self.path = self.derivatives_path / f"{self.name}/{self.atlas.name}/{suffix}"
            self.dataframe_path = self.path / 'db/db.csv'
            # print(self.path)
            if not os.path.exists(self.path) and make_subdir:
                os.makedirs(self.path, exist_ok=True)
                print(f"Created directory: {self.path}")
            self.files = self._list_files()
        
        def subjects_from_prefix(self, prefix, return_values=False):
            self.subjects = self.extract_subid_from_path(prefix, keep_prefix=True)
            if return_values:
                return self.subjects
            
        def _list_files(self):
            filelist = [os.path.join(self.path, f) for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f))]
            filelist.sort()
            return filelist
        
        def extract_subid_from_path(self, prefix, keep_prefix=False):
            if self.path is None:
                raise ValueError("Path is not specified. use set_derivatives_path.")
            if self.files is None:
                raise ValueError("Files are not specified. use set_derivatives_path.")
            if self.files is None:
                self.files = self._list_files()
            seq = [re.search(rf'({prefix}\d+)', f).group(0) for f in self.files]
            if keep_prefix == False:
                return [seq.split(prefix)[1] for seq in seq]
            return seq

class PairwiseInteraction(BaseDerivatives):
    """
    Class for computing pairwise interactions.
    """

    def __init__(self,
                 metric,
                 atlas,
                 extraction_method = 'mean',
                 path=None,
                 fisher_transform=False,
                 ):
        
        self.atlas = atlas
        self.name = 'pairwise_interactions'
        self.metric = metric
        self.extraction_method = extraction_method
        self.dimensionality_reduction = None
        self.path = path
        self.files = None
        self.subjects = None
        self.dataframe = None
        self.fisher_transform = fisher_transform

        """
        Initialize the pairwise interaction object.
        metric: str
            The method used to compute the pairwise interaction.
            supported values: "pearsonr", "mutual-information", "ar-1", "mdcor"
            also supports custom functions.

        """

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return self.__compute_individual(X)

    def save(self, file_path=None):

        if file_path is None:
            file_path = self.path
            if file_path is None:
                raise ValueError("Path is not specified.")

        pass

    def load(self, file_path=None, filter=None, save_as_dataframe=False):
        if file_path is None:
            if self.path is None:
                raise ValueError("Path is not specified.")
            file_path = self.path
        self.dataframe_path = self.path / 'db/db.csv'
        if os.path.isfile(self.dataframe_path):
            print(f"Loading dataframe from {self.dataframe_path}")
            self.__data = pd.read_csv(self.dataframe_path, index_col=0, header=0, low_memory=False)
            if filter is not None:
                self.__data = self.__data.filter(filter, axis=0)
            # TODO: check that all required subjects are present
            warnings.warn('nothing checks that the loaded db is up to date or that it matches the filter', UserWarning)
            return self.__data
        if self.files is None:
            self.files = self._list_files()
        if filter is not None:
            self.files = [f for f in self.files if any(filt in f for filt in filter)]
            self.subjects = [s for s in self.subjects if any(filt in s for filt in filter)]
        print(f"Loading files...")
        data = np.array([self.load_individual(file=file) for file in tqdm(self.files, mininterval=1)])
        print(f"Loaded {len(data)} files.")
        self.__data = pd.DataFrame(data, index=self.subjects)

        if save_as_dataframe:
            os.makedirs(self.path / 'db', exist_ok=True)
            self.__data.to_csv(self.path / 'db/db.csv' , index=True)
        return self.__data

    def get_data(self, as_dataframe=False):
        if self.__data is None:
            raise ValueError("Data is not computed yet.")
        if as_dataframe:
            return pd.DataFrame(self.__data, index=self.__index, columns=self.__columns)
        return self.__data
    
    def fit_individual(self, preceding, index):
        self.preceding = preceding
        if index not in self.preceding.get_data_keys():
            self.preceding.fit_individual(self.preceding.preceding, index)

        self.__data[index] = self.__compute_individual(self.preceding.transform_individual(index))
        return
    
    def transform_individual(self, index):
        if index not in self.__data.keys():
            raise ValueError("Data is not computed yet. use fit_individual or fit_transform_individual.")
        return self.__data[index]
    
    def fit_transform_individual(self, preceding, index):
        self.fit_individual(preceding, index)
        return self.transform_individual(index)

    
    # def get_individual_data(self, index):
    #     if self.metric == "mdcor":
    #         file = self.preceding[index]
    #         return self.__compute_individual()

    def get_individual_data(self, index):
        if index not in self.__data.keys():
            raise ValueError("Data is not computed yet. use fit_individual or fit_transform_individual.")
        return self.__data[index]
    
    def __compute_individual(self, data):
        """
        Compute the pairwise interaction of the given data.
        data: 
        """

        if type(data) == list or type(data) == np.ndarray:
            data = data
        elif os.path.isfile(data):
            if data.endswith(".csv"):
                data = np.loadtxt(data, delimiter=",")
        #     elif data.endswith(".nii"):
        #         data = nib.load(data).get_fdata()

        if self.metric == "pearsonr":
            out = np.corrcoef(data)[np.triu_indices(data.shape[0], k=1)]
            if self.fisher_transform:
                out = np.arctanh(out)
            return out
        elif self.metric == "mutual-information":
            pass
        elif self.metric == "ar-1":
            return ar_connectivity(data, lag=1, time_first=False)
        elif self.metric == "ar-3":
            return ar_connectivity(data, lag=3, time_first=False)
        elif self.metric == "mdcor":
            return multivariate_distance_correlation(data)
        elif self.metric == "multivariate_integration":
            return multivariate_integration(data, self.atlas.macro_labels)
        
        ###POTENTIAL issue: returns a matrix that is then reshape afterwards, redundant info

        else:
            pass
        
    def get_data_keys(self):
        return self.__data.keys()


class ParcellatedTimeseries(BaseDerivatives):
    """
    WORK IN PROGRESS

    Class for computing parcellated timeseries.
    """

    def __init__(self, 
                 atlas, 
                 extraction_method='mean',
                 derivatives_path=None
                 ):
        """
        Initialize the parcellated timeseries object.
        atlas: neuroginius.atlas.Atlas object
            The atlas used to parcellate the brain.

        preceding: IDerivatives object or Nifti1Image compatible format
        """
        self.name = 'parcellated_timeseries'
        self.atlas = atlas
        self.extraction_method = extraction_method
        self.derivatives_path = Path(derivatives_path)
        self.path = None
        self.files = None
        if self.derivatives_path is not None:
            self.path = self.derivatives_path / f"parcellated_timeseries/{self.atlas.name}/{self.extraction_method}"
            os.makedirs(self.path, exist_ok=True)
            self.files = self._list_files()
        # self.files = self._list_files()
        self.subjects = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        Compute the parcellated timeseries of the given data.
        """
        if self.extraction_method == 'multivariate':
            return split_multivariate_timeseries(X, self.atlas)
        else:
            return parcellate(X, self.atlas)
        
    def load(self):
        data = [self.load_individual(file) for file in self.files]
        return np.array(data)




    def save(self, file_path=None):
        if file_path is None:
            #defaults to a subfolder in the derivatives folder
            file_path = self.path + f"/{self.extraction_method}"
        np.savetxt(file_path, self.get_data(), delimiter=",")


    def get_data(self, as_dataframe=False):
        # if self.extraction_method == 'multivariate':
        #     return self.__split_timeseries()
        if self.__data is None:
            raise ValueError("Data is not computed yet.")
        if as_dataframe:
            return pd.DataFrame(self.__data, index=self.__index, columns=self.__columns)
        return self.__data
    
    def get_data_keys(self):
        return self.__data.keys()
    
    def get_individual_data(self, index):
        if self.extraction_method == 'multivariate':
            if index not in self.__data.keys():
                raise ValueError("Data is not computed yet. use fit_individual or fit_transform_individual.")
            return self.__data[index]

    def fit_individual(self, preceding, index):
        if index not in self.preceding.get_data_keys():
            self.preceding.fit_individual(self.preceding.preceding, index)

        if self.extraction_method == 'multivariate':
            self.__data[index] = split_multivariate_timeseries(self.preceding.transform_individual(index), self.atlas)
            return
        
    def transform_individual(self, index):
        if index not in self.__data.keys():
            raise ValueError("Data is not computed yet. use fit_individual or fit_transform_individual.")
        return self.__data[index]
    
    def fit_transform_individual(self, preceding, index):
        self.fit_individual(preceding, index)
        return self.transform_individual(index)