from abc import ABC, abstractmethod
import numpy as np
from nilearn.image import load_img
import os
import pandas as pd
import re
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator
from tqdm.auto import tqdm
from neuroginius.pairwise_interactions import multivariate_distance_correlation
from neuroginius.parcellate import split_multivariate_timeseries

from dask import delayed

class IDerivatives(ABC):
    """
    Abstract base class for derivatives.
    """

    @abstractmethod
    def transform(self):
        """
        Compute the derivative of the given data.
        This method must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def save(self, file_path):
        """
        Save the computed derivative to a file.
        This method must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def load(self, file_path):
        """
        Load the derivative from a file.
        This method must be implemented by subclasses.
        """
        pass

# class SharedMethodMixin:
#     def shared_method(self):
#         return "This is a shared method"

class PairwiseInteraction(BaseEstimator, IDerivatives):
    """
    Class for computing pairwise interactions.
    """

    def __init__(self,
                 metric,          
                 path=None,
                 compute_if_missing=False):
        
        self.path = path
        self.metric = metric
        self.compute_if_missing = compute_if_missing
        self.__data = {}
        self.__index = None
        self.__columns = None
        """
        Initialize the pairwise interaction object.
        metric: str
            The method used to compute the pairwise interaction.
            supported values: "pearsonr", "mutual-information", "ar-1", "mdcor"
            also supports custom functions.

        """

    def fit(self, X, y=None):
        pass

    def transform(self, input, return_data=False, **kwargs):
        
        self.preceding = input

        subset = kwargs.get("subset", None)

        index = kwargs.get("index", None)
        if subset is not None:
            index = np.array(index)[subset].squeeze()

        if index is not None:
            self.__index = index

        columns = kwargs.get("columns", None)

        if columns is not None:
            self.__columns = columns

        multiprocessing = kwargs.get("multiprocessing", None)

        filelist = self.preceding._list_files() if subset is None else np.array(self.preceding._list_files())[subset]

        result = [self.__compute_individual(f) for f in tqdm(filelist)]

        self.__data = np.array(result)
        if return_data:
            return pd.DataFrame(result, index=index, columns=columns)

    def save(self, file_path=None):

        if file_path is None:
            file_path = self.path
            if file_path is None:
                raise ValueError("Path is not specified.")

        pass

    def load(self, file_path=None):
        if file_path is None:
            if self.path is None:
                raise ValueError("Path is not specified.")
            file_path = self.path
        pass

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
    
    def __compute_individual(self, input):
        """
        Compute the pairwise interaction of the given data.
        input: 
        """

        if type(input) == list:
            input = input
        elif os.path.isfile(input):
            if input.endswith(".csv"):
                input = np.loadtxt(input, delimiter=",")
        #     elif input.endswith(".nii"):
        #         input = nib.load(input).get_fdata()

        if self.metric == "pearsonr":
            return np.corrcoef(input)[np.triu_indices(input.shape[0], k=1)]
        elif self.metric == "mutual-information":
            pass
        elif self.metric == "ar-1":
            pass
        elif self.metric == "mdcor":
            return multivariate_distance_correlation(input)

        else:
            pass
        
    def get_data_keys(self):
        return self.__data.keys()


class ParcellatedTimeseries(BaseEstimator, IDerivatives):
    """
    WORK IN PROGRESS

    Class for computing parcellated timeseries.
    """

    def __init__(self, 
                 atlas, 
                 derivatives_path,
                 extraction_method='mean',
                 preceding=None,
                 compute_if_missing=False):
        """
        Initialize the parcellated timeseries object.
        atlas: neuroginius.atlas.Atlas object
            The atlas used to parcellate the brain.

        preceding: IDerivatives object or Nifti1Image compatible format
        """

        if extraction_method == 'multivariate' and preceding is None:
            raise ValueError("Preceding must be specified when extraction method is multivariate.")

        self.atlas = atlas
        self.extraction_method = extraction_method
        self.__data = {}

        self.derivatives_path = derivatives_path
        self.__path = derivatives_path + f"/parcellated_timeseries/{self.atlas.name}/{self.extraction_method}"
        os.makedirs(self.__path, exist_ok=True)

        if type(preceding) == list:
            self.preceding = BottomDerivative(preceding)

        # self.preceding = preceding

        self.compute_if_missing = compute_if_missing
        self.filelist = self._list_files()

    def fit(self, X, y=None):
        pass

    def transform(self):
        """
        Compute the parcellated timeseries of the given data.
        """
        pass

    def save(self, file_path=None):
        if file_path is None:
            #defaults to a subfolder in the derivatives folder
            file_path = self.__path + f"/{self.extraction_method}"
        np.savetxt(file_path, self.get_data(), delimiter=",")

    def load(self, file_path):
        pass

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

    def extract_subid_from_path(self, prefix, keep_prefix=False):
        seq = [f.split(prefix)[1] for f in self.filelist]
        seq = [re.match(r"(\d+)", el).group(0) for el in seq]
        if keep_prefix:
            return [prefix + el for el in seq]
        return seq
    
    def _list_files(self):
        if self.extraction_method == 'multivariate':
            return None
        path = self.__path
        filelist = [os.path.join(path, f) for f in os.listdir(path)]
        filelist.sort()
        return filelist
    
    # def _save_individual(self, input, output):
    #     pass
    



class FunctionalConnectivity(IDerivatives):
    """
    Class for computing functional connectivity.
    """

    def __init__(self, 
                 method, 
                 extraction_method=None,
                 atlas=None,                 
                 path=None,
                 compute_if_missing=False):
        """
        Initialize the functional connectivity object.
        extraction_method: str
            The method used to extract the time series data.
            supported values: "mean", "pc1"
            default: None
        atlas: neuroginius.atlas.Atlas object
            The atlas used to parcellate the brain.
            

    
        """        
        if extraction_method is not None and atlas is None:
            raise ValueError("Atlas must be specified when extraction method is specified.")

        self.method = method
        self.atlas = atlas
        self.extraction_method = extraction_method
        self.path = path
        self.compute_if_missing = compute_if_missing

    def transform(self):
        """
        Compute the functional connectivity of the given data.
        """
        pass

    def save(self, file_path):
        """
        Save the computed functional connectivity to a file.
        """
        pass

    def load(self):
        """
        Load the functional connectivity.

        """

        if self.path is None:
            if self.compute_if_missing:
                return(self.compute())
            else:
                raise ValueError("""Path is not specified.
                                Provide path to an existing file 
                                or set compute_if_missing to True.""")  
        
        return np.loadtxt(self.path, delimiter=",")
    
class BottomDerivative:
    def __init__(self, data):
        if type(data) == list:
            self.__data = {i: el for i, el in enumerate(data)}

    def get_data_keys(self):
        return self.__data.keys()
    
    def transform_individual(self, index):
        return self.__data[index]