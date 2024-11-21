import os
import numpy as np
import pandas as pd



class Dataset:

    @classmethod
    def from_name(self, name):
        if name not in datasets_mapping.keys():
            raise ValueError(f"Unknown dataset: {name}. Supported datasets are: ishare, hcp")
        return datasets_mapping[name]
    
    @classmethod
    def from_bids(self, path):
        return BIDSDataset(path)
    
class ISHARE(Dataset):

    def __init__(self):
        self.name = "ishare"
        self.derivatives_path = None

    def get_data(self):
        # Placeholder for actual data retrieval logic
        # Replace this with the actual data retrieval logic
        return None
    
    
    def get_derivatives(self, type, subtype=None, subset=None):
        pass

    def set_derivatives_path(self, path):
        self.derivatives_path = path

    def add_processing_pipeline(self, pipeline, input):
        self.__pipeline_input = input
        self.__pipeline = pipeline


    
class HCP(Dataset):
    
        def __init__(self):
            self.name = "hcp"
    
        def get_data(self):
            # Placeholder for actual data retrieval logic
            # Replace this with the actual data retrieval logic
            return None
        
class BIDSDataset(Dataset):
    
        def __init__(self, path):
            self.path = path
    
        def get_data(self):
            # Placeholder for actual data retrieval logic
            # Replace this with the actual data retrieval logic
            return None

class DataRetriever:

    def __init__(self,):
        pass

    def from_dataset(self, dataset):
        if dataset == "ishare":
            pass
        elif dataset == "hcp":
            pass
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        




class DataFactory:

    def __init__(self, base_path):
        self.base_path = base_path

    def get_data(self, atlas, fc_method, cognitive_criterion):
        file_path = self._get_file_path(atlas, fc_method, cognitive_criterion)
        
        if not os.path.exists(file_path):
            self._compute_data(atlas, fc_method, cognitive_criterion, file_path)
        
        return self._load_data(file_path)

    def _get_file_path(self, atlas, fc_method, cognitive_criterion):
        file_name = f"{atlas}_{fc_method}_{cognitive_criterion}.csv"
        return os.path.join(self.base_path, file_name)

    def _compute_data(self, atlas, fc_method, cognitive_criterion, file_path):
        # Placeholder for actual data computation logic
        # Replace this with the actual computation based on atlas, fc_method, and cognitive_criterion
        data = np.random.rand(100, 10)  # Example data
        df = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(10)])
        df.to_csv(file_path, index=False)

    def _load_data(self, file_path):
        return pd.read_csv(file_path)


datasets_mapping = {
    "ishare": ISHARE(),
    "hcp": HCP(),

}