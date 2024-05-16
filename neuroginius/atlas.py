from sklearn.utils import Bunch
from nilearn import plotting
from nilearn.maskers import NiftiMapsMasker, NiftiLabelsMasker
from pathlib import Path


import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from nilearn import datasets
import pandas as pd
import nibabel as nib
import numpy as np
import os


atlas_mapping = {
    "harvard-oxford": lambda : datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm"),
    "schaefer": lambda : datasets.fetch_atlas_schaefer_2018(resolution_mm=2),
    "schaefer200": lambda : datasets.fetch_atlas_schaefer_2018(n_rois=200, resolution_mm=2),
    "schaefer100": lambda : datasets.fetch_atlas_schaefer_2018(n_rois=100, resolution_mm=2),
    "difumo": lambda : datasets.fetch_atlas_difumo(legacy_format=False),
    "smith": datasets.fetch_atlas_smith_2009,
    "msdl": datasets.fetch_atlas_msdl
}

is_soft_mapping = {
    "schaefer200": False,
    "msdl": True,
    "GINNA": False
}

class Atlas(Bunch):
    @classmethod
    def from_kwargs(cls, name, soft, **atlas_kwargs) -> None:
        new = cls(**atlas_kwargs)

        new.is_soft = soft
        new.name = name
        return new
    
    @classmethod
    def from_name(cls, name, soft=None):
        if soft is None:
            soft = is_soft_mapping[name] 
        atlas_kwargs = atlas_mapping[name]()
        new = cls(**atlas_kwargs)
        new.is_soft = soft
        new.name = name
        new.labels_ = atlas_kwargs["labels"]
        return new

    def get_coords(self):
        if "region_coords" in self.keys():
            return self.region_coords
        elif self.is_soft:
            return plotting.find_probabilistic_atlas_cut_coords(self.maps)
        else:
            return plotting.find_parcellation_cut_coords(self.maps)
    
    def overlay(self):
        raise NotImplementedError()

    def plot(self, **plotting_kwargs):
        if self.is_soft:
            return plotting.plot_prob_atlas(self.maps, title=self.name, **plotting_kwargs)
        else:
            return plotting.plot_roi(self.maps, title=self.name, **plotting_kwargs)

    def fit_masker(self, **masker_kw):
        if self.is_soft:
            masker = NiftiMapsMasker(
                maps_img=self.maps,
                **masker_kw
            )
        else:
            masker = NiftiLabelsMasker(
                labels_img=self.maps,
                #labels=self.labels, # TODO Test that
                **masker_kw
            )
        masker.fit()
        return masker

    @property
    def labels(self):
        # The issue is that for difumo the labels
        # are in a data frame
        # Check type of labels instead?
        if self.name == "difumo":
            return self.labels_.difumo_names.to_list()
        else:
            return self.labels_

    @property
    def macro_labels(self):
        if "networks" in self.keys():
            return self.networks
        l = self.labels
        return list(map(lambda x: str(x).split("_")[2], l))