import matplotlib.pyplot as plt
from neuroginius.parcellate import parcellate
from neuroginius.atlas import Atlas
import numpy as np
import os 
import pandas as pd
from pathlib import Path
import seaborn as sns
from scipy.stats import pearsonr
from tqdm.auto import tqdm

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

class NeurosynthDecoder:
    def __init__(self, atlas, metric='pearsonr'):
        self.atlas = atlas
        self.metric = metric
        self.neurosynth_maps = [os.path.join(ROOT_DIR.parent.parent / 'data/neurosynth/maps_3d', f)
                                for f in os.listdir(ROOT_DIR.parent.parent / 'data/neurosynth/maps_3d')]
        self.neurosynth_maps.sort()
        self.maps_names = [f.split('/')[-1].split('.')[0][:-5] for f in self.neurosynth_maps]

    def decode(self, map, do_parcellate=True, save_parcellated_db=True):
        """
        Decoding using spatial similarity assessment with Neurosynth meta-analytic maps.

        Parameters
        ----------
        map : str
            Path to the map to decode.
        parcellate : bool, optional
            Whether to parcellate the map. Default is True.

        Returns
        -------
        dict
            Dictionary with decoding results.
        """
        self.map = map

        maps_to_process = [self.map] if not isinstance(self.map, list) else self.map

        processed_maps = []

        for current_map in maps_to_process:

            map_name = current_map.split('/')[-1].split('.')[0]
            print(map_name)

            if not isinstance(self.atlas, Atlas):
                raise ValueError("Atlas must be an instance of neuroginius.atlas.Atlas")

            if do_parcellate:
                current_map = parcellate(current_map, self.atlas)
                save_path = ROOT_DIR.parent.parent / f'data/neurosynth/parcellated/{self.atlas.name}'
                if save_parcellated_db:
                    os.makedirs(save_path, exist_ok=True)
                    if os.path.exists(save_path / f'neurosynth_{self.atlas.name}.csv'):
                        print('loading existing parcellated neurosynth database')
                        self.neurosynth_maps = pd.read_csv(save_path / f'neurosynth_{self.atlas.name}.csv', index_col=0)
                    else:
                        print(f"Parcellating Neurosynth maps using atlas {self.atlas.name}")
                        list_maps = []
                        for map in tqdm(self.neurosynth_maps):
                            list_maps.append(parcellate(map, self.atlas))

                        self.neurosynth_maps = pd.DataFrame(np.array(list_maps).squeeze(), index=self.maps_names, columns=self.atlas.labels) 
                        print(self.neurosynth_maps.shape)

                        self.neurosynth_maps.to_csv(save_path / f'neurosynth_{self.atlas.name}.csv')
                        savefile = save_path / f'neurosynth_{self.atlas.name}.csv'
                        print(f"Parcellated Neurosynth maps saved at {savefile}")

            if self.metric == 'pearsonr':
                decoding = {map_name: pearsonr(current_map.ravel(), self.neurosynth_maps.loc[map_name], axis=None)[0] for map_name in self.maps_names}

            processed_maps.append(pd.DataFrame.from_dict(decoding, orient='index', columns=[f'{map_name}']))
        self.decoding = pd.concat(processed_maps, axis=1)
        return self.decoding
        
    def plot(self, axes=None, **snskwargs):

        cmap = snskwargs.pop('cmap', 'coolwarm')
        figsize = snskwargs.pop('figsize', (16, int(2 * self.decoding.shape[1])))
        # hierarchical clustering of neurosynth maps
        g = sns.clustermap(self.decoding.T,
                           cmap=cmap,
                           cbar_kws={'label': f'{self.metric}'}, 
                           figsize=figsize,
                           **snskwargs)
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=65, fontsize=10, va='top', ha='right')
        plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=10, va='center', ha='left')
        # fig.add_axes(g.ax_heatmap)

        # g = sns.heatmap(self.decoding.T, cmap='coolwarm', cbar_kws={'label': f'{self.metric}'})
        # g.set_xticklabels(g.get_xticklabels(), rotation=60, ha='right')
        return g
