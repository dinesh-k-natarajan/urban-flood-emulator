import os
import json
import random
from os.path import join, isfile
from typing import Union, Tuple, Dict, List

import torch
import numpy as np
import rasterio as rs
from torch.utils.data import Dataset

# Only using 20, 40, 60, 80, 100 mm/h rain events
ALLOWED_RAINS = [20, 40, 60, 80, 100]

# Global min and max values in dataset (2048x2048)
## For DEM, find relative elevation gain of each image first. Then apply scaling.
## These values were computed for the cities training and validation sets.
DEM_GLOBAL_MIN = 0.0
DEM_GLOBAL_MAX = 355.96002197265625
## These values were computed for 20,40,60,80,100 mm/h rain scenarios in the dataset.
RAIN_GLOBAL_MIN = 0.0
RAIN_GLOBAL_MAX = 32.029998779296875
## These values are assigned based on knowledge about slope
SLOPE_GLOBAL_MIN = 0.0
SLOPE_GLOBAL_MAX = 90.0
## These values are assigned based on knowledge about aspect
ASPECT_GLOBAL_MIN = 0.0
ASPECT_GLOBAL_MAX = 360.0
## These values were computed for the cities in training and validation sets
## After applying Cube-root transformation to curvatures
CURVATURE_GLOBAL_MIN = -38.52592849731445
CURVATURE_GLOBAL_MAX = 34.18069839477539
## After applying Log transformation to flow accumulation
FLACC_GLOBAL_MIN = 0.0
FLACC_GLOBAL_MAX = 15.073884010314941
## After Clipping to non-infinity maximum value of twi
TWI_GLOBAL_MIN = -4.168573379516602
TWI_GLOBAL_MAX = 28.091699600219727
## Define min-max scaling parameters
RANGE_MIN = -1
RANGE_MAX = 1

# Dataset class that combines the different data in the FIP dataset into an ML dataset
class FloodDataset(Dataset):
    """
    FloodDataset class inherited from the abstract class torch.utils.data.Dataset
    Architecture is summarized as follows:
        Inputs: DEM, time-series of total rainfall
        Targets: water depths (clipped at 5 m)
    """
    def __init__(
            self,
            name: str, 
            cities_list: List,
            city_dir: Union[str, os.PathLike],
            rainfall_dir: Union[str, os.PathLike],
        ) -> None:
        """Constructor of the FIP Dataset 1 class

        Args:
            name (str): Name for the dataset like train, validation or test dataset
            cities_list(List): List of cities in the dataset
            city_dir (str, path): Path to the directory containing city-wise FIP outputs
            rainfall_dir (str, path): Path to the directory containing rainfall events
        """
        super(FloodDataset, self).__init__()
        # Save arguments
        self.name = name
        self.cities_list = cities_list
        self.city_dir = city_dir
        self.rainfall_dir = rainfall_dir
        # Load the paths to FIP output samples for each city
        self.samples_list = []
        for city in self.cities_list:
            samples = [join(self.city_dir, city, "simulations", f"{city}_roi_rain_sim_180min_{rain}mmh.tif") for rain in ALLOWED_RAINS]
            self.samples_list += samples
        # Shuffle the list of samples
        random.shuffle(self.samples_list)
        print(f"Creating a FIPRegDataset object for {self.name} dataset...")
            
    def __getitem__(self, idx) -> Tuple[Dict['str', torch.Tensor], Dict['str', torch.Tensor]]:
        """
        Returns data sample corresponding to the index argument. The sample index corresponds to the axis 0 of the data arrays.
        The conversion of numpy arrays to torch.Tensor will be done automatically by the collate_fn of the dataloader during batching.

        Returns:
            Tuple[Dict['str', torch.Tensor], Dict['str', torch.Tensor]]: Tuple of dictionaries containing
            torch.Tensors of the input and target datasources.
        """
        # Get the path to the FIP output sample
        sample_path = self.samples_list[idx]
        # Load DEM data
        dem = self.load_dem(idx)
        ## Apply DEM scaling
        ### Find relative elevation gain of each DEM
        dem_min = np.min(dem)
        relative_dem = dem - (dem_min * np.ones_like(dem))
        ### Apply min-max scaling based on global min-max values in training+val dataset
        scaled_dem = (relative_dem - DEM_GLOBAL_MIN) / (DEM_GLOBAL_MAX - DEM_GLOBAL_MIN)
        scaled_dem = scaled_dem * (RANGE_MAX - RANGE_MIN) + RANGE_MIN
        
        # Load additional terrain features
        ## Find which DEM corresponds to the data sample
        fname_parts = sample_path.split('/')[-1].split('.')[0].split('_')
        roi_idx = fname_parts.index("roi")
        city_name = "_".join(fname_parts[:roi_idx])
        dem_path = join(self.city_dir, city_name, "terrain_features", f"{city_name}_roi.tif")
        ## Load ATFs from tif file
        with rs.open(dem_path) as f:
            # Addition of extra dims to the 2D array: dim 0 = # image channels
            slope = np.expand_dims(f.read(2), axis=0).astype(dtype='float32')
            aspect = np.expand_dims(f.read(3), axis=0).astype(dtype='float32')
            curvature = np.expand_dims(f.read(4), axis=0).astype(dtype='float32')
            flacc = np.expand_dims(f.read(5), axis=0).astype(dtype='float32')
            twi = np.expand_dims(f.read(6), axis=0).astype(dtype='float32')
            
        # Apply scaling to slope
        scaled_slope = (slope - SLOPE_GLOBAL_MIN) / (SLOPE_GLOBAL_MAX - SLOPE_GLOBAL_MIN)
        scaled_slope = scaled_slope * (RANGE_MAX - RANGE_MIN) + RANGE_MIN
        # Apply scaling to aspect
        scaled_aspect = (aspect - ASPECT_GLOBAL_MIN) / (ASPECT_GLOBAL_MAX - ASPECT_GLOBAL_MIN)
        scaled_aspect = scaled_aspect * (RANGE_MAX - RANGE_MIN) + RANGE_MIN
        # Apply transformations and then scaling to curvature
        transf_curvature = np.cbrt(curvature)
        scaled_curvature = (transf_curvature - CURVATURE_GLOBAL_MIN) / (CURVATURE_GLOBAL_MAX - CURVATURE_GLOBAL_MIN)
        scaled_curvature = scaled_curvature * (RANGE_MAX - RANGE_MIN) + RANGE_MIN
        # Apply transformations and then scaling to flow accumulation
        transf_flacc = np.log(flacc)
        scaled_flacc = (transf_flacc - FLACC_GLOBAL_MIN) / (FLACC_GLOBAL_MAX - FLACC_GLOBAL_MIN)
        scaled_flacc = scaled_flacc * (RANGE_MAX - RANGE_MIN) + RANGE_MIN
        # Apply transformations and then scaling to twi
        transf_twi = np.clip(twi, a_min=np.min(twi), a_max=np.nanmax(twi[twi!=np.inf]))
        scaled_twi = (transf_twi - TWI_GLOBAL_MIN) / (TWI_GLOBAL_MAX - TWI_GLOBAL_MIN)
        scaled_twi = scaled_twi * (RANGE_MAX - RANGE_MIN) + RANGE_MIN
        
        ## Assemble the DEM and ATF into one array with 6 channels
        dem_atf = np.concatenate(
            (
                scaled_dem,
                scaled_slope,
                scaled_aspect,
                scaled_curvature,
                scaled_flacc,
                scaled_twi
            ),
            axis=0
        )
        
        # Load rain data 
        rain_event = "_".join(sample_path.split('/')[-1].split('.')[0].split('_')[-4:])
        rain_path = join(self.rainfall_dir, f"{rain_event}_vals.txt")
        assert isfile(rain_path), f"Rain data file at {rain_path} does not exist!"
        with open(rain_path) as f:
            rain = np.array(json.load(f), dtype='float32')
        ## Apply rainfall scaling
        ### Apply min-max scaling based on global min-max values from all 37 rain scenarios
        scaled_rain = (rain - RAIN_GLOBAL_MIN) / (RAIN_GLOBAL_MAX - RAIN_GLOBAL_MIN)
        scaled_rain = scaled_rain * (RANGE_MAX - RANGE_MIN) + RANGE_MIN
                
        # Load water depth data
        with rs.open(sample_path) as f:
            # Addition of extra dims to the 2D array: dim 0 = # image channels
            water_depth = np.expand_dims(f.read(1), axis=0).astype(dtype='float32')
        ## Feature preprocessing of water depths
        np.clip(water_depth, a_min=0., a_max=5., out=water_depth)
        
        # Assemble dicts to return inputs and targets
        inputs = {
            'dem': dem_atf, 
            'rain': scaled_rain
        }
        targets = {
            'water_depth': water_depth,
        }
        return inputs, targets
    
    def load_dem(self, idx):
        """This method loads the DEM of the specific data sample

        Args:
            idx (int): Index of the data sample

        Returns:
            numpy array: 3D numpy array containing the DEM with shape (1,W,H)
        """
        sample_path = self.samples_list[idx]
        ## Find which DEM corresponds to the data sample
        fname_parts = sample_path.split('/')[-1].split('.')[0].split('_')
        roi_idx = fname_parts.index("roi")
        city_name = "_".join(fname_parts[:roi_idx])
        dem_path = join(self.city_dir, city_name, "dem", f"{city_name}_roi.tif")
        ## Load tif file
        with rs.open(dem_path) as f:
            # Addition of extra dims to the 2D array: dim 0 = # image channels
            dem = np.expand_dims(f.read(1), axis=0).astype(dtype='float32')
        
        return dem
    
    def __len__(self) -> int:
        """
        Returns the number of samples in the Florest dataset

        Returns:
            int: Number of samples in the dataset
        """       
        return len(self.samples_list)
    
    def get_input_channels(self) -> int:
        """Returns the number of channels in the DEM input

        Returns:
            int: Number of channels in the DEM input
        """
        inp, _ = self[0]
        return inp['dem'].shape[0]
    
    def get_identifiers(self, idx) -> Dict['str','str']:
        """
        Returns the identifiers for the data sample, i.e., the city name and rainfall event

        Returns:
            Dict['str', 'str']: {
                                'city': String of the city name
                                'rain_event': String describing the rainfall event
            }
        """
        sample_path = self.samples_list[idx]
        # Get city name
        ## Find which DEM corresponds to the data sample
        fname_parts = sample_path.split('/')[-1].split('.')[0].split('_')
        roi_idx = fname_parts.index("roi")
        city_name = "_".join(fname_parts[:roi_idx])
        # Get rainfall scenario
        rain_event = "_".join(sample_path.split('/')[-1].split('.')[0].split('_')[-4:])
        return {
            'city': city_name,
            'rain_event': rain_event
        }
    
    def unscale_inputs(self, idx):
        """This method undoes the scaling applied to the inputs

        Args:
            idx (int): Index of the data sample

        Returns:
            Dict['str', torch.Tensor]: Dictionary containing
            torch.Tensors of the unscaled input datasources.
        """
        inputs, _ = self[idx]
        # Unscale the DEM
        scaled_dem = inputs['dem']
        unscaled_dem = (scaled_dem - RANGE_MIN) / (RANGE_MAX - RANGE_MIN)
        unscaled_dem = (unscaled_dem * (DEM_GLOBAL_MAX - DEM_GLOBAL_MIN)) + DEM_GLOBAL_MIN
        # Unscale the rain
        scaled_rain = inputs['rain']
        unscaled_rain = (scaled_rain - RANGE_MIN) / (RANGE_MAX - RANGE_MIN)
        unscaled_rain = (unscaled_rain * (RAIN_GLOBAL_MAX - RAIN_GLOBAL_MIN)) + RAIN_GLOBAL_MIN

        return {
            'dem': unscaled_dem,
            'rain': unscaled_rain
        }
