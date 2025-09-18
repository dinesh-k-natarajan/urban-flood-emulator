import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from os.path import join
from pathlib import Path

import pandas as pd
import numpy as np
import rasterio as rs
import richdem as rd

# Helper function
def compute_twi(elevation, flacc):
    """Function to compute the Topographic Wetness Index (TWI)
    
    Args:
        elevation (np.ndarray): Elevation as 2D array
        flacc (np.ndarray): Flow Accumulation as 2D array 

    Returns:
        np.ndarray: TWI as 2D array
    """
    # Calculate the slope in radians
    slope_rad = rd.TerrainAttribute(elevation, attrib='slope_radians')
    # Calculate TWI
    upslope_area = flacc
    twi = np.log(upslope_area / np.tan(slope_rad))
    
    return twi


def main():
    # SETTINGS
    FILL_DEPRESSIONS = True

    # Define paths to the dataset
    data_dir = "./data/" # TODO: define the path to your data directory here
    city_dir = join(data_dir, "city_level_data")

    # Load training, validation and test city names
    sets_file = join(data_dir, "sets.csv")
    sets_df = pd.read_csv(sets_file)
    train_cities = sets_df[sets_df["set"] == "train"]["City"].to_list()
    val_cities = sets_df[sets_df["set"] == "val"]["City"].to_list()
    test_cities = sets_df[sets_df["set"] == "test"]["City"].to_list()
    all_cities = train_cities + val_cities + test_cities

    for city_name in all_cities:
        print(150*"=")
        print(f"Extracting terrain features for {city_name}...")
        # Load the DEM
        dem_path = join(city_dir, city_name, "dem", f"{city_name}_roi.tif")
        ## Load DEM from tiff file
        with rs.open(dem_path) as f:
            dem_arr = f.read(1)
        dem = rd.rdarray(dem_arr, no_data=-9999)
        
        if FILL_DEPRESSIONS:
            ## Depression-filling of the DEM
            dem_fd = rd.FillDepressions(dem, epsilon=True, in_place=False)
        else:
            dem_fd = dem

        ###################################################################
        # Compute Terrain attributes using richdem
        ###################################################################
        ## Slope
        slope = rd.TerrainAttribute(dem, attrib='slope_degrees')
        ## Aspect
        aspect = rd.TerrainAttribute(dem, attrib='aspect')
        ## Curvature 
        curvature = rd.TerrainAttribute(dem, attrib='curvature')
        
        ## Flow Accumulation
        ### Get flow accumulation with no explicit weighting. Default will be 1
        flacc = rd.FlowAccumulation(dem_fd, method='Quinn')
        ## Compute TWI
        twi = compute_twi(dem_fd, flacc)
        print("Number of infinity values in TWI:", np.count_nonzero(twi==np.inf))
        
        ################################################################################
        # Save the extracted features in a GeoTIFF file with all the bands
        ################################################################################
        # Load the DEM and metainfo from tif file using rasterio
        with rs.open(dem_path, 'r') as dem_file:
            src_meta = dem_file.meta
            dem_data = dem_file.read(1)
        # Data to write into tif file
        data_arrs = [dem_data, slope, aspect, curvature, flacc, twi]
        # Update the source profile
        src_meta.update(count=len(data_arrs))
        # Save tif file with dem and extracted features
        terrain_features_dir = join(city_dir, city_name, 'terrain_features')
        Path(terrain_features_dir).mkdir(parents=True, exist_ok=True)
        terrain_features_path = join(terrain_features_dir, f"{city_name}_roi.tif")
        
        with rs.open(terrain_features_path, 'w', **src_meta) as res_file:
            for band_num, data_arr in enumerate(data_arrs):
                res_file.write_band(band_num+1, data_arr.astype(rs.float32))
        print(f"Saved file to {terrain_features_path}")
        
        # ## Load the saved file and look at the plots for a sanity check
        # with rs.open(terrain_features_path, 'r') as loaded_file:
        #     dem_l = loaded_file.read(1)
        #     slope_l = loaded_file.read(2)
        #     aspect_l = loaded_file.read(3)
        #     curvature_l = loaded_file.read(4)
        #     flacc_l = loaded_file.read(5)
        #     twi_l = loaded_file.read(6)
                
        ################################################################################
        # Transform the extracted features, without scaling
        ################################################################################
        ## DEM
        dem_transf = dem
        ## Slope
        slope_transf = slope
        ## Aspect
        aspect_transf = aspect
        ## Curvature
        curvature_transf = np.cbrt(curvature)
        ## Flow Accumulation
        flacc_transf = np.log(flacc)
        ## TWI
        ### Clip inf values to the non-infinity max twi value
        twi_clip = np.clip(twi, a_min=np.min(twi), a_max=np.nanmax(twi[twi != np.inf]))
        twi_transf = twi_clip
        
        ## NOTE: Not saving these transformed features as they can be easily transformed in the Dataset class
            

if __name__=='__main__':
    main()
