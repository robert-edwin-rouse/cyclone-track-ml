"""
This module defines resources, both computational and data, the relevant
features, and training configurations / hyperparameters.
"""

import torch
import torch.optim as optim
import os


# =============================================================================
# Project Initialisation
# =============================================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
era5_data_dir = os.path.join(base_dir, "data", "era5")

os.makedirs(data_dir, exist_ok=True)
os.makedirs(era5_data_dir, exist_ok=True)

cds_api_url = "https://cds.climate.copernicus.eu/api"
# cds_api_key = "########################"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Data Acquisition & Feature Selection
# =============================================================================
lat_lon = [90, -180, -90, 180]

years = [x for x in range(2010, 2021)]
months = [x for x in range(1, 13)]
days = [x for x in range(1, 32)]
hours = [6*x for x in range(0, 4)]

surface_dataset = "reanalysis-era5-single-levels"
sst_variables = ["sea_surface_temperature",
                 "2m_temperature",]
sst_var_codes = ["sst", "t2m"]
sst_path = "era5_sst_data.nc"
sst_zarr_path = 'era5_sst.zarr'

pressure_dataset = "reanalysis-era5-pressure-levels"
pressure_levels = [1000, 750, 500]
pressure_variables = ["relative_humidity",
                      "temperature",
                      "u_component_of_wind",
                      "v_component_of_wind",
                      "vorticity"]
pressure_var_codes = ["r", "t", "u", "v", "vo"]
pressure_path = "era5_pressure_data.nc"
pressure_zarr_path = "era5_pressure.zarr"


# =============================================================================
# Data Labelling & Output Configuration
# =============================================================================
nm_to_km = 1.852
grid_res = 1/125
area_growth_factor = 3
lifestages = ['Storm - Nondeveloping',
              'Cyclolysis',
              'Cyclogenesis',
              'Active Cyclone',]
output_resolution = 0.125
train_set_percent = 0.85
valid_set_percent = 0.05
test_set_percent = 0.1

train_data_zarr = 'train_dataset.zarr'
val_data_zarr = 'val_dataset.zarr'
test_data_zarr = 'test_dataset.zarr'

normalisation_path = "normalisation_parameters.nc"

# =============================================================================
# Training & Model Configuration
# =============================================================================
optimiser = optim.AdamW
epochs = 64
batch_size = 256
num_workers = 8
learning_rate = 0.00001
weight_decay = 1e-5
dropout = 0.1

model_detect_path = 'cyclone-detect-ml.jit'
model_track_path = 'cyclone-track-ml.jit'
