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
cds_api_key = "########################"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Data Acquisition & Feature Selection
# =============================================================================
lon_lat_extents = [90, -180, -90, 180]

years = [x for x in range(2000,2025)]
months = [x for x in range(1,13)]
days = [x for x in range(1,32)]
hours = [6*x for x in range(0,4)]

surface_dataset = "reanalysis-era5-single-levels"
surface_variables = ["2m_temperature",
                     "sea_surface_temperature",
                     "land_sea_mask"]
surface_var_codes = ["","","",""]
surface_path = "era5_surface_data.nc"

pressure_dataset = "reanalysis-era5-pressure-levels"
pressure_levels = [1000, 750, 500]
pressure_variables = ["relative_humidity",
                      "temperature",
                      "u_component_of_wind",
                      "v_component_of_wind",
                      "vorticity"]
pressure_var_codes = ["r","t","u","v","vo"]
pressure_path = "era5_pressure_data.nc"


# =============================================================================
# Training & Model Configuration
# =============================================================================
optimiser = optim.adamw
epochs = 1024
batch_size = 32
learning_rate = 0.0001
weight_decay = 1e-5
dropout = 0.1

model_detect_path = 'cyclone-detect-ml.pt'
model_track_path = 'cyclone-track-ml.pt'
