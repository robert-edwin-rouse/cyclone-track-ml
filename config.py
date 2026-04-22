"""
This module defines resources, both computational and data, along with 
"""

import torch
import torch.optim as optim
import os


# =============================================================================
# Resource Acquisition
# =============================================================================
input_data = '.nc'
output_data =  '.nc'
model_path = 'cyclone-track-ml.pt'

cds_api_url = "https://cds.climate.copernicus.eu/api" 
cds_api_key = "########################"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Data & Feature Selection
# =============================================================================
longitude_extent = []
latitute_extent = []

years = [x for x in range(2000,2025)]
months = [x for x in range(1,13)]
days = [x for x in range(1,32)]
hours = [6*x for x in range(0,4)]

pressure_levels = [1000, 750, 500]
variables = []

# =============================================================================
# Training Configuration
# =============================================================================
optimiser = optim.adamw
epochs = 1024
batch_size = 32
learning_rate = 0.0001
weight_decay = 1e-5
dropout = 0.1
