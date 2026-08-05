"""
Diagnostic tool to check the contents of ERA5 files are viable.
"""

import xarray as xr
from dask.distributed import Client, LocalCluster


def validate_era5():
    cluster = LocalCluster(
        n_workers=16,
        threads_per_worker=1,
        memory_limit='7.5GB')
    client = Client(cluster)
    print('Dask Dashboard running at: {client.dashboard_link}')
    
    corrupt_files = []
    datasets = ['pressure','sst','rain']
    for d in datasets:
        base_filename = 'data/era5/era5_' + d + '_{year}_{month}.nc'
        for year in range (2010, 2021):
            for month in range(1, 13):
                filename = base_filename.format(year=year, month=month)
                try:
                    ds = xr.open_dataset(filename,
                                         engine="h5netcdf",
                                         chunks={'valid_time': 1460})
                    ds_mean = ds.mean()
                    ds_mean.load()
                    print(f'Successfully processed {filename}')
                except:
                    print(f'Error reading {filename}')
                    corrupt_files.append(filename)
    print('List of corrupted or missing files: {corrupt_files}')
    client.close()
    cluster.close()


if __name__ == 'main':
    validate_era5()