"""
Concatenates multiple ERA5 datasets where the total size of all files exceeds
available memory.
"""

from pathlib import Path
import re
import shutil
import warnings
import xarray as xr
import zarr


warnings.filterwarnings("ignore",
                        message=".*does not have a Zarr V3 specification.*")
DATA_DIR = Path("data/era5")


def get_sort_key(file_path: Path) -> tuple[int, int]:
    match = re.search(r"(\d{4})_(\d{1,2})\.nc$", file_path.name)
    if match:
        return tuple(map(int, match.groups()))
    return (0, 0)


def combine_files_to_zarr_sequential(file_pattern: str, output_zarr: Path):
    """Sorts files numerically by (year, month) and appends sequentially into Zarr."""
    print(f"\nProcessing files matching '{file_pattern}' in '{DATA_DIR}'...")

    files = sorted(DATA_DIR.glob(file_pattern), key=get_sort_key)
    if not files:
        print(f"No files found for pattern: {file_pattern} in {DATA_DIR}")
        return
    print(f"Found {len(files)} files to combine.")
    if output_zarr.exists():
        print(f"Deleting existing store at {output_zarr}...")
        shutil.rmtree(output_zarr)

    for i, file_path in enumerate(files, 1):
        with xr.open_dataset(file_path, engine="netcdf4") as ds:
            # Clear NetCDF encoding chunks
            for var in ds.variables:
                ds[var].encoding.pop("chunks", None)
                ds[var].encoding.pop("preferred_chunks", None)
            if i == 1:
                print(f"[{i}/{len(files)}] Initializing Zarr: {file_path.name}")
                ds.to_zarr(output_zarr,
                           mode="w",
                           consolidated=False,
                           compute=True,
                           zarr_format=2)
            else:
                print(f"[{i}/{len(files)}] Appending: {file_path.name}",
                      end="\r")
                ds.to_zarr(output_zarr,
                           mode="a",
                           append_dim="valid_time",
                           consolidated=False,
                           compute=True,
                           zarr_format=2)

    print(f"\nConsolidating Zarr metadata for {output_zarr.name}...")
    try:
        zarr.consolidate_metadata(str(output_zarr))
    except (AttributeError, TypeError):
        zarr.convenience.consolidate_metadata(str(output_zarr))
    print(f"Successfully created {output_zarr}.")


if __name__ == "__main__":
    xr.set_options(file_cache_maxsize=1)
    datasets = {
        "pressure": (
            "era5_pressure_*.nc",
            DATA_DIR / "era5_pressure.zarr",
        ),
        "sst": ("era5_sst_*.nc", DATA_DIR / "era5_sst.zarr"),
        "rain": ("era5_rain_*.nc", DATA_DIR / "era5_rain.zarr"),
    }
    for var_name, (pattern, zarr_path) in datasets.items():
        combine_files_to_zarr_sequential(pattern, zarr_path)