'''
Text pending.
'''

import xarray as xr
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import dask.array as darray
import config
import os
import shutil


# =============================================================================
# Load in ERA5 data
# =============================================================================
pressure_data = xr.open_dataset(  # 'haiyan_pressure.nc',
                                config.pressure_path,
                                engine='h5netcdf',
                                chunks={'valid_time': 20})
pressure_vars = pressure_data[config.pressure_var_codes]
pressure_array = pressure_vars.to_array(dim='var')
pressure_array = pressure_array.transpose('valid_time', 'latitude',
                                          'longitude', 'var', 'pressure_level')
pressure_array = pressure_array.stack(channel=('var', 'pressure_level'))

surface_array = xr.open_dataset(  # 'haiyan_surface.nc',
                                config.surface_path,
                                engine='h5netcdf',
                                chunks={'valid_time': 20})


# =============================================================================
# Transform SST using land-temperature mask
# =============================================================================
sst_mask = (np.isnan(surface_array.sst.isel(valid_time=0))).astype(int)
sst_mask = (sst_mask.expand_dims(valid_time=surface_array.valid_time, axis=0).
            expand_dims(channel=[-2], axis=-1).
            transpose('valid_time', 'latitude', 'longitude', 'channel'))

sst = surface_array.sst.fillna(surface_array.t2m)
sst = (sst.expand_dims(channel=[-1], axis=-1).
       transpose('valid_time', 'latitude', 'longitude', 'channel'))


# =============================================================================
# Chunk data
# =============================================================================
t_chunks, y_chunks, x_chunks, _ = pressure_array.chunks
n_channel = pressure_array.sizes['channel'] + sst.sizes['channel'] + \
    sst_mask.sizes['channel']
chunk_dict = {'valid_time': t_chunks,
              'latitude':   y_chunks,
              'longitude':  x_chunks,
              'channel':    n_channel, }

pressure_array = pressure_array.astype('float32')
sst_mask = sst_mask.astype('float32')
sst = sst.astype('float32')

n_pressure_ch = pressure_array.sizes['channel']
pressure_array = pressure_array.assign_coords(
    channel=np.arange(n_pressure_ch, dtype=np.int64))

sst_mask = sst_mask.assign_coords(channel=np.array([-2], dtype=np.int64))
sst = sst.assign_coords(channel=np.array([-1], dtype=np.int64))

feature_array = xr.concat([pressure_array, sst_mask, sst],
                          dim='channel', coords='minimal',
                          join='exact', compat='override',).chunk(chunk_dict)


# =============================================================================
# Load in IBTRACS data, define radii, and lifestage classifications
# =============================================================================
df = pd.read_csv('ibtracs.since1980.list.v04r01.csv',
                 usecols=['SID', 'ISO_TIME', 'LAT', 'LON', 'USA_STATUS',
                          'USA_WIND', 'USA_PRES', 'USA_SSHS', 'USA_R34_NE',
                          'USA_R34_SE', 'USA_R34_SW', 'USA_R34_NW'])
df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
df['USA_SSHS'] = pd.to_numeric(df['USA_SSHS'], errors='coerce')
df['USA_SSHS'] = df['USA_SSHS'].fillna(-1)
df['USA_R34_NE'] = pd.to_numeric(df['USA_R34_NE'], errors='coerce').fillna(0)
df['USA_R34_SE'] = pd.to_numeric(df['USA_R34_SE'], errors='coerce').fillna(0)
df['USA_R34_SW'] = pd.to_numeric(df['USA_R34_SW'], errors='coerce').fillna(0)
df['USA_R34_NW'] = pd.to_numeric(df['USA_R34_NW'], errors='coerce').fillna(0)
df['Effective_Radius'] = df[['USA_R34_NE', 'USA_R34_SE',
                             'USA_R34_SW', 'USA_R34_NW']].max(axis=1)
df['Effective_Radius'] = df['Effective_Radius'].apply(lambda x: 20 if x < 20
                                                      else x)
df['Grid_Radius'] = (df['Effective_Radius'] *
                     config.area_growth_factor *
                     config.nm_to_km *
                     config.grid_res)
df = df.dropna(subset=['ISO_TIME']).sort_values(['SID', 'ISO_TIME'])


def lifestage(sequence, labels, threshold):
    values = np.asarray(sequence)
    result = []
    last_threshold_cross = -1
    for i, val in enumerate(values):
        if val >= threshold:
            last_threshold_cross = i
    if last_threshold_cross == -1:
        return [labels[1]] * len(values)
    result = []
    seen_above_threshold = False
    for i, val in enumerate(values):
        if val >= threshold:
            seen_above_threshold = True
            result.append(labels[2])
        elif not seen_above_threshold:
            result.append(labels[1])
        elif i > last_threshold_cross:
            result.append(labels[0])
        else:
            result.append(labels[2])
    return result


cyclones = []
for sid, grp in df.groupby('SID'):
    if 1 not in grp.USA_SSHS.values:
        grp['Classification'] = config.lifestages[0]
    if 1 in grp.USA_SSHS.values:
        grp['Classification'] = lifestage(grp['USA_SSHS'],
                                          config.lifestages[1:4], 1)
    cyclones.append(grp)
cyclones = pd.concat(cyclones, ignore_index=True)


# =============================================================================
# Create cyclone lifestage image masks
# =============================================================================
times = pd.to_datetime(surface_array.valid_time.values)
class_map = {cls: i + 1 for i, cls in enumerate(config.lifestages)}
lat_max, lon_min, lat_min, lon_max = config.lat_lon

def latlon_to_pix(lat, lon, lat_max, lon_min, img_lat, img_lon, pixels):
    y = int((lat_max - lat) * pixels)
    x = int((lon - lon_min) * pixels)
    y = np.clip(y, 0, img_lat - 1)
    x = np.clip(x, 0, img_lon - 1)
    return x, y

def cyclone_segmentation(cyclones, times, class_map):
    from dask import delayed

    pixels = 1 / config.output_resolution
    img_lat = int(abs(config.lat_lon[0] - config.lat_lon[2]) * pixels)
    img_lon = int(abs(config.lat_lon[1] - config.lat_lon[3]) * pixels)

    latitudes = np.linspace(lat_max, lat_min, img_lat)
    longitudes = np.linspace(lon_min, lon_max, img_lon)

    grouped = cyclones.groupby('ISO_TIME')
    groups_dict = {pd.Timestamp(ts): grp for ts, grp in grouped}

    try:
        time_chunks = tuple(t_chunks)
    except Exception:
        time_chunks = (len(times),)

    delayed_blocks = []
    start = 0
    for block_len in time_chunks:
        end = start + block_len

        @delayed
        def _compute_block(s=start, e=end):
            block = np.zeros((e - s, img_lat, img_lon), dtype=np.uint8)
            for ti in range(s, e):
                t = pd.Timestamp(times[ti])
                if t not in groups_dict:
                    continue
                grp = groups_dict[t]
                mask = block[ti - s]
                for _, row in grp.iterrows():
                    lat = float(row['LAT'])
                    lon = float(row['LON'])
                    radius_deg = float(row.get('Grid_Radius', 0.0))
                    class_label = class_map.get(row['Classification'], 0)

                    cx, cy = latlon_to_pix(lat, lon, lat_max, lon_min,
                                           img_lat, img_lon, pixels)
                    r_pix = max(1, int(round(radius_deg * pixels)))

                    y0 = max(0, cy - r_pix)
                    y1 = min(img_lat, cy + r_pix + 1)
                    x0 = max(0, cx - r_pix)
                    x1 = min(img_lon, cx + r_pix + 1)

                    yy = np.arange(y0, y1)[:, None]
                    xx = np.arange(x0, x1)[None, :]
                    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
                    circular = dist2 <= (r_pix ** 2)

                    sub = mask[y0:y1, x0:x1]
                    sub[circular] = np.maximum(sub[circular], class_label)
                    mask[y0:y1, x0:x1] = sub

                block[ti - s] = mask
            return block

        delayed_blocks.append(_compute_block())
        start = end

    dask_blocks = []
    start = 0
    for db, block_len in zip(delayed_blocks, time_chunks):
        shape = (block_len, img_lat, img_lon)
        dask_blocks.append(darray.from_delayed(db, shape=shape, dtype=np.uint8))
        start += block_len

    masks_dask = darray.concatenate(dask_blocks, axis=0)
    label_values = [0] + sorted({int(v) for v in class_map.values()})
    one_hot_layers = [(masks_dask == lv).astype('float32') for lv in label_values]
    one_hot_dask = darray.stack(one_hot_layers, axis=3)
    channel_coords = np.array(label_values, dtype=np.int64)
    mask_da = xr.DataArray(
        one_hot_dask,
        dims=('valid_time', 'latitude', 'longitude', 'channel'),
        coords={
            'valid_time': times,
            'latitude': latitudes,
            'longitude': longitudes,
            'channel': channel_coords,
        },
        name='cyclone_masks_onehot'
    )

    full_class_map = {0: 'Background'}
    for name, lbl in class_map.items():
        full_class_map[int(lbl)] = name
    mask_da.attrs['class_map'] = full_class_map
    mask_da.attrs['channel_labels'] = list(channel_coords.tolist())

    return mask_da


storm_masks = cyclone_segmentation(cyclones, times, class_map)


# =============================================================================
# Merge input-output labels and write to zarr
# =============================================================================
storm_masks = storm_masks.rename({'latitude': 'label_latitude',
                                  'longitude': 'label_longitude',
                                  'channel': 'label_channel'})
full_ds = xr.Dataset({"inputs": feature_array, "labels": storm_masks})


def _split_indices(num_items, train_frac, valid_frac, test_frac):
    n_train = int(num_items * train_frac)
    n_valid = int(num_items * valid_frac)
    n_test = num_items - n_train - n_valid
    if n_test < 0:
        raise ValueError(
            f"Invalid split fractions: {train_frac}, {valid_frac}, {test_frac}"
        )
    idx = np.arange(num_items)
    return idx[:n_train], idx[n_train:n_train + n_valid], idx[n_train + n_valid:]


def _compute_normalisation_stats(train_inputs):
    axes = ('valid_time', 'latitude', 'longitude')
    stats = xr.Dataset({
        'min': train_inputs.min(dim=axes).compute(),
        'max': train_inputs.max(dim=axes).compute(),
        'mean': train_inputs.mean(dim=axes).compute(),
    })
    stats['range'] = stats['max'] - stats['min']
    stats['range'] = stats['range'].where(stats['range'] != 0, 1.0)
    return stats


# Split the full dataset according to config fractions.
num_times = full_ds.sizes['valid_time']
train_idx, valid_idx, test_idx = _split_indices(
    num_times,
    config.train_set_percent,
    config.valid_set_percent,
    config.test_set_percent,
)

train_ds_raw = full_ds.isel(valid_time=train_idx)
normalisation_stats = _compute_normalisation_stats(train_ds_raw['inputs'])

full_ds['inputs'] = (
    full_ds['inputs'] - normalisation_stats['mean']
) / normalisation_stats['range']

train = full_ds.isel(valid_time=train_idx)
valid = full_ds.isel(valid_time=valid_idx)
test = full_ds.isel(valid_time=test_idx)

normalization_cache = {
    'train_idx': train_idx,
    'valid_idx': valid_idx,
    'test_idx': test_idx,
    'stats': normalisation_stats,
}

def save_ds_splits_to_zarr(train, valid, test, base_dir):
    os.makedirs(base_dir, exist_ok=True)

    for filename, ds in [
        ("train_data.zarr", train),
        ("valid_data.zarr", valid),
        ("test_data.zarr", test),
    ]:
        path = os.path.join(base_dir, filename)
        if os.path.exists(path):
            shutil.rmtree(path)

        # Extract raw data and rebuild completely from scratch
        inputs_data = ds['inputs'].data
        labels_data = ds['labels'].data
        
        # Build with only numeric coordinates, no index variables
        ds_clean = xr.Dataset(
            {
                'inputs': (('valid_time', 'latitude', 'longitude', 'channel'), inputs_data),
                'labels': (('valid_time', 'label_latitude', 'label_longitude', 'label_channel'), labels_data),
            },
            coords={
                'valid_time': np.arange(len(ds['valid_time']), dtype=np.float32),
                'latitude': ds['latitude'].values.astype(np.float32),
                'longitude': ds['longitude'].values.astype(np.float32),
                'channel': ds['channel'].values.astype(np.int64),
                'label_latitude': ds['label_latitude'].values.astype(np.float32),
                'label_longitude': ds['label_longitude'].values.astype(np.float32),
                'label_channel': ds['label_channel'].values.astype(np.int64),
            }
        )
        
        # Rechunk to uniform sizes for Zarr compatibility
        ds_clean = ds_clean.chunk({'valid_time': 20, 'latitude': 721, 'longitude': 1440, 'channel': 17})
        ds_clean.to_zarr(path, mode="w")


save_ds_splits_to_zarr(train, valid, test, config.data_dir)