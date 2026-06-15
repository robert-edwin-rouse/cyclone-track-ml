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


# =============================================================================
# Load in ERA5 data
# =============================================================================
pressure_data = xr.open_dataset('haiyan_pressure.nc',
                                # config.pressure_path,
                                engine='h5netcdf',
                                chunks={'valid_time': 20})
pressure_vars = pressure_data[config.pressure_var_codes]
pressure_array = pressure_vars.to_array(dim='var')
pressure_array = pressure_array.transpose('valid_time', 'latitude',
                                          'longitude', 'var', 'pressure_level')
pressure_array = pressure_array.stack(channel=('var', 'pressure_level'))

surface_array = xr.open_dataset('haiyan_surface.nc',
                                # config.surface_path,
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
n_channel = pressure_array.sizes['channel']
chunk_dict = {'valid_time': t_chunks,
              'latitude':   y_chunks,
              'longitude':  x_chunks,
              'channel':    n_channel, }

pressure_array = pressure_array.astype('float32').chunk(chunk_dict)
sst_mask = sst_mask.astype('float32').chunk(chunk_dict)
sst = sst.astype('float32').chunk(chunk_dict)

pressure_array = pressure_array.assign_coords(
    channel=pressure_array.channel.values)
sst_mask = sst_mask.assign_coords(channel=np.array([-2], dtype=np.int64))
sst = sst.assign_coords(channel=np.array([-1], dtype=np.int64))

pressure_array = xr.concat([pressure_array, sst_mask, sst],
                           dim='channel', coords='minimal',
                           join='exact', compat='override',)


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
        return [labels[0]] * len(values)
    result = []
    seen_above_threshold = False
    for i, val in enumerate(values):
        if val >= threshold:
            seen_above_threshold = True
            result.append(labels[1])
        elif not seen_above_threshold:
            result.append(labels[0])
        elif i > last_threshold_cross:
            result.append(labels[2])
        else:
            result.append(labels[1])
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
    """
    Create cyclone masks as a dask-backed xarray.DataArray chunked along time.
    This builds mask blocks per time-chunk (using the module-level t_chunks
    from the ERA5 pressure array) via dask.delayed so the resulting DataArray
    has the same time-chunking as `pressure_array`.
    """
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
# Merge input-output labels
# =============================================================================
storm_masks = storm_masks.rename({'latitude': 'label_latitude',
                                  'longitude': 'label_longitude',
                                  'channel': 'label_channel'})
full_ds = xr.Dataset({"inputs": pressure_array, "labels": storm_masks})


# # Train/Val split (90/10) along time
# N = full_ds.dims["time"]
# rng = np.random.default_rng(42)
# perm = rng.permutation(N)
# n_tr = int(0.9 * N)
# train_idx = perm[:n_tr]
# val_idx   = perm[n_tr:]

# train_ds = full_ds.isel(time=train_idx)
# val_ds   = full_ds.isel(time=val_idx)

# print("Train/Val time sizes:", train_ds.dims["time"], val_ds.dims["time"])



# def normalize_features(X_train, X_val, normalize_channels=None):
#     """
#     Normalise features using training statistics. (for precipitation)
    
#     Args:
#         X_train: Training features tensor
#         X_val: Validation features tensor
#         normalize_channels: List of channel indices to normalise (None = normalise first 11)
    
#     Returns:
#         Normalised X_train, X_val, normalisation statistics
#     """
#     if normalize_channels is None:
#         normalize_channels = config.NORMALIZE_CHANNELS_FIRST_11
    
#     # Extract channels to normalise
#     X_train_normalized = X_train[:, normalize_channels, :, :]
    
#     # Compute per-feature min, max and mean
#     xmin_train = torch.amin(X_train_normalized, dim=(0, 2, 3)).view(1, -1, 1, 1)
#     xmax_train = torch.amax(X_train_normalized, dim=(0, 2, 3)).view(1, -1, 1, 1)
#     xavg_train = torch.mean(X_train_normalized, dim=(0, 2, 3)).view(1, -1, 1, 1)
    
#     # Normalise training data
#     X_train_normalized = (X_train_normalized - xavg_train) / (xmax_train - xmin_train)
    
#     # Combine normalised channels with unnormalised channels
#     if len(normalize_channels) == X_train.shape[1]:
#         X_train = X_train_normalized
#     else:
#         # Keep other channels unchanged
#         other_channels = [i for i in range(X_train.shape[1]) if i not in normalize_channels]
#         X_train = torch.cat((X_train_normalized, X_train[:, other_channels, :, :]), dim=1)
    
#     # Apply same normalisation to validation data
#     X_val_normalized = X_val[:, normalize_channels, :, :]
#     X_val_normalized = (X_val_normalized - xavg_train) / (xmax_train - xmin_train)
    
#     if len(normalize_channels) == X_val.shape[1]:
#         X_val = X_val_normalized
#     else:
#         X_val = torch.cat((X_val_normalized, X_val[:, other_channels, :, :]), dim=1)
    
#     norm_stats = {
#         'xmin': xmin_train,
#         'xmax': xmax_train,
#         'xavg': xavg_train,
#         'normalize_channels': normalize_channels
#     }
    
#     return X_train, X_val, norm_stats