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
pressure_data = xr.open_dataset(config.pressure_path,
                                engine='h5netcdf',
                                chunks='auto',)
pressure_vars = pressure_data[config.pressure_var_codes]
pressure_array = pressure_vars.to_array(dim='var')
pressure_array = pressure_array.transpose('valid_time', 'latitude',
                                          'longitude', 'var', 'pressure_level')
pressure_array = pressure_array.stack(channel=('var', 'pressure_level'))

surface_array = xr.open_dataset(config.surface_path,
                                engine='h5netcdf',
                                chunks='auto',)


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
    pixels = 1 / config.output_resolution
    img_lat = int(abs(config.lat_lon[0] - config.lat_lon[2]) * pixels)
    img_lon = int(abs(config.lat_lon[1] - config.lat_lon[3]) * pixels)
    storm_masks = np.zeros((len(times), img_lat, img_lon), dtype=np.uint8)

    latitudes = np.linspace(lat_max, lat_min, img_lat)
    longitudes = np.linspace(lon_min, lon_max, img_lon)


    lat_scale = img_lat / (lat_max - lat_min)
    lon_scale = img_lon / (lon_max - lon_min)

    grouped = cyclones.groupby('ISO_TIME')
    time_to_idx = {t: i for i, t in enumerate(times)}

    for ts, group in grouped:
        if pd.Timestamp(ts) not in time_to_idx:
            try:
                idx = times.get_loc(pd.to_datetime(ts))
            except Exception:
                continue
        else:
            idx = time_to_idx[pd.Timestamp(ts)]

        mask = storm_masks[idx]

        for _, row in group.iterrows():
            lat = float(row['LAT'])
            lon = float(row['LON'])
            radius_deg = float(row.get('Grid_Radius', 0.0))
            class_label = class_map.get(row['Classification'], 0)

            cx, cy = latlon_to_pix(lat, lon, lat_max, lon_min, img_lat, img_lon, pixels)
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

        storm_masks[idx] = mask

    mask_da = xr.DataArray(
        storm_masks,
        dims=('time', 'latitude', 'longitude'),
        coords={'time': times, 'latitude': latitudes, 'longitude': longitudes},
        name='cyclone_mask'
    )
    mask_da.attrs['class_map'] = class_map

    return mask_da
