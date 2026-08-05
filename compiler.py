import os
import gc
import shutil
import numpy as np
import pandas as pd
import xarray as xr
import zarr
import dask
from tqdm import tqdm
from dask import delayed
import dask.array as darray
from dask.diagnostics import ProgressBar
import config


def latlon_to_pix(lat, lon, lat_max, lon_min, img_lat, img_lon, pixels):
    """Converts latitude and longitude coordinates to pixel indices."""
    y = int((lat_max - lat) * pixels)
    x = int((lon - lon_min) * pixels)
    y = np.clip(y, 0, img_lat - 1)
    x = np.clip(x, 0, img_lon - 1)
    return x, y


def lifestage(sequence, labels, threshold):
    """
    Classifies storm sequence into lifestages based on intensity threshold
    crossing.
    """
    values = np.asarray(sequence)
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


def cyclone_segmentation(cyclones, times, class_map, time_chunk_len=10):
    """
    Generates memory-safe one-hot encoded cyclone lifestage segmentation
    masks using Dask.
    """
    pixels = 1 / config.output_resolution
    lat_max, lon_min, lat_min, lon_max = config.lat_lon

    img_lat = int(abs(lat_max - lat_min) * pixels)
    img_lon = int(abs(lon_max - lon_min) * pixels)

    latitudes = np.linspace(lat_max, lat_min, img_lat)
    longitudes = np.linspace(lon_min, lon_max, img_lon)

    grouped = cyclones.groupby('ISO_TIME')
    groups_dict = {pd.Timestamp(ts): grp for ts, grp in grouped}

    num_times = len(times)
    time_chunks = []
    curr = 0
    while curr < num_times:
        chunk = min(time_chunk_len, num_times - curr)
        time_chunks.append(chunk)
        curr += chunk

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

                block[ti - s] = mask
            return block

        delayed_blocks.append(_compute_block())
        start = end

    dask_blocks = []
    for db, block_len in zip(delayed_blocks, time_chunks):
        shape = (block_len, img_lat, img_lon)
        dask_blocks.append(darray.from_delayed(db, shape=shape, dtype=np.uint8))

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


def _split_indices(num_items, train_frac, valid_frac, test_frac):
    """Calculates sequential index splits based on split percentages."""
    n_train = int(num_items * train_frac)
    n_valid = int(num_items * valid_frac)
    n_test = num_items - n_train - n_valid
    if n_test < 0:
        raise ValueError(f"Invalid split fractions: {train_frac}, {valid_frac}, {test_frac}")
    idx = np.arange(num_items)
    return idx[:n_train], idx[n_train:n_train + n_valid], idx[n_train + n_valid:]


def subsample_data(indices, seed=42, subsample_frac=0.3):
    """
    Randomly subsamples a fraction of index positions and sorts them 
    chronologically.
    """
    rng = np.random.default_rng(seed)
    n_sample = max(1, len(indices) // (1/subsample_frac))
    sampled = rng.choice(indices, size=n_sample, replace=False)
    return np.sort(sampled)


def export_ds_to_zarr_batched(ds_split, out_path, batch_size=10):
    """
    Memory-safe batch exporter to stream write dataset splits into
    Zarr stores.
    """
    if os.path.exists(out_path):
        shutil.rmtree(out_path)

    total_times = ds_split.sizes['valid_time']
    if total_times == 0:
        return

    n_lat = ds_split.sizes['latitude']
    n_lon = ds_split.sizes['longitude']
    n_in_ch = ds_split.sizes['channel']

    n_lbl_lat = ds_split.sizes['label_latitude']
    n_lbl_lon = ds_split.sizes['label_longitude']
    n_lbl_ch = ds_split.sizes['label_channel']

    # Initialize empty Zarr store direct structure
    root = zarr.open_group(out_path, mode='w')

    # Create chunked Zarr target arrays
    inputs_z = root.create_array(
        name='inputs',
        shape=(total_times, n_lat, n_lon, n_in_ch),
        chunks=(10, n_lat, n_lon, n_in_ch),
        dtype='float32'
    )

    labels_z = root.create_array(
        name='labels',
        shape=(total_times, n_lbl_lat, n_lbl_lon, n_lbl_ch),
        chunks=(10, n_lbl_lat, n_lbl_lon, n_lbl_ch),
        dtype='float32'
    )

    # Save coordinate arrays
    root.create_array('valid_time', data=np.arange(total_times, dtype=np.float32))
    root.create_array('latitude', data=ds_split['latitude'].values.astype(np.float32))
    root.create_array('longitude', data=ds_split['longitude'].values.astype(np.float32))
    root.create_array('channel', data=ds_split['channel'].values.astype(np.int64))
    root.create_array('label_latitude', data=ds_split['label_latitude'].values.astype(np.float32))
    root.create_array('label_longitude', data=ds_split['label_longitude'].values.astype(np.float32))
    root.create_array('label_channel', data=ds_split['label_channel'].values.astype(np.int64))

    # Process and write in small time batches to prevent RAM buildup
    for start_idx in tqdm(range(0, total_times, batch_size),
                          desc=f"Writing {os.path.basename(out_path)}",
                          unit="batch"):
        end_idx = min(start_idx + batch_size, total_times)

        # Compute strictly current batch slice
        batch_inputs = ds_split['inputs'].isel(valid_time=slice(start_idx, end_idx)).compute().values
        batch_labels = ds_split['labels'].isel(valid_time=slice(start_idx, end_idx)).compute().values

        inputs_z[start_idx:end_idx] = batch_inputs
        labels_z[start_idx:end_idx] = batch_labels

        del batch_inputs, batch_labels
        gc.collect()


def run_compiler():
    print("=" * 60)
    print("STARTING PIPELINE: CYCLONE SEGMENTATION MASK GENERATOR")
    print("=" * 60)

    # =========================================================================
    # 1. Load Input Datasets from Zarr (Memory-safe lazy loading)
    # =========================================================================
    print("\n[1/8] Loading Datasets from Zarr Stores...")
    pressure_ds = xr.open_zarr(config.pressure_zarr_path)
    sst_ds = xr.open_zarr(config.sst_zarr_path)

    # Select requested pressure levels safely
    p_vars = []
    for var in config.pressure_var_codes:
        for level in config.pressure_levels:
            da_var = pressure_ds[var].sel(pressure_level=level).drop_vars('pressure_level')
            da_var.name = f"{var}_{level}"
            p_vars.append(da_var)

    pressure_combined = xr.merge(p_vars).to_array(dim='channel')
    pressure_combined = pressure_combined.transpose('valid_time',
                                                    'latitude',
                                                    'longitude',
                                                    'channel')

    # Assign integer indexing to channels
    n_pressure_ch = pressure_combined.sizes['channel']
    pressure_combined = pressure_combined.assign_coords(channel=np.arange(n_pressure_ch, dtype=np.int64))

    # =========================================================================
    # 2. Mask & SST Transformation
    # =========================================================================
    print("\n[2/8] Creating Sea-Land Mask & Infilling SST...")
    sst = sst_ds['sst']
    t2m = sst_ds['t2m']

    first_sst_slice = sst.isel(valid_time=0)
    mask_2d = xr.where(first_sst_slice.isnull(), 1.0, 0.0).fillna(1.0).astype('float32')
    sst_mask = mask_2d.broadcast_like(sst).astype('float32')
    sst_mask = sst_mask.expand_dims(channel=[-2]).transpose('valid_time',
                                                    'latitude',
                                                    'longitude',
                                                    'channel')
    sst_mask = sst_mask.assign_coords(channel=np.array([-2],
                                                       dtype=np.int64))

    sst_filled = xr.where(sst.notnull(), sst, t2m).fillna(0.0).astype('float32')
    sst_filled = sst_filled.expand_dims(channel=[-1]).transpose('valid_time',
                                                    'latitude',
                                                    'longitude',
                                                    'channel')
    sst_filled = sst_filled.assign_coords(channel=np.array([-1],
                                                           dtype=np.int64))

    n_channel = pressure_combined.sizes['channel'] + sst_filled.sizes['channel'] + sst_mask.sizes['channel']
    chunk_dict = {
        'valid_time': 10,
        'latitude': pressure_combined.sizes['latitude'],
        'longitude': pressure_combined.sizes['longitude'],
        'channel': n_channel
    }

    feature_array = xr.concat(
        [pressure_combined, sst_mask, sst_filled],
        dim='channel',
        coords='minimal',
        join='exact',
        compat='override'
    ).chunk(chunk_dict)

    # =========================================================================
    # 3. Parsing IBTRACS & Classifying Cyclone Lifestages
    # =========================================================================
    print("\n[3/8] Parsing IBTRACS Track Data...")
    df = pd.read_csv(
        'ibtracs.since1980.list.v04r01.csv',
        low_memory=False,
        usecols=[
            'SID', 'ISO_TIME', 'LAT', 'LON', 'USA_STATUS',
            'USA_WIND', 'USA_PRES', 'USA_SSHS', 'USA_R34_NE',
            'USA_R34_SE', 'USA_R34_SW', 'USA_R34_NW'
        ]
    )
    df = df.drop(0, errors='ignore')
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
    df['USA_SSHS'] = pd.to_numeric(df['USA_SSHS'], errors='coerce').fillna(-1)
    df['LAT'] = pd.to_numeric(df['LAT'])
    df['LON'] = pd.to_numeric(df['LON'])

    df['USA_R34_NE'] = pd.to_numeric(df['USA_R34_NE'], errors='coerce').fillna(0)
    df['USA_R34_SE'] = pd.to_numeric(df['USA_R34_SE'], errors='coerce').fillna(0)
    df['USA_R34_SW'] = pd.to_numeric(df['USA_R34_SW'], errors='coerce').fillna(0)
    df['USA_R34_NW'] = pd.to_numeric(df['USA_R34_NW'], errors='coerce').fillna(0)

    df['Effective_Radius'] = df[['USA_R34_NE', 'USA_R34_SE', 'USA_R34_SW', 'USA_R34_NW']].max(axis=1)
    df['Effective_Radius'] = df['Effective_Radius'].apply(lambda x: 20 if x < 20 else x)
    df['Grid_Radius'] = df['Effective_Radius'] * config.area_growth_factor * config.nm_to_km * config.grid_res
    df = df.dropna(subset=['ISO_TIME']).sort_values(['SID', 'ISO_TIME'])

    print("Classifying Storm Lifestages...")
    cyclones = []
    for sid, grp in tqdm(df.groupby('SID'), desc="Classifying Storms", unit="storm"):
        if 1 not in grp.USA_SSHS.values:
            grp['Classification'] = config.lifestages[0]
        else:
            grp['Classification'] = lifestage(grp['USA_SSHS'], config.lifestages[1:4], 1)
        cyclones.append(grp)
    cyclones = pd.concat(cyclones, ignore_index=True)

    # =========================================================================
    # 4. Constructing Cyclone Lifestage Mask
    # =========================================================================
    print("\n[4/8] Generating Dask-backed Cyclone Segmentation Masks...")
    times = pd.to_datetime(pressure_ds.valid_time.values)
    class_map = {cls: i + 1 for i, cls in enumerate(config.lifestages)}

    storm_masks = cyclone_segmentation(cyclones, times, class_map, time_chunk_len=10)
    storm_masks = storm_masks.rename({
        'latitude': 'label_latitude',
        'longitude': 'label_longitude',
        'channel': 'label_channel'
    })

    full_ds = xr.Dataset({"inputs": feature_array, "labels": storm_masks})

    # =========================================================================
    # 5. Dataset Splitting & Random Subsampling
    # =========================================================================
    print("\n[5/8] Splitting Dataset into Train / Validation / Test...")
    num_times = full_ds.sizes['valid_time']
    train_idx, valid_idx, test_idx = _split_indices(
        num_times,
        config.train_set_percent,
        config.valid_set_percent,
        config.test_set_percent
    )

    print("Randomly sampling 1/3 of input-output pairs per split...")
    train_sampled_idx = subsample_data(train_idx)
    valid_sampled_idx = subsample_data(valid_idx)
    test_sampled_idx = subsample_data(test_idx)

    print(f" Train set size:      {len(train_idx)} -> {len(train_sampled_idx)} samples")
    print(f" Validation set size: {len(valid_idx)} -> {len(valid_sampled_idx)} samples")
    print(f" Test set size:       {len(test_idx)} -> {len(test_sampled_idx)} samples")

    # =========================================================================
    # 6. Memory-Safe Compute Normalization Statistics
    # =========================================================================
    print("\n[6/8] Calculating Normalization Statistics on Sampled Training Split...")
    train_inputs = full_ds['inputs'].isel(valid_time=train_sampled_idx)

    axes = ('valid_time', 'latitude', 'longitude')
    min_da = train_inputs.min(dim=axes)
    max_da = train_inputs.max(dim=axes)
    mean_da = train_inputs.mean(dim=axes)

    with ProgressBar():
        print(" Computing min/max/mean using Dask...")
        min_val, max_val, mean_val = dask.compute(min_da, max_da, mean_da)

    range_val = max_val - min_val
    range_val = xr.where(range_val != 0, range_val, 1.0)

    # Apply min-max/mean normalization lazily
    full_ds['inputs'] = (full_ds['inputs'] - mean_val) / range_val

    # Save normalization metadata
    os.makedirs(config.data_dir, exist_ok=True)
    norm_ds = xr.Dataset({'mean': mean_val, 'min': min_val, 'max': max_val, 'range': range_val})
    norm_ds.to_netcdf(os.path.join(config.data_dir, "norm_params.nc"))

    # =========================================================================
    # 7. Materializing Sampled Split Datasets
    # =========================================================================
    print("\n[7/8] Subsetting Sampled Dataset Splits...")
    train_ds = full_ds.isel(valid_time=train_sampled_idx)
    valid_ds = full_ds.isel(valid_time=valid_sampled_idx)
    test_ds = full_ds.isel(valid_time=test_sampled_idx)

    # =========================================================================
    # 8. Memory-Safe Batched Export to Zarr
    # =========================================================================
    print("\n[8/8] Exporting Sampled Split Datasets to Zarr Stores...")

    export_ds_to_zarr_batched(train_ds, os.path.join(config.data_dir, "train_data.zarr"), batch_size=10)
    export_ds_to_zarr_batched(valid_ds, os.path.join(config.data_dir, "valid_data.zarr"), batch_size=10)
    export_ds_to_zarr_batched(test_ds, os.path.join(config.data_dir, "test_data.zarr"), batch_size=10)

    gc.collect()
    print("\n=== COMPILER PIPELINE COMPLETE ===")


if __name__ == '__main__':
    run_compiler()