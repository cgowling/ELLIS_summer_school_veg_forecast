#!/usr/bin/env python
# coding: utf-8

# # Sketching a dataloader

# In[8]:


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import xarray as xr
import os


# In[12]:


class NDVIDataset(Dataset):
    def __init__(self, chirps, era5, ndvi, copernicus, dem, seq_len=4, stats=None):
        """
        stats (dict or None): precomputed means/stds to reuse across datasets.
        If None, compute on the fly.
        """
        self.seq_len = seq_len

        # --- Extract variables ---
        print("NDVI Dataloader")
        print("  Extracting data")
        self.ndvi_data = ndvi["NDVI_VNP43C4_GHA"].values  # (T, lat, lon)
        self.chirps_data = chirps["CHIRPS_mm_per_week"].values
        self.era5_data = np.stack([
            era5["potential_evaporation_sum"].values,
            era5["volumetric_soil_water_layer_1"].values,
            era5["volumetric_soil_water_layer_2"].values,
            era5["volumetric_soil_water_layer_3"].values,
            era5["temperature_2m"].values
        ], axis=-1)  # (T, lat, lon, 5)

        self.landcover = copernicus["band_data"].values  # (lat, lon)
        self.dem_data = dem["band_data"].values

        self.time_len, self.lat_len, self.lon_len = self.ndvi_data.shape

        # --- Mask shrubland (20) + herbaceous (30) ---
        print("  Building land cover mask")
        valid_mask = np.isin(self.landcover, [20, 30])
        self.valid_pixels = np.argwhere(valid_mask)  # (N, 2)

        if stats is None:
            print("  Computing standardisation statistics")
            # reshape to (time, lat*lon)
            ndvi_flat = self.ndvi_data.reshape(self.time_len, -1)  # (T, L*W)
            chirps_flat = self.chirps_data.reshape(self.time_len, -1)
            era5_flat = self.era5_data.reshape(self.time_len, -1, self.era5_data.shape[-1])  # (T, L*W, 5)
            dem_flat = self.dem_data.reshape(-1)  # (L*W,)

            # apply mask to pixel axis
            mask_flat = valid_mask.reshape(-1)

            ndvi_vals = ndvi_flat[:, mask_flat].reshape(-1)
            chirps_vals = chirps_flat[:, mask_flat].reshape(-1)
            era5_vals = era5_flat[:, mask_flat, :].reshape(-1, self.era5_data.shape[-1])
            dem_vals = dem_flat[mask_flat].reshape(-1)

            self.means = {
                "ndvi": np.nanmean(ndvi_vals),
                "chirps": np.nanmean(chirps_vals),
                "era5": np.nanmean(era5_vals, axis=0),
                "dem": np.nanmean(dem_vals),
            }
            self.stds = {
                "ndvi": np.nanstd(ndvi_vals),
                "chirps": np.nanstd(chirps_vals),
                "era5": np.nanstd(era5_vals, axis=0),
                "dem": np.nanstd(dem_vals),
            }
        else:
            print("  Using cached standardisation statistics")
            self.means = stats["means"]
            self.stds = stats["stds"]

        print("-> Dataset ready :)")

    def __len__(self):
        return len(self.valid_pixels) * (self.time_len - self.seq_len)

    def __getitem__(self, idx):
        t = np.random.randint(self.seq_len, self.time_len)
        pixel_idx = np.random.randint(0, len(self.valid_pixels))
        i, j = self.valid_pixels[pixel_idx]

        # --- Dynamic history ---
        ndvi_hist = self.ndvi_data[t-self.seq_len:t, i, j]
        chirps_hist = self.chirps_data[t-self.seq_len:t, i, j]
        era5_hist = self.era5_data[t-self.seq_len:t, i, j, :]

        ndvi_hist = (ndvi_hist - self.means["ndvi"]) / self.stds["ndvi"]
        chirps_hist = (chirps_hist - self.means["chirps"]) / self.stds["chirps"]
        era5_hist = (era5_hist - self.means["era5"]) / self.stds["era5"]

        dynamic = np.concatenate([
            ndvi_hist[:, None],
            chirps_hist[:, None],
            era5_hist
        ], axis=-1)  # (seq_len, 7)

        # --- Static features ---
        dem_val = (self.dem_data[i, j] - self.means["dem"]) / self.stds["dem"]

        lc_val = self.landcover[i, j]
        if lc_val == 20:  # shrubland
            lc_onehot = np.array([1, 0], dtype=np.float32)
        elif lc_val == 30:  # herbaceous
            lc_onehot = np.array([0, 1], dtype=np.float32)

        static = np.concatenate([[dem_val], lc_onehot])  # (3,)

        # --- Target NDVI ---
        target = self.ndvi_data[t, i, j]
        target = (target - self.means["ndvi"]) / self.stds["ndvi"]

        return {
            "dynamic": torch.tensor(dynamic, dtype=torch.float32),
            "static": torch.tensor(static, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
        }


# # Gap Filling NDVI

# In[ ]:


def fill_ndvi_gaps(ndvi: xr.Dataset) -> xr.Dataset:

    ndvi = ndvi.sortby('lat')
    NDVI_filled = ndvi.interpolate_na(
        dim="lat", method="nearest", fill_value="extrapolate"
    )
    NDVI_filled = NDVI_filled.interpolate_na(
        dim="lon", method="nearest", fill_value="extrapolate"
    )
    NDVI_filled = NDVI_filled.sortby('lat', ascending=False)
    return NDVI_filled


# In[ ]:


"""ndvi_filled = fill_ndvi_gaps(ndvi_data)
save_path = os.path.join(data_path, "NDVI_VNP43C4_v002_GHA_GAP_FILLED.nc")
ndvi_filled.to_netcdf(save_path)"""


# # Example Usage

# In[15]:


import os
import xarray as xr

data_path = "/ptmp/mp002/ellis/veg_forecast/"

_NAME_COPERNICUS = "COPERNICUS_Landcover_2019_GHA_reprojected.nc"
_NAME_CHIRPS = "CHIRPS_v002_weekly_sum_VNP43C4_GHA.nc"
_NAME_ERA5 = "ERA5_land_weekly_aggregated_GHA_VNP43C4.nc"
_NAME_NDVI = "NDVI_VNP43C4_v002_GHA_GAP_FILLED.nc"
_NAME_DEM = "srtm_5km_projection.nc"

copernicus_data = xr.open_dataset(os.path.join(data_path, _NAME_COPERNICUS))
chirps_data = xr.open_dataset(os.path.join(data_path, _NAME_CHIRPS))
era5_data = xr.open_dataset(os.path.join(data_path, _NAME_ERA5))
ndvi_data = xr.open_dataset(os.path.join(data_path, _NAME_NDVI))
dem_data = xr.open_dataset(os.path.join(data_path, _NAME_DEM))

# DROP FIRST THREE DATAPOINTS TO MATCH NDVI TIMESERIES
chirps_data = chirps_data.isel(time=slice(3, None))
era5_data = era5_data.isel(time=slice(3, None))


# In[ ]:


# Step 2: create a dataset to compute stats
tmp_dataset = NDVIDataset(chirps_data, era5_data, ndvi_data, copernicus_data, dem_data)
stats = {"means": tmp_dataset.means, "stds": tmp_dataset.stds}

# Step 3: create training/validation datasets with cached stats
train_dataset = NDVIDataset(chirps_data, era5_data, ndvi_data, copernicus_data, dem_data, stats=stats)
val_dataset = NDVIDataset(chirps_data, era5_data, ndvi_data, copernicus_data, dem_data, stats=stats)

loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
batch = next(iter(loader))
print("Dynamic:", batch["dynamic"].shape)  # (64, 4, 7) → [NDVI, CHIRPS, 5xERA5]
print("Static:", batch["static"].shape)    # (64, 1)   → DEM
print("Target:", batch["target"].shape)    # (64,)


# In[14]:


ndvi_data = ndvi_data["NDVI_VNP43C4_GHA"].values
time_len, lat_len, lon_len = ndvi_data.shape
ndvi_flat = ndvi_data.reshape(time_len, -1)

landcover = copernicus_data["band_data"].values
valid_mask = np.isin(landcover, [20, 30])
mask_flat = valid_mask.reshape(-1)
ndvi_vals = ndvi_flat[:, mask_flat].reshape(-1)