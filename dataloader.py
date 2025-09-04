import torch
from torch.utils.data import Dataset
import numpy as np

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
        nan_mask = np.any(np.isnan(self.era5_data), axis=(0, -1))  # shape (lat, lon)
        valid_mask = valid_mask & (~nan_mask)  # keep only non-NaN pixels
        self.valid_pixels = np.argwhere(valid_mask)  # (N, 2)
        self.valid_pixels = self.valid_pixels.reshape(-1, 2)

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
        while True:  # keep resampling until a valid sample is found
            t = np.random.randint(self.seq_len, self.time_len)
            pixel_idx = np.random.randint(0, len(self.valid_pixels))
            i, j = self.valid_pixels[pixel_idx]

            # --- Dynamic history ---
            ndvi_hist = self.ndvi_data[t-self.seq_len:t, i, j]
            chirps_hist = self.chirps_data[t-self.seq_len:t, i, j]
            era5_hist = self.era5_data[t-self.seq_len:t, i, j, :]

            ndvi_hist = (ndvi_hist - self.means["ndvi"]) / (self.stds["ndvi"] + 1e-6)
            chirps_hist = (chirps_hist - self.means["chirps"]) / (self.stds["chirps"] + 1e-6)
            era5_hist = (era5_hist - self.means["era5"]) / (self.stds["era5"] + 1e-6)

            dynamic = np.concatenate([
                ndvi_hist[:, None],
                chirps_hist[:, None],
                era5_hist
            ], axis=-1)  # (seq_len, 7)

            # --- Static features ---
            dem_val = (self.dem_data[i, j] - self.means["dem"]) / (self.stds["dem"] + 1e-6)

            lc_val = self.landcover[i, j]
            if lc_val == 20:  # shrubland
                lc_onehot = np.array([1, 0], dtype=np.float32)
            elif lc_val == 30:  # herbaceous
                lc_onehot = np.array([0, 1], dtype=np.float32)
            else:
                lc_onehot = np.array([0, 0], dtype=np.float32)  # fallback

            static = np.concatenate([[dem_val], lc_onehot])  # (3,)

            # --- Target NDVI ---
            target = self.ndvi_data[t, i, j]
            target = (target - self.means["ndvi"]) / (self.stds["ndvi"] + 1e-6)

            # --- Convert to torch ---
            sample = {
                "dynamic": torch.tensor(dynamic, dtype=torch.float32),
                "static": torch.tensor(static, dtype=torch.float32),
                "target": torch.tensor(target, dtype=torch.float32),
            }

            # --- Check validity ---
            if (torch.isnan(sample["dynamic"]).any() or
                torch.isnan(sample["static"]).any() or
                torch.isnan(sample["target"]).any() or
                torch.isinf(sample["dynamic"]).any() or
                torch.isinf(sample["static"]).any() or
                torch.isinf(sample["target"]).any()):
                # Bad sample → resample
                print("Bad sample, resampling...")
                continue

            return sample