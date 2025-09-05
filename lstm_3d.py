import xarray as xr
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import random
import numpy as np
from tqdm import trange
from sklearn.metrics import r2_score

from torch.utils.data import Dataset, DataLoader
import os

print("Started")

class NDVIDataset(Dataset):
    def __init__(self, chirps, era5, ndvi, copernicus, dem, seq_len=10, stats=None):
        """
        stats (dict or None): precomputed means/stds to reuse across datasets.
        If None, compute on the fly.
        """
        self.seq_len = seq_len

        # --- Extract variables ---
        print("NDVI Dataloader")
        # print("  Extracting data")
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
        # print("  Building land cover mask")
        valid_mask = np.isin(self.landcover, [20, 30])
        self.valid_pixels = np.argwhere(valid_mask)  # (N, 2)
        
        # Precompute all valid (t, pixel) pairs
        self.index_pairs = []
        for t in range(self.seq_len, self.time_len - 1):  # t+1 must be valid
            for pixel_idx in range(len(self.valid_pixels)):
                self.index_pairs.append((t, pixel_idx))

        if stats is None:
            # print("  Computing standardisation statistics")
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
        # return len(self.valid_pixels) * (self.time_len - self.seq_len)
        return len(self.index_pairs)

    def __getitem__(self, idx):
        num_bad = 0
        while True:  # keep resampling until a valid sample is found
            # t, pixel_idx = self.index_pairs[idx]
            # i, j = self.valid_pixels[pixel_idx]
            t = np.random.randint(self.seq_len, self.time_len-10)
            # if t == self.seq_len or (t+1) == self.seq_len:
            #     continue
            pixel_idx = np.random.randint(0, len(self.valid_pixels))
            i, j = self.valid_pixels[pixel_idx]

            # --- Dynamic history ---
            ndvi_hist = self.ndvi_data[t-self.seq_len:t, i, j]
            # ndvi_targ = self.ndvi_data[t-self.seq_len+1:t+1, i, j]
            chirps_hist = self.chirps_data[t-self.seq_len:t, i, j]
            era5_hist = self.era5_data[t-self.seq_len:t, i, j, :]

            ndvi_hist = (ndvi_hist - self.means["ndvi"]) / self.stds["ndvi"]
            # ndvi_targ = (ndvi_targ - self.means["ndvi"]) / self.stds["ndvi"]
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
                lc_onehot = np.array([[1, 0]], dtype=np.float32)
            elif lc_val == 30:  # herbaceous
                lc_onehot = np.array([[0, 1]], dtype=np.float32)
                
            dem_seq = np.repeat(dem_val, self.seq_len).reshape(-1,1)
            lc_onehot_seq = np.repeat(lc_onehot, self.seq_len, axis=0)
            dynamic = np.concatenate([
                ndvi_hist[:, None],
                chirps_hist[:, None],
                era5_hist,
                dem_seq,
                lc_onehot_seq
            ], axis=-1)  # (seq_len, 10)

            static = np.concatenate([[dem_val], lc_onehot.reshape(-1)])  # (3,)

            # --- Target NDVI ---
            target = self.ndvi_data[t+1, i, j].reshape(-1,1)
            target = (target - self.means["ndvi"]) / self.stds["ndvi"]
            # target = ndvi_targ[:, None]

            # return {
            #     "dynamic": torch.tensor(dynamic, dtype=torch.float32),
            #     # "static": torch.tensor(static, dtype=torch.float32),
            #     "target": torch.tensor(target, dtype=torch.float32),
            # }
            # --- Check validity ---
            
            if (torch.isnan(torch.tensor(dynamic, dtype=torch.float32)).any() or
                torch.isnan(torch.tensor(target, dtype=torch.float32)).any() or
                torch.isinf(torch.tensor(dynamic, dtype=torch.float32)).any() or
                torch.isinf(torch.tensor(target, dtype=torch.float32)).any()):
                # Bad sample → resample
                # print("Bad sample, resampling...", end="", flush=True)
                num_bad =+ 1
                continue
            # print(num_bad)
            return torch.tensor(dynamic, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

# # Gap Filling NDVI


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



data_path = "/ptmp/mp002/ellis/veg_forecast/"

_NAME_COPERNICUS = "COPERNICUS_Landcover_2019_GHA_reprojected.nc"
_NAME_CHIRPS = "CHIRPS_v002_weekly_sum_VNP43C4_GHA.nc"
_NAME_ERA5 = "ERA5_land_weekly_aggregated_GHA_VNP43C4.nc"
_NAME_NDVI = "NDVI_VNP43C4_v002_GHA_GAP_FILLED.nc"
_NAME_DEM = "srtm_5km_projection.nc"

copernicus_data = xr.open_dataset(os.path.join(data_path, _NAME_COPERNICUS)).isel(band=0)
chirps_data = xr.open_dataset(os.path.join(data_path, _NAME_CHIRPS))
era5_data = xr.open_dataset(os.path.join(data_path, _NAME_ERA5))
ndvi_data = xr.open_dataset(os.path.join(data_path, _NAME_NDVI))
dem_data = xr.open_dataset(os.path.join(data_path, _NAME_DEM)).isel(band=0)

# display(copernicus_data)

# DROP FIRST THREE DATAPOINTS TO MATCH NDVI TIMESERIES
chirps_data = chirps_data.isel(time=slice(3, None))
era5_data = era5_data.isel(time=slice(3, None))

# TRAINING - VALIDATION - TEST SPLIT
print("\nTrain-val-test splitting")
train_years = slice("2012", "2017")
val_years = slice("2018", "2019")
test_years = slice("2020", "2020")

chirps_train = chirps_data.sel(time=train_years)
era5_train   = era5_data.sel(time=train_years)
ndvi_train   = ndvi_data.sel(time=train_years)
chirps_val   = chirps_data.sel(time=val_years)
era5_val     = era5_data.sel(time=val_years)
ndvi_val     = ndvi_data.sel(time=val_years)
chirps_test  = chirps_data.sel(time=test_years)
era5_test    = era5_data.sel(time=test_years)
ndvi_test    = ndvi_data.sel(time=test_years)
print("Done!")

# INITIALIZE DATALOADERS
print("\nCreating torch datasets and dataloaders")

train_dataset = NDVIDataset(chirps_train, era5_train, ndvi_train, copernicus_data, dem_data)
val_dataset   = NDVIDataset(chirps_val, era5_val, ndvi_val, copernicus_data, dem_data,
                           stats={"means": train_dataset.means,
                                  "stds": train_dataset.stds},
                           )
test_dataset  = NDVIDataset(chirps_test, era5_test, ndvi_test, copernicus_data, dem_data,
                           stats={"means": train_dataset.means,
                                  "stds": train_dataset.stds},
                           )
# sample= train_dataset[0]['dynamic']
# test = train_dataset[:]['dynamic']
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=10)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=10)
print("Done!")
# sample = train_dataset[0]
# print(len(train_loader))
# # Define the model

# --- Reproducibility ---
def set_seed(seed=42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Model ---
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)        # [B, T, H]
        last = out[:, -1, :]         # last time step
        yhat = self.fc(last)         # [B, 1]
        return yhat

# --- Init function ---
def init_model(input_size, hidden_size=64, num_layers=1, dropout=0.0, seed=42, device=None):
    set_seed(seed)
    model = LSTMRegressor(input_size, hidden_size, num_layers, dropout)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    return model.to(device)
cuda = "cuda" if torch.cuda.is_available() else "cpu"
print(cuda)
# Example: initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = init_model(10, hidden_size=64, seed=42, device=device)

#Train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, max_epochs=20, patience=10):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    for epoch in trange(1, max_epochs+1, desc="Training"):
        # --- Training ---
        print(epoch)
        model.train()
        train_loss = 0.0
        for count, (xb, yb) in enumerate(train_loader):
            # print(count)
            if count == 1000:
                break
            yb = yb.squeeze(-1)
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for count, (xb, yb) in enumerate(val_loader):
                if count == 500:
                    break
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                # print(loss)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        # Record
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Adjust LR
        scheduler.step(val_loss)
        # print(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    model.load_state_dict(best_model_state)
    return model, train_losses, val_losses

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

model, train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device)


# NEW: Evaluation function for the test set
def evaluate_model(model, test_loader, criterion, device):
    """Evaluates the model on the test dataset."""
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            
            preds = model(xb)
            loss = criterion(preds, yb)
            test_loss += loss.item() * xb.size(0)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())

    # Note: len(test_loader.dataset) might be slow if __len__ is complex.
    # An alternative is to sum the batch sizes.
    test_loss /= len(test_loader.dataset)
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    r2 = r2_score(all_targets.flatten(), all_preds.flatten())
    # r2 = r2_score(all_targets, all_preds)
    #TODO inverse_transform

    print(f"\n--- Test Set Evaluation ---")
    print(f"Test Loss (MSE): {test_loss:.4f}")
    print(f"Test R^2 Score:  {r2:.4f}")
    print(f"---------------------------\n")

    return test_loss, r2


# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode='min', factor=0.5, patience=3
# )

# # Train the model
# model, train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device)

# Evaluate the final model on the test dataset
test_loss, test_r2 = evaluate_model(model, test_loader, criterion, device)

print(test_loss, test_r2)












# torch.save()
# Evaluate the model
def evaluate_model(model, X_t, y_t):
    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_t).cpu().numpy().ravel()
    # inverse transform
    # (ndvi_hist - self.means["ndvi"]) / self.stds["ndvi"]
    preds = (preds_scaled*ndvi_train.stds["ndvi"] + ndvi_train.means["ndvi"]).reshape(-1, 1).ravel()
    truth = (y_t*ndvi_train.stds["ndvi"] + ndvi_train.means["ndvi"]).ravel()
    # preds = preds_scaled.ravel()
    # truth = y_t.ravel()
    r2 = r2_score(truth, preds)
    return r2, truth, preds

# # Evaluate
# train_x, train_y = train_dataset.__main__(1)
# val_x, val_y = val_loader
# test_x, test_y = test_dataset[0][0].shape
# r2_train, y_true_train, y_pred_train = evaluate_model(model, train_x, train_y)
# r2_val,   y_true_val,   y_pred_val   = evaluate_model(model, val_x, val_y)
# r2_test,  y_true_test,  y_pred_test  = evaluate_model(model, test_x, test_y)
# print(r2_train, r2_val, r2_test)
# # r2_train, y_true_train, y_pred_train = evaluate_model(model, X_train_t, y_train_t, dates_train)
# # r2_val,   y_true_val,   y_pred_val   = evaluate_model(model, X_val_t, y_val_t, dates_val)
# # r2_test,  y_true_test,  y_pred_test  = evaluate_model(model, X_test_t, y_test_t, dates_test)



# #Figure
# fig, ax = plt.subplots(figsize=(12, 4))

# # Observed GPP for the whole period
# ax.plot(df.index, df[target_col], color="black", linewidth=0.7, label="Observed")

# # Predictions by split
# ax.plot(y_true_train.index, y_pred_train, color="tab:blue", alpha=0.7, label=f"Predicted (Train, R2={r2_train:.2f})")
# ax.plot(y_true_val.index,   y_pred_val,   color="tab:orange", alpha=0.7, label=f"Predicted (Val, R2={r2_val:.2f})")
# ax.plot(y_true_test.index,  y_pred_test,  color="tab:green", alpha=0.7, label=f"Predicted (Test, R2={r2_test:.2f})")

# # Highlight test period
# ax.axvspan(y_true_test.index[0], y_true_test.index[-1], color="gray", alpha=0.1)

# # Labels and title
# ax.set_title("Observed vs Predicted NDVI", fontsize=12)
# ax.set_ylabel("NDVI (-))")
# ax.set_xlabel("Date")

# # Legend
# ax.legend(loc="lower left", fontsize=10, frameon=False, ncol=4)

# plt.tight_layout()
# plt.show()