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

# from captum.attr import Saliency, IntegratedGradients
# import pre_process_timeseries as pre_pro
# import rasterio

# """"""""""""""""""""
# # Paths to DATA
# """"""""""""""""""""
#  Load data arrays and tiffs

path_to_NDVI_data = "/ptmp/mp002/ellis/veg_forecast/NDVI_VNP43C4_v002_GHA.nc"
path_to_CHIRPS_data = "//ptmp/mp002/ellis/veg_forecast/CHIRPS_v002_weekly_sum_VNP43C4_GHA.nc"
path_elevation = "/ptmp/mp002/ellis/veg_forecast/srtm_5km_projection.nc"
path_landcover = "/ptmp/mp002/ellis/veg_forecast/COPERNICUS_Landcover_2019_GHA_reprojected.nc"
# path_to_era5 = "/ptmp/mp002/ellis/veg_forecast/COPERNICUS_Landcover_2019_GHA_reprojected.nc"

# """"""""""""""""""""""""""""""""""""""""""""""""""
# # LOAD Data
# """"""""""""""""""""""""""""""""""""""""""""""""""
# SINGLE pixel from near DOLO (District in Somalia)
#
with xr.open_dataset(path_elevation) as dtm:
    display(dtm)
    plt.pcolormesh(dtm.band_data.sortby('y').isel(band=0), vmax=2500)
    plt.colorbar()

with xr.open_dataset(path_to_NDVI_data) as NDVI:
    display(NDVI)
    ndvi = NDVI.NDVI_VNP43C4_GHA.sortby('lat')
    plt.pcolormesh(ndvi.count('time'))

with xr.open_dataset(path_to_CHIRPS_data) as PREC:
    display(PREC)
    prec_sort = PREC.CHIRPS_mm_per_week.sortby('lat')
    prec = prec_sort.sel(time=slice("2012-01-28", "2024-12-29"))
    display(prec)
# # Full descriptive names for titles
# full_names = {
#     "air_temp": "Air Temperature",
#     "shortwave_rad": "Downward Shortwave Radiation",
#     "precip": "Precipitation",
#     "vpd": "Vapour Pressure Deficit",
#     "gpp": "Gross Primary Production"
# }

# # Units
# units = {
#     "air_temp": "°C",
#     "shortwave_rad": "W/m2",
#     "precip": "mm/day",
#     "vpd": "kPa",
#     "gpp": "gC/m2/day"
# }

# vars_to_plot = feature_cols + [target_col]

# fig, axes = plt.subplots(len(vars_to_plot), 1, figsize=(8, 7), sharex=True)
# for ax, var in zip(axes, vars_to_plot):
#     df[var].plot(ax=ax, linewidth=1)
#     ax.set_title(f"{full_names[var]} ({units[var]})", fontsize=10, loc="left")
# plt.tight_layout()
# plt.show()
lon = 350
lat = 150
prec2d = prec.isel(lon=lon, lat=lat).values
ndvi2d = ndvi.isel(lon=lon, lat=lat).values
ndvi2d_target = ndvi.isel(lon=lon, lat=lat).values
dates = ndvi.isel(lon=lon, lat=lat).time.values

len(ndvi2d)
df = pd.DataFrame(prec2d)
# df['time']
df['ndvi2d'] = ndvi2d
# df['ndvi2d_target'] = ndvi2d
df = df.rename(columns={0: "prec2d"})
df = df.set_index(dates)

# Select features and target
# feature_cols = ["prec2d", "ndvi2d", "dem", "lc"]
# target_col = "ndvi2d_target"
feature_cols = ["prec2d"]
target_col = "ndvi2d"

seq_len = 15  # number of weeks in each input sequence
train_years = (2000, 2017)
val_years   = (2018, 2019)
test_years  = (2020)

# Split by year
df_train = df.loc[str(train_years[0]):str(train_years[1])]
df_val   = df.loc[str(val_years[0]):str(val_years[1])]
df_test  = df.loc[str(test_years)]


print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

# Scaling
X_scaler = StandardScaler().fit(df_train[feature_cols])
y_scaler = StandardScaler().fit(df_train[[target_col]])

def scale_df(df, X_scaler, y_scaler):
    out = df.copy()
    out[feature_cols] = X_scaler.transform(df[feature_cols])
    out[target_col] = y_scaler.transform(df[[target_col]])
    return out

df_train_scaled = scale_df(df_train, X_scaler, y_scaler)
df_val_scaled   = scale_df(df_val, X_scaler, y_scaler)
df_test_scaled  = scale_df(df_test, X_scaler, y_scaler)
# df_train_scaled = df_train
# df_val_scaled   = df_val
# df_test_scaled  = df_test



def create_sequences(data, seq_len, feature_cols, target_col):
    X, y, dates = [], [], []
    arr_X = data[feature_cols].values
    arr_y = data[target_col].values
    idx = data.index
    for i in range(len(data) - seq_len + 1):
        X.append(arr_X[i:i+seq_len])
        y.append(arr_y[i+seq_len-1])    # same-day target
        dates.append(idx[i+seq_len-1])  # date of target
    return np.array(X), np.array(y), dates

# Create sequences
X_train, y_train, dates_train = create_sequences(df_train_scaled, seq_len, feature_cols, target_col)
X_val,   y_val,   dates_val   = create_sequences(df_val_scaled, seq_len, feature_cols, target_col)
X_test,  y_test,  dates_test  = create_sequences(df_test_scaled, seq_len, feature_cols, target_col)



# Convert to tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
X_val_t   = torch.tensor(X_val, dtype=torch.float32)
y_val_t   = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)
X_test_t  = torch.tensor(X_test, dtype=torch.float32)
y_test_t  = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

print(f"Input shape:\n  Train - {X_train_t.shape}\n  Val   - {X_val_t.shape}\n  Test  - {X_test_t.shape}")
print(f"Target shape:\n  Train - {y_train_t.shape}\n  Val   - {y_val_t.shape}\n  Test  - {y_test_t.shape}")

# Define the model

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

# Example: initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = init_model(len(feature_cols), hidden_size=64, seed=42, device=device)

#Train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, max_epochs=30, patience=10):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    for epoch in trange(1, max_epochs+1, desc="Training"):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for count, (xb, yb) in enumerate(train_loader):
            if count == 1000:
                continue
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
                if count == 100:
                    continue
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        # Record
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Adjust LR
        scheduler.step(val_loss)

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

# DataLoaders
batch_size = 64
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)
# for i, j in enumerate(train_loader):
#     print(j[0].shape, j[1].shape)
#     break
# sample = train_loader[0]

model, train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device)

# Evaluate the model
def evaluate_model(model, X_t, y_t, dates, y_scaler):
    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_t).cpu().numpy().ravel()
    # inverse transform
    preds = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
    truth = y_scaler.inverse_transform(y_t.cpu().numpy().ravel().reshape(-1, 1)).ravel()
    # preds = preds_scaled.ravel()
    # truth = y_t.ravel()
    r2 = r2_score(truth, preds)
    return r2, pd.Series(truth, index=dates), pd.Series(preds, index=dates)

# Evaluate
r2_train, y_true_train, y_pred_train = evaluate_model(model, X_train_t, y_train_t, dates_train, y_scaler)
r2_val,   y_true_val,   y_pred_val   = evaluate_model(model, X_val_t, y_val_t, dates_val, y_scaler)
r2_test,  y_true_test,  y_pred_test  = evaluate_model(model, X_test_t, y_test_t, dates_test, y_scaler)
# r2_train, y_true_train, y_pred_train = evaluate_model(model, X_train_t, y_train_t, dates_train)
# r2_val,   y_true_val,   y_pred_val   = evaluate_model(model, X_val_t, y_val_t, dates_val)
# r2_test,  y_true_test,  y_pred_test  = evaluate_model(model, X_test_t, y_test_t, dates_test)



#Figure
fig, ax = plt.subplots(figsize=(12, 4))

# Observed GPP for the whole period
ax.plot(df.index, df[target_col], color="black", linewidth=0.7, label="Observed")

# Predictions by split
ax.plot(y_true_train.index, y_pred_train, color="tab:blue", alpha=0.7, label=f"Predicted (Train, R2={r2_train:.2f})")
ax.plot(y_true_val.index,   y_pred_val,   color="tab:orange", alpha=0.7, label=f"Predicted (Val, R2={r2_val:.2f})")
ax.plot(y_true_test.index,  y_pred_test,  color="tab:green", alpha=0.7, label=f"Predicted (Test, R2={r2_test:.2f})")

# Highlight test period
ax.axvspan(y_true_test.index[0], y_true_test.index[-1], color="gray", alpha=0.1)

# Labels and title
ax.set_title("Observed vs Predicted NDVI", fontsize=12)
ax.set_ylabel("NDVI (-))")
ax.set_xlabel("Date")

# Legend
ax.legend(loc="lower left", fontsize=10, frameon=False, ncol=4)

plt.tight_layout()
plt.show()