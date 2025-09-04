# LOAD MODULES
print("Loading modules")

import torch
from dataloader import NDVIDataset
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import xarray as xr
import os
import torch.nn as nn
print("Done!")

# LOAD DATA
print("\nLoading data")

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

# DROP FIRST THREE DATAPOINTS TO MATCH NDVI TIMESERIES

chirps_data = chirps_data.isel(time=slice(3, None))
era5_data = era5_data.isel(time=slice(3, None))
print("Done!")

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

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=128, shuffle=False)
print("Done!")

# DEFINE SIMPLE LINEAR MODEL

class LinearNDVIModel(nn.Module):
    def __init__(self, seq_len=4, dynamic_features=7, static_features=3):
        super().__init__()
        # Flatten dynamic + static into one vector per sample
        self.linear = nn.Linear(seq_len * dynamic_features + static_features, 1)

    def forward(self, dynamic, static):
        # dynamic: (batch, seq_len, features)
        batch_size = dynamic.shape[0]
        x = torch.cat([dynamic.view(batch_size, -1), static], dim=1)
        out = self.linear(x)
        return out.squeeze(1)  # (batch,)
    
# TRAINING LOOP
print("\nTraining loop for linear model")

# STRATEGY 1: SUBSAMPLE DATASETS
class SubsampledNDVIDataset(NDVIDataset):
    def __init__(self, *args, subsample_factor=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.subsample_factor = subsample_factor
        self.original_len = len(self.valid_pixels) * (self.time_len - self.seq_len)
        self.subsampled_len = self.original_len // subsample_factor
        
    def __len__(self):
        return self.subsampled_len
    
    def __getitem__(self, idx):
        # Map subsampled index back to full dataset index
        actual_idx = (idx * self.subsample_factor) % self.original_len
        return super().__getitem__(actual_idx)

# Create subsampled datasets
print("Creating subsampled datasets (10x smaller)...")
train_dataset_sub = SubsampledNDVIDataset(
    chirps_train, era5_train, ndvi_train, copernicus_data, dem_data,
    subsample_factor=10
)
val_dataset_sub = SubsampledNDVIDataset(
    chirps_val, era5_val, ndvi_val, copernicus_data, dem_data,
    subsample_factor=10,
    stats={"means": train_dataset_sub.means, "stds": train_dataset_sub.stds}
)

# Create new dataloaders with subsampled data
train_loader = DataLoader(train_dataset_sub, batch_size=256, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset_sub, batch_size=256, shuffle=False, num_workers=2)

print(f"Train dataset size: {len(train_dataset_sub):,}")
print(f"Val dataset size: {len(val_dataset_sub):,}")

# STRATEGY 2: TRAINING WITH EARLY STOPPING
import time
from collections import defaultdict

device = "cuda" if torch.cuda.is_available() else "cpu"
model = LinearNDVIModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Training parameters
n_epochs = 50
patience = 5
min_delta = 1e-4
max_batches_per_epoch = 500  # Limit batches per epoch

# Early stopping variables
best_val_loss = float('inf')
patience_counter = 0
history = defaultdict(list)

print(f"Starting training with early stopping (max {max_batches_per_epoch} batches/epoch)...")

for epoch in range(n_epochs):
    start_time = time.time()
    
    # TRAINING with limited batches per epoch
    model.train()
    train_loss = 0
    batch_count = 0
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_count >= max_batches_per_epoch:
            break
            
        dynamic = batch["dynamic"].to(device)
        static = batch["static"].to(device)
        target = batch["target"].to(device)
        
        optimizer.zero_grad()
        pred = model(dynamic, static)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * dynamic.size(0)
        batch_count += 1
        
        # Progress indicator
        if batch_idx % 50 == 0:
            print(f"\r  Batch {batch_idx+1}/{min(len(train_loader), max_batches_per_epoch)}, Loss: {loss.item():.4f}", end="", flush=True)
    
    avg_train_loss = train_loss / (batch_count * train_loader.batch_size)
    
    # VALIDATION (also limit batches)
    model.eval()
    val_loss = 0
    val_batches = 0
    max_val_batches = 100
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_val_batches:
                break
                
            dynamic = batch["dynamic"].to(device)
            static = batch["static"].to(device)
            target = batch["target"].to(device)
            pred = model(dynamic, static)
            loss = criterion(pred, target)
            val_loss += loss.item() * dynamic.size(0)
            val_batches += 1
    
    avg_val_loss = val_loss / (val_batches * val_loader.batch_size)
    
    # Record history
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    
    epoch_time = time.time() - start_time
    print(f"\nEpoch {epoch+1}/{n_epochs} ({epoch_time:.1f}s)")
    print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    
    # Early stopping check
    if avg_val_loss < best_val_loss - min_delta:
        best_val_loss = avg_val_loss
        patience_counter = 0
        print(f"  -> New best validation loss! Saving model...")
        torch.save(model.state_dict(), 'best_linear_model.pth')
    else:
        patience_counter += 1
        print(f"  -> No improvement (patience: {patience_counter}/{patience})")
        
    if patience_counter >= patience:
        print(f"\nEarly stopping triggered after {epoch+1} epochs")
        break

print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")

# Load best model for final evaluation
model.load_state_dict(torch.load('best_linear_model.pth'))

# TEST EVALUATION
print("\nEvaluation on test set with best model")

# Create subsampled test dataset
test_dataset_sub = SubsampledNDVIDataset(
    chirps_test, era5_test, ndvi_test, copernicus_data, dem_data,
    subsample_factor=10,
    stats={"means": train_dataset_sub.means, "stds": train_dataset_sub.stds}
)
test_loader = DataLoader(test_dataset_sub, batch_size=256, shuffle=False, num_workers=2)

model.eval()
test_loss = 0
total_samples = 0

with torch.no_grad():
    for batch in test_loader:
        dynamic = batch["dynamic"].to(device)
        static = batch["static"].to(device)
        target = batch["target"].to(device)
        pred = model(dynamic, static)
        test_loss += criterion(pred, target).item() * dynamic.size(0)
        total_samples += dynamic.size(0)

test_loss /= total_samples
print(f"Final Test MSE: {test_loss:.4f}")

# Plot training history (optional)
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Training History')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Validation MSE')
    plt.title('Validation Loss')
    plt.axhline(y=best_val_loss, color='r', linestyle='--', label=f'Best: {best_val_loss:.4f}')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
except ImportError:
    print("Matplotlib not available for plotting")
    print(f"Training history: {len(history['train_loss'])} epochs completed")