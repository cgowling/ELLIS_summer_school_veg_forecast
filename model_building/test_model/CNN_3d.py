import torch
import torch.nn as nn
import numpy as np 
import xarray as xr
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import datetime
import os
from sklearn.metrics import r2_score, root_mean_squared_error
import functools

print = functools.partial(print, flush=True)


def get_ndvi_test_array(path_to_NDVI_data):
    """

    :param path_to_NDVI_data: path to .nc file containing raw NDVI data
    :return: Subset of the NDVI data, nan replaced with linear  interpolation

    - loads the netcdf file containing weekly NDVI data from 2012-end of 2024
    - extracts subset of data last 2 years, 30 by 36 pixels
    - check for nan values
    - sort by ascending latidue values (for interpolation)
    - interploate nan values (leaves non noan values alone)
    - return to decending lat values
    - check no nans remaining
    - extract values into an array and return array
    """

    #  Test a sub sample , approx 2 years of data and top left corner
    NDVI = xr.open_dataarray(path_to_NDVI_data)
    print(NDVI.shape)



    #  need to have ascending coordinates for the interpolation to work
    NDVI_subset = NDVI.sortby('lat')
    #  Approx Dolo bounding box
    # min_lat = 3.5611
    # max_lat = 4.321199999999996
    # min_lon = 41.817600000000006
    # max_lon = 42.5176

    min_lat = 2.954
    max_lat = 4.321199999999996
    min_lon = 41.017600000000006
    max_lon = 42.6176

    NDVI_subset = NDVI_subset.isel(time=slice(-520, -1))

    NDVI_subset = NDVI_subset.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))
    print(NDVI_subset.shape)
    ndvi_array_raw = NDVI_subset.values  #
    print("shape of raw array", ndvi_array_raw.shape)
    print("number of nans", np.count_nonzero(np.isnan(ndvi_array_raw)))
    print("dataset size", ndvi_array_raw.size)

    print("percentage nans",np.count_nonzero(100*(np.isnan(ndvi_array_raw))/ ndvi_array_raw.size) )

    print("raw NDVI min:", np.nanmin(ndvi_array_raw))
    print("raw NDVI max:", np.nanmax(ndvi_array_raw))
    print("raw NDVI contains NaNs:", np.isnan(ndvi_array_raw).any())

    # Fill NaNs by interpolating over lat & lon (spatial)
    NDVI_filled = NDVI_subset.interpolate_na(dim="lat", method="nearest",fill_value="extrapolate")  # extrapolate allows nan values at the edge to be replaced
    NDVI_filled = NDVI_filled.interpolate_na(dim="lon", method="nearest", fill_value="extrapolate")


    NDVI_filled = NDVI_filled.sortby('lat', ascending=False)

    # Double-check for remaining NaNs
    print("NaNs remaining:", NDVI_filled.isnull().sum().item())

    # Convert to numpy array
    ndvi_array = NDVI_filled.values.astype(np.float32)

    return ndvi_array

class NDVIDataset(Dataset):
    def __init__(self, ndvi_data, input_steps=12, forecast_steps=4):
        """
        ndvi_data: 3D numpy or torch array with shape (time, height, width)
        """
        self.ndvi = torch.tensor(ndvi_data, dtype=torch.float32)
        self.input_steps = input_steps
        self.forecast_steps = forecast_steps
        self.total_time = self.ndvi.shape[0]
        self.num_samples = self.total_time - input_steps - forecast_steps + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = self.ndvi[idx:idx+self.input_steps]      # shape: [12, H, W]
        y = self.ndvi[idx+self.input_steps:idx+self.input_steps+self.forecast_steps]  # [4, H, W]
        x = x.unsqueeze(0)  # add channel dim -> [1, 12, H, W]
        return x, y  # x: [1, 12, H, W], y: [4, H, W]


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class ConvBlockSpatial(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UNet3DNDVIForecaster(nn.Module):
    def __init__(self, in_channels=1, future_steps=4):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, 16)
        self.spatial_pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.temporal_pool1 = nn.MaxPool3d(kernel_size=(2, 1, 1))
        # self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        # self.down1 = nn.Conv3d(32, 32, kernel_size=3, stride=(2, 1, 1), padding=(1, 1, 1))  # → T:6

        self.enc2 = ConvBlock(16, 32)
        self.spatial_pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.temporal_pool2 = nn.MaxPool3d(kernel_size=(2, 1, 1))



        # Bottleneck
        self.bottleneck = ConvBlock(32, 64)

        # Decoder

        self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec2 = ConvBlockSpatial(64, 32)
        # self.dec2 = ConvBlock(64, 32)

        self.up1 = nn.ConvTranspose3d(32, 16, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec1 = ConvBlockSpatial(32, 16)
        # self.dec1 = ConvBlock(32, 16)

        # Final output layer
        self.final = nn.Conv3d(16, future_steps, kernel_size=(1, 3, 3), padding=(0, 1, 1))


    def forward(self, x):
        # x: [B, 1, T, H, W]
        enc1 = self.enc1(x)     # -> [B, 16, T, H, W]
        # print("enc1 shape", enc1.shape)
        spatial_pool1 = self.spatial_pool1(enc1) # -> [B, 32, T, H/2, W/2]
        temporal_pool1 = self.temporal_pool1(spatial_pool1)  # -> [B, 32, T/2, H/2, W/2]

        enc2 = self.enc2(temporal_pool1)
        spatial_pool2 = self.spatial_pool2(enc2) # -> [B, 32, T, H/4, W/4]
        temporal_pool2 = self.temporal_pool2(spatial_pool2)  # -> [B, 32, T/4, H/4, W/4]
        # print("enc2 shape", enc2.shape)

        bottleneck = self.bottleneck(temporal_pool2)  # -> [B, 64, T/4, H/4, W/4]
        # print("bottleneck", bottleneck.shape)
        #  compress temporal dimension
        bottleneck = torch.mean(bottleneck, dim=2, keepdim=True)  # shape: [B, 64, 1, H/4, W/4]

        u2 = self.up2(bottleneck)
        # print("u2", u2.shape, "enc2",enc2.shape)
        u2 = torch.cat((u2, torch.mean(enc2, dim=2, keepdim=True)), dim=1)# skip connection
        # u2 = torch.cat((u2, enc2), dim=1)# skip connection
        dec2 = self.dec2(u2)

        u1 = self.up1(dec2)
        # print("u1", u1.shape, "enc1", enc1.shape)
        u1 = torch.cat((u1, torch.mean(enc1, dim=2, keepdim=True)), dim=1)# skip connection
        # u1 = torch.cat((u1, enc1), dim=1)# skip connection
        dec1 = self.dec1(u1)
        # print("dec1",dec1.shape)

        # dec1= torch.mean(dec1, dim=2, keepdim=True)  # shape: [B, 6, 1, H, W]
        out = self.final(dec1)  # -> [B, future_steps, 1, H, W]
        # print(out.shape)
        # out = out[:, :, -1, :, :]
        # out = out.mean(dim=2)

        out = out.squeeze(2)    # remove time dim → [B, future_steps, H, W]

        return out

def evaluate_model(model, val_loader, device="cpu"):
    model.eval()
    all_preds = [[] for _ in range(4)]  # for each lead time
    all_targets = [[] for _ in range(4)]

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            preds = model(batch_x).squeeze(2)  # shape: [B, 4, H, W]

            for i in range(4):  # lead times 1 to 4
                pred_i = preds[:, i, :, :].cpu().numpy().flatten()
                target_i = batch_y[:, i, :, :].cpu().numpy().flatten()

                all_preds[i].append(pred_i)
                all_targets[i].append(target_i)

    # Compute R² and RMSE for each lead time
    for i in range(4):
        y_true = np.concatenate(all_targets[i])
        y_pred = np.concatenate(all_preds[i])

        r2 = r2_score(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred, squared=False)

        print(f"Lead time {i+1} week(s): R² = {r2:.4f}, RMSE = {rmse:.4f}")


def plot_predictions_vs_actuals(model, val_loader, device="cpu", sample_size=10000):
    model.eval()
    all_preds = [[] for _ in range(4)]
    all_targets = [[] for _ in range(4)]

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            preds = model(batch_x).squeeze(2)  # [B, 4, H, W]

            for i in range(4):  # lead times 1 to 4
                pred_i = preds[:, i, :, :].cpu().numpy().flatten()
                target_i = batch_y[:, i, :, :].cpu().numpy().flatten()

                all_preds[i].append(pred_i)
                all_targets[i].append(target_i)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i in range(4):
        y_pred = np.concatenate(all_preds[i])
        y_true = np.concatenate(all_targets[i])

        # Optional: sample for clarity
        if len(y_true) > sample_size:
            idx = np.random.choice(len(y_true), size=sample_size, replace=False)
            y_true = y_true[idx]
            y_pred = y_pred[idx]

        r2 = r2_score(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)

        ax = axes[i]
        ax.scatter(y_true, y_pred, alpha=0.3, s=5, label=f'R²={r2:.3f}, RMSE={rmse:.4f}')
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        ax.set_title(f'Lead Time {i+1} Week(s)')
        ax.set_xlabel("Actual NDVI")
        ax.set_ylabel("Predicted NDVI")
        ax.legend()

    plt.tight_layout()
    # plt.show(block=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fig.savefig(f"results/CNN_results_{timestamp}.png")


#  Load NDVI arrays for 2012-2025
path_to_NDVI_data = "/ptmp/mp002/ellis/veg_forecast/NDVI_VNP43C4_v002_GHA.nc"


ndvi_array = get_ndvi_test_array(path_to_NDVI_data)

# Example input: batch of NDVI with 12 historical time steps, HxW spatial resolution
input_steps = 12
height = ndvi_array .shape[1]
width = ndvi_array .shape[2]
future_steps = 4



os.makedirs("checkpoints", exist_ok=True)
os.makedirs("../results", exist_ok=True)


# ----------------------------
# DataLoader
# ----------------------------
dataset = NDVIDataset(ndvi_array, input_steps=input_steps, forecast_steps=future_steps)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4)

model = UNet3DNDVIForecaster(in_channels=1, future_steps=future_steps)


#  Is the following just a repetition?
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# -----------------------------
# Training loop
# -----------------------------
num_epochs = 10
for epoch in range(num_epochs):
    print("Epoch", epoch)
    model.train()
    epoch_loss = 0

    for batch_x, batch_y in train_loader:
        # print("input shape", batch_x.shape)
        # x: [4, 1, 12, H, W] [num_batches, num_channels, input_steps, height, width ]
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)# NDVI future targets  for 4 samples torch.Size([4, 4, H, W])
        # print("Target shape:", batch_y.shape)

        optimizer.zero_grad()
        output = model(batch_x)  # shape: [B, 4, H, W]
        # print("Output shape:", output.shape)


        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
    if (epoch + 1) % 5 == 0:  # every 5 epochs
        torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch + 1}.pth")




# evaluate_model(model, val_loader, device=device)

plot_predictions_vs_actuals(model, val_loader, device=device)
