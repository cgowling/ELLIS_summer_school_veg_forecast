import xarray as xr
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import pre_process_timeseries as pre_pro
import rasterio

def open_dataset_bounding_box(filepath, variable_name, min_lat, max_lat, min_lon, max_lon):
    ds = xr.open_dataset(filepath)
    ds_raw = ds[variable_name]
    ds_bbox = ds_raw.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))
    return ds_bbox

# """"""""""""""""""""
# # Paths to DATA
# """"""""""""""""""""
#  Load data arrays and tiffs

path_to_NDVI_data = "/ptmp/mp002/ellis/veg_forecast/NDVI_VNP43C4_v002_GHA.nc"
path_to_CHIRPS_data = "//ptmp/mp002/ellis/veg_forecast/CHIRPS_v002_weekly_sum_VNP43C4_GHA.nc"
path_elevation = "/ptmp/mp002/ellis/veg_forecast/srtm_5km_projection.tif"
path_landcover = "/ptmp/mp002/ellis/veg_forecast/COPERNICUS_Landcover_2019_GHA_reprojected.tif"


# """"""""""""""""""""""""""""""""""""""""""""""""""
# # LOAD Data
# """"""""""""""""""""""""""""""""""""""""""""""""""
# SINGLE pixel from near DOLO (District in Somalia)
#
df_raw = pre_pro.load_raw_time_series_for_pixel(lat=4.321199999999996, lon = 42.5176, filepath= path_to_NDVI_data, variable_name="NDVI" )
df_raw_CHIRPS = pre_pro.load_raw_time_series_for_pixel(lat=4.321199999999996, lon = 42.5176, filepath= path_to_NDVI_data, variable_name= "CHIRPS")

elevation = rasterio.open(path_elevation).read(1)
landcover = rasterio.open(path_landcover).read(1)


# Load data arrays for  Bounding box cluster3
min_lat =5.8462
max_lat =  1.16
min_lon = 39.0926
max_lon = 43.3669

#DOLO bounding box
# min_lat = 3.5611
# max_lat = 4.321199999999996
# min_lon = 41.817600000000006
# max_lon = 42.5176


ds_bbox_ndvi = open_dataset_bounding_box(path_to_NDVI_data , 'NDVI_VNP43C4_GHA', min_lat, max_lat, min_lon, max_lon)

ds_bbox_chirps = open_dataset_bounding_box(path_to_CHIRPS_data , 'CHIRPS_mm_per_week', min_lat, max_lat, min_lon, max_lon)


# """"""""""""""""""""""""""""""""""""""""""""""""""
# Create NDVI and CHIRPS  stacked dataframe
# """"""""""""""""""""""""""""""""""""""""""""""""""

# Flatten spatial dimensions into one and  optionally drop NaNs
df_ndvi = ds_bbox_ndvi.stack(pixel=['lat', 'lon', 'time']).to_dataframe().reset_index(drop= True)
# df_ndvi = df_ndvi.dropna(subset=['NDVI_VNP43C4_GHA'])

df_chirps = ds_bbox_chirps.stack(pixel=['lat', 'lon', 'time']).to_dataframe().reset_index(drop= True)
# df_chirps = ds_bbox_chirps.dropna(subset=["CHIRPS_weekly_sum"])


# Create a  month  and year column
df_ndvi['month'] = pd.to_datetime(df_ndvi['time']).dt.month
df_ndvi['year'] = pd.to_datetime(df_ndvi['time']).dt.year

df_chirps['month'] = pd.to_datetime(df_chirps ['time']).dt.month
df_chirps['year'] = pd.to_datetime(df_chirps ['time']).dt.year

# """"""""""""""""""""""""""""""""""""""""""""""""""
# # Plot Violin  plots distribution in values over all time for each month
# """"""""""""""""""""""""""""""""""""""""""""""""""

# """"""""""""""""""""""""""""""""""""""""""""""""""
#  NDVI
# """"""""""""""""""""""""""""""""""""""""""""""""""
plt.figure(figsize=(12, 6))
sns.violinplot(x='month', y='NDVI_VNP43C4_GHA', data=df_ndvi, palette="viridis", hue='month', legend=False)

plt.title(f"Monthly NDVI Distribution for BBox cluster 3")
plt.xlabel("Month")
plt.ylabel("NDVI")
plt.xticks(ticks=range(0, 12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.tight_layout()
plt.show(block=True)

# """"""""""""""""""""""""""""""""""""""""""""""""""
#  PRECIPITATION
# """"""""""""""""""""""""""""""""""""""""""""""""""

monthly_precipitation = df_chirps.groupby(
    ['lat', 'lon', 'month','year'], as_index=False
)['CHIRPS_mm_per_week'].sum()
monthly_precipitation.rename(columns={'CHIRPS_mm_per_week': 'CHIRPS_monthly_total_precipitation'}, inplace=True)


plt.figure(figsize=(12, 6))
sns.violinplot(x='month', y="CHIRPS_monthly_total_precipitation", data=monthly_precipitation, palette="viridis", hue='month', legend=False)


plt.title(f"Monthly PrecipitationDistribution for BBox cluster 3")
plt.xlabel("Month")
plt.ylabel("CHIRPS precipitation")
plt.xticks(ticks=range(0, 12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

plt.tight_layout()
plt.show(block=True)

plt.figure(figsize=(12, 6))

# """"""""""""""""""""""""""""""""""""""""""""""""""
#  Comparing monthly  median NDVI value year on year
# """"""""""""""""""""""""""""""""""""""""""""""""""

years = df_ndvi['year'].unique().tolist()

for i, year in enumerate(years):
    print(year)

    data_year = df_ndvi[df_ndvi["year"] == year]
    data_year_month_spatial_group = data_year.groupby([ 'month'], as_index=False)['NDVI_VNP43C4_GHA'].median()
    sns.lineplot(x='month', y='NDVI_VNP43C4_GHA', data=data_year_month_spatial_group, label=year)
plt.legend()
plt.tight_layout()
plt.show(block=True)


# """"""""""""""""""""""""""""""""""""""""""""""""""
#  Violin plot for a single year
# """"""""""""""""""""""""""""""""""""""""""""""""""
year = 2020
ndvi_year = ds_bbox_ndvi.sel(time=str(year))

# Flatten spatial dimensions into one and drop NaNs
df_year = ndvi_year.stack(pixel=['lat', 'lon', 'time']).to_dataframe().reset_index(drop= True)
df_year = df_year.dropna(subset=['NDVI_VNP43C4_GHA'])

# Extract month
df_year['month'] = pd.to_datetime(df_year['time']).dt.month

# Plot boxplots
plt.figure(figsize=(12, 6))
sns.violinplot(x='month', y='NDVI_VNP43C4_GHA', data=df_year, palette="viridis", hue='month', legend=False)
plt.title(f"Monthly NDVI Distribution for BBox cluster 3 {year}")
plt.xlabel("Month")
plt.ylabel("NDVI")
plt.xticks(ticks=range(0, 12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.tight_layout()
plt.show(block=True)