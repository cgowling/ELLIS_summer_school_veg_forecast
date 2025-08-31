# Using machine learning methods to forecast drought in East Africa with multi-modal remote sensing data

## 1. Getting setup on the cluster

**ELLIS summer school documentation on the MPCDF cluster can be found here:** 

- Linux/ MacOS-X users: https://pad.gwdg.de/s/2KKJPuo1W
- Windows users: https://pad.gwdg.de/s/Gqm31-5j6

**Raven cluster documentation:**
https://docs.mpcdf.mpg.de/doc/computing/raven-user-guide.html

If not on an internal network, depending on your setup this should require your password and OTP (Summer school documentation above for setting up ssh config file )
```
$ ssh <user>@gate.mpcdf.mpg.de
```

From within network or after the above login, if on external network 
```
$ ssh <user>@raven.mpcdf.mpg.de
```

## 2. Set up  an ssh Key to connect cluster <---> git 
Once you are logged into Raven you need to set up an ssh key for connecting with git.

Below are some very brief instructions if you need a memory jog, if you want more detailed instructions see here: https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent

1. Check for exisiting keys 
``` shell
$ ls -al ~/.ssh
```

2. Generate ssh key 
```shell
$ ssh-keygen -t rsa -b 4096 -C "<your_email>@example.com"
```

3. Create a config file  (unless you already have one)

```shell
emacs ~/.ssh/config
```

4. Add this to your config file, ensuring the file specified matches the public key you just created. 
```text
Host github.com
  AddKeysToAgent yes
  UseKeychain yes
  IdentityFile ~/.ssh/id_ed25519
```

5. Copy the contents of your public key and add a new key to  your github account

## 3. Clone ELLIS_summer_school_veg_forecast repository to your Raven account
Once you've logged in you should be in your home directory which has a path like `/u/<userid>`. From here clone the research challenge repository 
```
$ git clone git@github.com:cgowling/ELLIS_summer_school_veg_forecast.git
```

## 4. Create conda environment 
Navigate into the repository you have just cloned 

```shell
$ cd ELLIS_summer_school_veg_forecast
```
(From Raven documentation)
Due to the hierarchical module environment, many libraries and applications only appear after loading a compiler, and subsequently also a MPI module (_the order is important here: first, load compiler, then load the MPI_). These modules provide libraries and software consistently compiled with/for the user-selected combination of compiler and MPI library.
```shell
$ module load intel/21.2.0 impi/2021.2 anaconda/3/2023.03

```
You may wish to use a different compiler or MPI combination, but this should work for setting up the environment (see Raven docs for more info).


There is a specification file in the repository that should have most of the packages you need. To build a conda environment:
```shell
$ conda create --name myenv --file spec-file.txt
```

Once your environment has finished building, test the environment by activating it
```shell
$ conda activate myenv
```

Have a look at what packages are available 
```shell
$ conda list
```

## 5. Test run Batch script 

Now that the initial setup is complete we can test the setup by running a small batch run.

Raven documentation "The batch system on the HPC cluster Raven is the open-source workload manager Slurm (Simple Linux Utility for Resource management). To run test or production jobs, submit a job script (see'BATCH_scripts/train_ndvi_simple.slurm' ) to Slurm, which will find and allocate the resources required for your job (e.g. the compute nodes to run your job on)."

Have a look at the files  'BATCH_scripts/train_ndvi_simple.slurm' and 'model_building/test_model/CNN3_d.py'

This is example uses a small geographical subset of the NDVI dataset, and forecasts NDVI values 4 weeks in advance based on the 12 previous weeks, where the model is a 3D CNN (This is more of a demonstration of how to run code from within a BATCH script rather than an example model. )

See the Raven documentation for more example SLURM Batch scripts.

To run this test job navigate to the BATCH_scripts folder 
```shell
$ cd BATCH_scripts
# Output and error files in the above slurm file have been specified to be written to the log_files folder
$ mkdir log_files # this will be ignored by git 
```

To submit a job use the sbatch command followed by your batch script. You'll note this run is only scheduled for 10 minutes, it is a good idea to start small  when testing out your setup.
```
sbatch train_ndvi_simple.slurm
```

To check on your run, replace <jobid> with your  jobid which is printed out when a job is submitted 

```
$ scontrol show jobid -dd <jobid>
```
##  6. Data 

The data for this challenge can be found at the shared directory 

```
/ptmp/mp002/ellis/veg_forecast
```
The data has been reprojected so all datasets match the NDVI (Normalised difference vegetation index) dataset resolution and projection of 5km 

The datasets roughly cover a bounding box over the Greater Horn of Africa 

**GHA**
```python
geometry = [21.282,-5.618,55.411,24.293]
```
You may want to focus your efforts on this region covered by the PASSAGE research project, a cross border region covering parts of Kenya, Ethiopia, Uganda, South Sudan and Somalia. In these regions pastoralist communities rely on rain-red agriculture like grasslands to sustain their livestock.  A reliable vegetation condition forecast could enable communities and decision makers to take action early to reduce or prevent the impacts of droughts. 

**PASSAGE bbox**
```python
geometry = [32.098 ,7.186,43.330,  0.754]
```

**Data summary**

| Data source                 | Indicator(s)                  | Units                   | Source resolution | Resampling method | Indicator label                 |
|-----------------------------|-------------------------------|-------------------------|-------------------|-------------------|---------------------------------|
| VIIRS: VNP43C4 v002         | NDVI                          | NA                      | ~5km              | NA                | 'NDVI_VNP43C4_GHA'              |
| CHIRPS v002                 | Sum Precipitation             | mm per week             | ~5km              | nearest           | 'CHIRPS_mm_per_week'            |
| Era5 land                   | total evaporation sum         | m of water equivalent   | ~9km              | nearest           | 'total_evaporation_sum'         |
|                             | potential evaporation sum     | m per week              | ~9km              | nearest           | 'potential_evaporation_sum'     |
|                             | total precipitation sum       | m per week              | "                 | "                 | 'total_precipitation_sum'       |
|                             | volumetric soil water layer 1 | Volume fraction         | "                 | "                 | 'volumetric_soil_water_layer_1' |
|                             | volumetric soil water layer 2 | Volume fraction         | "                 | "                 | 'volumetric_soil_water_layer_2' |
|                             | volumetric soil water layer 3 | Volume fraction         | "                 | "                 | 'volumetric_soil_water_layer_3' |
|                             | volumetric soil water layer 4 | Volume fraction         | "                 | "                 | 'volumetric_soil_water_layer_4' |
|                             | temperature_2m'               | K                       | "                 | "                 | temperature_2m'                 |
| NASA SRTM Digital Elevation | Elevation                     | metres                  | 30 m              | mean              | NA                              |
| Copernicus Landcover 2019   | Landcover                     | discrete classification | 100m              | mode              | NA                              |


See discrete landcover labels here https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_Landcover_100m_Proba-V-C3_Global#bands 
