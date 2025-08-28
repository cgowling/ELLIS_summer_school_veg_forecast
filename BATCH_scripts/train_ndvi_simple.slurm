#!/bin/bash -l
#SBATCH --job-name=ndvi_forecast_test
# Standard output and error:
#SBATCH -o ./log_files/out.%j
#SBATCH -e ./log_files/err.%j
#SBATCH -D ./
#SBATCH --ntasks=1         # launch job on a single core
#SBATCH --cpus-per-task=1  #   on a shared node
#SBATCH --mem=2000MB       # memory limit for the job
#SBATCH --time=0:10:00

module purge

# Load necessary modules
module load gcc/10 impi/2021.2
module load anaconda/3/2021.05

conda deactivate 
#source ~/.bashrc

# Activate  conda environment
conda activate myenv_build

# Check the python from the conda environment is being used
which python3

# Change to your working directory
cd /u/mp006/ELLIS_summer_school_veg_forecast/model_building/test_model
pwd

# Set number of OMP threads to fit the number of available cpus, if applicable.
export OMP_NUM_THREADS=1

# Run the training script
srun python3 CNN_3d.py
