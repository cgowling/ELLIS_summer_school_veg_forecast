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
module load gcc/10 impi/2021.2
module load anaconda/3/2021.05


source ~/.bashrc
# Activate environment
conda activate test_env


# Optional: change to your working directory
cd ELLIS_summer_school_veg_forecast/model_building/
pwd

# Set number of OMP threads to fit the number of available cpus, if applicable.
export OMP_NUM_THREADS=1



# Run the training script
srun python CNN_3d.py




