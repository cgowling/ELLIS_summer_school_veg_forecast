#!/bin/bash
# Job name
#SBATCH --job-name=ndvi_forecast_test
# Standard output and error:
#SBATCH -o ./out.%j
#SBATCH -e ./err.%j
# Initial working directory:
#SBATCH -D ./
#SBATCH --time=00:10:00
#SBATCH --nodes=1            # Request 1 or more full nodes
#SBATCH --constraint="gpu"   #   providing GPUs.
#SBATCH --gres=gpu:a100:4    # Request 4 GPUs per node.
#SBATCH --ntasks-per-node=4  # Run one task per GPU
#SBATCH --cpus-per-task=18   #   using 18 cores each.
#SBATCH --mem=32G

# Load required modules (if using module system)
module purge

module load intel/21.2.0 impi/2021.2 anaconda/3/2023.03

# Activate environment
conda activate test_env

# Optional: change to your working directory
cd ELLIS_summer_school_veg_forecast/model_building/
pwd

# Run the training script
srun python CNN_3d.py