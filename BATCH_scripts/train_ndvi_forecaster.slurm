#!/bin/bash
# Job name
#SBATCH --job-name=ndvi_forecast_test
# Standard output and error:
#SBATCH -o ./log_files/out.%j
#SBATCH -e ./log_files/err.%j
# Initial working directory:
#SBATCH -D ./
#SBATCH --time=00:02:00
#SBATCH --nodes=1            # Request 1 or more full nodes
#SBATCH --constraint="gpu"   #   providing GPUs.
#SBATCH --gres=gpu:a100:4    # Request 4 GPUs per node.
#SBATCH --ntasks-per-node=4  # Run one task per GPU
#SBATCH --cpus-per-task=18   #   using 18 cores each.
#SBATCH --mem=32G

module purge
module load gcc/10 impi/2021.2
module load anaconda/3/2021.05

source ~/.bashrc
# Activate environment
conda deactivate
which python3

conda activate test_env
cd
$echo pwd

# Optional: change to your working directory
cd /u/mp006/ELLIS_summer_school_veg_forecast/model_building/
pwd

# Set number of OMP threads to fit the number of available cpus, if applicable.
export OMP_NUM_THREADS=1
which python3


# Run the training script
srun python3 CNN_3d.py
