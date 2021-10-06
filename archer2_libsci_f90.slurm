#!/bin/bash

# Slurm job options (job-name, compute nodes, job time)
#SBATCH --job-name=LibSci-f90
#SBATCH --time=0:20:0
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1

# Replace [budget code] below with your budget code (e.g. t01)
#SBATCH --account=z19
#SBATCH --partition=standard
#SBATCH --qos=standard

# Setup the job environment (this module needs to be loaded before any other modules)
module load epcc-job-env

cat $0

export OMP_NUM_THREADS=1

ldd libsci_f90.x
srun --distribution=block:block --hint=nomultithread ./libsci_f90.x
