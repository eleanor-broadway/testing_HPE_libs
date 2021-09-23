#!/bin/bash -l
#SBATCH --job-name=mytest
#SBATCH --time=00:05:00
#SBATCH --ntasks=48
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --output=out.log.j%j
#SBATCH --error=out.log.j%j

echo "The current job ID is $SLURM_JOB_ID"
echo "Running on $SLURM_JOB_NUM_NODES nodes: $SLURM_JOB_NODELIST"
echo "Using $SLURM_NTASKS_PER_NODE tasks per node"
echo "A total of $SLURM_NTASKS tasks is used"

export OMP_NUM_THREADS=2
echo "Running on $HOSTNAME"

echo "Launching parallel job."
aprun -B ./test
echo "Parallel execution finished."
