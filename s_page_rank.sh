#!/bin/bash

#SBATCH --output=opagerank.out
#SBATCH --partition general
#SBATCH --nodes=2      
#SBATCH --ntasks-per-node=8
#SBATCH --time 00:50:00
#SBATCH --job-name page_rank_mpi

module load openmpi/4.1.7-3ilj

cd $SLURM_SUBMIT_DIR
# Run using sbatch --nodes=4 --ntasks-per-node=16 to override the default values
srun page_rank web-Google.pm