#!/bin/bash

#SBATCH --output=opagerank.out
#SBATCH --partition debug
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=64
#SBATCH --time 00:10:00
#SBATCH --job-name page_rank_mpi

module load openmpi/4.1.7-3ilj

cd $SLURM_SUBMIT_DIR
# Run using sbatch --nodes=4 --ntasks-per-node=16 to override the default values
srun mod_page_rank web-Google.pm