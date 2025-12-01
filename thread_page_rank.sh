#!/bin/bash

#SBATCH --output=thread_pagerank.out
#SBATCH --partition debug
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time 00:20:00
#SBATCH --job-name threaded_page_rank_mpi

module load openmpi/4.1.7-3ilj

cd $SLURM_SUBMIT_DIR

export OMP_NUM_THREADS=16
export OMP_PROC_BIND=close
export OMP_PLACES=cores

srun --cpu-bind=cores openmp_page_rank web-Google.pm