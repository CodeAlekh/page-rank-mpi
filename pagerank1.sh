#!/bin/bash

#SBATCH --output=pagerank1.out
#SBATCH --partition general
#SBATCH --nodes=1     
#SBATCH --ntasks-per-node=1
#SBATCH --time 00:50:00
#SBATCH --job-name page_rank_mpi_1

module load openmpi/4.1.7-3ilj

cd ..
srun page_rank web-Google.pm