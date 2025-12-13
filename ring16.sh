#!/bin/bash

#SBATCH --partition general
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 32
#SBATCH --time 01:30:00
#SBATCH --job-name ring-16
#SBATCH --mail-user bwitkowski@unm.edu
#SBATCH --mail-type ALL

module load openmpi
srun -n 1 ./ring.out
