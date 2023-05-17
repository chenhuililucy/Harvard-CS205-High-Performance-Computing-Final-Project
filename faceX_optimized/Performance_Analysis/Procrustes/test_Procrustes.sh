#!/usr/bin/env bash
#SBATCH --job-name=test
#SBATCH --output=test.out
#SBATCH --error=test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:15:00

# Load required modules
module load git/2.17.0-fasrc01 python/3.9.12-fasrc01 gcc/12.1.0-fasrc01 openmpi/4.1.3-fasrc01 opencv/3.4.3-fasrc01 centos6/0.0.1-fasrc01 pango/1.28.4-fasrc01 eigen/3.3.7-fasrc01 OpenBLAS/0.3.7-fasrc01

make clean
make Procrustes

./Procrustes
