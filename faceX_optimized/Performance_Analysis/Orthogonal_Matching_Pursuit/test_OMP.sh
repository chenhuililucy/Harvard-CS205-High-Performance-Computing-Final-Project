#!/usr/bin/env bash
#SBATCH --job-name=test
#SBATCH --output=test_%j.out
#SBATCH --error=test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:20:00

# Load required modules

module load gcc/12.1.0-fasrc01 
module load openmpi/4.1.3-fasrc01
module load centos6/0.0.1-fasrc01
module load pango/1.28.4-fasrc01
module load opencv/3.4.3-fasrc01
module load gcc/12.1.0-fasrc01 OpenBLAS/0.3.7-fasrc01
module load eigen/3.3.7-fasrc01

make clean && make

./facex-train sample_config2.txt


