#!/bin/bash

#SBATCH --job-name=facex_train
#SBATCH --output=facex_train.out
#SBATCH --error=facex_train.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=8G

# load all relevant modules
module purge
module load gcc/12.1.0-fasrc01 openmpi/4.1.3-fasrc01
module load centos6/0.0.1-fasrc01
module load pango/1.28.4-fasrc01
module load opencv/3.4.3-fasrc01
module load eigen/3.3.7-fasrc01
module load OpenBLAS/0.3.7-fasrc01

# Build the project
make clean
make

module load python/3.9.12-fasrc01

for ((j=250; j<=4000; j*=2)); do
    python generate_labels.py $j

    # loop through the iterations
    for (( i=1; i<=10;i++ )); do
    # run the command with the current iteration
	srun --ntasks $i -c1 --mem-per-cpu=8G  ./facex-train config.txt
    done
done

python plot.py
