#!/usr/bin/env bash
#SBATCH --job-name=facex_train
#SBATCH --output=facex_train-%j.out
#SBATCH --error=facex_train-%j.err
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=2G


module load gcc/12.1.0-fasrc01 
module load openmpi/4.1.3-fasrc01
module load centos6/0.0.1-fasrc01
module load pango/1.28.4-fasrc01
module load opencv/3.4.3-fasrc01
module load gcc/12.1.0-fasrc01 OpenBLAS/0.3.7-fasrc01
module load eigen/3.3.7-fasrc01


make clean && make

rm -rf ./logs/*.log
rm -rf ./combined_logs/*.log
rm -rf ./plots/*.png

#!/bin/bash
python generate_labels.py 4000
srun --mem-per-cpu=8G --ntasks 5 ./facex-train model_config.txt