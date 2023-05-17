#!/bin/bash
#SBATCH --job-name=Covariance_test
#SBATCH --output=Covariance_test_%j.out
#SBATCH --error=Covariance_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --mem-per-cpu=2G

set -e

module load python/3.9.12-fasrc01
module load centos6/0.0.1-fasrc01
module load pango/1.28.4-fasrc01
module load opencv/3.4.3-fasrc01
module load gcc/12.1.0-fasrc01 openmpi/4.1.3-fasrc01

rm -rf ./*.log
make clean && make
max_iter=30000

for ((i=1000; i<=$max_iter; i*=2)); do
    # run the command with the current iteration
    ./main "$i"
done


python3 plot.py
