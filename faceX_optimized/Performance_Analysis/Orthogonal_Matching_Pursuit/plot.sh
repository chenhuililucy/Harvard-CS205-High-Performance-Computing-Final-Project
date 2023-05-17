#!/bin/bash
make clean && make

rm *.log

# set the number of tasks to 5 if no argument is provided
if [ -z "$1" ]; then
    ntasks=5
else
    ntasks=$1
fi

# calculate the maximum number of iterations
max_iter=$((1<<$ntasks))

# loop through the iterations
for (( i=1; i<=max_iter; i*=2 )); do
    # run the command with the current iteration
    srun --ntasks $i ./facex-train sample_config2.txt
done

python3 plot_strong_scaling.py