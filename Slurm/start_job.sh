#!/bin/bash
cd ../project

python3 -m src.tuning.main -variant gridsearch -r worm_grid_slurm -n 10000 -p 4 -w $1 -b $2 -adv $3 -u
