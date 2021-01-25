#!/bin/bash
cd ../project

python3 -m src.tuning.main -variant gridsearch -r worm_grid_slurm -n 10000 -p 4 -b 1 -w 0 -adv reinforce -u