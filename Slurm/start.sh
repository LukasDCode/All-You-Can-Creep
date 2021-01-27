#!/bin/bash
cd ../project
#python3 -m src.worm.train a2c -n 7500 -t 50 -w "$1" -a "$2" -g "$3" -e "$4" -ef "$5" -net "$6" -adv "$7" -r "$8" -s 1000
python3 -m src.tuning.main -n 10000 -r worm_grid_slurm -p 4 -w 100 -b 1 -adv td -variant gridsearch -u
