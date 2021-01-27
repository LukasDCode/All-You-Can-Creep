#!/bin/bash
#cd ../project

#python3 -m src.tuning.main -variant gridsearch -r worm_grid_slurm -n 10000 -p 4 -b 1 -w 0 -adv reinforce -u
#python3 -m src.tuning.main -variant gridsearch -r worm_grid_slurm -n 10000 -p 4 -b 1 -w 20 -adv reinforce -u
#python3 -m src.tuning.main -variant gridsearch -r worm_grid_slurm -n 10000 -p 4 -b 1 -w 40 -adv a2c -u
#python3 -m src.tuning.main -variant gridsearch -r worm_grid_slurm -n 10000 -p 4 -b 1 -w 60 -adv a2c -u
#python3 -m src.tuning.main -variant gridsearch -r worm_grid_slurm -n 10000 -p 4 -b 1 -w 80 -adv td -u
#python3 -m src.tuning.main -variant gridsearch -r worm_grid_slurm -n 10000 -p 4 -b 1 -w 100 -adv td -u

#python3 -m src.tuning.main -variant gridsearch -r worm_grid_slurm -n 10000 -p 4 -b 10 -w 0 -adv reinforce -u
#python3 -m src.tuning.main -variant gridsearch -r worm_grid_slurm -n 10000 -p 4 -b 10 -w 20 -adv reinforce -u
#python3 -m src.tuning.main -variant gridsearch -r worm_grid_slurm -n 10000 -p 4 -b 10 -w 40 -adv a2c -u
#python3 -m src.tuning.main -variant gridsearch -r worm_grid_slurm -n 10000 -p 4 -b 10 -w 60 -adv a2c -u
#python3 -m src.tuning.main -variant gridsearch -r worm_grid_slurm -n 10000 -p 4 -b 10 -w 80 -adv td -u
#python3 -m src.tuning.main -variant gridsearch -r worm_grid_slurm -n 10000 -p 4 -b 10 -w 100 -adv td -u

adv=(reinforce a2c td)
batch_size=(1 10)
index=0

for b in "${!batch_size[@]}"; do
  for a in "${!adv[@]}"; do
    sbatch --partition=All start_job.sh "$index" "${batch_size[b]}" "${adv[a]}"
    index=$((index+20))
  done
done

