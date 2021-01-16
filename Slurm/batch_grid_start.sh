#!/bin/bash

adv=(3step, a2c, td, reinforce)
network=(split multihead)
index=0
parallel=0
dir=ball_test_denny
errorbuffer=15

for a in "${!adv[@]}"; do
    for g in "${!network[@]}"; do
        echo Starting job with "${adv[a]}" "${network[g]}" 
        sbatch --partition=All grid_start.sh $index "${adv[$a]}" "${network[$g]}" "$parallel" "$dir" 
        index=$((index + parallel + errorbuffer))
  done
done

