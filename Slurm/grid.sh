#!/bin/bash
alphas=(0.001)
gammas=(0.999 0.9999 0.99999)
entropy=(0.9 0.5 0.1 0.01 0.001)
entropy_fall=(1.0)

len_g="${#gammas[@]}"
len_e="${#entropy[@]}"

index=$1
net=$2
adv=$3
dir=$4

for a in "${!alphas[@]}"; do
  for g in "${!gammas[@]}"; do
    for e in "${!entropy[@]}"; do
      for f in "${!entropy_fall[@]}"; do
        echo Starting job with "${alphas[a]}" "${gammas[g]}" "${entropy[e]}"
        sbatch --partition=All start.sh $index "${alphas[$a]}" "${gammas[$g]}" "${entropy[$e]}" "${entropy_fall[$f]}" "$net" "$adv" "$dir"  
        index=$((index+1))
      done
    done
  done
done


