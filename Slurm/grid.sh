#!/bin/bash
alphas=(0.001)
gammas=(0.999 0.9999 0.99999 1)
entropy=(1e-7 1e-6 1e-5 1e-4)

len_g="${#gammas[@]}"
len_e="${#entropy[@]}"

index=0
for a in "${!alphas[@]}"; do
  for g in "${!gammas[@]}"; do
    for e in "${!entropy[@]}"; do
      echo Starting job with "${alphas[a]}" "${gammas[g]}" "${entropy[e]}"
      /bin/bash -c "sbatch --partition=All start.sh $index ${alphas[$a]} ${gammas[$g]} ${entropy[$e]}"
      index=$((index+1))
    done
  done
done


