#!/bin/bash
conda activate autonome
cd ../project
python3 -m src.worm.train -n 5000 -w $1 -a $2 -g $3 -e $4 -r results -s 100  

