#!/bin/bash
conda activate autonome
cd ../project
python3 -m src.worm.train -n 10
