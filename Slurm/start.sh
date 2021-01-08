#!/bin/bash
cd ../project
python3 -m src.worm.train -n 10000  -w "$1" -a "$2" -g "$3" -e "$4" -ef 1 -r worm_results -s 1000

