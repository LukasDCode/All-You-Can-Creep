#!/bin/bash
cd ../project
python3 -m src.worm.train a2c -n 7500 -t 50 -w "$1" -a "$2" -g "$3" -e "$4" -ef "$5" -net "$6" -adv "$7" -r "$8" -s 1000

