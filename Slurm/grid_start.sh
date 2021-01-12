#!/bin/bash
cd ../project
python3 -m src.tuning.main -variant gridsearch -n 1000 -t 50 -w "$1"  -adv "$2" -net "$3" -p "$4" -r "$5" -s 1000

