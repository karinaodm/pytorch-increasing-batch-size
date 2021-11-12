#!/bin/bash

python3 train.py --n-exp=0 --pretrained --batch-size=102 --use-cudnn-benchmark

python3 train.py --n-exp=1 --pretrained --batch-size=111

python3 train.py --n-exp=2 --pretrained --batch-size=214 --use-amp
python3 train.py --n-exp=3 --pretrained --batch-size=199 --use-apex-amp=O1
python3 train.py --n-exp=4 --pretrained --batch-size=212 --use-apex-amp=O2
python3 train.py --n-exp=5 --pretrained --batch-size=218 --use-apex-amp=O3

python3 train.py --n-exp=6 --pretrained --batch-size=260 --use-amp --use-checkpointing=1
python3 train.py --n-exp=7 --pretrained --batch-size=265 --use-amp --use-checkpointing=2
python3 train.py --n-exp=8 --pretrained --batch-size=285 --use-amp --use-checkpointing=3

python3 train.py --n-exp=9 --pretrained --batch-size=480 --use-checkpointing=optimal --use-amp

python3 train.py --n-exp=10 --pretrained --batch-size=480 --use-checkpointing=optimal --use-amp --accumulation-steps=2