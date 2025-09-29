#!/bin/bash

num_envs=(2 4 8 16 32 64 128)
seeds=(1 2 3 4 5 6 7 8 9 10)

for n in "${num_envs[@]}"; do
    for s in "${seeds[@]}"; do
        echo "Running with $n environments and seed $s"
        python main.py --num_envs $n --seed $s
    done
done
