#!/bin/bash

evo_algs=("CMA_ES" "SimulatedAnnealing" "SimpleES")
seeds=(1 2 3 4 5 6 7 8 9 10)

for alg in "${evo_algs[@]}"; do
    for s in "${seeds[@]}"; do
        echo "Running with algorithm $alg and seed $s"
        python main.py --evo_alg $alg --seed $s
    done
done
