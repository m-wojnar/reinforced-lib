#!/bin/bash

NS3_DIR="${NS3_DIR:=$HOME/ns-3.37}"

AGENT="DQN"
AGENT_TYPE="discrete"
NUM_REPS=15
N_WIFI=55
SCENARIO="convergence"
SEED=42

for (( i = 0; i < NUM_REPS - 1; i += 1)); do
    echo "Run simulation $i!"

    if [[ $i -eq 0 ]]; then
        python3 main.py --ns3Path="$NS3_DIR" --agent="$AGENT" --pythonSeed="$SEED" --rng="$SEED" --runId="$i" --agentType="$AGENT_TYPE" --nWifi="$N_WIFI" --scenario="$SCENARIO"
    else
        python3 main.py --ns3Path="$NS3_DIR" --agent="$AGENT" --rng="$(( SEED + i ))" --runId="$i" --agentType="$AGENT_TYPE" --nWifi="$N_WIFI" --scenario="$SCENARIO"
    fi
done
