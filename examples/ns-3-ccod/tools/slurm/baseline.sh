#!/bin/bash

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <scenario> <nWifi> <seed>"
  exit 1
fi

NS3_DIR="${NS3_DIR:=$HOME/ns-3.37}"
RLIB_DIR="${RLIB_DIR:=$HOME/reinforced-lib/examples/ns-3-ccod}"

cd "$NS3_DIR/build/scratch"

SCENARIO=$1
N_WIFI=$2
SEED=$3

NUM_REPS=10

for (( i = 1; i <= NUM_REPS; i += 1)); do
  CSV_PATH="$RLIB_DIR/outputs/CSMA_${SCENARIO}_${N_WIFI}_run${i}.csv"
  echo "Testing CSMA/CA ${SCENARIO} ${N_WIFI} simulation [${i}/${NUM_REPS}]"
  ./ns3.37-ccod-sim-optimized --csvPath="$CSV_PATH" --agentType="discrete" --dryRun="true" --envStepTime="0.01" --historyLength="300" --nonZeroStart="true" --nWifi="$N_WIFI" --scenario="$SCENARIO" --simTime="60.0" --RngRun="$SEED"

  SEED=$(( SEED + 1 ))
done
