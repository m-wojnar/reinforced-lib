#!/usr/bin/scl enable devtoolset-11 rh-python38 -- /bin/bash -l

NS3_DIR="${NS3_DIR:=$HOME/ns-3.37}"
RLIB_DIR="${RLIB_DIR:=$HOME/reinforced-lib/examples/ns-3-ccod}"

cd "$NS3_DIR/build/scratch"

SIM_TIME=60
SCENARIO="convergence"
N_WIFI=55

NUM_REPS=10
SEED=300

for (( i = 1; i <= NUM_REPS; i += 1)); do
  CSV_PATH="$RLIB_DIR/outputs/CSMA_${SCENARIO}_${N_WIFI}_run${i}.csv"
  echo "Testing CSMA/CA ${SCENARIO} ${N_WIFI} simulation [${i}/${NUM_REPS}]"
  ./ns3.37-ccod-sim-optimized --csvPath="$CSV_PATH" --agentType="discrete" --dryRun="true" --envStepTime="0.01" --historyLength="300" --nonZeroStart="true" --nWifi="$N_WIFI" --scenario="$SCENARIO" --simTime="$SIM_TIME" --RngRun="$SEED"

  SEED=$(( SEED + 1 ))
done
