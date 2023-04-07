#!/usr/bin/scl enable devtoolset-11 rh-python38 -- /bin/bash -l

NS3_DIR="${NS3_DIR:=$HOME/ns-3.37}"
RLIB_DIR="${RLIB_DIR:=$HOME/reinforced-lib/examples/ns-3-ccod}"

cd "$RLIB_DIR"

AGENT="DQN"
AGENT_TYPE="discrete"

SCENARIO="basic"
N_WIFI=5

LAST_RUN=14
NUM_REPS=10
SEED=200

LOAD_PATH="$RLIB_DIR/checkpoints/${AGENT}_${SCENARIO}_${N_WIFI}_test.pkl.lz4"

for (( i = 1; i <= NUM_REPS; i += 1)); do
  echo "Testing ${AGENT} ${SCENARIO} ${N_WIFI} simulation [${i}/${NUM_REPS}]"
  python3 main.py --ns3Path="$NS3_DIR" --agent="$AGENT" --agentType="$AGENT_TYPE" --sampleOnly --nWifi="$N_WIFI" --scenario="$SCENARIO" --pythonSeed="$SEED" --rng="$SEED" --loadPath="$LOAD_PATH"

  SEED=$(( SEED + 1 ))
done
