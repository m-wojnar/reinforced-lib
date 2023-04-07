#!/usr/bin/scl enable devtoolset-11 rh-python38 -- /bin/bash -l

NS3_DIR="${NS3_DIR:=$HOME/ns-3.37}"
RLIB_DIR="${RLIB_DIR:=$HOME/reinforced-lib/examples/ns-3-ccod}"
TOOLS_DIR="${TOOLS_DIR:=$HOME/reinforced-lib/examples/ns-3-ccod/tools}"

cd "$RLIB_DIR"

AGENT="DQN"
AGENT_TYPE="discrete"

SCENARIO="basic"
N_WIFI=5

LAST_RUN=14
NUM_REPS=10
SEED=200

CHECKPOINT_LOAD_PATH="$RLIB_DIR/checkpoints/${AGENT}_${SCENARIO}_${N_WIFI}_run_${LAST_RUN}.pkl.lz4"
CHECKPOINT_SAVE_PATH="$RLIB_DIR/checkpoints/${AGENT}_${SCENARIO}_${N_WIFI}_test.pkl.lz4"
LOAD_PATH="$CHECKPOINT_SAVE_PATH"

python3 "$TOOLS_DIR/create_test_checkpoint.py" --loadPath="$CHECKPOINT_LOAD_PATH" --savePath="$CHECKPOINT_SAVE_PATH" --agent="$AGENT"

for (( i = 1; i <= NUM_REPS; i += 1)); do
  echo "Testing ${AGENT} ${SCENARIO} ${N_WIFI} simulation [${i}/${NUM_REPS}]"
  python3 main.py --ns3Path="$NS3_DIR" --agent="$AGENT" --agentType="$AGENT_TYPE" --sampleOnly --nWifi="$N_WIFI" --scenario="$SCENARIO" --pythonSeed="$SEED" --rng="$SEED" --loadPath="$LOAD_PATH"

  SEED=$(( SEED + 1 ))
done
