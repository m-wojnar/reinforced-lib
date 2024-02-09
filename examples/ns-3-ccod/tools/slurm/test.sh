#!/bin/bash

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <agent> <scenario> <nWifi> <seed>"
  exit 1
fi

NS3_DIR="${NS3_DIR:=$HOME/ns-3.37}"
RLIB_DIR="${RLIB_DIR:=$HOME/reinforced-lib/examples/ns-3-ccod}"

cd "$RLIB_DIR"

AGENT=$1

if [[ $AGENT == "DDPG" ]]; then
  AGENT_TYPE="continuous"
elif [[ $AGENT == "DDQN" ]]; then
  AGENT_TYPE="discrete"
else
  echo "Invalid agent type: $AGENT"
  exit 2
fi

SCENARIO=$2
N_WIFI=$3
SEED=$4

LAST_RUN=14
NUM_REPS=10
MEMPOOL_KEY=1234

CHECKPOINT_LOAD_PATH="$RLIB_DIR/checkpoints/${AGENT}_${SCENARIO}_${N_WIFI}_run_${LAST_RUN}.pkl.lz4"
CHECKPOINT_SAVE_PATH="$RLIB_DIR/checkpoints/${AGENT}_${SCENARIO}_${N_WIFI}_test.pkl.lz4"
LOAD_PATH="$CHECKPOINT_SAVE_PATH"

python3 create_test_checkpoint.py --loadPath="$CHECKPOINT_LOAD_PATH" --savePath="$CHECKPOINT_SAVE_PATH" --agent="$AGENT"

for (( i = 1; i <= NUM_REPS; i += 1)); do
  CSV_PATH="$RLIB_DIR/outputs/${AGENT}_${SCENARIO}_${N_WIFI}_run${i}.csv"
  echo "Testing ${AGENT} ${SCENARIO} ${N_WIFI} simulation [${i}/${NUM_REPS}]"
  python3 main.py --ns3Path="$NS3_DIR" --agent="$AGENT" --agentType="$AGENT_TYPE" --sampleOnly --nWifi="$N_WIFI" --scenario="$SCENARIO" --pythonSeed="$SEED" --seed="$SEED" --loadPath="$LOAD_PATH" --csvPath="$CSV_PATH" --mempoolKey="$MEMPOOL_KEY"

  SEED=$(( SEED + 1 ))
done
