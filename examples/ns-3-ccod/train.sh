#!/usr/bin/scl enable devtoolset-11 rh-python38 -- /bin/bash -l

NS3_DIR="${NS3_DIR:=$HOME/ns-3.37}"
RLIB_DIR="${RLIB_DIR:=$HOME/reinforced-lib/examples/ns-3-ccod}"

cd "$RLIB_DIR"

AGENT="DQN"
AGENT_TYPE="discrete"
NUM_REPS=15
N_WIFI=5
SCENARIO="basic"
SEED=100
PYTHON_SEED=100

for (( i = 0; i < NUM_REPS - 1; i += 1)); do
  if [[ i -gt 0 ]]; then
    PYTHON_SEED="-1"
  fi

  echo "Run simulation $i!"
  python3 main.py --ns3Path="$NS3_DIR" --agent="$AGENT" --agentType="$AGENT_TYPE" --nWifi="$N_WIFI" --scenario="$SCENARIO" --pythonSeed="$PYTHON_SEED" --rng="$(( SEED + i ))" --runId="$i"
done

python3 main.py --ns3Path="$NS3_DIR" --training="False" --agent="$AGENT" --agentType="$AGENT_TYPE" --nWifi="$N_WIFI" --scenario="$SCENARIO" --pythonSeed="-1" --rng="$(( SEED + NUM_REPS - 1 ))" --runId="$(( NUM_REPS - 1 ))"
