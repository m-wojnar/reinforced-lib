#!/usr/bin/scl enable devtoolset-11 rh-python38 -- /bin/bash -l

TOOLS_DIR="${TOOLS_DIR:=$HOME/reinforced-lib/examples/ns-3-ra/tools}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
TASKS_PER_NODE=5

MANAGERS=("EGreedy" "ThompsonSampling" "UCB")
MANAGERS_LEN=${#MANAGERS[@]}

SHIFT=0
SEED_SHIFT=100
BASE_MEMPOOL=3000

LOSS_MODEL="LogDistance"

### Basic scenarios

run_equal_distance() {
  N_REP=10
  N_POINTS=13
  DISTANCE=$1

  for (( i = 0; i < MANAGERS_LEN; i++ )); do
    MANAGER=${MANAGERS[$i]}
    ARRAY_SHIFT=0

    for (( j = 0; j < N_POINTS; j++)); do
      N_WIFI=$(( j == 0 ? 1 : 4 * j))
      SIM_TIME=$(( 10 * N_WIFI + 50 ))

      START=$ARRAY_SHIFT
      END=$(( ARRAY_SHIFT + N_REP - 1 ))

      MEMPOOL_SHIFT=$(( SHIFT + BASE_MEMPOOL ))
      ARRAY_SHIFT=$(( ARRAY_SHIFT + N_REP ))

      sbatch --ntasks-per-node="$TASKS_PER_NODE" -p gpu --array=$START-$END "$TOOLS_DIR/slurm/distance/ml.sh" "$SEED_SHIFT" "$MANAGER" "$MANAGER" "$N_WIFI" "$DISTANCE" "$SIM_TIME" "$LOSS_MODEL" "$MEMPOOL_SHIFT"
    done

    SHIFT=$(( SHIFT + N_POINTS * N_REP ))
  done
}

run_rwpm() {
  N_REP=40
  N_WIFI=10
  SIM_TIME=1000
  NODE_SPEED=$1

  START=0
  END=$(( N_REP - 1 ))

  for (( i = 0; i < MANAGERS_LEN; i++ )); do
    MANAGER=${MANAGERS[$i]}

    MEMPOOL_SHIFT=$(( SHIFT + BASE_MEMPOOL ))

    sbatch --ntasks-per-node="$TASKS_PER_NODE" -p gpu --array=$START-$END "$TOOLS_DIR/slurm/rwpm/ml.sh" "$SEED_SHIFT" "$MANAGER" "$MANAGER" "$N_WIFI" "$SIM_TIME" "$NODE_SPEED" "$LOSS_MODEL" "$MEMPOOL_SHIFT"

    SHIFT=$(( SHIFT + N_REP ))
  done

}

run_moving() {
  N_REP=20
  VELOCITY=$1
  SIM_TIME=$2
  INTERVAL=$3

  START=0
  END=$(( N_REP - 1 ))

  for (( i = 0; i < MANAGERS_LEN; i++ )); do
    MANAGER=${MANAGERS[$i]}

    MEMPOOL_SHIFT=$(( SHIFT + BASE_MEMPOOL ))

    sbatch --ntasks-per-node="$TASKS_PER_NODE" -p gpu --array=$START-$END "$TOOLS_DIR/slurm/moving/ml.sh" "$SEED_SHIFT" "$MANAGER" "$MANAGER" "$VELOCITY" "$SIM_TIME" "$INTERVAL" "$LOSS_MODEL" "$MEMPOOL_SHIFT"

    SHIFT=$(( SHIFT + N_REP ))
  done
}

run_power() {
  N_REP=20

  DELTA=$1
  INTERVAL=$2
  VELOCITY=$3
  START_POS=$4

  START=0
  END=$(( N_REP - 1 ))

  for (( i = 0; i < MANAGERS_LEN; i++ )); do
    MANAGER=${MANAGERS[$i]}
    MANAGER_NAME=${MANAGERS_NAMES[$i]}

    MEMPOOL_SHIFT=$(( SHIFT + BASE_MEMPOOL ))

    sbatch --ntasks-per-node="$TASKS_PER_NODE" -p gpu --array=$START-$END "$TOOLS_DIR/slurm/power/ml.sh" "$SEED_SHIFT" "$MANAGER" "$MANAGER" "$DELTA" "$INTERVAL" "$VELOCITY" "$START_POS" "$LOSS_MODEL" "$MEMPOOL_SHIFT"

    SHIFT=$(( SHIFT + N_REP ))
  done
}

### Run section

echo -e "\nQueue equal distance (d=0) scenario"
run_equal_distance 0

echo -e "\nQueue equal distance (d=20) scenario"
run_equal_distance 20

echo -e "\nQueue moving station (v=1) scenario"
run_moving 1 60 1

echo -e "\nQueue moving station (v=2) scenario"
run_moving 2 30 "0.5"

echo -e "\nQueue static stations scenario"
run_rwpm 0

echo -e "\nQueue mobile stations scenario"
run_rwpm "1.4"

echo -e "\nQueue power (delta=5, interval=4, v=0, start=5) scenario"
run_power 5 4 0 5

echo -e "\nQueue power (delta=15, interval=8, v=0, start=5) scenario"
run_power 15 8 0 5
