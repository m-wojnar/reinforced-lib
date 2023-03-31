#!/usr/bin/scl enable devtoolset-11 rh-python38 -- /bin/bash -l

TOOLS_DIR="${TOOLS_DIR:=$HOME/reinforced-lib/examples/ns-3-ra/tools}"

MANAGERS=("ns3::IdealWifiManager" "ns3::MinstrelHtWifiManager")
MANAGERS_NAMES=("Ideal" "Minstrel")
MANAGERS_LEN=${#MANAGERS[@]}

LOSS_MODEL="LogDistance"
SEED_SHIFT=100

### Basic scenarios

run_equal_distance() {
  N_POINTS=9
  DISTANCE=$1

  for (( i = 0; i < MANAGERS_LEN; i++ )); do
    MANAGER=${MANAGERS[$i]}
    MANAGER_NAME=${MANAGERS_NAMES[$i]}
    ARRAY_SHIFT=0

    for (( j = 0; j < N_POINTS; j++)); do
      N_WIFI=$(( j == 0 ? 1 : 2 * j))
      SIM_TIME=$(( 10 * N_WIFI + 50 ))
      N_REP=$(( N_WIFI <= 4 ? 6 : (N_WIFI / 2) * (N_WIFI / 2) ))

      START=$ARRAY_SHIFT
      END=$(( ARRAY_SHIFT + N_REP - 1 ))

      ARRAY_SHIFT=$(( ARRAY_SHIFT + N_REP ))

      sbatch -p gpu --array=$START-$END "$TOOLS_DIR/slurm/distance/classic.sh" "$SEED_SHIFT" "$MANAGER" "$MANAGER_NAME" "$N_WIFI" "$DISTANCE" "$SIM_TIME" "$LOSS_MODEL"
    done
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
    MANAGER_NAME=${MANAGERS_NAMES[$i]}

    sbatch -p gpu --array=$START-$END "$TOOLS_DIR/slurm/rwpm/classic.sh" "$SEED_SHIFT" "$MANAGER" "$MANAGER_NAME" "$N_WIFI" "$SIM_TIME" "$NODE_SPEED" "$LOSS_MODEL"
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
    MANAGER_NAME=${MANAGERS_NAMES[$i]}

    sbatch -p gpu --array=$START-$END "$TOOLS_DIR/slurm/moving/classic.sh" "$SEED_SHIFT" "$MANAGER" "$MANAGER_NAME" "$VELOCITY" "$SIM_TIME" "$INTERVAL" "$LOSS_MODEL"
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

    sbatch -p gpu --array=$START-$END "$TOOLS_DIR/slurm/power/classic.sh" "$SEED_SHIFT" "$MANAGER" "$MANAGER_NAME" "$DELTA" "$INTERVAL" "$VELOCITY" "$START_POS" "$LOSS_MODEL"
  done
}

### Run section

echo -e "\nQueue equal distance (d=1) scenario"
run_equal_distance 1

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

echo -e "\nQueue power (delta=5, interval=4, v=0, start=8) scenario"
run_power 5 4 0 8

echo -e "\nQueue power (delta=15, interval=8, v=0, start=8) scenario"
run_power 15 8 0 8
