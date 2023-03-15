#!/usr/bin/scl enable devtoolset-11 rh-python38 -- /bin/bash -l

NS3_DIR="${NS3_DIR:=$HOME/ns-3.37}"
TOOLS_DIR="${TOOLS_DIR:=$HOME/reinforced-lib/examples/ns-3-ra/tools}"

cd "$NS3_DIR"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NS3_DIR/build/lib
cd build/scratch

SEED_SHIFT=$1
MANAGER=$2
MANAGER_NAME=$3
VELOCITY=$4
SIM_TIME=$5
INTERVAL=$6
LOSS_MODEL=$7

SEED=$(( SEED_SHIFT + SLURM_ARRAY_TASK_ID ))

CSV_PATH="$TOOLS_DIR/outputs/moving_${MANAGER_NAME}_v${VELOCITY}_s${SEED}.csv"
WARMUP_TIME=5

./ns3.37-ra-sim-optimized --wifiManager="$MANAGER" --wifiManagerName="$MANAGER_NAME" --velocity="$VELOCITY" --simulationTime="$SIM_TIME" --warmupTime="$WARMUP_TIME" --logEvery="$INTERVAL" --lossModel="$LOSS_MODEL" --RngRun="$SEED" --csvPath="$CSV_PATH"
