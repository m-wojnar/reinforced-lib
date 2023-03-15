#!/usr/bin/scl enable devtoolset-11 rh-python38 -- /bin/bash -l

NS3_DIR="${NS3_DIR:=$HOME/ns-3.37}"
TOOLS_DIR="${TOOLS_DIR:=$HOME/reinforced-lib/examples/ns-3-ra/tools}"

cd "$NS3_DIR"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NS3_DIR/build/lib
cd build/scratch

SEED_SHIFT=$1
MANAGER=$2
MANAGER_NAME=$3
DELTA=$4
INTERVAL=$5
VELOCITY=$6
START_POS=$7
LOSS_MODEL=$8

SEED=$(( SEED_SHIFT + SLURM_ARRAY_TASK_ID ))

CSV_PATH="$TOOLS_DIR/outputs/power_${MANAGER_NAME}_v${VELOCITY}_d${DELTA}_i${INTERVAL}_s${SEED}.csv"
WARMUP_TIME=5

./ns3.37-ra-sim-optimized --wifiManager="$MANAGER" --wifiManagerName="$MANAGER_NAME" --deltaPower="$DELTA" --intervalPower="$INTERVAL" --velocity="$VELOCITY" --initialPosition="$START_POS" --simulationTime="56" --warmupTime="$WARMUP_TIME" --logEvery="0.5" --lossModel="$LOSS_MODEL" --RngRun="$SEED" --csvPath="$CSV_PATH"
