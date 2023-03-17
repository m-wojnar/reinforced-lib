#!/usr/bin/scl enable devtoolset-11 rh-python38 -- /bin/bash -l

RLIB_DIR="${RLIB_DIR:=$HOME/reinforced-lib/examples/ns-3-ra}"
TOOLS_DIR="${TOOLS_DIR:=$HOME/reinforced-lib/examples/ns-3-ra/tools}"
NS3_DIR="${NS3_DIR:=$HOME/ns-3.38}"

cd "$RLIB_DIR"

SEED_SHIFT=$1
MANAGER=$2
MANAGER_NAME=$3
DELTA=$4
INTERVAL=$5
VELOCITY=$6
START_POS=$7
LOSS_MODEL=$8
MEMPOOL_SHIFT=$9

SEED=$(( SEED_SHIFT + SLURM_ARRAY_TASK_ID ))
MEMPOOL_KEY=$(( MEMPOOL_SHIFT + SLURM_ARRAY_TASK_ID ))

CSV_PATH="$TOOLS_DIR/outputs/power_${MANAGER_NAME}_v${VELOCITY}_d${DELTA}_i${INTERVAL}_s${SEED}.csv"
WARMUP_TIME=5

python3 main.py --mobilityModel="Distance" --agent="$MANAGER" --ns3Path="$NS3_DIR" --wifiManagerName="$MANAGER_NAME" --deltaPower="$DELTA" --intervalPower="$INTERVAL" --velocity="$VELOCITY" --initialPosition="$START_POS" --simulationTime="55" --warmupTime="$WARMUP_TIME" --logEvery="0.5" --lossModel="$LOSS_MODEL" --seed="$SEED" --csvPath="$CSV_PATH" --mempoolKey="$MEMPOOL_KEY"
