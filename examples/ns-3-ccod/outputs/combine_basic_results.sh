#!/bin/bash

if [[ $# -ne 1 ]]; then
  echo "Please provide the path to the results of the basic scenario!"
  echo "Usage: $0 <results dir>"
  exit 1
fi

RLIB_DIR="${RLIB_DIR:=$HOME/reinforced-lib/examples/ns-3-ccod}"
RESULTS_DIR="$1"
OUTPUT_FILE="basic_results.csv"

cd "$RESULTS_DIR"

AGENTS=("CSMA" "DDQN" "DDPG")
N_WIFIS=(5 15 30 50)
N_REPS=10

for (( i = 0; i < ${#AGENTS[@]}; i += 1 )); do
  AGENT="${AGENTS[$i]}"

  for (( j = 0; j < ${#N_WIFIS[@]}; j += 1 )); do
    N_WIFI="${N_WIFIS[$j]}"
    FILE="${AGENT}_basic_${N_WIFI}_test.out"

    RUN_DESC=`yes "${AGENT},basic,${N_WIFI}" | head -n ${N_REPS}`
    SEEDS=`cat "$FILE" | grep "RngRun" | awk '{print $NF}'`
    THROUGHPUT=`cat "$FILE" |  grep -E "Sent mbytes.*Throughput" | awk '{print $NF}'`

    paste -d, <(echo "$RUN_DESC") <(echo "$SEEDS") <(echo "") <(echo "$THROUGHPUT") >> "$OUTPUT_FILE"
  done
done

mv "$OUTPUT_FILE" "$RLIB_DIR/outputs"
