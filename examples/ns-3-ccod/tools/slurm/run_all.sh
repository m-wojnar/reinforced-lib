#!/bin/bash

TOOLS_DIR="${TOOLS_DIR:=$HOME/reinforced-lib/examples/ns-3-ccod/tools}"
cd "$TOOLS_DIR/slurm"

for agent in "CSMA" "DDPG" "DDQN"; do
  for nWifi in 5 15 30 50; do
    if [[ $agent == "CSMA" ]]; then
      ./baseline.sh "basic" $nWifi 300
    else
      ./train.sh $agent "basic" $nWifi 200
      ./test.sh $agent "basic" $nWifi 300
    fi
  done

  if [[ $agent == "CSMA" ]]; then
    ./baseline.sh "convergence" 55 400
  else
    ./train.sh $agent "convergence" 55 200
    ./test.sh $agent "convergence" 55 400
  fi
done
