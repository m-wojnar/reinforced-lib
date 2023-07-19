#!/bin/bash

python3 examples/ns-3-ra/main.py --mobilityModel="Distance" --agent="UCB" --ns3Path="/Users/wciezobka/ncn/ns-3.37/ns-3-dev" --wifiManagerName="UCB" --velocity="1" --simulationTime="25" --warmupTime="5" --logEvery="1" --lossModel="LogDistance" --seed="42" --csvPath="$HOME/rlib-ns3-integration-test.csv" --mempoolKey="2138"
