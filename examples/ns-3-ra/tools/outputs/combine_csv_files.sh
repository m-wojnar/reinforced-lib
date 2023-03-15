#!/bin/bash

TOOLS_DIR="${TOOLS_DIR:=$HOME/reinforced-lib/examples/ns-3-ra/tools}"
OUTPUT_FILE="all_results.csv"

cd "$TOOLS_DIR/outputs"
echo "wifiManager,lossModel,mobilityModel,channelWidth,minGI,velocity,delta,interval,seed,nWifi,nWifiReal,position,time,throughput" > "$OUTPUT_FILE"

for file in *.csv; do
    if [[ "$file" != "$OUTPUT_FILE" ]]; then
        cat "$file" >> "$OUTPUT_FILE"
    fi
done
