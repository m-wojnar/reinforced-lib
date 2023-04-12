#!/bin/bash

RLIB_DIR="${RLIB_DIR:=$HOME/reinforced-lib/examples/ns-3-ccod}"
OUTPUT_FILE="all_results.csv"

cd "$RLIB_DIR/outputs"
echo "agent,time,throughput" > "$OUTPUT_FILE"

for file in *.csv; do
    if [[ "$file" != "$OUTPUT_FILE" ]]; then
        cat "$file" >> "$OUTPUT_FILE"
    fi
done

mv "$OUTPUT_FILE" ../tools/plots/