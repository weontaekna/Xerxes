#!/bin/bash
# DCS-CXL Multi-Switch Scalability Sweep
# Produces CSV results showing DCS near-linear scaling vs host-limited baseline.
set -e
cd "$(dirname "$0")"

OUTPUT_CSV="output/scalability-results.csv"
mkdir -p output configs/generated

echo "mode,num_switches,num_ops,transfer_size,switch_delay,host_overhead,orch_id,wall_time_ns,effective_bw_gbps,agg_bw_gbps,avg_latency_ns" > "$OUTPUT_CSV"

NUM_OPS=50
TRANSFER_SIZE=4096
SWITCH_DELAY=25

echo "=== DCS-CXL Multi-Switch Scalability Sweep ==="

for NUM_SW in 1 2 4 8; do
    echo ""
    echo "--- $NUM_SW switches ---"

    # DCS mode
    TAG="dcs-sw${NUM_SW}"
    python3 configs/dcs-cxl-multi-switch.py -m dcs --switches $NUM_SW \
        -n $NUM_OPS -s $TRANSFER_SIZE --switch_delay $SWITCH_DELAY \
        --log /dev/null > configs/generated/${TAG}.toml

    STATS=$(./build/Xerxes configs/generated/${TAG}.toml 2>&1)

    # Parse each orchestrator's stats
    echo "$STATS" | python3 -c "
import sys, re
text = sys.stdin.read()
# Find all orchestrator blocks
blocks = re.findall(r'(Orch-\d+)#\d+ stats:.*?(?=\w+#\d+ stats:|$)', text, re.DOTALL)
for block in blocks:
    match = re.search(r'Orch-(\d+)', block)
    orch_id = match.group(1) if match else '?'
    wt = re.search(r'Total wall time \(ns\): (\d+)', block)
    bw = re.search(r'Effective bandwidth \(GB/s\): ([0-9.e+-]+)', block)
    lat = re.search(r'Average op latency \(ns\): ([0-9.e+-]+)', block)
    wall = wt.group(1) if wt else '0'
    bandwidth = bw.group(1) if bw else '0'
    latency = lat.group(1) if lat else '0'
    print(f'dcs,${NUM_SW},${NUM_OPS},${TRANSFER_SIZE},${SWITCH_DELAY},0,{orch_id},{wall},{bandwidth},,{latency}')
" >> "$OUTPUT_CSV"

    echo "  DCS: $(echo "$STATS" | grep -c 'Total operations completed') orchestrators active"

    # Baseline modes: vary host CPU overhead
    for HOST_OH in 0 100 500 1000; do
        TAG="base-sw${NUM_SW}-oh${HOST_OH}"
        python3 configs/dcs-cxl-multi-switch.py -m baseline --switches $NUM_SW \
            -n $NUM_OPS -s $TRANSFER_SIZE --switch_delay $SWITCH_DELAY \
            --host_overhead $HOST_OH \
            --log /dev/null > configs/generated/${TAG}.toml

        STATS=$(./build/Xerxes configs/generated/${TAG}.toml 2>&1)

        AGG_BW=$(echo "$STATS" | grep -A1 "Aggregate" | grep "Bandwidth" | awk -F: '{print $2}' | tr -d ' ')
        AVG_LAT=$(echo "$STATS" | grep -A2 "Aggregate" | grep "Average latency" | awk -F: '{print $2}' | tr -d ' ')
        echo "baseline,${NUM_SW},${NUM_OPS},${TRANSFER_SIZE},${SWITCH_DELAY},${HOST_OH},,0,${AGG_BW},,${AVG_LAT}" >> "$OUTPUT_CSV"

        echo "  Baseline (oh=${HOST_OH}ns): ${AGG_BW} GB/s"
    done
done

echo ""
echo "Results saved to $OUTPUT_CSV"
