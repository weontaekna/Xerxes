#!/bin/bash
# Run DCS-CXL D2D experiment: compare orchestrated vs host-mediated transfers.
#
# Usage: ./run_dcs_experiment.sh [num_ops] [transfer_size]

set -e
cd "$(dirname "$0")"

NUM_OPS=${1:-100}
TRANSFER_SIZE=${2:-4096}
SWITCH_DELAY=${3:-25}

mkdir -p output configs/generated

echo "=== DCS-CXL D2D Experiment ==="
echo "  num_ops=$NUM_OPS, transfer_size=$TRANSFER_SIZE, switch_delay=${SWITCH_DELAY}ns"
echo ""

# Generate and run DCS-CXL (Orchestrator) config
echo "--- Generating DCS-CXL config ---"
python3 configs/dcs-cxl-d2d.py \
    -m dcs -n $NUM_OPS -s $TRANSFER_SIZE \
    --switch_delay $SWITCH_DELAY \
    --log "output/dcs-n${NUM_OPS}-s${TRANSFER_SIZE}-sw${SWITCH_DELAY}.csv" \
    > configs/generated/dcs-d2d.toml

echo "--- Running DCS-CXL simulation ---"
./build/Xerxes configs/generated/dcs-d2d.toml 2> output/dcs-stats.txt
echo ""
echo "DCS-CXL stats:"
cat output/dcs-stats.txt
echo ""

# Generate and run baseline (host-mediated) config
echo "--- Generating baseline config ---"
python3 configs/dcs-cxl-d2d.py \
    -m baseline -n $NUM_OPS -s $TRANSFER_SIZE \
    --switch_delay $SWITCH_DELAY \
    --log "output/baseline-n${NUM_OPS}-s${TRANSFER_SIZE}-sw${SWITCH_DELAY}.csv" \
    > configs/generated/baseline-d2d.toml

echo "--- Running baseline simulation ---"
./build/Xerxes configs/generated/baseline-d2d.toml 2> output/baseline-stats.txt
echo ""
echo "Baseline stats:"
cat output/baseline-stats.txt
echo ""

echo "=== Results saved to output/ ==="
echo "  DCS-CXL:  output/dcs-n${NUM_OPS}-s${TRANSFER_SIZE}-sw${SWITCH_DELAY}.csv"
echo "  Baseline:  output/baseline-n${NUM_OPS}-s${TRANSFER_SIZE}-sw${SWITCH_DELAY}.csv"
