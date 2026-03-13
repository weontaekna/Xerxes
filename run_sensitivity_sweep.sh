#!/bin/bash
# Sensitivity sweep for DCS-CXL paper Figure 6.
# Sweeps: transfer_size, max_outstanding, switch_delay
# Uses dcs-cxl-d2d.py config with 4KB block size, 50 ops.

set -e
cd "$(dirname "$0")"

ESF="$(pwd)"
CFGDIR="$ESF/configs"
OUTDIR="$ESF/output"
RESULTS="$OUTDIR/sensitivity-results.csv"

mkdir -p "$OUTDIR"
echo "param,param_value,mode,wall_time_ns,effective_bw_gbps,agg_bw_gbps" > "$RESULTS"

# Helper: run one config and extract stats
run_one() {
    local param="$1" val="$2" mode="$3" toml="$4"
    local out
    out=$(timeout 60 "$ESF/build/Xerxes" "$toml" 2>&1) || true

    if [ "$mode" = "dcs" ] || [ "$mode" = "dcs-pipe" ]; then
        local wt=$(echo "$out" | grep "Total wall time" | head -1 | sed 's/.*: //')
        local bw=$(echo "$out" | grep "Effective bandwidth" | head -1 | sed 's/.*: //')
        echo "$param,$val,$mode,${wt:-0},${bw:-0}," >> "$RESULTS"
        echo "  $mode: BW=${bw:-0} GB/s"
    else
        local bw=$(echo "$out" | grep "Bandwidth" | tail -1 | sed 's/.*: //')
        local lat=$(echo "$out" | grep "Average latency" | tail -1 | sed 's/.*: //')
        echo "$param,$val,$mode,,,$bw" >> "$RESULTS"
        echo "  baseline: BW=${bw:-0} GB/s"
    fi
}

NUM_OPS=50
BLOCK=4096

echo "=== Transfer Size Sweep ==="
for XFER in 256 1024 4096 16384 65536; do
    echo "Transfer size: $XFER bytes"
    for MODE in dcs baseline; do
        TOML="/tmp/sens-xfer-${XFER}-${MODE}.toml"
        cd "$CFGDIR"
        python3 dcs-cxl-d2d.py -m $MODE -n $NUM_OPS -s $XFER --switch_delay 25 --max_outstanding 32 --block_size $BLOCK > "$TOML" 2>/dev/null
        cd "$ESF"
        run_one "transfer_size" "$XFER" "$MODE" "$TOML"
    done
done

echo "=== Max Outstanding Sweep ==="
for MO in 8 16 32 64; do
    echo "Max outstanding: $MO"
    for MODE in dcs baseline; do
        TOML="/tmp/sens-mo-${MO}-${MODE}.toml"
        cd "$CFGDIR"
        python3 dcs-cxl-d2d.py -m $MODE -n $NUM_OPS -s 4096 --switch_delay 25 --max_outstanding $MO --block_size $BLOCK > "$TOML" 2>/dev/null
        cd "$ESF"
        run_one "max_outstanding" "$MO" "$MODE" "$TOML"
    done
done

echo "=== Switch Delay Sweep ==="
for SD in 10 25 50 100; do
    echo "Switch delay: $SD ns"
    for MODE in dcs baseline; do
        TOML="/tmp/sens-sd-${SD}-${MODE}.toml"
        cd "$CFGDIR"
        python3 dcs-cxl-d2d.py -m $MODE -n $NUM_OPS -s 4096 --switch_delay $SD --max_outstanding 32 --block_size $BLOCK > "$TOML" 2>/dev/null
        cd "$ESF"
        run_one "switch_delay" "$SD" "$MODE" "$TOML"
    done
done

echo ""
echo "=== Results ==="
cat "$RESULTS"
