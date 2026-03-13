#!/bin/bash
# Run all DCS-CXL experiments for paper figures
# Usage: bash run_all_experiments.sh [section]
# Sections: storage, llm-multigpu, llm-streaming, scalability-extended, all

set -e
cd "$(dirname "$0")"

BINARY=build/Xerxes
PYTHON=python3
OUTDIR=output
GENDIR=configs/generated
mkdir -p "$OUTDIR" "$GENDIR"

if [ ! -f "$BINARY" ]; then
    echo "ERROR: Binary not found. Run: cd build && cmake .. && make -j\$(nproc)"
    exit 1
fi

section="${1:-all}"

run_sim() {
    local tag="$1"
    local toml="$2"
    echo "  Running: $tag"
    $BINARY "$toml" 2>&1 | grep -E "(Bandwidth|latency|completed|Duration|Effective)" || true
}

# ============================================================
# Section 1: Storage Virtualization Workload (NEW)
# ============================================================
if [[ "$section" == "storage" || "$section" == "all" ]]; then
    echo "=== Storage Virtualization Experiments ==="

    for io_size in 4096 65536 131072; do
        for mode in dcs baseline; do
            for overhead in 0 100 500; do
                tag="storage-${mode}-io${io_size}-oh${overhead}"
                echo "[$tag]"
                oh_arg=""
                if [ "$mode" == "baseline" ]; then
                    oh_arg="--host_overhead $overhead"
                elif [ "$overhead" -gt 0 ]; then
                    continue  # DCS doesn't use host overhead
                fi
                $PYTHON configs/dcs-cxl-storage-virt.py -m "$mode" \
                    --io_size "$io_size" --num_ios 1000 --queue_depth 32 \
                    --num_pairs 2 $oh_arg \
                    --log "$OUTDIR/${tag}.csv" > "$GENDIR/${tag}.toml" 2>/dev/null
                run_sim "$tag" "$GENDIR/${tag}.toml"
            done
        done
    done

    # Queue depth sensitivity
    for qd in 1 4 8 16 32 64 128; do
        for mode in dcs baseline; do
            tag="storage-${mode}-qd${qd}"
            $PYTHON configs/dcs-cxl-storage-virt.py -m "$mode" \
                --io_size 4096 --num_ios 500 --queue_depth "$qd" \
                --num_pairs 2 \
                --log "$OUTDIR/${tag}.csv" > "$GENDIR/${tag}.toml" 2>/dev/null
            run_sim "$tag" "$GENDIR/${tag}.toml"
        done
    done

    # Multi-pair scalability
    for pairs in 1 2 4; do
        for mode in dcs baseline; do
            for overhead in 0 500; do
                oh_arg=""
                if [ "$mode" == "baseline" ]; then
                    oh_arg="--host_overhead $overhead"
                elif [ "$overhead" -gt 0 ]; then
                    continue
                fi
                tag="storage-${mode}-pairs${pairs}-oh${overhead}"
                $PYTHON configs/dcs-cxl-storage-virt.py -m "$mode" \
                    --io_size 4096 --num_ios 500 --queue_depth 32 \
                    --num_pairs "$pairs" $oh_arg \
                    --log "$OUTDIR/${tag}.csv" > "$GENDIR/${tag}.toml" 2>/dev/null
                run_sim "$tag" "$GENDIR/${tag}.toml"
            done
        done
    done

    echo "=== Storage Virtualization Done ==="
fi

# ============================================================
# Section 2: LLM Multi-GPU (remaining models)
# ============================================================
if [[ "$section" == "llm-multigpu" || "$section" == "all" ]]; then
    echo "=== LLM Multi-GPU Experiments ==="

    for model in llama-7b llama-13b llama-70b; do
        for gpus in 1 2 4; do
            case $model in
                llama-7b)  layers_per=8 ;;
                llama-13b) layers_per=10 ;;
                llama-70b) layers_per=10 ;;
            esac
            for mode in dcs baseline; do
                tag="llm-${mode}-${model}-d${gpus}"
                echo "[$tag]"
                $PYTHON configs/dcs-cxl-llm-kv.py -m "$mode" \
                    --model "$model" --prefill_tokens 512 \
                    --num_decode_gpus "$gpus" \
                    --num_layers_per_transfer "$layers_per" \
                    --block_size 4096 \
                    --log "$OUTDIR/${tag}.csv" > "$GENDIR/${tag}.toml" 2>/dev/null
                run_sim "$tag" "$GENDIR/${tag}.toml"
            done
        done
    done

    echo "=== LLM Multi-GPU Done ==="
fi

# ============================================================
# Section 3: Extended Scalability (more switch counts + overhead)
# ============================================================
if [[ "$section" == "scalability-extended" || "$section" == "all" ]]; then
    echo "=== Extended Scalability Experiments ==="

    for sw in 1 2 3 4 6 8; do
        for mode in dcs baseline; do
            for overhead in 0 100 250 500 1000; do
                oh_arg=""
                if [ "$mode" == "baseline" ]; then
                    oh_arg="--host_overhead $overhead"
                elif [ "$overhead" -gt 0 ]; then
                    continue
                fi
                tag="scale-${mode}-sw${sw}-oh${overhead}"
                echo "[$tag]"
                $PYTHON configs/dcs-cxl-multi-switch.py -m "$mode" \
                    --switches "$sw" -n 100 -s 4096 \
                    --max_outstanding 32 $oh_arg \
                    --log "$OUTDIR/${tag}.csv" > "$GENDIR/${tag}.toml" 2>/dev/null
                run_sim "$tag" "$GENDIR/${tag}.toml"
            done
        done
    done

    echo "=== Extended Scalability Done ==="
fi

echo ""
echo "All requested experiments complete. Results in $OUTDIR/"
