#!/bin/bash
# Sweep LLM KV cache transfer experiments across models, token counts, and host overhead.
# Uses 4KB block size for tractable simulation of large transfers.

set -e
cd "$(dirname "$0")"

OUTPUT_DIR="output"
RESULTS_FILE="$OUTPUT_DIR/llm-sweep-results.csv"
BLOCK_SIZE=4096
MAX_OUTSTANDING=32
MAX_CLOCK=500000000

mkdir -p "$OUTPUT_DIR"

echo "model,prefill_tokens,mode,host_overhead,total_kv_mb,num_ops,total_blocks,wall_time_ns,effective_bw_gbps" > "$RESULTS_FILE"

# Models and token counts to sweep
MODELS=("llama-70b" "llama-13b" "llama-7b")
TOKEN_COUNTS=(64 128 256 512)
HOST_OVERHEADS=(0 100 500 1000)

for model in "${MODELS[@]}"; do
    for tokens in "${TOKEN_COUNTS[@]}"; do
        echo "=== DCS: $model, ${tokens} tokens ==="
        TOML_FILE="/tmp/llm-dcs-${model}-t${tokens}.toml"
        INFO_FILE="/tmp/llm-dcs-${model}-t${tokens}.info"

        cd configs
        python3 dcs-cxl-llm-kv.py -m dcs --model "$model" \
            --prefill_tokens "$tokens" \
            --block_size "$BLOCK_SIZE" \
            --max_outstanding "$MAX_OUTSTANDING" \
            --max_clock "$MAX_CLOCK" \
            > "$TOML_FILE" 2>"$INFO_FILE"
        cd ..

        # Extract info
        total_kv_mb=$(grep "Total KV cache" "$INFO_FILE" | sed 's/.*: //' | sed 's/ MB//')
        num_ops=$(grep "Number of ops" "$INFO_FILE" | sed 's/.*: //')
        total_blocks=$(grep "Total blocks" "$INFO_FILE" | sed 's/.*: //')

        # Run simulation
        SIM_OUTPUT=$(timeout 300 build/Xerxes "$TOML_FILE" 2>&1 || true)

        # Parse orchestrator stats
        wall_time=$(echo "$SIM_OUTPUT" | grep "Total wall time" | sed 's/.*: //')
        eff_bw=$(echo "$SIM_OUTPUT" | grep "Effective bandwidth" | sed 's/.*: //')

        if [ -n "$wall_time" ] && [ -n "$eff_bw" ]; then
            echo "$model,$tokens,dcs,0,$total_kv_mb,$num_ops,$total_blocks,$wall_time,$eff_bw" >> "$RESULTS_FILE"
            echo "  DCS: wall_time=${wall_time}ns, BW=${eff_bw} GB/s"
        else
            echo "  DCS: TIMEOUT or ERROR"
            echo "$model,$tokens,dcs,0,$total_kv_mb,$num_ops,$total_blocks,TIMEOUT,TIMEOUT" >> "$RESULTS_FILE"
        fi

        for overhead in "${HOST_OVERHEADS[@]}"; do
            echo "=== Baseline: $model, ${tokens} tokens, ${overhead}ns overhead ==="
            TOML_FILE="/tmp/llm-baseline-${model}-t${tokens}-oh${overhead}.toml"

            cd configs
            python3 dcs-cxl-llm-kv.py -m baseline --model "$model" \
                --prefill_tokens "$tokens" \
                --block_size "$BLOCK_SIZE" \
                --max_outstanding "$MAX_OUTSTANDING" \
                --host_overhead "$overhead" \
                --max_clock "$MAX_CLOCK" \
                > "$TOML_FILE" 2>/dev/null
            cd ..

            SIM_OUTPUT=$(timeout 300 build/Xerxes "$TOML_FILE" 2>&1 || true)

            # For baseline, parse aggregate stats from requester
            wall_time=""
            eff_bw=""
            agg_line=$(echo "$SIM_OUTPUT" | grep -A5 "Aggregate")
            if [ -n "$agg_line" ]; then
                eff_bw=$(echo "$SIM_OUTPUT" | grep "Bandwidth" | head -1 | sed 's/.*: //')
                # Compute wall time from CSV if available
                csv_file="$OUTPUT_DIR/llm-baseline-${model}-t${tokens}-d1-b${BLOCK_SIZE}.csv"
                if [ -f "$csv_file" ] && [ -s "$csv_file" ]; then
                    wall_time=$(tail -1 "$csv_file" | cut -d',' -f1)
                fi
            fi

            if [ -n "$eff_bw" ]; then
                echo "$model,$tokens,baseline,$overhead,$total_kv_mb,$num_ops,$total_blocks,${wall_time:-N/A},$eff_bw" >> "$RESULTS_FILE"
                echo "  Baseline (${overhead}ns): BW=${eff_bw} GB/s"
            else
                echo "  Baseline (${overhead}ns): TIMEOUT or ERROR"
                echo "$model,$tokens,baseline,$overhead,$total_kv_mb,$num_ops,$total_blocks,TIMEOUT,TIMEOUT" >> "$RESULTS_FILE"
            fi
        done
    done
done

echo ""
echo "=== Results ==="
cat "$RESULTS_FILE"
