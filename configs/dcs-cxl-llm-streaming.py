#!/usr/bin/env python3
"""DCS-CXL LLM KV Cache Streaming Transfer.

Models "eager" KV transfer: as each layer completes prefill, its KV cache
is immediately transferred to the decode GPU. This overlaps prefill
computation with KV data movement, reducing effective transfer time.

Key insight: With DCS-CXL, the orchestrator can begin transferring layer N's
KV cache while the GPU is computing layer N+1's prefill. The host CPU
is not involved, so there's no contention for GPU DMA engines.

Comparison modes:
  - batch: Wait for all layers, then transfer entire KV cache at once
  - streaming: Transfer each layer's KV cache as soon as prefill completes
  - baseline: Host-mediated batch transfer (traditional approach)

The streaming advantage depends on:
  - T_layer (prefill time per layer) vs T_xfer_layer (transfer time per layer)
  - If T_xfer_layer < T_layer: streaming hides ALL transfer time
  - If T_xfer_layer > T_layer: streaming hides T_layer per layer

For LLaMA-70B: T_layer ≈ 55ms/80 ≈ 0.69ms, T_xfer_layer ≈ 33ms/8ops = 4.1ms
→ Streaming hides ~0.69ms per layer = ~55ms total → effective TTFT: 48ms + 55ms decode prep

For LLaMA-7B: T_layer ≈ 12ms/32 ≈ 0.375ms, T_xfer_layer ≈ 42ms/16ops = 2.6ms
→ Streaming hides ~0.375ms per layer = ~12ms total
"""

import argparse
import sys

# GPU compute times per layer (from published benchmarks)
# T_prefill_per_layer = T_prefill_total / n_layers
MODEL_PARAMS = {
    "llama-7b": {
        "n_layers": 32, "n_kv_heads": 32, "head_dim": 128,
        "prefill_512_ms": 12,   # total prefill time for 512 tokens
        "decode_first_ms": 3,
    },
    "llama-13b": {
        "n_layers": 40, "n_kv_heads": 40, "head_dim": 128,
        "prefill_512_ms": 18,
        "decode_first_ms": 5,
    },
    "llama-70b": {
        "n_layers": 80, "n_kv_heads": 8, "head_dim": 128,
        "prefill_512_ms": 55,
        "decode_first_ms": 15,
    },
}

# Transfer bandwidth from ESF simulations (GB/s)
# With optimal chunking (~20MB ops), 4KB blocks
TRANSFER_BW = {
    "dcs_1sw": 5.05,       # DCS-CXL, 1 switch
    "dcs_4sw": 20.0,       # DCS-CXL, 4 switches
    "baseline_0": 2.56,    # Host-mediated, 0ns overhead
    "baseline_100": 0.64,  # Host-mediated, 100ns overhead
    "baseline_500": 0.128, # Host-mediated, 500ns overhead
}


def compute_streaming_ttft(model_name, tokens, system_bw_key):
    """Compute TTFT under streaming transfer mode."""
    p = MODEL_PARAMS[model_name]
    bw = TRANSFER_BW[system_bw_key]

    n_layers = p["n_layers"]
    kv_per_token_per_layer = 2 * p["n_kv_heads"] * p["head_dim"] * 2
    kv_per_layer = kv_per_token_per_layer * tokens
    total_kv = kv_per_layer * n_layers

    t_prefill_ms = p["prefill_512_ms"] * (tokens / 512)
    t_decode_ms = p["decode_first_ms"]
    t_layer_ms = t_prefill_ms / n_layers

    # Transfer time for one layer's KV cache
    t_xfer_layer_ms = (kv_per_layer / (bw * 1e9)) * 1000

    # Batch mode: transfer after all layers complete
    t_xfer_batch_ms = (total_kv / (bw * 1e9)) * 1000
    ttft_batch = t_prefill_ms + t_xfer_batch_ms + t_decode_ms

    # Streaming mode: overlap transfer with prefill
    # First layer: prefill layer 0, then start transfer
    # Each subsequent layer: max(t_layer, t_xfer_layer) gap
    # After last layer: wait for last layer's transfer + decode
    if t_xfer_layer_ms <= t_layer_ms:
        # Transfer faster than compute: fully hidden after first layer
        # Total extra time: just the last layer's transfer
        t_xfer_streaming_ms = t_xfer_layer_ms  # Only last layer exposed
    else:
        # Transfer slower than compute: partial hiding
        # Each layer: t_xfer_layer - t_layer exposed beyond compute
        exposed_per_layer = t_xfer_layer_ms - t_layer_ms
        # First (n-1) layers overlap with next layer's compute
        # Last layer's transfer is fully exposed
        t_xfer_streaming_ms = (n_layers - 1) * exposed_per_layer + t_xfer_layer_ms

    ttft_streaming = t_prefill_ms + t_xfer_streaming_ms + t_decode_ms

    return {
        "model": model_name,
        "tokens": tokens,
        "system": system_bw_key,
        "bw_gbps": bw,
        "kv_total_mb": total_kv / (1024 * 1024),
        "kv_per_layer_kb": kv_per_layer / 1024,
        "t_prefill_ms": t_prefill_ms,
        "t_layer_ms": t_layer_ms,
        "t_xfer_layer_ms": t_xfer_layer_ms,
        "t_xfer_batch_ms": t_xfer_batch_ms,
        "t_xfer_streaming_ms": t_xfer_streaming_ms,
        "ttft_batch_ms": ttft_batch,
        "ttft_streaming_ms": ttft_streaming,
        "t_decode_ms": t_decode_ms,
        "streaming_speedup": ttft_batch / ttft_streaming if ttft_streaming > 0 else float('inf'),
        "hidden_pct": (1 - t_xfer_streaming_ms / t_xfer_batch_ms) * 100,
    }


def main():
    parser = argparse.ArgumentParser(description="Streaming TTFT model")
    parser.add_argument("--model", default="all", choices=["all"] + list(MODEL_PARAMS.keys()))
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--csv", action="store_true")
    args = parser.parse_args()

    models = list(MODEL_PARAMS.keys()) if args.model == "all" else [args.model]
    systems = list(TRANSFER_BW.keys())

    if args.csv:
        print("model,tokens,system,bw_gbps,kv_total_mb,t_prefill_ms,t_layer_ms,"
              "t_xfer_layer_ms,t_xfer_batch_ms,t_xfer_streaming_ms,"
              "ttft_batch_ms,ttft_streaming_ms,streaming_speedup,hidden_pct")

    for model in models:
        if not args.csv:
            p = MODEL_PARAMS[model]
            kv_per_tok = 2 * p["n_kv_heads"] * p["head_dim"] * 2 * p["n_layers"]
            print(f"\n{'='*90}")
            print(f"Model: {model}, {p['n_layers']} layers, {args.tokens} tokens, "
                  f"KV total: {kv_per_tok * args.tokens / (1024*1024):.1f} MB")
            print(f"T_prefill={p['prefill_512_ms']*(args.tokens/512):.1f}ms, "
                  f"T_per_layer={p['prefill_512_ms']*(args.tokens/512)/p['n_layers']:.3f}ms")
            print(f"{'='*90}")
            print(f"{'System':<20} {'BW':>6} {'Batch':>8} {'Stream':>8} "
                  f"{'Speedup':>8} {'Hidden':>7} {'T_xfer_L':>9}")
            print(f"{'-'*20} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*9}")

        for sys_key in systems:
            r = compute_streaming_ttft(model, args.tokens, sys_key)
            if args.csv:
                print(f"{r['model']},{r['tokens']},{r['system']},{r['bw_gbps']},"
                      f"{r['kv_total_mb']:.1f},{r['t_prefill_ms']:.1f},"
                      f"{r['t_layer_ms']:.3f},{r['t_xfer_layer_ms']:.3f},"
                      f"{r['t_xfer_batch_ms']:.2f},{r['t_xfer_streaming_ms']:.2f},"
                      f"{r['ttft_batch_ms']:.2f},{r['ttft_streaming_ms']:.2f},"
                      f"{r['streaming_speedup']:.2f},{r['hidden_pct']:.1f}")
            else:
                print(f"{sys_key:<20} {r['bw_gbps']:>5.2f} {r['ttft_batch_ms']:>7.1f}ms "
                      f"{r['ttft_streaming_ms']:>7.1f}ms {r['streaming_speedup']:>7.2f}x "
                      f"{r['hidden_pct']:>5.1f}% {r['t_xfer_layer_ms']:>8.3f}ms")


if __name__ == "__main__":
    main()
