#!/usr/bin/env python3
"""TTFT (Time-to-First-Token) Model for Disaggregated LLM Inference.

Computes end-to-end TTFT under different KV cache transfer systems:
  TTFT = T_prefill + T_transfer + T_first_decode

Where:
  T_prefill: prefill computation time (GPU-bound, same for all systems)
  T_transfer: KV cache transfer from prefill to decode GPU
  T_first_decode: first decode step (GPU-bound, same for all systems)

Key insight: T_transfer is the ONLY component that differs between systems.
DCS-CXL reduces T_transfer by eliminating CPU overhead and enabling
multi-switch parallelism.
"""

import argparse
import sys

# GPU compute times (from published benchmarks)
# Source: various vLLM/TensorRT-LLM benchmarks on A100/H100
GPU_COMPUTE = {
    # model: {tokens: (prefill_ms, decode_first_ms)}
    "llama-70b": {
        64:   (8, 15),     # ~8ms prefill, ~15ms first decode
        128:  (15, 15),
        256:  (28, 15),
        512:  (55, 15),
        1024: (110, 15),
        2048: (220, 15),
        4096: (440, 15),
    },
    "llama-13b": {
        64:   (3, 5),
        128:  (5, 5),
        256:  (10, 5),
        512:  (18, 5),
        1024: (35, 5),
        2048: (70, 5),
    },
    "llama-7b": {
        64:   (2, 3),
        128:  (3, 3),
        256:  (6, 3),
        512:  (12, 3),
        1024: (22, 3),
        2048: (45, 3),
    },
}

# KV cache sizes (bytes per token)
KV_PER_TOKEN = {
    "llama-70b": 320 * 1024,    # 320KB (GQA, 80 layers)
    "llama-13b": 800 * 1024,    # 800KB (MHA, 40 layers)
    "llama-7b":  512 * 1024,    # 512KB (MHA, 32 layers)
}

# Transfer systems and their effective bandwidth (GB/s)
# Based on our ESF simulation results with 4KB blocks, chunked ops
TRANSFER_SYSTEMS = {
    "DCS-CXL (1 switch)":       5.05,
    "DCS-CXL (4 switches)":     20.0,   # ~4x scaling
    "DCS-CXL (4 decode GPUs)":  70.0,   # ~17.5 GB/s per GPU
    "Baseline CXL (0ns oh)":    2.56,
    "Baseline CXL (100ns oh)":  0.64,
    "Baseline CXL (500ns oh)":  0.128,
    "RDMA (100Gbps)":           12.0,   # ~12 GB/s practical
    "RDMA (400Gbps)":           45.0,   # ~45 GB/s practical
    "TraCT (real CXL)":         10.0,   # OSDI'25 reported
}


def compute_ttft(model, tokens, system_bw, t_prefill_ms, t_decode_ms):
    """Compute TTFT in ms."""
    kv_bytes = KV_PER_TOKEN[model] * tokens
    t_transfer_ms = (kv_bytes / (system_bw * 1e9)) * 1000  # bytes / (GB/s) * 1000
    ttft = t_prefill_ms + t_transfer_ms + t_decode_ms
    return ttft, t_transfer_ms


def main():
    parser = argparse.ArgumentParser(description="TTFT model for disaggregated LLM inference")
    parser.add_argument("--model", default="llama-70b", choices=KV_PER_TOKEN.keys())
    parser.add_argument("--tokens", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048])
    parser.add_argument("--csv", action="store_true", help="Output CSV format")
    args = parser.parse_args()

    model = args.model
    token_counts = [t for t in args.tokens if t in GPU_COMPUTE.get(model, {})]

    if not token_counts:
        print(f"No compute data for {model} at those token counts", file=sys.stderr)
        sys.exit(1)

    if args.csv:
        print("model,tokens,system,kv_mb,t_prefill_ms,t_transfer_ms,t_decode_ms,ttft_ms,transfer_pct")

    for tokens in token_counts:
        t_prefill, t_decode = GPU_COMPUTE[model][tokens]
        kv_mb = KV_PER_TOKEN[model] * tokens / (1024 * 1024)

        if not args.csv:
            print(f"\n{'='*80}")
            print(f"Model: {model}, Prefill tokens: {tokens}, KV cache: {kv_mb:.1f} MB")
            print(f"T_prefill: {t_prefill} ms, T_first_decode: {t_decode} ms")
            print(f"{'='*80}")
            print(f"{'System':<30} {'T_xfer(ms)':>10} {'TTFT(ms)':>10} {'Xfer%':>6} {'vs DCS':>8}")
            print(f"{'-'*30} {'-'*10} {'-'*10} {'-'*6} {'-'*8}")

        dcs_ttft = None
        for sys_name, bw in TRANSFER_SYSTEMS.items():
            ttft, t_transfer = compute_ttft(model, tokens, bw, t_prefill, t_decode)
            xfer_pct = (t_transfer / ttft) * 100

            if dcs_ttft is None and "DCS-CXL (1 switch)" in sys_name:
                dcs_ttft = ttft

            if args.csv:
                print(f"{model},{tokens},{sys_name},{kv_mb:.1f},{t_prefill},{t_transfer:.2f},{t_decode},{ttft:.2f},{xfer_pct:.1f}")
            else:
                speedup = f"{ttft/dcs_ttft:.2f}x" if dcs_ttft else "—"
                print(f"{sys_name:<30} {t_transfer:>10.2f} {ttft:>10.2f} {xfer_pct:>5.1f}% {speedup:>8}")


if __name__ == "__main__":
    main()
