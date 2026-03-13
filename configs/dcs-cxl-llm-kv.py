#!/usr/bin/env python3
"""DCS-CXL LLM KV Cache Transfer Workload.

Models disaggregated LLM inference where a prefill GPU produces KV cache
and sends it to a decode GPU via CXL fabric, orchestrated by DCS-CXL Engine.

KV Cache Sizes (FP16, per token per layer):
  2 × n_kv_heads × head_dim × 2 bytes
  LLaMA-3-70B (GQA): 2 × 8 × 128 × 2 = 4,096 bytes/token/layer
  LLaMA-2-70B (MHA): 2 × 64 × 128 × 2 = 32,768 bytes/token/layer
  LLaMA-2-13B: 2 × 40 × 128 × 2 = 20,480 bytes/token/layer
  LLaMA-2-7B:  2 × 32 × 128 × 2 = 16,384 bytes/token/layer

Per-request KV cache (all layers):
  LLaMA-3-70B (80 layers, GQA): 4KB × 80 = 320KB/token → 512 tokens = 160MB
  LLaMA-2-70B (80 layers, MHA): 32KB × 80 = 2.5MB/token → 512 tokens = 1.25GB
  LLaMA-2-13B (40 layers): 20KB × 40 = 800KB/token → 512 tokens = 400MB
  LLaMA-2-7B  (32 layers): 16KB × 32 = 512KB/token → 512 tokens = 256MB

Transfer pattern: Prefill GPU finishes all layers, then entire KV cache
is transferred to decode GPU(s). This is multiple large D2D transfers.

Topology:
  [Host] --+
           |
       [Switch-0] ---- [Switch-1]
       /   |   \\         /   |
  [PrefillGPU] [Orch]  [DecodeGPU-0] [DecodeGPU-1...]
"""

from mkcfg import utils, devices
import argparse

parser = argparse.ArgumentParser(description="DCS-CXL LLM KV cache transfer")
parser.add_argument("-m", "--mode", type=str, default="dcs",
                    choices=["dcs", "baseline"],
                    help="dcs = orchestrated D2D, baseline = host-mediated")
parser.add_argument("--model", type=str, default="llama-70b",
                    choices=["llama-7b", "llama-13b", "llama-70b"],
                    help="LLM model (determines KV cache size)")
parser.add_argument("--prefill_tokens", type=int, default=512,
                    help="number of tokens in prefill (determines KV cache size)")
parser.add_argument("--num_decode_gpus", type=int, default=1,
                    help="number of decode GPUs (KV cache split across them)")
parser.add_argument("--num_layers_per_transfer", type=int, default=0,
                    help="layers per D2D transfer op (0 = all layers as one op)")
parser.add_argument("--block_size", type=int, default=4096,
                    help="block size for transfers (64=cache line, 4096=page)")
parser.add_argument("--switch_delay", type=int, default=25)
parser.add_argument("--max_outstanding", type=int, default=32)
parser.add_argument("--host_overhead", type=int, default=0,
                    help="host CPU overhead per request in ns (baseline only)")
parser.add_argument("--log", type=str, default="")
parser.add_argument("--max_clock", type=int, default=500000000)
args = parser.parse_args()

# Model parameters
MODEL_PARAMS = {
    "llama-7b":   {"n_layers": 32, "n_kv_heads": 32, "head_dim": 128},  # MHA
    "llama-13b":  {"n_layers": 40, "n_kv_heads": 40, "head_dim": 128},  # MHA
    "llama-70b":  {"n_layers": 80, "n_kv_heads": 8,  "head_dim": 128},  # GQA (LLaMA-3)
}

params = MODEL_PARAMS[args.model]
n_layers = params["n_layers"]
# KV cache size per token per layer: 2 (K+V) × n_kv_heads × head_dim × 2 (FP16)
kv_per_token_per_layer = 2 * params["n_kv_heads"] * params["head_dim"] * 2
kv_per_request = kv_per_token_per_layer * n_layers * args.prefill_tokens

# How to split into transfer operations
if args.num_layers_per_transfer == 0:
    # One big transfer per decode GPU
    transfer_size = kv_per_request // args.num_decode_gpus
    num_ops = args.num_decode_gpus
else:
    # Multiple transfers, each covering a group of layers
    layers_per_group = args.num_layers_per_transfer
    num_groups = (n_layers + layers_per_group - 1) // layers_per_group
    transfer_size = kv_per_token_per_layer * layers_per_group * args.prefill_tokens
    num_ops = num_groups * args.num_decode_gpus

MODE = args.mode
BLOCK_SIZE = args.block_size
SWITCH_DELAY = args.switch_delay
MAX_OUTSTANDING = args.max_outstanding
HOST_OVERHEAD = args.host_overhead
LOG = args.log or f"output/llm-{MODE}-{args.model}-t{args.prefill_tokens}-d{args.num_decode_gpus}-b{BLOCK_SIZE}.csv"

# With 4KB blocks, packet count drops 64x vs 64B (e.g., 160MB / 4KB = 40960 packets vs 2.5M)
blocks_per_transfer = (transfer_size + BLOCK_SIZE - 1) // BLOCK_SIZE
total_blocks = num_ops * blocks_per_transfer

import sys
print(f"# Model: {args.model}", file=sys.stderr)
print(f"# KV cache per token per layer: {kv_per_token_per_layer} bytes", file=sys.stderr)
print(f"# Total KV cache: {kv_per_request / (1024*1024):.1f} MB", file=sys.stderr)
print(f"# Transfer size per op: {transfer_size / 1024:.1f} KB", file=sys.stderr)
print(f"# Number of ops: {num_ops}", file=sys.stderr)
print(f"# Block size: {BLOCK_SIZE} bytes", file=sys.stderr)
print(f"# Total blocks: {total_blocks}", file=sys.stderr)
print(f"# Host overhead: {HOST_OVERHEAD} ns (baseline only)", file=sys.stderr)

# Device setup
prefill_gpu = devices.DRAMsim3Interface("PrefillGPU")
prefill_gpu.capacity = 1 << 34  # 16GB
prefill_gpu.start = 0
prefill_gpu.wr_ratio = 0.0  # Read source

sw0 = devices.Switch("Switch-0")
sw0.delay = SWITCH_DELAY
sw1 = devices.Switch("Switch-1")
sw1.delay = SWITCH_DELAY

cfg = utils.Config()
cfg.max_clock = args.max_clock
cfg.log_name = LOG
cfg.log_level = "INFO"

if MODE == "dcs":
    all_devs = [prefill_gpu, sw0, sw1]

    decode_gpus = []
    orchs = []
    for i in range(args.num_decode_gpus):
        dgpu = devices.DRAMsim3Interface(f"DecodeGPU-{i}")
        dgpu.capacity = 1 << 34
        dgpu.start = (1 << 34) + i * (1 << 32)
        dgpu.wr_ratio = 1.0  # Write destination
        decode_gpus.append(dgpu)
        all_devs.append(dgpu)

    # One orchestrator per decode GPU
    for i in range(args.num_decode_gpus):
        if args.num_layers_per_transfer == 0:
            ops_for_this_gpu = 1
            xfer = transfer_size
        else:
            ops_for_this_gpu = num_ops // args.num_decode_gpus
            xfer = transfer_size

        orch = devices.Orchestrator(f"Orch-{i}")
        orch.num_ops = ops_for_this_gpu
        orch.transfer_size = xfer
        orch.src_device = "PrefillGPU"
        orch.dst_device = f"DecodeGPU-{i}"
        orch.src_base_addr = 0
        orch.dst_base_addr = (1 << 34) + i * (1 << 32)
        orch.max_outstanding = MAX_OUTSTANDING
        orch.max_outstanding_reads = MAX_OUTSTANDING
        orch.max_outstanding_writes = MAX_OUTSTANDING
        orch.block_size = BLOCK_SIZE
        orch.pipeline = True
        orchs.append(orch)
        all_devs.append(orch)

    cfg.add_devices(all_devs)

    cfg.connect(sw0, prefill_gpu)
    for orch in orchs:
        cfg.connect(orch, sw0)
    for dgpu in decode_gpus:
        cfg.connect(sw1, dgpu)
    cfg.connect(sw0, sw1)

elif MODE == "baseline":
    blocks_per_op = (transfer_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    total_requests = num_ops * blocks_per_op

    host = devices.Requester("Host")
    host.interleave_type = "stream"
    host.interleave_param = total_requests
    host.q_capacity = MAX_OUTSTANDING
    host.block_size = BLOCK_SIZE
    host.burst_size = 1
    host.issue_delay = HOST_OVERHEAD
    host.coherent = False

    decode_gpus = []
    all_devs = [prefill_gpu, sw0, sw1, host]
    for i in range(args.num_decode_gpus):
        dgpu = devices.DRAMsim3Interface(f"DecodeGPU-{i}")
        dgpu.capacity = 1 << 34
        dgpu.start = (1 << 34) + i * (1 << 32)
        dgpu.wr_ratio = 1.0
        decode_gpus.append(dgpu)
        all_devs.append(dgpu)

    cfg.add_devices(all_devs)

    cfg.connect(host, sw0)
    cfg.connect(sw0, prefill_gpu)
    for dgpu in decode_gpus:
        cfg.connect(sw1, dgpu)
    cfg.connect(sw0, sw1)

print(cfg)
